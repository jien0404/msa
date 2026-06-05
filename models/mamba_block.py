import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    try:
        from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .mixer import Attention

class Block_sm_v1(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
            use_mlp=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = nn.Linear(dim, dim)

        self.sigmoid = nn.Sigmoid()
        self.sigmoid_proj = nn.Linear(dim, dim)
        self.final_proj = nn.Linear(dim, dim)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
            use_checkpoint=False, mm_A=None, mm_x=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        mm_A = None
        mm_x = None
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        inner_res = self.sigmoid(self.sigmoid_proj(hidden_states))
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        # if self.use_mlp:
        #     hidden_states = self.mlp(hidden_states)

        hidden_states = self.final_proj(hidden_states * inner_res)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class TextGuidedFusionBlock(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
            use_mlp=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer_AT = mixer_cls(dim * 2)
        self.mixer_VT = mixer_cls(dim * 2)
        self.mixer_T = mixer_cls(dim)

        self.norm_AT = norm_cls(dim * 2)
        self.norm_VT = norm_cls(dim * 2)
        self.norm_T = norm_cls(dim)
        self.proj_VT = nn.Linear(dim * 2, dim)
        self.proj_AT = nn.Linear(dim * 2, dim)

        self.drop_path = nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm_AT, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.use_mlp = use_mlp

        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid_proj = nn.Linear(dim, dim)
        # self.final_proj = nn.Linear(dim, dim)

    def forward(
            self, hidden_states_T, hidden_states_V, hidden_states_A, residual_T=None, residual_V=None, residual_A=None,
            inference_params=None, use_checkpoint=False,
    ):

        fused_add_norm_fn = layer_norm_fn
        hidden_states_T, residual_T = fused_add_norm_fn(
            hidden_states_T if residual_T is None else self.drop_path(hidden_states_T),
            self.norm_T.weight,
            self.norm_T.bias,
            residual=residual_T,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm_T.eps,
        )
        if residual_V is not None:
            hidden_states_A = hidden_states_A + residual_A
        if residual_V is not None:
            hidden_states_V = hidden_states_V + residual_V
        residual_V, residual_A = hidden_states_V, hidden_states_A
        hidden_states_AT = torch.cat([hidden_states_A, hidden_states_T], dim=-1)
        hidden_states_VT = torch.cat([hidden_states_V, hidden_states_T], dim=-1)

        hidden_states_AT = fused_add_norm_fn(
            hidden_states_AT,
            self.norm_AT.weight,
            self.norm_AT.bias,
            residual=None,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm_AT.eps,
        )
        hidden_states_VT = fused_add_norm_fn(
            hidden_states_VT,
            self.norm_VT.weight,
            self.norm_VT.bias,
            residual=None,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm_VT.eps,
        )

        hidden_states_AT = self.mixer_AT(hidden_states_AT, inference_params=inference_params)
        hidden_states_VT = self.mixer_VT(hidden_states_VT, inference_params=inference_params)
        hidden_states_T = self.mixer_T(hidden_states_T, inference_params=inference_params)
        T_cls_token = hidden_states_T[:, 0, :].unsqueeze(dim=1)
        hidden_states_AT = self.proj_AT(hidden_states_AT) + T_cls_token
        hidden_states_VT = self.proj_VT(hidden_states_VT) + T_cls_token

        return hidden_states_T, hidden_states_VT, hidden_states_AT, residual_T, residual_V, residual_A

    class TextGuidedFusionBlock(nn.Module):
        def __init__(
                self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
                use_mlp=False,
        ):
            """
            Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

            This Block has a slightly different structure compared to a regular
            prenorm Transformer block.
            The standard block is: LN -> MHA/MLP -> Add.
            [Ref: https://arxiv.org/abs/2002.04745]
            Here we have: Add -> LN -> Mixer, returning both
            the hidden_states (output of the mixer) and the residual.
            This is purely for performance reasons, as we can fuse add and LayerNorm.
            The residual needs to be provided (except for the very first block).
            """
            super().__init__()
            self.residual_in_fp32 = residual_in_fp32
            self.fused_add_norm = fused_add_norm
            self.mixer_AT = mixer_cls(dim * 2)
            self.mixer_VT = mixer_cls(dim * 2)
            self.mixer_T = mixer_cls(dim)

            self.norm_AT = norm_cls(dim * 2)
            self.norm_VT = norm_cls(dim * 2)
            self.norm_T = norm_cls(dim)
            self.proj_VT = nn.Linear(dim * 2, dim)
            self.proj_AT = nn.Linear(dim * 2, dim)

            self.drop_path = nn.Identity()
            if self.fused_add_norm:
                assert RMSNorm is not None, "RMSNorm import fails"
                assert isinstance(
                    self.norm_AT, (nn.LayerNorm, RMSNorm)
                ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            self.use_mlp = use_mlp

            # self.sigmoid = nn.Sigmoid()
            # self.sigmoid_proj = nn.Linear(dim, dim)
            # self.final_proj = nn.Linear(dim, dim)

        def forward(
                self, hidden_states_T, hidden_states_V, hidden_states_A, residual_T=None, residual_V=None,
                residual_A=None,
                inference_params=None, use_checkpoint=False,
        ):

            fused_add_norm_fn = layer_norm_fn
            hidden_states_T, residual_T = fused_add_norm_fn(
                hidden_states_T if residual_T is None else self.drop_path(hidden_states_T),
                self.norm_T.weight,
                self.norm_T.bias,
                residual=residual_T,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm_T.eps,
            )
            if residual_V is not None:
                hidden_states_A = hidden_states_A + residual_A
            if residual_V is not None:
                hidden_states_V = hidden_states_V + residual_V
            residual_V, residual_A = hidden_states_V, hidden_states_A
            hidden_states_AT = torch.cat([hidden_states_A, hidden_states_T], dim=-1)
            hidden_states_VT = torch.cat([hidden_states_V, hidden_states_T], dim=-1)

            hidden_states_AT = fused_add_norm_fn(
                hidden_states_AT,
                self.norm_AT.weight,
                self.norm_AT.bias,
                residual=None,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm_AT.eps,
            )
            hidden_states_VT = fused_add_norm_fn(
                hidden_states_VT,
                self.norm_VT.weight,
                self.norm_VT.bias,
                residual=None,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm_VT.eps,
            )

            hidden_states_AT = self.mixer_AT(hidden_states_AT, inference_params=inference_params)
            hidden_states_VT = self.mixer_VT(hidden_states_VT, inference_params=inference_params)
            hidden_states_T = self.mixer_T(hidden_states_T, inference_params=inference_params)
            T_cls_token = hidden_states_T[:, 0, :].unsqueeze(dim=1)
            hidden_states_AT = self.proj_AT(hidden_states_AT) + T_cls_token
            hidden_states_VT = self.proj_VT(hidden_states_VT) + T_cls_token

            return hidden_states_T, hidden_states_VT, hidden_states_AT, residual_T, residual_V, residual_A

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class TextGuidedFusionBlock_v2(nn.Module):
    """
    Each instance is "anchored" on one primary modality (main) and takes the
    other two (V, A) as context.  The primary modality is processed with a
    prenorm + residual path; the two context modalities get a simpler norm.
    The primary CLS token is added to both context outputs so they are steered
    by the primary modality's representation.

    Returns 7 values:
        hidden_states_main, hidden_states_V, hidden_states_A,
        residual_T, residual_V, residual_A, cls_token   (B, seq, D)
    cls_token[:, 0, :] is used by MSAmba_v5_c1 for sub-loss supervision.
    """

    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
            residual_in_fp32=False, drop_path=0., use_mlp=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.mixer_main = mixer_cls(dim)
        self.mixer_V    = mixer_cls(dim)
        self.mixer_A    = mixer_cls(dim)

        self.norm_main = norm_cls(dim)
        self.norm_V    = norm_cls(dim)
        self.norm_A    = norm_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self, hidden_states_main, hidden_states_V, hidden_states_A,
            residual_T=None, residual_V=None, residual_A=None,
            inference_params=None, use_checkpoint=False,
    ):
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_main, RMSNorm) else layer_norm_fn

        # primary modality: prenorm + residual
        hidden_states_main, residual_T = fused_add_norm_fn(
            hidden_states_main if residual_T is None else self.drop_path(hidden_states_main),
            self.norm_main.weight, self.norm_main.bias,
            residual=residual_T, prenorm=True,
            residual_in_fp32=self.residual_in_fp32, eps=self.norm_main.eps,
        )

        # context modalities: simple residual add then norm (no prenorm path)
        if residual_V is not None:
            hidden_states_V = hidden_states_V + residual_V
        if residual_A is not None:
            hidden_states_A = hidden_states_A + residual_A
        residual_V = hidden_states_V
        residual_A = hidden_states_A

        norm_fn = rms_norm_fn if isinstance(self.norm_V, RMSNorm) else layer_norm_fn
        hidden_states_V = norm_fn(
            hidden_states_V, self.norm_V.weight, self.norm_V.bias,
            residual=None, prenorm=False,
            residual_in_fp32=self.residual_in_fp32, eps=self.norm_V.eps,
        )
        norm_fn = rms_norm_fn if isinstance(self.norm_A, RMSNorm) else layer_norm_fn
        hidden_states_A = norm_fn(
            hidden_states_A, self.norm_A.weight, self.norm_A.bias,
            residual=None, prenorm=False,
            residual_in_fp32=self.residual_in_fp32, eps=self.norm_A.eps,
        )

        hidden_states_main = self.mixer_main(hidden_states_main, inference_params=inference_params)
        hidden_states_V    = self.mixer_V(hidden_states_V,    inference_params=inference_params)
        hidden_states_A    = self.mixer_A(hidden_states_A,    inference_params=inference_params)

        # steer context with primary CLS token
        cls_token = hidden_states_main[:, 0:1, :]   # (B, 1, D)
        hidden_states_V = hidden_states_V + cls_token
        hidden_states_A = hidden_states_A + cls_token

        return hidden_states_main, hidden_states_V, hidden_states_A, residual_T, residual_V, residual_A, hidden_states_main


class TextGuidedFusionBlock_v3(nn.Module):
    """
    Single shared fusion block: text guides both video and audio by adding the
    text CLS token to their outputs.  All three modalities go through independent
    prenorm + Mamba paths.

    Returns 6 values:
        hidden_states_T, hidden_states_V, hidden_states_A,
        residual_T, residual_V, residual_A
    """

    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
            residual_in_fp32=False, drop_path=0., use_mlp=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.mixer_T = mixer_cls(dim)
        self.mixer_V = mixer_cls(dim)
        self.mixer_A = mixer_cls(dim)

        self.norm_T = norm_cls(dim)
        self.norm_V = norm_cls(dim)
        self.norm_A = norm_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self, hidden_states_T, hidden_states_V, hidden_states_A,
            residual_T=None, residual_V=None, residual_A=None,
            inference_params=None, use_checkpoint=False,
    ):
        def _prenorm(x, residual, norm):
            fn = rms_norm_fn if isinstance(norm, RMSNorm) else layer_norm_fn
            return fn(
                x if residual is None else self.drop_path(x),
                norm.weight, norm.bias,
                residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32, eps=norm.eps,
            )

        hidden_states_T, residual_T = _prenorm(hidden_states_T, residual_T, self.norm_T)
        hidden_states_V, residual_V = _prenorm(hidden_states_V, residual_V, self.norm_V)
        hidden_states_A, residual_A = _prenorm(hidden_states_A, residual_A, self.norm_A)

        hidden_states_T = self.mixer_T(hidden_states_T, inference_params=inference_params)
        hidden_states_V = self.mixer_V(hidden_states_V, inference_params=inference_params)
        hidden_states_A = self.mixer_A(hidden_states_A, inference_params=inference_params)

        cls_T = hidden_states_T[:, 0:1, :]   # (B, 1, D)
        hidden_states_V = hidden_states_V + cls_T
        hidden_states_A = hidden_states_A + cls_T

        return hidden_states_T, hidden_states_V, hidden_states_A, residual_T, residual_V, residual_A


class Block_GLCE(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
            use_mlp=False, seq_len=50
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = nn.Linear(dim, dim)

        self.seq_len = seq_len
        self.local_extractor = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.global_extractor = nn.Linear(self.seq_len, self.seq_len)
        self.layer_norm_2 = norm_cls(dim)
        self.mixer_b = mixer_cls(dim)  # backward Mamba for bidirectional scan


    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
            use_checkpoint=False, mm_A=None, mm_x=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states_t = hidden_states.permute(0, 2, 1)
            # print(hidden_states_t.shape)
            hidden_states_t = self.global_extractor(hidden_states_t)
            hidden_states = hidden_states_t.permute(0, 2, 1) + hidden_states + self.local_extractor(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.layer_norm_2.weight,
                self.layer_norm_2.bias,
                residual=None,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.layer_norm_2.eps,
            )
        if use_checkpoint:
            h_fwd = checkpoint.checkpoint(self.mixer,   hidden_states,        inference_params)
            h_bwd = checkpoint.checkpoint(self.mixer_b, hidden_states.flip(1), inference_params).flip(1)
        else:
            h_fwd = self.mixer(hidden_states, inference_params=inference_params)
            h_bwd = self.mixer_b(hidden_states.flip(1), inference_params=inference_params).flip(1)
        hidden_states = h_fwd + h_bwd
        if self.use_mlp:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class Block_ISM(nn.Module):
    """
    Intra-modal Selective Mamba (ISM) block cho MSAmba.
    Interface giống Block_GLCE: forward(hidden_states, residual) → (hidden_states, residual)

    Cấu trúc: Add→Norm→[Global + Local context]→Norm2→Mamba→FFN
      - Global branch : Linear trên chiều seq (nắm bắt global dependency)
      - Local branch  : Conv1d kernel=3 (context cục bộ)
      - Mamba         : SSM trên full sequence
      - FFN           : Pre-norm feed-forward (ISM-style)
    """

    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
            residual_in_fp32=False, drop_path=0., use_mlp=False, seq_len=51,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.norm  = norm_cls(dim)
        self.norm2 = norm_cls(dim)
        self.mixer   = mixer_cls(dim)
        self.mixer_b = mixer_cls(dim)  # backward Mamba for bidirectional scan

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # GLCE-style context extraction
        self.global_extractor = nn.Linear(seq_len, seq_len)
        self.local_extractor  = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        # Pre-norm FFN (ISM-style)
        self.norm_ff = norm_cls(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None,
            inference_params=None, use_checkpoint=False, **kwargs,
    ):
        # ── 1. Add + Norm ─────────────────────────────────────────────────────
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight, self.norm.bias,
                residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps,
            )

        # ── 2. Global + Local context (GLCE) ──────────────────────────────────
        h_global = self.global_extractor(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        h_local  = self.local_extractor(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_states = hidden_states + h_global + h_local

        # second norm before Mamba
        if not self.fused_add_norm:
            hidden_states = self.norm2(hidden_states)
        else:
            fused_fn = rms_norm_fn if isinstance(self.norm2, RMSNorm) else layer_norm_fn
            hidden_states = fused_fn(
                hidden_states, self.norm2.weight, self.norm2.bias,
                residual=None, prenorm=False,
                residual_in_fp32=self.residual_in_fp32, eps=self.norm2.eps,
            )

        # ── 3. Bidirectional Mamba SSM ────────────────────────────────────────
        if use_checkpoint:
            h_fwd = checkpoint.checkpoint(self.mixer,   hidden_states,        inference_params)
            h_bwd = checkpoint.checkpoint(self.mixer_b, hidden_states.flip(1), inference_params).flip(1)
        else:
            h_fwd = self.mixer(hidden_states, inference_params=inference_params)
            h_bwd = self.mixer_b(hidden_states.flip(1), inference_params=inference_params).flip(1)
        hidden_states = h_fwd + h_bwd

        # ── 4. Pre-norm FFN ───────────────────────────────────────────────────
        hidden_states = hidden_states + self.ff(self.norm_ff(hidden_states))

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

