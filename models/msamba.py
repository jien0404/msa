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

try:
    from mamba_ssm.modules.mamba2_simple import Mamba2Simple
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    Mamba2 = None
    print("Error: Failed to import Mamba2")

from .bert import BertTextEncoder
from einops import repeat
from .almt_layer import Transformer, CrossTransformer, HhyperLearningEncoder
from transformers import RobertaModel, HubertModel, Data2VecAudioModel
from .mamba_block import Block_GLCE
from .transformer import TransformerEncoder

import copy

# from mamba_ssm.modules.mlp import GatedMLP
# from mamba_ssm.modules.mha import MHA

from .mamba_block import Block_sm_v1, TextGuidedFusionBlock, TextGuidedFusionBlock_v2, TextGuidedFusionBlock_v3, Block_ISM
from .almt_layer import HhyperLearningEncoder, CrossTransformer
from .mixer import MambaVisionMixer, BiMambaVisionMixer

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, \
        mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None


class Block(nn.Module):
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
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        if self.use_mlp:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class FusionBlock(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
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

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
            use_checkpoint=False
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
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class Block_new(nn.Module):
    def __init__(
            self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
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
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
        mamba_type="mamba2",
        use_mlp=False,
        block_type="Block",
        seq_len=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    if mamba_type == "mamba":
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    elif mamba_type == "mamba2":
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, headdim=4, **ssm_cfg, **factory_kwargs)
    elif mamba_type == "mixer":
        mixer_cls = partial(MambaVisionMixer, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    elif mamba_type == "bimamba":
        # Vim-style bimamba: single in_proj/out_proj, internal bidirectional scan
        mixer_cls = partial(BiMambaVisionMixer, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        raise ValueError("wrong mamba type, could only be mamba, mamba2, mixer, or bimamba")
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    if block_type == 'Block':
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_mlp=use_mlp,
        )
        block.layer_idx = layer_idx
    elif block_type == 'Transformer':
        block = TransformerEncoder(embed_dim=d_model, num_heads=8, layers=2)
    elif block_type == 'Block_sm_v1':
        block = Block_sm_v1(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_mlp=use_mlp,
        )
        block.layer_idx = layer_idx
    elif block_type == 'Block_GLCE':
        block = Block_GLCE(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_mlp=use_mlp, seq_len=seq_len,
            use_bimamba=(mamba_type != 'bimamba'),
        )
        block.layer_idx = layer_idx
    elif block_type == 'Block_ISM':
        block = Block_ISM(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_mlp=use_mlp, seq_len=seq_len,
            use_bimamba=(mamba_type != 'bimamba'),
        )
        block.layer_idx = layer_idx
    elif block_type == 'TextGuidedFusionBlock':
        block = TextGuidedFusionBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_mlp=use_mlp,
        )
    elif block_type == 'TextGuidedFusionBlock_v2':
        block = TextGuidedFusionBlock_v2(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_mlp=use_mlp,
        )
    elif block_type == 'TextGuidedFusionBlock_v3':
        block = TextGuidedFusionBlock_v3(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_mlp=use_mlp,
        )
    else:
        raise ValueError("Wrong block_type passed.")

    return block


def create_fusion_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
        mamba_type="mamba2",
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    if mamba_type == "mamba":
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    elif mamba_type == "mamba2":
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, headdim=4, **ssm_cfg, **factory_kwargs)
    else:
        raise ValueError("wrong mamba type, could only be mamba or mamba2")
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = FusionBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx

    return block


def create_block_new(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        # ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        # if ssm_layer not in ["Mamba1", "Mamba2"]:
        #     raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        ssm_layer = "Mamba2"
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            bimamba=bimamba,
            headdim=4,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block_new(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class MSAmba_v5_c1(nn.Module):
    def __init__(self, dataset, bert_pretrained='bert-base-uncased', sm_depth=2, use_checkpoint=False,
                 checkpoint_num=0, mamba_type='mamba', cross_modal_fusion=None, fusion_depth=None, sub_loss=False,
                 use_mlp=False, use_con_loss=False, use_roberta=False, sm_block_type='Block'):
        super(MSAmba_v5_c1, self).__init__()

        self.sm_embed_dim = 128
        self.seq_len = 50#50
        self.mamba_type = mamba_type
        self.sm_depth = sm_depth
        self.checkpoint_num = checkpoint_num
        self.use_checkpoint = use_checkpoint
        self.cross_modal_fusion = cross_modal_fusion
        self.fusion_depth = fusion_depth
        self.sub_loss = sub_loss
        self.use_mlp = use_mlp
        self.use_con_loss = use_con_loss
        self.use_roberta = use_roberta
        self.sm_block_type = sm_block_type

        if not self.use_roberta:
            self.text_model = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)
        else:
            self.text_model = RobertaModel.from_pretrained('roberta-base')
        # print(dataset, 11111111111111111)
        # mosi
        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(5, 128)
            self.proj_v0 = nn.Linear(20, 128)
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(74, 128)
            self.proj_v0 = nn.Linear(35, 128)
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(33, 128)
            self.proj_v0 = nn.Linear(709, 128)
        else:
            assert False, "DatasetName must be mosi, mosei or sims."

        bimamba = True
        rms_norm = RMSNorm is not None
        if self.sm_block_type == 'Transformer':
            self.layers_video = nn.ModuleList(
                TransformerEncoder(self.sm_embed_dim, 8, 2) for i in range(sm_depth)
            )
            self.layers_audio = nn.ModuleList(
                TransformerEncoder(self.sm_embed_dim, 8, 2) for i in range(sm_depth)
            )
            self.layers_text = nn.ModuleList(
                TransformerEncoder(self.sm_embed_dim, 8, 2) for i in range(sm_depth)
            )
        elif self.sm_block_type == 'LSTM':
            pass
        else:
            self.layers_video = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type,
                                 use_mlp=self.use_mlp, block_type=sm_block_type, seq_len=51)
                    for i in range(sm_depth)
                ]
            )
            self.layers_audio = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type,
                                 use_mlp=self.use_mlp, block_type=sm_block_type, seq_len=51)
                    for i in range(sm_depth)
                ]
            )
            self.layers_text = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type,
                                 use_mlp=self.use_mlp, block_type=sm_block_type, seq_len=51)
                    for i in range(sm_depth)
                ]
            )

        # self.apply(segm_init_weights)
        # self.layers_text.apply(segm_init_weights)
        # self.layers_video.apply(segm_init_weights)
        # self.layers_audio.apply(segm_init_weights)

        # # mamba init
        self.layers_text.apply(
            partial(
                _init_weights,
                n_layer=self.sm_depth,
                **({}),
            )
        )
        self.layers_audio.apply(
            partial(
                _init_weights,
                n_layer=self.sm_depth,
                **({}),
            )
        )
        self.layers_video.apply(
            partial(
                _init_weights,
                n_layer=self.sm_depth,
                **({}),
            )
        )

        if self.cross_modal_fusion == 'videomamba':
            self.fusion_layer = nn.ModuleList([
                create_block(self.sm_embed_dim * 3, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                             residual_in_fp32=True,
                             fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type)
                for i in range(self.fusion_depth)
            ])
            self.fusion_layer.apply(
                partial(
                    _init_weights,
                    n_layer=self.fusion_depth,
                    **({}),
                )
            )
            self.fusion_mlp = nn.ModuleList([
                nn.Linear(128 * 3, 128 * 3, bias=False) for i in range(self.fusion_depth)
            ])
            self.fusion_layer_norm = nn.ModuleList([
                nn.LayerNorm(128 * 3, eps=1e-5) for i in range(self.fusion_depth)
            ])
        elif self.cross_modal_fusion == 'dual_modal_concat':
            self.fusion_layer_ta = nn.ModuleList([
                create_fusion_block(self.sm_embed_dim * 2)
            ])
            # self.fusion_layer_tv = nn.ModuleList([])
            pass
        elif self.cross_modal_fusion == 'cross_mamba':
            # self.fusion_layer_cross = nn.ModuleList(
            #     [
            #         create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
            #                      residual_in_fp32=True,
            #                      fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type)
            #         for i in range(self.fusion_depth)
            #     ]
            # )
            self.fusion_layer_T = create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                               residual_in_fp32=True,
                                               fused_add_norm=True, bimamba=bimamba, layer_idx=0, mamba_type=mamba_type)
            self.fusion_layer_A = create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                               residual_in_fp32=True,
                                               fused_add_norm=True, bimamba=bimamba, layer_idx=0, mamba_type=mamba_type)
            self.fusion_layer_V = create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                               residual_in_fp32=True,
                                               fused_add_norm=True, bimamba=bimamba, layer_idx=0, mamba_type=mamba_type)
        elif self.cross_modal_fusion == 'cross_mamba_multi':
            # bimamba = False
            self.fusion_layer_T = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type)
                    for i in range(self.fusion_depth)
                ]
            )
            self.fusion_layer_A = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type)
                    for i in range(self.fusion_depth)
                ]
            )
            self.fusion_layer_V = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type)
                    for i in range(self.fusion_depth)
                ]
            )
        elif self.cross_modal_fusion == 'text_guided_fusion':
            self.fusion_layers_T = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=False, layer_idx=i, mamba_type='mixer',
                                 block_type='TextGuidedFusionBlock_v2')
                    for i in range(self.fusion_depth)
                ]
            )
            self.fusion_layers_A = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=True, layer_idx=i, mamba_type='mixer',
                                 block_type='TextGuidedFusionBlock_v2')
                    for i in range(self.fusion_depth)
                ]
            )
            self.fusion_layers_V = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=True, layer_idx=i, mamba_type='mixer',
                                 block_type='TextGuidedFusionBlock_v2')
                    for i in range(self.fusion_depth)
                ]
            )
            self.sub_fc_T_cls = nn.Linear(self.sm_embed_dim, 1)
            self.sub_fc_A_cls = nn.Linear(self.sm_embed_dim, 1)
            self.sub_fc_V_cls = nn.Linear(self.sm_embed_dim, 1)
            # self.fusion_layer_AT = nn.ModuleList(
            #     [
            #         create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
            #                      residual_in_fp32=True,
            #                      fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type,
            #                      block_type='TextGuidedFusionBlock')
            #         for i in range(self.fusion_depth)
            #     ]
            # )
        elif self.cross_modal_fusion == 'text_guided_fusion_v3':
            # self.cls_token_at = nn.Parameter(torch.zeros(1, 1, self.sm_embed_dim*2))
            # self.cls_token_vt = nn.Parameter(torch.zeros(1, 1, self.sm_embed_dim * 2))
            self.fusion_layers = nn.ModuleList(
                [
                    create_block(self.sm_embed_dim, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=rms_norm,
                                 residual_in_fp32=True,
                                 fused_add_norm=True, bimamba=bimamba, layer_idx=i, mamba_type=mamba_type,
                                 block_type='TextGuidedFusionBlock_v3')
                    for i in range(self.fusion_depth)
                ]
            )

        self.fc = nn.Linear(128 * 3, 1)
        # self.fc = nn.Linear(128, 1)

        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, self.sm_embed_dim))
        self.pos_embed_audio = nn.Parameter(torch.zeros(1, 50 + 1, self.sm_embed_dim))
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, self.sm_embed_dim))
        self.pos_embed_video = nn.Parameter(torch.zeros(1, 50 + 1, self.sm_embed_dim))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, self.sm_embed_dim))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, 50 + 1, self.sm_embed_dim))

        trunc_normal_(self.pos_embed_text, std=.02)
        trunc_normal_(self.pos_embed_audio, std=.02)
        trunc_normal_(self.pos_embed_video, std=.02)
        #
        # self.apply(
        #     partial(
        #         _init_weights,
        #         n_layer=2,
        #         **({}),
        #         n_residuals_per_layer=1
        #     )
        # )

        # alpha = nn.Softmax()(torch.randn(6*self.fusion_depth))
        # alpha = torch.ones(6*self.fusion_depth)
        # self.alpha = nn.Parameter(alpha)

        if self.sub_loss:
            self.sub_fc_A = nn.Linear(self.sm_embed_dim, 1)
            self.sub_fc_V = nn.Linear(self.sm_embed_dim, 1)
            self.sub_fc_T = nn.Linear(self.sm_embed_dim, 1)
        if self.use_con_loss:
            self.con_fc_A = nn.Linear(self.sm_embed_dim, self.sm_embed_dim)
            self.con_fc_V = nn.Linear(self.sm_embed_dim, self.sm_embed_dim)
            self.con_fc_T = nn.Linear(self.sm_embed_dim, self.sm_embed_dim)

        # self.alpha_proj = nn.ModuleList([
        #     nn.Linear(51, 1)
        #     for i in range(self.fusion_depth)
        # ])

    def forward(self, x_visual, x_audio, x_text, inference_params=None, test_flops=False, get_feat=False):
        b = x_visual.size(0)
        x_text = self.text_model(x_text)


        x_visual = self.proj_v0(x_visual)
        x_audio = self.proj_a0(x_audio)
        x_text = self.proj_l0(x_text)

        cls_token_audio = self.cls_token_audio.expand(x_audio.shape[0], -1, -1)
        cls_token_video = self.cls_token_video.expand(x_audio.shape[0], -1, -1)
        cls_token_text = self.cls_token_text.expand(x_audio.shape[0], -1, -1)
        x_audio = torch.cat((cls_token_audio, x_audio), dim=1) + self.pos_embed_audio
        x_visual = torch.cat((cls_token_video, x_visual), dim=1) + self.pos_embed_video
        x_text = torch.cat((cls_token_text, x_text), dim=1) + self.pos_embed_text

        residual_video, residual_audio, residual_text = None, None, None
        hidden_states_video, hidden_states_audio, hidden_states_text = x_visual, x_audio, x_text
        for idx, layer in enumerate(zip(self.layers_video, self.layers_audio, self.layers_text)):
            layer_video, layer_audio, layer_text = layer

            if self.sm_block_type == 'Transformer' or self.sm_block_type == 'LSTM':
                hidden_states_video = layer_video(hidden_states_video)
                hidden_states_audio = layer_audio(hidden_states_audio)
                hidden_states_text = layer_text(hidden_states_text)
            else:
                if self.use_checkpoint and idx < self.checkpoint_num:
                    hidden_states_video, residual_video = layer_video(
                        hidden_states_video, residual_video, inference_params=inference_params,
                        use_checkpoint=True
                    )
                    hidden_states_audio, residual_audio = layer_audio(
                        hidden_states_audio, residual_audio, inference_params=inference_params,
                        use_checkpoint=True
                    )
                    hidden_states_text, residual_text = layer_text(
                        hidden_states_text, residual_text, inference_params=inference_params,
                        use_checkpoint=True
                    )
                else:
                    hidden_states_video, residual_video = layer_video(
                        hidden_states_video, residual_video, inference_params=inference_params,
                    )
                    hidden_states_audio, residual_audio = layer_audio(
                        hidden_states_audio, residual_audio, inference_params=inference_params,
                    )
                    hidden_states_text, residual_text = layer_text(
                        hidden_states_text, residual_text, inference_params=inference_params,
                    )

        # [BS, SEQ, DIM]

        if self.sub_loss and not test_flops:
            sub_output_A, sub_output_V, sub_output_T = self.sub_fc_A(hidden_states_audio[:, 0, :]), \
                                                       self.sub_fc_V(hidden_states_video[:, 0, :]), \
                                                       self.sub_fc_T(hidden_states_text[:, 0, :])
        else:
            sub_output_A, sub_output_V, sub_output_T = None, None, None

        if self.use_con_loss:
            con_output_A, con_output_V, con_output_T = self.con_fc_A(hidden_states_audio[:, 0, :]), \
                                                       self.con_fc_V(hidden_states_video[:, 0, :]), \
                                                       self.con_fc_T(hidden_states_text[:, 0, :])

        # coupled-or-cross mamba block for cross-modal fusion:
        if self.cross_modal_fusion == 'cross_mamba':
            hidden_states_video, _ = self.fusion_layer_V(hidden_states_video,
                                                         mm_A=[self.alpha[0] * self.fusion_layer_A.mixer.A_log,
                                                               self.alpha[1] * self.fusion_layer_T.mixer.A_log],
                                                         mm_X=[hidden_states_text, hidden_states_audio])
            hidden_states_audio, _ = self.fusion_layer_A(hidden_states_audio,
                                                         mm_A=[self.alpha[2] * self.fusion_layer_V.mixer.A_log,
                                                               self.alpha[3] * self.fusion_layer_T.mixer.A_log],
                                                         mm_X=[hidden_states_text, hidden_states_video])
            hidden_states_text, _ = self.fusion_layer_T(hidden_states_text,
                                                        mm_A=[self.alpha[4] * self.fusion_layer_A.mixer.A_log,
                                                              self.alpha[5] * self.fusion_layer_V.mixer.A_log],
                                                        mm_X=[hidden_states_video, hidden_states_audio])
            x = torch.cat([hidden_states_video[:, 0, :], hidden_states_audio[:, 0, :], hidden_states_text[:, 0, :]],
                          dim=1)

        elif self.cross_modal_fusion == 'cross_mamba_multi':
            residual_text, residual_audio, residual_video = None, None, None
            for idx, layer in enumerate(zip(self.fusion_layer_V, self.fusion_layer_A, self.fusion_layer_T)):
                alpha_idx = idx * 6
                # print('A_log shape: ', self.fusion_layer_A[idx].mixer.A_log.shape)
                # print('h shape: ', hidden_states_audio.shape)
                hidden_states_video, residual_video = self.fusion_layer_V[idx](hidden_states_video, residual_video,
                                                                               mm_A=[
                                                                                   # self.alpha_proj[alpha_idx](torch.mean(hidden_states_audio, dim=-1))*
                                                                                   self.fusion_layer_A[idx].mixer.A_log,
                                                                                   # self.alpha_proj[alpha_idx+1](torch.mean(hidden_states_text, dim=-1))*
                                                                                   self.fusion_layer_T[
                                                                                       idx].mixer.A_log],
                                                                               mm_x=[hidden_states_text,
                                                                                     hidden_states_audio])
                hidden_states_audio, residual_audio = self.fusion_layer_A[idx](hidden_states_audio, residual_audio,
                                                                               mm_A=[
                                                                                   # self.alpha_proj[alpha_idx+2](torch.mean(hidden_states_text, dim=-1))*
                                                                                   self.fusion_layer_V[idx].mixer.A_log,
                                                                                   # self.alpha_proj[alpha_idx+3](torch.mean(hidden_states_video, dim=-1))*
                                                                                   self.fusion_layer_T[
                                                                                       idx].mixer.A_log],
                                                                               mm_x=[hidden_states_text,
                                                                                     hidden_states_video])
                hidden_states_text, residual_text = self.fusion_layer_T[idx](hidden_states_text, residual_text,
                                                                             mm_A=[
                                                                                 # self.alpha_proj[alpha_idx+4](torch.mean(hidden_states_video, dim=-1))*
                                                                                 self.fusion_layer_A[idx].mixer.A_log,
                                                                                 # self.alpha_proj[alpha_idx+5](torch.mean(hidden_states_audio, dim=-1))*
                                                                                 self.fusion_layer_V[
                                                                                     idx].mixer.A_log],
                                                                             mm_x=[hidden_states_video,
                                                                                   hidden_states_audio])
            # residual_late = None
            # hidden_states = torch.cat([hidden_states_video, hidden_states_audio, hidden_states_text], dim=-1)
            # for idx, late_fusion_layer in enumerate(self.late_fusion_layers):
            #     hidden_states, residual_late = late_fusion_layer(hidden_states, residual_late)
            #
            # x = hidden_states[:, 0, :]
            x = torch.cat([hidden_states_video[:, 0, :], hidden_states_audio[:, 0, :], hidden_states_text[:, 0, :]],
                          dim=1)
            # pass
            # print(x.shape)
        elif self.cross_modal_fusion == 'text_guided_fusion':
            residual_T_T, residual_V_T, residual_A_T = None, None, None
            residual_T_A, residual_V_A, residual_A_A = None, None, None
            residual_T_V, residual_V_V, residual_A_V = None, None, None
            preserve_text_token = hidden_states_text[:, 0, :]
            preserve_audio_token = hidden_states_audio[:, 0, :]
            preserve_video_token = hidden_states_video[:, 0, :]
            hidden_states_video_T, hidden_states_video_V, hidden_states_video_A = hidden_states_video, hidden_states_video, hidden_states_video
            hidden_states_audio_T, hidden_states_audio_V, hidden_states_audio_A = hidden_states_audio, hidden_states_audio, hidden_states_audio
            hidden_states_text_T, hidden_states_text_V, hidden_states_text_A = hidden_states_text, hidden_states_text, hidden_states_text

            for idx, layer in enumerate(zip(self.fusion_layers_T, self.fusion_layers_A, self.fusion_layers_V)):
                layer_T, layer_A, layer_V = layer
                hidden_states_text_T, hidden_states_video_T, hidden_states_audio_T, residual_T_T, residual_V_T, residual_A_T, T_cls_token = layer_T(
                    hidden_states_text_T, hidden_states_video_T, hidden_states_audio_T, residual_T=residual_T_T,
                    residual_V=residual_V_T, residual_A=residual_A_T,
                    inference_params=None)
                hidden_states_video_V, hidden_states_text_V, hidden_states_audio_V, residual_V_V, residual_T_V, residual_A_V, V_cls_token = layer_V(
                    hidden_states_video_V, hidden_states_text_V, hidden_states_audio_V, residual_T=residual_V_V,
                    residual_V=residual_T_V, residual_A=residual_A_V,
                    inference_params=None)
                hidden_states_audio_A, hidden_states_video_A, hidden_states_text_A, residual_A_A, residual_V_A, residual_T_A, A_cls_token = layer_A(
                    hidden_states_audio_A, hidden_states_video_A, hidden_states_text_T, residual_T=residual_A_A,
                    residual_V=residual_V_A, residual_A=residual_T_A,
                    inference_params=None)

            if not test_flops:
                sub_output_A_cls, sub_output_V_cls, sub_output_T_cls = self.sub_fc_A_cls(A_cls_token[:, 0, :]), \
                                                           self.sub_fc_V_cls(V_cls_token[:, 0, :]), \
                                                           self.sub_fc_T_cls(T_cls_token[:, 0, :])
            else:
                sub_output_A_cls, sub_output_V_cls, sub_output_T_cls = None, None, None
            x = torch.cat([preserve_video_token+hidden_states_video_T[:, 0, :]+hidden_states_video_A[:, 0, :],
                           preserve_audio_token+hidden_states_audio_T[:, 0, :]+hidden_states_audio_V[:, 0, :],
                           preserve_text_token+hidden_states_text_A[:, 0, :]+hidden_states_text_V[:, 0, :]],
                          dim=1)
            # x = torch.cat([preserve_video_token+hidden_states_video[:, 0, :], preserve_audio_token+hidden_states_audio[:, 0, :], preserve_text_token],
                          # dim=1)
        elif self.cross_modal_fusion == 'text_guided_fusion_v3':
            residual_T, residual_V, residual_A = None, None, None
            preserve_text_token = hidden_states_text[:, 0, :]
            # cls_token_at = self.cls_token_at.expand(x_audio.shape[0], -1, -1)
            # cls_token_vt = self.cls_token_vt.expand(x_audio.shape[0], -1, -1)
            # hidden_states_audio = torch.cat([cls_token_at, hidden_states_audio], dim=1)
            # hidden_states_video = torch.cat([cls_token_vt, hidden_states_video], dim=1)
            for idx, layer in enumerate(self.fusion_layers):
                hidden_states_text, hidden_states_video, hidden_states_audio, residual_T, residual_V, residual_A = layer(
                    hidden_states_text, hidden_states_video, hidden_states_audio, residual_T=residual_T,
                    residual_V=residual_V, residual_A=residual_A,
                    inference_params=None, )
            x = torch.cat([hidden_states_video[:, 0, :], hidden_states_audio[:, 0, :], preserve_text_token],
                          dim=1)
        else:
            # no mamba block for final fusion, just simple concatenation:
            sub_output_A_cls, sub_output_V_cls, sub_output_T_cls = None, None, None
            x = torch.cat([hidden_states_video[:, 0, :], hidden_states_audio[:, 0, :], hidden_states_text[:, 0, :]],
                          dim=1)

        if get_feat:
            return x

        x = self.fc(x)

        if not self.sub_loss and not self.use_con_loss:
            return {'output': x}
        elif self.sub_loss and not self.use_con_loss:
            return {'output': x, 'sub_output_A': sub_output_A, 'sub_output_T': sub_output_T,
                    'sub_output_V': sub_output_V, 'cls_A':sub_output_A_cls, 'cls_V':sub_output_V_cls,
                    'cls_T':sub_output_T_cls}
        elif not self.sub_loss and self.use_con_loss:
            return {'output': x, 'Feature_v': con_output_V, 'Feature_a': con_output_A, 'Feature_t': con_output_T}
        else:
            return {'output': x, 'sub_output_A': sub_output_A, 'sub_output_T': sub_output_T,
                    'sub_output_V': sub_output_V,
                    'Feature_v': con_output_V, 'Feature_a': con_output_A, 'Feature_t': con_output_T}


class CHMFusion(nn.Module):
    """
    Cross-modal Hybrid Mamba (CHM) Fusion — bước ④ trong diagram.

    Ba nhánh BSSM song song:
      • BSSM^AL : Mamba trên Concat[H_a, H_l] (2D) → proj D → F_al + CLS_cross_AL
      • BSSM^VL : Mamba trên Concat[H_v, H_l] (2D) → proj D → F_vl + CLS_cross_VL
      • BSSM^L  : Mamba trên H_l (centralized)      →          F_l  + CLS_central_L

    Sau đó Self-Attention hybrid với inject CLS(H_l) để cross-modal refinement.
    Output: F_al, F_vl, F_l (B, TL, D) và 3 CLS tokens (B, D).
    """

    def __init__(self, dim, token_len, mamba_type='mamba', depth=1):
        super().__init__()
        D  = dim
        TL = token_len
        rms_norm = RMSNorm is not None

        def _mamba_block(d_in, idx=0):
            return create_block(d_in, ssm_cfg=None, norm_epsilon=1e-5,
                                rms_norm=rms_norm, residual_in_fp32=True,
                                fused_add_norm=True, bimamba=True,
                                layer_idx=idx, mamba_type=mamba_type,
                                block_type='Block_ISM', seq_len=TL)

        # BSSM cho từng nhánh (depth layers)
        self.bssm_al = nn.ModuleList([_mamba_block(D * 2, i) for i in range(depth)])
        self.bssm_vl = nn.ModuleList([_mamba_block(D * 2, i) for i in range(depth)])
        self.bssm_l  = nn.ModuleList([_mamba_block(D,     i) for i in range(depth)])

        # projection 2D → D sau BSSM concat
        self.proj_al = nn.Linear(D * 2, D)
        self.proj_vl = nn.Linear(D * 2, D)

        # Self-Attention hybrid (cross-modal refinement)
        # dùng CrossTransformerEncoder từ almt_layer
        self.refine_al = CrossTransformer(
            source_num_frames=TL, tgt_num_frames=TL, dim=D, depth=1, heads=8, mlp_dim=D)
        self.refine_vl = CrossTransformer(
            source_num_frames=TL, tgt_num_frames=TL, dim=D, depth=1, heads=8, mlp_dim=D)

        self.norm_out = nn.LayerNorm(D)

    def _run_stack(self, x, stack):
        residual = None
        for block in stack:
            x, residual = block(x, residual)
        if residual is not None:
            x = x + residual
        return x

    def forward(self, h_hyper, h_l, h_a, h_v):
        """
        h_hyper : (B, TL, D)  — output AHL
        h_l     : (B, TL, D)  — language tokens (h_t_last)
        h_a     : (B, TL, D)  — audio tokens
        h_v     : (B, TL, D)  — video tokens

        Returns:
            F_al, F_vl, F_l         : (B, TL, D)
            cls_cross_AL, cls_cross_VL, cls_central_L : (B, D)
        """
        # ── BSSM^AL : audio-language ──────────────────────────────────────────
        x_al = torch.cat([h_a, h_l], dim=-1)          # (B, TL, 2D)
        x_al = self._run_stack(x_al, self.bssm_al)
        x_al = self.proj_al(x_al)                      # (B, TL, D)

        # ── BSSM^VL : video-language ──────────────────────────────────────────
        x_vl = torch.cat([h_v, h_l], dim=-1)          # (B, TL, 2D)
        x_vl = self._run_stack(x_vl, self.bssm_vl)
        x_vl = self.proj_vl(x_vl)                      # (B, TL, D)

        # ── BSSM^L : language centralized ────────────────────────────────────
        # dùng h_hyper (output AHL) làm "centralized language"
        x_l = self._run_stack(h_hyper, self.bssm_l)    # (B, TL, D)

        # ── inject CLS(H_l) vào 2 nhánh → cross-modal refinement ─────────────
        # CLS token của nhánh L dùng làm query để refine AL và VL
        F_al = self.refine_al(x_l, x_al)[:, 1:]        # (B, TL, D)  bỏ prepended CLS
        F_vl = self.refine_vl(x_l, x_vl)[:, 1:]        # (B, TL, D)
        F_l  = self.norm_out(x_l)                       # (B, TL, D)

        cls_cross_AL  = F_al[:, 0]   # (B, D)
        cls_cross_VL  = F_vl[:, 0]   # (B, D)
        cls_central_L = F_l[:, 0]    # (B, D)

        return F_al, F_vl, F_l, cls_cross_AL, cls_cross_VL, cls_central_L


class MSAmba_ALMT(nn.Module):
    """
    ALMT-Mamba: tích hợp pipeline ALMT vào MSAmba.

    Pipeline (theo diagram):
      ① Modality Embedding : Linear → prepend CLS+8 tokens → (B, 9, 128)
      ② Language Encoder   : Block_ISM × sm_depth (save hidden) → h_t_list + cls_intra_L
      ③ AHL Module         : HhyperLearningEncoder (giữ nguyên từ ALMT)
      ④ CHM Fusion         : BSSM^AL + BSSM^VL + BSSM^L + Self-Attn hybrid
      ⑤ Prediction + Loss  : Concat[CLS_intra_L, CLS_cross_AL, CLS_cross_VL, CLS_central_L]
                             → FC(4D, 1)  |  L_total = L_MSE + λ(L_aux_ISM + L_aux_CHM)
    """

    TOKEN_LEN = 8   # số learnable token sau CLS
    DIM       = 128

    def __init__(self, dataset, bert_pretrained='bert-base-uncased',
                 sm_depth=2, mamba_type='mamba',
                 fusion_layer_depth=2, AHL_depth=3,
                 sub_loss=False, sub_loss_lambda=0.5):
        super().__init__()

        self.sub_loss        = sub_loss
        self.sub_loss_lambda = sub_loss_lambda
        self.AHL_depth       = AHL_depth
        D  = self.DIM
        TL = self.TOKEN_LEN

        # ── ① BERT + Linear projection ───────────────────────────────────────
        self.text_model = BertTextEncoder(
            use_finetune=True, transformers='bert', pretrained=bert_pretrained)

        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(768, D)
            self.proj_a0 = nn.Linear(5,   D)
            self.proj_v0 = nn.Linear(20,  D)
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(768, D)
            self.proj_a0 = nn.Linear(74,  D)
            self.proj_v0 = nn.Linear(35,  D)
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(768, D)
            self.proj_a0 = nn.Linear(33,  D)
            self.proj_v0 = nn.Linear(709, D)
        else:
            raise ValueError("dataset must be mosi / mosei / sims")

        # CLS token + 8 learnable tokens + pos embedding (per modality)
        # seq_len = 50 (aligned_50); total = 1 + TL + 50 = 59
        SEQ = 50
        for mod in ('l', 'a', 'v'):
            setattr(self, f'cls_{mod}',
                    nn.Parameter(torch.zeros(1, 1, D)))
            setattr(self, f'tokens_{mod}',
                    nn.Parameter(torch.zeros(1, TL, D)))
            setattr(self, f'pos_{mod}',
                    nn.Parameter(torch.randn(1, 1 + TL + SEQ, D)))

        # ── ② ISM encoder (per modality) ─────────────────────────────────────
        rms_norm = RMSNorm is not None

        def _make_ism_stack(depth, seq_len):
            return nn.ModuleList([
                create_block(D, ssm_cfg=None, norm_epsilon=1e-5,
                             rms_norm=rms_norm, residual_in_fp32=True,
                             fused_add_norm=True, bimamba=True,
                             layer_idx=i, mamba_type=mamba_type,
                             block_type='Block_ISM', seq_len=seq_len)
                for i in range(depth)
            ])

        # projection ISM: seq_len = 1+TL+SEQ = 59  (CLS+tokens+raw)
        self.ism_proj_l = _make_ism_stack(sm_depth, 1 + TL + SEQ)
        self.ism_proj_a = _make_ism_stack(sm_depth, 1 + TL + SEQ)
        self.ism_proj_v = _make_ism_stack(sm_depth, 1 + TL + SEQ)

        # language encoder ISM: seq_len = TL = 8  (tokens only, no raw seq)
        # save_hidden manually → AHL_depth states needed → depth = AHL_depth-1
        self.ism_lang = _make_ism_stack(AHL_depth - 1, TL)
        self.pos_lang = nn.Parameter(torch.randn(1, TL, D))

        # ── ③ AHL Module ─────────────────────────────────────────────────────
        self.h_hyper      = nn.Parameter(torch.ones(1, TL, D))
        self.h_hyper_layer = HhyperLearningEncoder(
            dim=D, depth=AHL_depth, heads=8, dim_head=16, dropout=0.)

        # ── ④ CHM Fusion ──────────────────────────────────────────────────────
        self.chm_fusion = CHMFusion(
            dim=D, token_len=TL, mamba_type=mamba_type, depth=fusion_layer_depth)

        # ── ⑤ Heads ───────────────────────────────────────────────────────────
        # main: Concat[CLS_intra_L, CLS_cross_AL, CLS_cross_VL, CLS_central_L] → FC(4D, 1)
        self.cls_head = nn.Linear(D * 4, 1)
        if self.sub_loss:
            # L_aux_ISM: từ cls_intra_L, cls_a, cls_v
            self.aux_head_ism = nn.Linear(D, 1)
            # L_aux_CHM: từ 3 CLS tokens của CHM
            self.aux_head_chm = nn.Linear(D, 1)

        trunc_normal_(self.pos_lang, std=.02)
        for mod in ('l', 'a', 'v'):
            trunc_normal_(getattr(self, f'pos_{mod}'), std=.02)

    def _run_ism_proj(self, x_raw, mod):
        """Linear proj → prepend CLS+tokens → ISM stack → trả (cls, tokens)."""
        b = x_raw.size(0)
        x = getattr(self, f'proj_{mod}0')(x_raw)          # (B, 50, 128)
        cls    = getattr(self, f'cls_{mod}').expand(b, -1, -1)   # (B,1,128)
        tokens = getattr(self, f'tokens_{mod}').expand(b, -1, -1)# (B,8,128)
        x = torch.cat([cls, tokens, x], dim=1)             # (B,59,128)
        x = x + getattr(self, f'pos_{mod}')[:, :x.size(1)]

        residual = None
        for block in getattr(self, f'ism_proj_{mod}'):
            x, residual = block(x, residual)

        if residual is not None:
            x = x + residual

        cls_out    = x[:, 0:1]                          # (B,1,128)
        tokens_out = x[:, 1:1+self.TOKEN_LEN]           # (B,8,128)
        return cls_out, tokens_out

    def _run_ism_lang(self, tokens):
        """ISM encoder trên 8 tokens, save hidden → h_t_list (len=AHL_depth)."""
        x = tokens + self.pos_lang[:, :tokens.size(1)]
        hidden_list = [x]
        residual = None
        for block in self.ism_lang:
            x, residual = block(x, residual)
            if residual is not None:
                hidden_list.append(x + residual)
            else:
                hidden_list.append(x)
        # hidden_list: [input, after_block_1, ..., after_block_(AHL_depth-1)]
        # len = AHL_depth
        return hidden_list

    def forward(self, x_visual, x_audio, x_text):
        b = x_visual.size(0)

        x_text = self.text_model(x_text)   # (B,50,768)

        # ── ① + ② projection ISM ─────────────────────────────────────────────
        cls_l, h_l = self._run_ism_proj(x_text,   'l')  # (B,1,128), (B,8,128)
        cls_a, h_a = self._run_ism_proj(x_audio,  'a')
        cls_v, h_v = self._run_ism_proj(x_visual, 'v')

        # ② language encoder → h_t_list cho AHL, cls_intra_L cho aux loss
        h_t_list    = self._run_ism_lang(h_l)                         # list len=AHL_depth
        cls_intra_L = (cls_l.squeeze(1) + h_t_list[-1].mean(1)) / 2  # (B,128)

        # ── ③ AHL ────────────────────────────────────────────────────────────
        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)    # (B,8,128)

        # ── ④ CHM Fusion ──────────────────────────────────────────────────────
        _, _, _, cls_cross_AL, cls_cross_VL, cls_central_L = self.chm_fusion(
            h_hyper, h_t_list[-1], h_a, h_v)

        # ── ⑤ Prediction ──────────────────────────────────────────────────────
        # Concat 4 CLS tokens → FC(4D, 1)
        feat   = torch.cat([cls_intra_L, cls_cross_AL, cls_cross_VL, cls_central_L], dim=-1)  # (B,4D)
        output = self.cls_head(feat)                                    # (B,1)

        result = {'output': output}
        if self.sub_loss:
            # L_aux_ISM: supervision trên từng modality CLS từ ISM projection
            result['sub_output_T'] = self.aux_head_ism(cls_intra_L)         # (B,1)
            result['sub_output_A'] = self.aux_head_ism(cls_a.squeeze(1))    # (B,1)
            result['sub_output_V'] = self.aux_head_ism(cls_v.squeeze(1))    # (B,1)
            # L_aux_CHM: supervision trên 3 CLS tokens của CHM
            result['cls_T'] = self.aux_head_chm(cls_central_L)              # (B,1)
            result['cls_A'] = self.aux_head_chm(cls_cross_AL)               # (B,1)
            result['cls_V'] = self.aux_head_chm(cls_cross_VL)               # (B,1)
        return result


def build_model(opt):
    if opt.datasetName == 'sims':
        l_pretrained = 'bert-base-chinese'
    else:
        l_pretrained = 'bert-base-uncased'

    if opt.project_name == 'MSAmba_v1':
        model = MSAmba_v1(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth)
    # elif opt.project_name == 'MSAmba_ALMT':
    #     model = MSAmba_ALMT(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth)
    elif opt.project_name == 'MSAmba_v2':
        model = MSAmba_v2(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                          mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                          fusion_depth=opt.fusion_depth)
    elif opt.project_name == 'MSAmba_v2_f1':
        model = MSAmba_v2_f1(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth)
    elif opt.project_name == 'MSAmba_v2_f2':
        model = MSAmba_v2_f2(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth)
    elif opt.project_name == 'MSAmba_v2_f3':
        model = MSAmba_v2_f3(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth)
    elif opt.project_name == 'MSAmba_v3':
        model = MSAmba_v3(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                          mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                          fusion_depth=opt.fusion_depth)
    elif opt.project_name == 'MSAmba_v4':  # added learnable adaptive fusion parameter based on v3
        model = MSAmba_v4(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                          mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                          fusion_depth=opt.fusion_depth)
    elif opt.project_name == 'MSAmba_v4_c1':  # added learnable adaptive fusion parameter based on v3
        model = MSAmba_v4_c1(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth, sub_loss=opt.sub_loss, use_mlp=opt.use_mlp)
    elif opt.project_name == 'MSAmba_v4_c2':  # added learnable adaptive fusion parameter based on v3
        model = MSAmba_v4_c2(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth, sub_loss=opt.sub_loss, use_mlp=opt.use_mlp,
                             use_con_loss=opt.use_con_loss, use_roberta=opt.use_roberta)
    elif opt.project_name == 'MSAmba_v4_c3':  # added learnable adaptive fusion parameter based on v3
        model = MSAmba_v4_c3(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth, sub_loss=opt.sub_loss, use_mlp=opt.use_mlp,
                             use_con_loss=opt.use_con_loss, use_roberta=opt.use_roberta)
    elif opt.project_name == 'MSAmba_v4_c4':  # added learnable adaptive fusion parameter based on v3
        model = MSAmba_v4_c4(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth, sub_loss=opt.sub_loss, use_mlp=opt.use_mlp,
                             use_con_loss=opt.use_con_loss, use_roberta=opt.use_roberta)
    elif opt.project_name == 'MSAmba_v4_c5':  # added learnable adaptive fusion parameter based on v3
        model = MSAmba_v4_c5(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth, sub_loss=opt.sub_loss, use_mlp=opt.use_mlp,
                             use_con_loss=opt.use_con_loss, use_roberta=opt.use_roberta,
                             sm_block_type=opt.sm_block_type)
    elif opt.project_name == 'MSAmba_v5_c1':  # added learnable adaptive fusion parameter based on v3
        model = MSAmba_v5_c1(dataset=opt.datasetName, bert_pretrained=l_pretrained, sm_depth=opt.single_modality_depth,
                             mamba_type=opt.mamba_type, cross_modal_fusion=opt.cross_modal_fusion,
                             fusion_depth=opt.fusion_depth, sub_loss=opt.sub_loss, use_mlp=opt.use_mlp,
                             use_con_loss=opt.use_con_loss, use_roberta=opt.use_roberta,
                             sm_block_type=opt.sm_block_type)
    elif opt.project_name.startswith('MSAmba_ALMT'):
        model = MSAmba_ALMT(dataset=opt.datasetName, bert_pretrained=l_pretrained,
                            sm_depth=opt.single_modality_depth,
                            mamba_type=opt.mamba_type,
                            fusion_layer_depth=opt.fusion_layer_depth,
                            AHL_depth=opt.AHL_depth,
                            sub_loss=opt.sub_loss,
                            sub_loss_lambda=opt.sub_loss_lambda)
    else:
        raise ValueError("Wrong project name in opt.")

    return model
