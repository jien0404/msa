#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

from pathlib import Path


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

        print('using mixer')

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """

        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class BiMambaVisionMixer(nn.Module):
    """Vim-style bidirectional Mamba.

    Khác với dùng hai Mamba riêng biệt, BiMambaVisionMixer:
    - Dùng MỘT in_proj duy nhất cho cả hai chiều (→ 2×d_inner)
    - Dùng MỘT out_proj duy nhất nhận tổng hai chiều
    - Mỗi chiều có SSM parameters riêng (A_log, D, x_proj, dt_proj, conv1d)

    So sánh params với 2×MambaVisionMixer:
    - Tiết kiệm: 1 × in_proj (d_model → d_inner) params
    - Kiến trúc đúng với Vim/VideoMamba-style bimamba
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            layer_idx=None,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model   = d_model
        self.d_state   = d_state
        self.d_conv    = d_conv
        self.expand    = expand
        self.d_inner   = int(self.expand * self.d_model)
        self.dt_rank   = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self._dh       = self.d_inner // 2   # shorthand: SSM operates on this many channels

        # Single in_proj — outputs 2×d_inner to cover both directions
        self.in_proj  = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # Single out_proj — takes d_inner (sum of fwd+bwd after gating)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Per-direction SSM parameters
        for tag in ('fwd', 'bwd'):
            dh = self._dh
            # depthwise conv for x branch
            setattr(self, f'conv1d_x_{tag}', nn.Conv1d(
                dh, dh, bias=bool(conv_bias), kernel_size=d_conv,
                groups=dh, **factory_kwargs))
            # depthwise conv for z (gating) branch
            setattr(self, f'conv1d_z_{tag}', nn.Conv1d(
                dh, dh, bias=bool(conv_bias), kernel_size=d_conv,
                groups=dh, **factory_kwargs))
            # input-dependent SSM parameters
            setattr(self, f'x_proj_{tag}',
                    nn.Linear(dh, self.dt_rank + d_state * 2, bias=False, **factory_kwargs))
            dt_proj = nn.Linear(self.dt_rank, dh, bias=True, **factory_kwargs)
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                nn.init.constant_(dt_proj.weight, dt_init_std)
            dt = torch.exp(
                torch.rand(dh, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            dt_proj.bias._no_reinit = True
            setattr(self, f'dt_proj_{tag}', dt_proj)
            # SSM state-space matrix A
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n", d=dh,
            ).contiguous()
            A_log = nn.Parameter(torch.log(A))
            A_log._no_weight_decay = True
            setattr(self, f'A_log_{tag}', A_log)
            # skip connection D
            D = nn.Parameter(torch.ones(dh, device=device))
            D._no_weight_decay = True
            setattr(self, f'D_{tag}', D)

    def _scan(self, xz, tag):
        """SSM scan for one direction.
        xz : (B, d_inner, L)  — x and z channels interleaved
        Returns y_gated : (B, d_inner, L)
        """
        dh = self._dh
        x, z = xz[:, :dh, :], xz[:, dh:, :]
        A = -torch.exp(getattr(self, f'A_log_{tag}').float())
        x = F.silu(F.conv1d(x, getattr(self, f'conv1d_x_{tag}').weight,
                            getattr(self, f'conv1d_x_{tag}').bias,
                            padding='same', groups=dh))
        z = F.silu(F.conv1d(z, getattr(self, f'conv1d_z_{tag}').weight,
                            getattr(self, f'conv1d_z_{tag}').bias,
                            padding='same', groups=dh))
        seqlen = x.shape[-1]
        x_dbl = getattr(self, f'x_proj_{tag}')(rearrange(x, "b d l -> (b l) d"))
        dt, B_ssm, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(getattr(self, f'dt_proj_{tag}')(dt), "(b l) d -> b d l", l=seqlen)
        B_ssm = rearrange(B_ssm, "(b l) s -> b s l", l=seqlen).contiguous()
        C     = rearrange(C,     "(b l) s -> b s l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x, dt, A, B_ssm, C,
            getattr(self, f'D_{tag}').float(),
            z=None,
            delta_bias=getattr(self, f'dt_proj_{tag}').bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        return torch.cat([y, z], dim=1)   # (B, d_inner, L)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, D)  →  (B, L, D)"""
        seqlen = hidden_states.shape[1]

        # Single in_proj → split into forward and backward halves
        xz_all = rearrange(self.in_proj(hidden_states), "b l d -> b d l")
        xz_fwd, xz_bwd = xz_all.chunk(2, dim=1)   # each (B, d_inner, L)

        # Forward scan
        y_fwd = self._scan(xz_fwd, 'fwd')

        # Backward scan: flip sequence → scan → flip back
        y_bwd = self._scan(xz_bwd.flip(-1), 'bwd').flip(-1)

        # Average fwd+bwd (Vim v2 convention: /2 keeps magnitude comparable to unidirectional)
        y = rearrange((y_fwd + y_bwd) / 2, "b d l -> b l d")
        return self.out_proj(y)


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
