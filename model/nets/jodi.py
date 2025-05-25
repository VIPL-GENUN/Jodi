# This file is modified from https://github.com/NVlabs/Sana

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma

import os
import torch
import torch.nn as nn
from timm.models.layers import DropPath

from model.builder import MODELS
from model.nets.modules import GLUMBConv
from model.nets.blocks import (
    Attention,
    CaptionEmbedder,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    PatchEmbedMS,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
    get_2d_sincos_pos_embed,
)
from model.nets.norms import RMSNorm
from model.utils import auto_grad_checkpoint
from utils.dist_utils import get_rank
from utils.import_utils import is_xformers_available
from utils.logger import get_root_logger

_xformers_available = False
if is_xformers_available():
    _xformers_available = True


class JodiBlock(nn.Module):
    """
    A Transformer block with global shared adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        qk_norm=False,
        attn_type="linear",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if attn_type == "flash":  # flash self attention
            self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, **block_kwargs)
        elif attn_type == "vanilla":  # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        elif attn_type == "linear":  # linear self attention
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=mlp_acts,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, r, role, mask=None, HW=None, **kwargs):
        B, L, N, C = x.shape  # L = 1 + num_conditions
        assert t.shape == (B, 6 * C), f"{t.shape} != {(B, 6 * C)}"
        assert r.shape == (B, L, 6 * C), f"{r.shape} != {(B, L, 6 * C)}"
        assert role.shape == (B, L), f"{role.shape} != {(B, L)}"

        adaln_params = [
            (self.scale_shift_table[None] + t.reshape(B, 6, -1) + r[:, i].reshape(B, 6, -1)).chunk(6, dim=1)
            for i in range(L)
        ]

        # Self attention
        # modulate seperately
        z = [t2i_modulate(self.norm1(x[:, i]), adaln_params[i][0], adaln_params[i][1]) for i in range(L)]
        # attention together
        ignore = torch.eq(role, 2).repeat_interleave(N, dim=1)  # (B, L * N)
        z = torch.cat(z, dim=1)  # (B, L * N, C)
        z = self.attn(z, HW=HW, ignore=ignore, block_id=kwargs.get("block_id", None))
        # modulate seperately
        z = z.reshape(B, L, N, C)
        z = [self.drop_path(adaln_params[i][2] * z[:, i]) for i in range(L)]
        z = torch.stack(z, dim=1)  # (B, L, N, C)
        x = x + z

        # Cross attention
        x = x.reshape(B, L * N, C)  # (B, L * N, C)
        x = x + self.cross_attn(x, y, mask)
        x = x.reshape(B, L, N, C)  # (B, L, N, C)

        # Mix-FFN
        # modulate seperately
        z = [t2i_modulate(self.norm2(x[:, i]), adaln_params[i][3], adaln_params[i][4]) for i in range(L)]
        # feedforward separately
        z = [self.mlp(z[i], HW=HW) for i in range(L)]
        # modulate seperately
        z = [self.drop_path(adaln_params[i][5] * z[i]) for i in range(L)]
        z = torch.stack(z, dim=1)  # (B, L, N, C)
        x = x + z
        return x


@MODELS.register_module()
class Jodi(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        drop_path=0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=300,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="linear",
        use_pe=False,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        num_conditions=1,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.class_dropout_prob = class_dropout_prob
        self.pe_interpolation = pe_interpolation
        self.use_pe = use_pe
        self.y_norm = y_norm
        self.fp32_attention = kwargs.get("use_fp32_attention", False)
        self.num_conditions = num_conditions
        self.h = self.w = 0

        # Patch embedding
        kernel_size = patch_embed_kernel or patch_size
        self.x_embedders = nn.ModuleList([
            PatchEmbedMS(patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True)
            for _ in range(1 + num_conditions)
        ])

        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Position embedding (dynamically computed in forward pass)
        self.base_size = input_size // patch_size
        self.pos_embed_ms = None

        # Caption embedding
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)

        # Role embedding
        self.role_embedder = nn.Embedding(3, hidden_size)  # 0: generated, 1: condition, 2: null
        self.role_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Domain embedding
        self.domain_embedding = nn.Parameter(torch.randn(1+num_conditions, hidden_size))

        # Transformer blocks
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                JodiBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=qk_norm,
                    attn_type=attn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_norm=cross_norm,
                )
                for i in range(depth)
            ]
        )

        # Final layer
        self.final_layers = nn.ModuleList([
            T2IFinalLayer(hidden_size, patch_size, self.out_channels)
            for _ in range(1 + num_conditions)
        ])

        # Weights initialization
        self.initialize()

        logger = get_root_logger(os.path.join(config.work_dir, "train_log.log")).warning if config else print
        if get_rank() == 0:
            logger(f"use pe: {use_pe}")
            logger(f"position embed interpolation: {self.pe_interpolation}")
            logger(f"base size: {self.base_size}")
            logger(f"attention type: {attn_type}")
            logger(f"autocast linear attn: {os.environ.get('AUTOCAST_LINEAR_ATTN', False)}")

    def forward(self, x, timestep, y, role, mask=None, clean_x=None, **kwargs):
        """
        Forward pass of Jodi.
        x: (N, 1+K, C, H, W) tensor of spatial inputs (latent representations of images and conditions)
        t: (N, ) tensor of diffusion timesteps
        y: (N, 1, 300, C) tensor of text embeddings
        role: (N, 1+K) tensor of role (0: generated, 1: condition, 2: null)
        clean_x: (N, 1+K, C, H, W) tensor of clean images (optional)
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        role = role.to(torch.long)
        assert x.shape[1] == 1 + self.num_conditions, f"{x.shape[1]} != {1 + self.num_conditions}"
        assert role.shape[1] == 1 + self.num_conditions, f"{role.shape[1]} != {1 + self.num_conditions}"
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        if clean_x is None:
            clean_x = torch.zeros_like(x)

        # Handle cfg stacking: the sampler won't stack model_kwargs (role and clean_x)
        # TODO: should find a better way to handle this
        if role.shape[0] * 2 == x.shape[0]:
            role = role.repeat(2, 1)
        if clean_x.shape[0] * 2 == x.shape[0]:
            clean_x = clean_x.repeat(2, 1, 1, 1, 1)

        # Replace role==1 with clean_x
        if torch.eq(role, 1).any():
            clean_x = clean_x.to(self.dtype)
            x = torch.where(torch.eq(role, 1)[..., None, None, None], clean_x, x)

        # Patch embedding
        x = torch.unbind(x, dim=1)  # tuple of (N, C, H, W)
        x = [x_embedder(x[i]) for i, x_embedder in enumerate(self.x_embedders)]
        x = torch.stack(x, dim=1)  # (N, 1+K, T, D)

        # Add positional embedding
        if self.use_pe:
            if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x[0].shape[1:]:
                self.pos_embed_ms = (
                    torch.from_numpy(
                        get_2d_sincos_pos_embed(
                            self.hidden_size,
                            (self.h, self.w),
                            pe_interpolation=self.pe_interpolation,
                            base_size=self.base_size,
                        )
                    ).unsqueeze(0).to(x[0].device).to(self.dtype)
                )
            x = torch.unbind(x, dim=1)  # tuple of (N, T, D)
            x = [_x + self.pos_embed_ms for _x in x]
            x = torch.stack(x, dim=1)  # (N, 1+K, T, D)

        # Replace role==2 with zero
        x = torch.where(torch.eq(role, 2)[..., None, None], 0., x)

        # Role embedding + Domain embedding
        r = self.role_embedder(role)  # (N, 1+K, D)
        r = r + self.domain_embedding[None]  # (N, 1+K, D)
        r0 = self.role_block(r)  # (N, 1+K, 6 * D)

        # Time embedding
        t = self.t_embedder(timestep)  # (N, D)
        t0 = self.t_block(t)  # (N, 6 * D)

        # Caption embedding
        force_drop_ids = (role[:, 0] == 1)
        if self.training and self.class_dropout_prob > 0:
            cfg_force_drop_ids = (torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob)
            force_drop_ids = force_drop_ids | cfg_force_drop_ids
        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)  # (N, 1, L, D)
        if self.y_norm:
            y = self.attention_y_norm(y)
        if mask is not None:
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1) if mask.shape[0] != y.shape[0] else mask
            mask = mask.squeeze(1).squeeze(1)
            mask = torch.where(force_drop_ids[:, None], 1, mask)
            if _xformers_available:
                y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = mask
        elif _xformers_available:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            raise ValueError(f"Attention type is not available due to _xformers_available={_xformers_available}.")

        # Transformer blocks
        for block_id, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block, x, y, t0, r0, role, y_lens, (self.h, self.w), **kwargs
            )  # (N, 1+K, T, D), support grad checkpoint

        # Final layer
        x = torch.unbind(x, dim=1)  # tuple of (N, T, D)
        x = [final_layer(x[i], t + r[:, i, :]) for i, final_layer in enumerate(self.final_layers)]

        # Unpatchify
        x = [self.unpatchify(_x) for _x in x]  # tuple of (N, out_channels, H, W)
        x = torch.stack(x, dim=1)  # (N, 1+K, out_channels, H, W)

        # Detach role==1 and role==2 to prevent gradient flow
        x = torch.where(torch.eq(role, 1)[..., None, None, None], x.detach(), x)
        x = torch.where(torch.eq(role, 2)[..., None, None, None], x.detach(), x)

        return x

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedders[0].patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    def initialize(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed:
        for x_embedder in self.x_embedders:
            w = x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Initialize role embedding:
        nn.init.normal_(self.role_embedder.weight, std=0.02)
        nn.init.normal_(self.role_block[1].weight, std=0.02)

        # Initialize domain embedding:
        nn.init.normal_(self.domain_embedding, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@MODELS.register_module()
def Jodi_600M_P1_D28(**kwargs):
    return Jodi(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@MODELS.register_module()
def Jodi_600M_P2_D28(**kwargs):
    return Jodi(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


@MODELS.register_module()
def Jodi_600M_P4_D28(**kwargs):
    return Jodi(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


@MODELS.register_module()
def Jodi_1600M_P1_D20(**kwargs):
    # 20 layers, 1648.48M
    return Jodi(depth=20, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)


@MODELS.register_module()
def Jodi_1600M_P2_D20(**kwargs):
    # 28 layers, 1648.48M
    return Jodi(depth=20, hidden_size=2240, patch_size=2, num_heads=20, **kwargs)
