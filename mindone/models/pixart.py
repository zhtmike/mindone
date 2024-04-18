from typing import Any, Dict, Optional, Union

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import XavierUniform, initializer

from mindone.models.modules.pos_embed import _get_2d_sincos_pos_embed_from_grid

from .dit import GELU, LayerNorm, Mlp, PatchEmbed, TimestepEmbedder
from .utils import constant_, exists, normal_, xavier_uniform_


def t2i_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def get_2d_sincos_pos_embed(
    embed_dim: int, nh: int, nw: Optional[int] = None, lewei_scale: float = 1.0, base_size: int = 16
) -> np.ndarray:
    nw = nh if nw is None else nw
    grid_h = np.arange(nh, dtype=np.float32) / (nh / base_size) / lewei_scale
    grid_w = np.arange(nw, dtype=np.float32) / (nw / base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, nh, nw])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


class DropPath(nn.Cell):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(p=drop_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ops.ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class CaptionEmbedder(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        uncond_prob: float = 0.0,
        act_layer=GELU(approximate="tanh"),
        token_num: int = 120,
    ) -> None:
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0.0,
        )
        self.y_embedding = Parameter(ops.randn((token_num, in_channels)) / in_channels**0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, caption: Tensor, force_drop_ids: Optional[Tensor] = None) -> Tensor:
        """
        Drops token embedding to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = ops.rand(caption.shape[0]) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = ops.where(drop_ids[:, None, None], self.y_embedding[None, ...], caption)
        return caption

    def construct(self, caption: Tensor, train: bool, force_drop_ids: Optional[Tensor] = None) -> Tensor:
        if (train and self.uncond_prob > 0) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class SizeEmbedder(TimestepEmbedder):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, hidden_size, has_bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def construct(self, s: Tensor, bs: int) -> Tensor:
        assert s.shape[0] == bs
        s = ops.reshape(s, (-1,))
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size)
        s_emb = self.mlp(s_freq)
        s_emb = ops.reshape(s_emb, (bs, -1))
        return s_emb


class Attention(nn.Cell):
    def __init__(self, dim_head: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.attn_drop = nn.Dropout(p=attn_drop)

    def construct(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        sim = ops.BatchMatMul(transpose_b=True)(q, k) * self.scale

        # use fp32 for exponential inside
        sim = sim.to(ms.float32)
        if exists(mask):
            mask = mask[:, None, None, :]
            sim = ops.masked_fill(sim, ~mask, -ms.numpy.inf)
        attn = ops.softmax(sim, axis=-1).astype(v.dtype)
        attn = self.attn_drop(attn)
        out = ops.matmul(attn, v)
        return out


class SelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if enable_flash_attention:
            raise NotImplementedError("Flash attention is not supported yet.")

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.attention = Attention(head_dim, attn_drop=attn_drop)

    @staticmethod
    def _rearange_out(x: Tensor) -> Tensor:
        # (b, h, n, d) -> (b, n, h*d)
        b, _, n, _ = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, -1))
        return x

    def construct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.num_heads
        B, N, _ = x.shape

        # (b, n, 3*h*d) -> (b, n, 3, h*d)  -> (3, b, n, h*d)
        qkv = self.qkv(x).reshape(B, N, 3, h, -1).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)

        out = self.attention(q, k, v, mask=mask)
        # (b, h, n, d) -> (b, n, h*d)
        out = self._rearange_out(out)

        return self.proj_drop(self.proj(out))


class CrossAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash_attention: bool = False,
    ):
        super().__init__()
        if enable_flash_attention:
            raise NotImplementedError("Flash attention is not supported yet.")

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv_linear = nn.Dense(dim, 2 * dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.attention = Attention(self.head_dim, attn_drop=attn_drop)

    @staticmethod
    def _rearange_out(x: Tensor) -> Tensor:
        # (b, h, n, d) -> (b, n, h*d)
        b, _, n, _ = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, -1))
        return x

    def construct(self, x: Tensor, cond: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.num_heads
        B, N, _ = x.shape

        q = self.q_linear(x).reshape(B, N, h, self.head_dim).permute((0, 2, 1, 3))
        kv = self.kv_linear(cond).reshape(B, -1, 2, h, self.head_dim).permute((2, 0, 3, 1, 4))
        k, v = kv.unbind(0)

        out = self.attention(q, k, v, mask=mask)
        out = self._rearange_out(out)
        return self.proj_drop(self.proj(out))


class T2IFinalLayer(nn.Cell):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Dense(hidden_size, patch_size * patch_size * out_channels, has_bias=True)
        self.scale_shift_table = Parameter(ops.randn((2, hidden_size)) / hidden_size**0.5)
        self.out_channels = out_channels

    def construct(self, x: Tensor, t: Tensor) -> Tensor:
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, axis=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PixArtBlock(nn.Cell):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0, **block_kwargs: Any
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = Parameter(ops.randn(6, hidden_size) / hidden_size**0.5)

    def construct(self, x: Tensor, y: Tensor, t: Tensor, mask_y: Tensor) -> Tensor:
        B = x.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, axis=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.cross_attn(x, y, mask=mask_y)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class PixArt(nn.Cell):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        pred_sigma: bool = True,
        drop_path: float = 0.0,
        caption_channels: int = 4096,
        lewei_scale: float = 1.0,
        model_max_length: int = 120,
        block_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        base_size = input_size // self.patch_size
        pos_embed = get_2d_sincos_pos_embed(
            hidden_size, int(self.x_embedder.num_patches**0.5), lewei_scale=self.lewei_scale, base_size=base_size
        )
        self.pos_embed = Tensor(pos_embed[None, ...], dtype=ms.float32)

        approx_gelu = lambda: GELU(approximate="tanh")
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        drop_path = np.linspace(0, drop_path, depth).tolist()  # stochastic depth decay rule
        self.blocks = nn.CellList(
            [
                PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i], **block_kwargs)
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)
        normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            constant_(block.cross_attn.proj.weight, 0)
            constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        nh, nw = h // self.patch_size, w // self.patch_size
        x = x.reshape((x.shape[0], nh, nw, self.patch_size, self.patch_size, c))
        x = ops.transpose(x, (0, 5, 1, 3, 2, 4))
        imgs = x.reshape((x.shape[0], c, nh * self.patch_size, nw * self.patch_size))
        return imgs

    @ms.jit
    def construct(self, x: Tensor, t: Tensor, y: Tensor, mask_y: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, L, C') tensor of text embeddings
        mask_y: (N, L) tensor of text mask
        """
        _, _, h, w = x.shape
        x = self.x_embedder(x) + self.pos_embed.to(x.dtype)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, L, D)

        for block in self.blocks:
            x = block(x, y, t0, mask_y=mask_y)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, h, w)  # (N, out_channels, H, W)
        return x

    @ms.jit
    def construct_with_cfg(
        self, x: Tensor, t: Tensor, y: Tensor, cfg_scale: Union[Tensor, float], mask_y: Optional[Tensor] = None
    ) -> Tensor:
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, t, y, mask_y=mask_y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)


class PixArtMS(PixArt):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        pred_sigma: bool = True,
        drop_path: float = 0.0,
        caption_channels: int = 4096,
        lewei_scale: float = 1.0,
        model_max_length: int = 120,
        block_kwargs: Dict[str, Any] = {},
    ):
        super(PixArt, self).__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(None, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        approx_gelu = lambda: GELU(approximate="tanh")
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        self.csize_embedder = SizeEmbedder(hidden_size // 3)
        self.ar_embedder = SizeEmbedder(hidden_size // 3)

        drop_path = np.linspace(0, drop_path, depth).tolist()  # stochastic depth decay rule
        self.blocks = nn.CellList(
            [
                PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i], **block_kwargs)
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)
        normal_(self.t_block[1].weight, std=0.02)
        normal_(self.csize_embedder.mlp[0].weight, std=0.02)
        normal_(self.csize_embedder.mlp[2].weight, std=0.02)
        normal_(self.ar_embedder.mlp[0].weight, std=0.02)
        normal_(self.ar_embedder.mlp[2].weight, std=0.02)

        # Initialize caption embedding MLP:
        normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            constant_(block.cross_attn.proj.weight, 0)
            constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)

    @ms.jit
    def construct(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        pos_embed: Tensor,
        csize: Tensor,
        ar: Tensor,
        mask_y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, L, C') tensor of text embeddings
        pos_embed: (N, L') or (1, L') tensor of positional embedding
        csize: (N, 2) tensor of size
        ar: (N, 1) tensor of aspect ratio
        mask_y: (N, L) tensor of text mask
        """
        n, _, h, w = x.shape

        x = self.x_embedder(x) + pos_embed.to(x.dtype)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        csize = self.csize_embedder(csize, n)  # (N, D)
        ar = self.ar_embedder(ar, n)
        t = t + ops.concat([csize, ar], axis=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, L, D)

        for block in self.blocks:
            x = block(x, y, t0, mask_y=mask_y)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, h, w)  # (N, out_channels, H, W)
        return x

    @ms.jit
    def construct_with_cfg(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        pos_embed: Tensor,
        csize: Tensor,
        ar: Tensor,
        cfg_scale: Union[Tensor, float],
        mask_y: Optional[Tensor] = None,
    ) -> Tensor:
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, t, y, pos_embed, csize, ar, mask_y=mask_y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)


def PixArt_XL_2(**kwargs):
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def PixArtMS_XL_2(**kwargs):
    return PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
