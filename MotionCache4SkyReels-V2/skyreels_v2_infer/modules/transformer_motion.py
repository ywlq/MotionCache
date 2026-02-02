import math
import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.backends.cuda import sdp_kernel
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention

from .attention import flash_attention
import time

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")
# False
DISABLE_COMPILE = True  # get os env

__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, group_idx, token_indices=None, freqs_cache=None):
    """

    Args:
    """
    n, c = x.size(2), x.size(3) // 2
    bs = x.size(0)
    actual_seq_len = x.size(1)

    if token_indices is not None and freqs_cache is not None:
        freqs_i = freqs_cache[token_indices]
    elif token_indices is None:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = grid_sizes.tolist()
        start_f = group_idx * f
        end_f = start_f + f

        freqs_i = torch.cat(
            [
                freqs[0][start_f:end_f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
    else:
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = grid_sizes.tolist()
        start_f = group_idx * f
        end_f = start_f + f

        freqs_full = torch.cat(
            [
                freqs[0][start_f:end_f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        freqs_i = freqs_full[token_indices]

    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, actual_seq_len, n, -1, 2))

    # apply rotary embedding
    x = torch.view_as_real(x * freqs_i).flatten(3)

    return x


def build_freqs_cache(freqs, grid_sizes, group_idx):
    """


    Args:

    Returns:
    """
    f, h, w = grid_sizes.tolist()
    c = freqs.size(1)
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    start_f = group_idx * f
    end_f = start_f + f

    freqs_cache = torch.cat(
        [
            freqs_split[0][start_f:end_f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)

    return freqs_cache


@torch.compile(dynamic=True, disable=DISABLE_COMPILE)
def fast_rms_norm(x, weight, eps):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x = x.type_as(x) * weight
    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return fast_rms_norm(x, self.weight, self.eps)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, layer_id=0, num_layers=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self._flag_ar_attention = False

        self.layer_id = layer_id
        self.num_layers = num_layers

        self.register_buffer('kv_cache', None)  # [B, L, nH, d]
        self.register_buffer('k_cache_even', None)
        self.register_buffer('v_cache_even',  None)
        self.register_buffer('k_cache_odd', None)
        self.register_buffer('v_cache_odd',  None)
        self.register_buffer('k_cache', None)
        self.register_buffer('v_cache', None)

        # {group_idx: freqs_cache_tensor}
        self.freqs_cache_dict = {}

    def set_ar_attention(self):
        self._flag_ar_attention = True

    def _alloc_kv(self, total_tokens, batch_size, device, dtype):
        return torch.zeros(
            batch_size,
            total_tokens,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_and_return_kv(self, q, k, v, cond_flag, group_idx, group_size, grid_hw, num_groups, batch_size,
                              update_mask_per_group_list=None): # , kv_cluster=None): # , use_kvrange: bool = False, use_rkv: bool = False):
        """
        group_idx : 0..4
        group_size: 5
        grid_hw   : H*W
        """
        total_tokens   = num_groups *group_size* grid_hw          # 25×P
        token_per_grp  = group_size * grid_hw  # 5×P
        start = group_idx * token_per_grp
        end   = start + k.size(1)

        buf_k = self.k_cache_even if cond_flag else self.k_cache_odd
        buf_v = self.v_cache_even if cond_flag else self.v_cache_odd

        if buf_k is None and buf_v is None:
            buf_k = self._alloc_kv(total_tokens, batch_size, k.device, k.dtype)
            buf_v = self._alloc_kv(total_tokens, batch_size, v.device, v.dtype)
            
        buf_k[:,start:end] = k.detach()
        buf_v[:,start:end] = v.detach()

        if cond_flag:
            self.k_cache_even = buf_k
            self.v_cache_even = buf_v
        else:
            self.k_cache_odd  = buf_k
            self.v_cache_odd  = buf_v

        # if not use_kvrange and not use_rkv:
        k_full = buf_k[:, :end]
        v_full = buf_v[:, :end]
        return k_full, v_full
    
    def forward(self, x, grid_sizes, freqs, block_mask, group_idx, cond_flag, num_groups,
                update_mask_per_group_list=None): # , kv_cluster=None): # , use_kvrange: bool = False, use_rkv: bool = False):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        # flop_count
        x = x.to(self.q.weight.dtype)
        q, k, v = qkv_fn(x)

        if not self._flag_ar_attention:
            q = rope_apply(q, grid_sizes, freqs, group_idx)
            k = rope_apply(k, grid_sizes, freqs, group_idx)
            #------------
            group_size = grid_sizes[0]
            grid_hw    = grid_sizes[1] * grid_sizes[2]
            k_full, v_full = self._update_and_return_kv(
                q, k, v, cond_flag, group_idx, group_size, grid_hw, num_groups, batch_size=b,
                update_mask_per_group_list=update_mask_per_group_list,
            )
            #------------
            x = flash_attention(q=q, k=k_full, v=v_full, window_size=self.window_size)
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

            with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                x = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=block_mask
                    )
                    .transpose(1, 2)
                    .contiguous()
                )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def forward_selective(self, x, important_indices_tensor, grid_sizes, freqs, group_idx, cond_flag, num_groups,
                         update_mask_per_group_list=None):
        """
        Args:
        """
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.head_dim

        token_indices_tensor = important_indices_tensor

        x = x.to(self.q.weight.dtype)
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        cache_key = group_idx
        if cache_key not in self.freqs_cache_dict:
            self.freqs_cache_dict[cache_key] = build_freqs_cache(freqs, grid_sizes, group_idx)
        freqs_cache = self.freqs_cache_dict[cache_key]

        q = rope_apply(q, grid_sizes, freqs, group_idx, token_indices=token_indices_tensor, freqs_cache=freqs_cache)
        k = rope_apply(k, grid_sizes, freqs, group_idx, token_indices=token_indices_tensor, freqs_cache=freqs_cache)

        group_size = grid_sizes[0]
        grid_hw = grid_sizes[1] * grid_sizes[2]

        buf_k = self.k_cache_even if cond_flag else self.k_cache_odd
        buf_v = self.v_cache_even if cond_flag else self.v_cache_odd

        token_per_grp = group_size * grid_hw
        total_tokens = num_groups * token_per_grp

        if buf_k is None and buf_v is None:
            buf_k = self._alloc_kv(total_tokens, b, k.device, k.dtype)
            buf_v = self._alloc_kv(total_tokens, b, v.device, v.dtype)

        start = group_idx * token_per_grp
        global_indices = start + token_indices_tensor

        # buf_k[:, global_indices] = k.detach()
        # buf_v[:, global_indices] = v.detach()

        # indices_expanded = global_indices.view(1, -1, 1, 1).expand(b, -1, n, d)
        # buf_k.scatter_(1, indices_expanded, k)
        # buf_v.scatter_(1, indices_expanded, v)

        k_src = k.detach().contiguous()
        v_src = v.detach().contiguous()
        buf_k.index_copy_(1, global_indices, k_src)
        buf_v.index_copy_(1, global_indices, v_src)

        if cond_flag:
            self.k_cache_even = buf_k
            self.v_cache_even = buf_v
        else:
            self.k_cache_odd = buf_k
            self.v_cache_odd = buf_v

        end = start + token_per_grp
        k_full = buf_k[:, :end]
        v_full = buf_v[:, :end]

        x = flash_attention(q=q, k=k_full, v=v_full, window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img)
        # compute attention
        x = flash_attention(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


def mul_add(x, y, z):
    return x.float() + y.float() * z.float()


def mul_add_add(x, y, z):
    return x.float() * (1 + y) + z


mul_add_compile = torch.compile(mul_add, dynamic=True, disable=DISABLE_COMPILE)
mul_add_add_compile = torch.compile(mul_add_add, dynamic=True, disable=DISABLE_COMPILE)


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        layer_id=0,
        num_layers=0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        self.layer_id = layer_id
        self.num_layers = num_layers

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, layer_id, num_layers) 
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)


    def set_ar_attention(self):
        self.self_attn.set_ar_attention()

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        block_mask,
        group_idx, 
        cond_flag,
        num_groups,
        update_mask_per_group_list=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        if e.dim() == 3:
            modulation = self.modulation  # 1, 6, dim
            with amp.autocast("cuda", dtype=torch.float32):
                e = (modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)  # 1, 6, 1, dim
            with amp.autocast("cuda", dtype=torch.float32):
                e = (modulation + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]

        # self-attention
        out = mul_add_add_compile(self.norm1(x), e[1], e[0])

        y = self.self_attn(
            out, grid_sizes, freqs, block_mask, group_idx, cond_flag, num_groups,
            update_mask_per_group_list=update_mask_per_group_list,
        )

        with amp.autocast("cuda", dtype=torch.float32):
            x = mul_add_compile(x, y, e[2])

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            dtype = context.dtype
            x = x + self.cross_attn(self.norm3(x.to(dtype)), context)
            y = self.ffn(mul_add_add_compile(self.norm2(x), e[4], e[3]).to(dtype))
            with amp.autocast("cuda", dtype=torch.float32):
                x = mul_add_compile(x, y, e[5])
            return x

        x = cross_attn_ffn(x, context, e)
        return x.to(torch.bfloat16)

    def forward_selective(
        self,
        x,
        important_indices_tensor,
        e,
        grid_sizes,
        freqs,
        context,
        block_mask,
        group_idx, 
        cond_flag,
        num_groups,
        update_mask_per_group_list=None,
    ):
        """
        
        Args:
        """
        r"""
        Args:
            x(Tensor): Shape [B, N_important, C]
            important_indices_tensor: Tensor[int] token indices in full sequence
        """

        if e.dim() == 3:
            modulation = self.modulation  # 1, 6, dim
            with amp.autocast("cuda", dtype=torch.float32):
                e = (modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)  # 1, 6, 1, dim
            with amp.autocast("cuda", dtype=torch.float32):
                e = (modulation + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]

        n_tokens = important_indices_tensor.size(0)
        out = mul_add_add_compile(self.norm1(x), e[1][:, :n_tokens, :], e[0][:, :n_tokens, :])
        y = self.self_attn.forward_selective(
            out, important_indices_tensor, grid_sizes, freqs, group_idx, cond_flag, num_groups,
            update_mask_per_group_list=update_mask_per_group_list,
        )

        with amp.autocast("cuda", dtype=torch.float32):
            x = mul_add_compile(x, y, e[2][:, :n_tokens, :])

        def cross_attn_ffn(x, context, e, n_tokens):
            dtype = context.dtype
            x = x + self.cross_attn(self.norm3(x.to(dtype)), context)
            y = self.ffn(mul_add_add_compile(self.norm2(x), e[4][:, :n_tokens, :], e[3][:, :n_tokens, :]).to(dtype))
            with amp.autocast("cuda", dtype=torch.float32):
                x = mul_add_compile(x, y, e[5][:, :n_tokens, :])
            return x

        x = cross_attn_ffn(x, context, e, n_tokens)
        return x.to(torch.bfloat16)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        with amp.autocast("cuda", dtype=torch.float32):
            if e.dim() == 2:
                modulation = self.modulation  # 1, 2, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)

            elif e.dim() == 3:
                modulation = self.modulation.unsqueeze(2)  # 1, 2, seq, dim
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
                e = [ei.squeeze(1) for ei in e]
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        inject_sample_info=False,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = 1
        self.flag_causal_attention = False
        self.block_mask = None

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, layer_id=i, num_layers=num_layers)
                for i in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False

        self.cpu_offloading = False

        self.inject_sample_info = inject_sample_info
        # initialize weights
        self.init_weights()

        self.group_size = 5
        self.num_groups = 5
        self.overlap = False
        self.overlap_frames = 0
        self.latent_width = 0
        self.latent_height = 0
        self.cnt_even = None
        self.cnt_odd = None
        self.cnt = 0
        self.inference_steps = 0
        self.optimize = True

        # =============== Distance Visualization Start ===============
        self.enable_distance_visualization = False

        self.distance_n_classes = 30
        self.distance_output_dir = "./result/distance_viz"
        self.distance_metric = "L1"
        self.distance_binning_method = "quantile"
        self.distance_save_steps = None
        self.current_niter = 0
        # =============== Distance Visualization End =================

        # =============== Residual Difference Visualization Start =================
        self.enable_residual_diff_visualization = False
        self.residual_diff_output_dir = "./result/residual_diff_viz"
        self.residual_diff_metric = "L1"
        self.residual_diff_save_steps = None
        # =============== Residual Difference Visualization End =================

        # =============== Selective Token Visualization Start =================
        self.enable_selective_viz = False
        self.selective_viz_output_dir = "./result/selective_viz"
        # =============== Selective Token Visualization End ================

        self.previous_output_even = {}  # {group_idx: Tensor [B, L, C]}
        self.previous_output_odd = {}

        self.previous_residual_even = {}  # {group_idx: Tensor [B, L, C]}
        self.previous_residual_odd = {}

        self.previous_e0_g_even = {}  # {group_idx: Tensor [B, 6, L, C]}
        self.previous_e0_g_odd = {}

        self.teacache_coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]

        # =============== Print Control Start =================
        self.print_only_group0 = True
        # =============== Print Control End =================

        # =============== Token-wise Cache Start =================
        self.token_cache_enabled = False
        self.token_cache_threshold = 0.10
        self.token_cache_warmup_steps = 5
        self.weight_min = 0.5
        self.weight_max = 2.0
        self.token_phase1_update_count = 3
        self.token_weight_gamma = 1.0

        self.token_accumulated_even = {}  # {group_idx: Tensor[L]}
        self.token_accumulated_odd = {}

        self.token_weights_even = {}  # {group_idx: Tensor[L]}
        self.token_weights_odd = {}

        self.token_phase1_counter_even = {}
        self.token_phase1_counter_odd = {}
        self.token_phase1_done_even = {}
        self.token_phase1_done_odd = {}

        self.prev_group_last_frame_even = {}  # {group_idx: Tensor [B, H*W, C]}
        self.prev_group_last_frame_odd = {}

        self.use_frame_diff_weight = False
        self.frame_diff_viz_enabled = False
        self.frame_diff_viz_output_dir = "./result/frame_diff_viz"

        self.weight_smooth_grid_size = 1
        # =============== Token-wise Cache End =================

        # =============== Static Weight Mask Start =================
        self.use_static_weight = False
        self.static_weight_value = 0.3
        # static_weight_regions = [((h1, w1), (h2, w2)), ((h1, w1), (h2, w2)), ...]
        self.static_weight_regions = [((0.0, 0.0), (0.5, 1.0))]
        # =============== Static Weight Mask End =================

        # =============== Input Diff Weight Start =================
        self.use_input_diff_weight = False
        self.input_diff_viz_enabled = False
        self.input_diff_viz_output_dir = "./result/input_diff_viz"

        self.prev_group_last_frame_input_even = {}  # {group_idx: Tensor [B, H*W, C]}
        self.prev_group_last_frame_input_odd = {}
        # =============== Input Diff Weight End =================

        # =============== Weight Normalization Mode =================
        self.weight_norm_mode = "mean"
        self.weight_floor = 0.3
        # =============== Weight Normalization Mode End =================

        # =============== Temporal Consistency Constraint =================
        self.use_temporal_consistency = False
        self.temporal_consistency_threshold = 0.5
        self.first_frame_full_weight = False
        self.group0_first_frame_mode = "ones"
        # =============== Temporal Consistency Constraint End =================

        # =============== Minimum Update Ratio =================
        self.min_update_ratio = 0.4
        # =============== Minimum Update Ratio End =================

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def zero_init_i2v_cross_attn(self):
        print("zero init i2v cross attn")
        for i in range(self.num_layers):
            self.blocks[i].cross_attn.v_img.weight.data.zero_()
            self.blocks[i].cross_attn.v_img.bias.data.zero_()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21, frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask



    def forward(self, x, t, context, update_mask_i ,clip_fea=None, y=None, fps=None):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        #-----------------
        group_size      = self.group_size   
        num_groups      = self.num_groups 
        overlap         = self.overlap           
        overlap_frames  = self.overlap_frames
        # update_mask_per_group = update_mask_i.view(-1, group_size)
        update_mask_per_group = update_mask_i.view(num_groups, group_size).any(dim=1)
        # should_forward_groupe = active_all_before_last = torch.flip(torch.cummax(torch.flip(active_this_step, dims=[0]), dim=0)[0], dims=[0])
        update_mask_per_group_list = [False]*num_groups
        for indx in range(num_groups):
            if update_mask_per_group[indx]==True:
                update_mask_per_group_list[indx] = True
        should_forward_groupe = [False]*num_groups
        for indx in range(num_groups-1, -1, -1):
            if update_mask_per_group_list[indx]==True:
                last_true = indx
                break
        for j in range(last_true+1):
            should_forward_groupe[j] = True

        #------------------------------------------------
        if self.optimize:
            for g in range(num_groups):
                if should_forward_groupe[g]:
                    cnt_vec = self.cnt_even if (self.cnt % 2 == 0) else self.cnt_odd
                    if cnt_vec[g] >= self.inference_steps:
                        should_forward_groupe[g] = False
            if self.overlap:
                if self.cnt <= 1:
                    should_forward_groupe[0] = True
                else:
                    should_forward_groupe[0] = False
        else:
            print('not using optimize')
        #------------------------------------------------

        if y is not None:
            x = torch.cat([x, y], dim=1)

        x = self.patch_embedding(x) # [1, 16, 25, 68, 120] -> [1, 1536, 25, 34, 60]
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)

        #-----------------
        self.latent_width = grid_sizes[2]
        self.latent_height = grid_sizes[1]
        token_per_frame = self.latent_width * self.latent_height
        token_per_group = group_size * token_per_frame
        #-----------------

        x = x.flatten(2).transpose(1, 2) # 1*51000*1536

        if self.flag_causal_attention:
            frame_num = grid_sizes[0]
            height = grid_sizes[1]
            width = grid_sizes[2]
            block_num = frame_num // self.num_frame_per_block
            range_tensor = torch.arange(block_num).view(-1, 1)
            range_tensor = range_tensor.repeat(1, self.num_frame_per_block).flatten()
            casual_mask = range_tensor.unsqueeze(0) <= range_tensor.unsqueeze(1)  # f, f
            casual_mask = casual_mask.view(frame_num, 1, 1, frame_num, 1, 1).to(x.device)
            casual_mask = casual_mask.repeat(1, height, width, 1, height, width)
            casual_mask = casual_mask.reshape(frame_num * height * width, frame_num * height * width)
            self.block_mask = casual_mask.unsqueeze(0).unsqueeze(0)

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            if t.dim() == 2:
                b, f = t.shape
                _flag_df = True
            else:
                _flag_df = False

            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(self.patch_embedding.weight.dtype) # torch.Size([1, 51000, 1536])
            )  # b, dim

            e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim  torch.Size([1, 6, 51000, 1536])


            if self.inject_sample_info:
                fps = torch.tensor(fps, dtype=torch.long, device=device)

                fps_emb = self.fps_embedding(fps).float()
                if _flag_df:
                    e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)

                else:
                    e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

            if _flag_df:
                e = e.view(b, f, 1, 1, self.dim)
                e0 = e0.view(b, f, 1, 1, 6, self.dim)
                e = e.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1).flatten(1, 3)
                e0 = e0.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).flatten(1, 3)
                e0 = e0.transpose(1, 2).contiguous()

            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        # text_embeddingx = context
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        x_chunks = torch.chunk(x,  num_groups, dim=1)
        e0_chunks = torch.chunk(e0, num_groups, dim=2)

        cond_flag = (self.cnt % 2 == 0)

        out_chunks = [torch.zeros_like(x_g) for x_g in x_chunks]
        
        for g, (x_g, e0_g) in enumerate(zip(x_chunks, e0_chunks)):
            if should_forward_groupe[g]==True:
                grid_sizes[0] =  group_size
                cnt_vec = self.cnt_even if cond_flag else self.cnt_odd
                step_num = cnt_vec[g]

                # ==================== Token Cache Logic Start ====================
                if self.token_cache_enabled and update_mask_per_group_list[g]:
                    L = x_g.size(1)
                    device = x_g.device

                    token_accumulated_dict = self._get_token_accumulated(cond_flag)
                    token_weights_dict = self._get_token_weights(cond_flag)
                    token_phase1_counter_dict = self._get_token_phase1_counter(cond_flag)
                    token_phase1_done_dict = self._get_token_phase1_done(cond_flag)
                    prev_output_dict = self._get_previous_output(cond_flag)
                    residual_dict = self._get_residual_dict(cond_flag)
                    previous_e0_g_dict = self._get_previous_e0_g(cond_flag)

                    branch = "cond" if cond_flag else "uncond"

                    if step_num < self.token_cache_warmup_steps:
                        ori_g = x_g.clone()

                        forward_start_time = time.time()
                        kwargs = dict(
                            e=e0_g,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            block_mask=self.block_mask,
                            group_idx=g,
                            cond_flag=cond_flag,
                            num_groups=num_groups,
                            update_mask_per_group_list=update_mask_per_group_list,
                        )
                        for block in self.blocks:
                            x_g = block(x_g, **kwargs)
                        forward_end_time = time.time()
                        forward_time = forward_end_time - forward_start_time

                        prev_output_dict[g] = x_g.clone()
                        residual_dict[g] = x_g - ori_g
                        previous_e0_g_dict[g] = e0_g.clone()

                        if g not in token_accumulated_dict:
                            token_accumulated_dict[g] = torch.zeros(L, device=device, dtype=torch.float32)

                        if g not in token_phase1_counter_dict:
                            token_phase1_counter_dict[g] = 0
                            token_phase1_done_dict[g] = False

                        if (g == 0 if self.print_only_group0 else True) and cond_flag:
                            print(f"  Group {g} step {step_num} ({branch}) - Token Warmup, Time: {forward_time:.3f}s")

                    elif step_num == self.token_cache_warmup_steps:
                        ori_g = x_g.clone()

                        forward_start_time = time.time()
                        kwargs = dict(
                            e=e0_g,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            block_mask=self.block_mask,
                            group_idx=g,
                            cond_flag=cond_flag,
                            num_groups=num_groups,
                            update_mask_per_group_list=update_mask_per_group_list,
                        )
                        for block in self.blocks:
                            x_g = block(x_g, **kwargs)
                        forward_end_time = time.time()
                        forward_time = forward_end_time - forward_start_time

                        prev_output_dict[g] = x_g.clone()
                        residual_dict[g] = x_g - ori_g
                        previous_e0_g_dict[g] = e0_g.clone()

                        token_accumulated_dict[g] = torch.zeros(L, device=device, dtype=torch.float32)

                        token_phase1_counter_dict[g] = 0
                        if self.token_phase1_update_count == 0:
                            token_phase1_done_dict[g] = True
                            weights = self._compute_weights_from_frame_diff(x_g, grid_sizes, g, cond_flag)
                            token_weights_dict[g] = weights
                        else:
                            token_phase1_done_dict[g] = False

                        if (g == 0 if self.print_only_group0 else True) and cond_flag:
                            print(f"  Group {g} step {step_num} ({branch}) - Token First step after warmup, Time: {forward_time:.3f}s")

                    else:
                        use_differentiated_weights = token_phase1_done_dict.get(g, False)

                        if use_differentiated_weights:
                            weights = token_weights_dict[g]
                        else:
                            weights = torch.ones(L, device=device, dtype=torch.float32)

                        prev_e0_g = previous_e0_g_dict[g]
                        if self.token_distance_mode == "global":
                            global_dist = self._compute_global_distance(e0_g, prev_e0_g)
                            distance_per_token = torch.full((L,), global_dist, device=device, dtype=torch.float32)
                        else:
                            distance_per_token = self._compute_per_token_distance(e0_g, prev_e0_g)

                        accumulated = token_accumulated_dict[g]
                        weighted_dist = distance_per_token * weights
                        accumulated = accumulated + weighted_dist

                        need_update_mask = accumulated >= self.token_cache_threshold

                        if self.use_temporal_consistency:
                            H_val = grid_sizes[1].item() if hasattr(grid_sizes[1], 'item') else grid_sizes[1]
                            W_val = grid_sizes[2].item() if hasattr(grid_sizes[2], 'item') else grid_sizes[2]
                            spatial_size = H_val * W_val
                            # need_update_mask shape: [L] where L = group_size * H * W
                            mask_reshaped = need_update_mask.view(group_size, spatial_size)
                            spatial_update_ratio = mask_reshaped.float().mean(dim=0)
                            consistent_mask = spatial_update_ratio > self.temporal_consistency_threshold  # [H*W]
                            need_update_mask = consistent_mask.unsqueeze(0).expand(group_size, -1).reshape(-1)

                        tokens_to_update = torch.where(need_update_mask)[0]

                        num_to_update = len(tokens_to_update)
                        update_ratio = num_to_update / L if L > 0 else 0.0
                        skip_forward_this_step = False

                        if self.min_update_ratio > 0 and 0 < update_ratio < self.min_update_ratio:
                            skip_forward_this_step = True
                            tokens_to_update = torch.tensor([], device=x_g.device, dtype=torch.long)
                            # if (g == 0 if self.print_only_group0 else True) and cond_flag:
                            #     print(f"  Group {g} step {step_num} ({branch}) - Token update ratio {update_ratio:.2%} < min_update_ratio {self.min_update_ratio:.2%}, skipping forward")
                        else:
                            accumulated[need_update_mask] = 0.0

                        token_accumulated_dict[g] = accumulated

                        previous_e0_g_dict[g] = e0_g.clone()

                        num_to_update = len(tokens_to_update)

                        forward_start_time = time.time()

                        if num_to_update == 0:
                            x_g = x_g + residual_dict[g]
                            forward_end_time = time.time()

                            if (g == 0 if self.print_only_group0 else True) and cond_flag:
                                phase_str = "Phase2" if use_differentiated_weights else "Phase1"
                                print(f"  Group {g} step {step_num} ({branch}) - Token {phase_str} All cached, Time: {forward_end_time - forward_start_time:.3f}s")

                        elif num_to_update == L:
                            ori_g = x_g.clone()
                            kwargs = dict(
                                e=e0_g,
                                grid_sizes=grid_sizes,
                                freqs=self.freqs,
                                context=context,
                                block_mask=self.block_mask,
                                group_idx=g,
                                cond_flag=cond_flag,
                                num_groups=num_groups,
                                update_mask_per_group_list=update_mask_per_group_list,
                            )
                            for block in self.blocks:
                                x_g = block(x_g, **kwargs)

                            residual_dict[g] = x_g - ori_g
                            forward_end_time = time.time()
                            if (g == 0 if self.print_only_group0 else True) and cond_flag:
                                phase_str = "Phase2" if use_differentiated_weights else "Phase1"
                                print(f"  Group {g} step {step_num} ({branch}) - Token {phase_str} Full forward ({num_to_update}/{L}), Time: {forward_end_time - forward_start_time:.3f}s")

                        else:
                            ori_g = x_g.clone()

                            x_g = x_g + residual_dict[g]

                            x_selected = ori_g[:, tokens_to_update, :]
                            x_updated = self._forward_selective_tokens(
                                x_selected, tokens_to_update, g,
                                e=e0_g,
                                grid_sizes=grid_sizes,
                                freqs=self.freqs,
                                context=context,
                                cond_flag=cond_flag,
                                num_groups=num_groups,
                                update_mask_per_group_list=update_mask_per_group_list,
                            )

                            x_g[:, tokens_to_update, :] = x_updated

                            residual_dict[g][:, tokens_to_update, :] = x_updated - ori_g[:, tokens_to_update, :]

                            forward_end_time = time.time()
                            if (g == 0 if self.print_only_group0 else True) and cond_flag:
                                phase_str = "Phase2" if use_differentiated_weights else "Phase1"
                                print(f"  Group {g} step {step_num} ({branch}) - Token {phase_str} Selective ({num_to_update}/{L}, {100.0*num_to_update/L:.1f}%), Time: {forward_end_time - forward_start_time:.3f}s")

                        if num_to_update > 0 and not token_phase1_done_dict.get(g, False):
                            token_phase1_counter_dict[g] = token_phase1_counter_dict.get(g, 0) + 1

                            if token_phase1_counter_dict[g] >= self.token_phase1_update_count:
                                token_phase1_done_dict[g] = True
                                if (g == 0 if self.print_only_group0 else True) and cond_flag:
                                    print(f"  Group {g} ({branch}) - Token Phase1 done ({token_phase1_counter_dict[g]}/{self.token_phase1_update_count}), switching to differentiated weights")

                        should_compute_weights = (
                            token_phase1_done_dict.get(g, False) or
                            (num_to_update > 0 and token_phase1_counter_dict.get(g, 0) >= self.token_phase1_update_count)
                        )

                        if should_compute_weights:
                            prev_output_dict[g] = x_g.clone()
                            weights = self._compute_weights_from_frame_diff(x_g, grid_sizes, g, cond_flag)
                            token_weights_dict[g] = weights
                # ==================== Token Cache Logic End ====================

                # ==================== Original Logic (No Cache) ====================
                else:
                    kwargs = dict(
                        e=e0_g,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        block_mask=self.block_mask,
                        group_idx=g,
                        cond_flag=cond_flag,
                        num_groups=num_groups,
                        update_mask_per_group_list=update_mask_per_group_list,
                    )

                    start_time = time.time()
                    for block in self.blocks:
                        x_g = block(x_g,**kwargs)

                    end_time = time.time()
                    branch = "cond" if cond_flag else "uncond"
                    if (g == 0 if self.print_only_group0 else True) and cond_flag:
                        print(f"  Group {g} step {cnt_vec[g]} ({branch}) - Original forward, Time: {end_time - start_time:.3f}s")
                
                if update_mask_per_group_list[g]==True:
                    cnt_vec[g] = cnt_vec[g]+1

                    if cond_flag:
                        self.cnt_even = cnt_vec
                    else:
                        self.cnt_odd = cnt_vec

                out_chunks[g] = x_g
            else:
                continue

        self.cnt = self.cnt + 1
        

        x = torch.cat(out_chunks, dim=1) # 1*51000*1536

        x = self.head(x, e) # 1*51000*64 

        grid_sizes[2] = self.latent_width
        grid_sizes[1] = self.latent_height
        grid_sizes[0] = group_size * num_groups

        # unpatchify
        x = self.unpatchify(x, grid_sizes) # 1*16*25*68*120

        return x.float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])

        return x

    def _forward_selective_tokens(self, x_important, important_indices_tensor, group_idx, 
                                  e, grid_sizes, freqs, context, cond_flag, num_groups,
                                  update_mask_per_group_list):
        """
        
        Args:
            e: modulation embeddings
            freqs: rope frequencies
        
        Returns:
        """
        for block in self.blocks:
            x_important = block.forward_selective(
                x_important, 
                important_indices_tensor,
                e=e,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                block_mask=None,
                group_idx=group_idx,
                cond_flag=cond_flag,
                num_groups=num_groups,
                update_mask_per_group_list=update_mask_per_group_list,
            )
        
        return x_important

    def set_distance_classification_config(self, n_classes=30, metric="L1", 
                                        binning_method="quantile"):
        """
        
        Args:
        """
        self.distance_n_classes = n_classes
        if metric in ["L1", "L2"]:
            self.distance_metric = metric
        else:
            print(f"Warning: Invalid metric '{metric}', using 'L1' instead")
            self.distance_metric = "L1"
        if binning_method in ["quantile", "uniform"]:
            self.distance_binning_method = binning_method
        else:
            print(f"Warning: Invalid binning_method '{binning_method}', using 'quantile' instead")
            self.distance_binning_method = "quantile"
        
        print(f"Distance classification config: n_classes={n_classes}, "
              f"metric={self.distance_metric}, binning={self.distance_binning_method}")

    def set_distance_visualization_config(self, enable=False, output_dir=None, save_steps=None):
        self.enable_distance_visualization = enable
        if output_dir is not None:
            self.distance_output_dir = output_dir
        
        if save_steps is not None:
            self.distance_save_steps = list(save_steps)
        else:
            self.distance_save_steps = None
        
        save_steps_info = f"save_steps={self.distance_save_steps}" if self.distance_save_steps else "save all steps"
        png_info = "save PNG" if enable else "no PNG"
        print(f"Distance visualization config: output_dir={self.distance_output_dir}, "
              f"{save_steps_info}, {png_info}")

    def set_residual_diff_visualization_config(self, enable=False, output_dir=None,
                                                metric="L1", save_steps=None):
        self.enable_residual_diff_visualization = enable
        if output_dir is not None:
            self.residual_diff_output_dir = output_dir

        if metric in ["L1", "L2", "cosine"]:
            self.residual_diff_metric = metric
        else:
            print(f"Warning: Invalid metric '{metric}', using 'L1' instead")
            self.residual_diff_metric = "L1"

        if save_steps is not None:
            self.residual_diff_save_steps = list(save_steps)
        else:
            self.residual_diff_save_steps = None

        if enable:
            save_steps_info = f"save_steps={self.residual_diff_save_steps}" if self.residual_diff_save_steps else "save all steps"
            print(f"Residual diff visualization enabled: output_dir={self.residual_diff_output_dir}, "
                  f"metric={self.residual_diff_metric}, {save_steps_info}")


    def visualize_selective_tokens(self, important_token_indices, group_size, h, w, 
                                   group_idx, step_num, is_cond):
        import os
        import numpy as np
        from PIL import Image
        
        rgb_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        selected_spatial_positions = set()
        
        for token_idx in important_token_indices:
            spatial_idx = token_idx % (h * w)
            selected_spatial_positions.add(spatial_idx)
        
        for spatial_idx in selected_spatial_positions:
            h_idx = spatial_idx // w
            w_idx = spatial_idx % w
            rgb_image[h_idx, w_idx] = [255, 0, 0]
        
        img = Image.fromarray(rgb_image)
        step_num_val = step_num.item() if hasattr(step_num, 'item') else step_num
        branch_name = "cond" if is_cond else "uncond"
        filename = f"niter_{self.current_niter}-group_{group_idx}-output_{branch_name}-step_{step_num_val}_selective.png"
        
        os.makedirs(self.selective_viz_output_dir, exist_ok=True)
        filepath = os.path.join(self.selective_viz_output_dir, filename)
        img.save(filepath)
        
        total_tokens = group_size * h * w
        selected_tokens = len(important_token_indices)
        selected_spatial = len(selected_spatial_positions)

    def set_selective_viz_config(self, enable=False, output_dir=None):
        self.enable_selective_viz = enable
        if output_dir is not None:
            self.selective_viz_output_dir = output_dir


    def set_token_cache_config(self, enable=False, threshold=0.10,
                               warmup_steps=5, weight_min=0.5,
                               weight_max=2.0, phase1_update_count=3,
                               distance_mode="token", weight_gamma=1.0,
                               weight_smooth_grid_size=1):
        self.token_cache_enabled = enable
        self.token_cache_threshold = threshold
        self.token_cache_warmup_steps = warmup_steps
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.token_phase1_update_count = phase1_update_count
        self.token_distance_mode = distance_mode  # "token" or "global"
        self.token_weight_gamma = weight_gamma
        self.weight_smooth_grid_size = weight_smooth_grid_size

        self.token_accumulated_even = {}
        self.token_accumulated_odd = {}
        self.token_weights_even = {}
        self.token_weights_odd = {}
        self.token_phase1_counter_even = {}
        self.token_phase1_counter_odd = {}
        self.token_phase1_done_even = {}
        self.token_phase1_done_odd = {}

        self.global_accumulated_even = {}  # {group_idx: float}
        self.global_accumulated_odd = {}


    def clear_token_cache(self):
        self.token_accumulated_even = {}
        self.token_accumulated_odd = {}
        self.token_weights_even = {}
        self.token_weights_odd = {}
        self.token_phase1_counter_even = {}
        self.token_phase1_counter_odd = {}
        self.token_phase1_done_even = {}
        self.token_phase1_done_odd = {}

    def set_static_weight_config(self, enable=False, weight_value=0.3, regions=None):
        self.use_static_weight = enable
        self.static_weight_value = weight_value
        if regions is not None:
            self.static_weight_regions = regions

        if enable:
            print(f"Static weight enabled: {len(self.static_weight_regions)} region(s), weight_value={weight_value}")
            for i, ((h1, w1), (h2, w2)) in enumerate(self.static_weight_regions):
                print(f"  Region {i}: height {h1:.0%}-{h2:.0%}, width {w1:.0%}-{w2:.0%}")

    def set_frame_diff_weight_config(self, enable=False, viz_enable=False, output_dir=None):
        self.use_frame_diff_weight = enable
        self.frame_diff_viz_enabled = viz_enable
        if output_dir is not None:
            self.frame_diff_viz_output_dir = output_dir

        self.prev_group_last_frame_even = {}
        self.prev_group_last_frame_odd = {}

        if enable:
            print(f"Frame diff weight enabled: viz={viz_enable}, output_dir={self.frame_diff_viz_output_dir}")

    def set_input_diff_weight_config(self, enable=False, viz_enable=False, output_dir=None):
        self.use_input_diff_weight = enable
        self.input_diff_viz_enabled = viz_enable
        if output_dir is not None:
            self.input_diff_viz_output_dir = output_dir

        self.prev_group_last_frame_input_even = {}
        self.prev_group_last_frame_input_odd = {}


    def set_weight_norm_mode(self, mode="mean", weight_floor=0.3):
        if mode not in ["mean", "max", "max_rescale"]:
            raise ValueError(f"Invalid weight_norm_mode: {mode}, must be 'mean', 'max' or 'max_rescale'")
        self.weight_norm_mode = mode
        self.weight_floor = weight_floor

    def set_temporal_consistency_config(self, enable=False, threshold=0.5, first_frame_full_weight=False, group0_first_frame_mode="ones"):
        self.use_temporal_consistency = enable
        self.temporal_consistency_threshold = threshold
        self.first_frame_full_weight = first_frame_full_weight
        self.group0_first_frame_mode = group0_first_frame_mode
        if enable:
            print(f"Temporal consistency constraint enabled: threshold={threshold}")
        if first_frame_full_weight:
            print(f"First frame full weight enabled")
        if group0_first_frame_mode != "ones":
            print(f"Group0 first frame mode: {group0_first_frame_mode}")

    def set_min_update_ratio(self, min_ratio=0.0):
        if min_ratio < 0 or min_ratio > 1:
            raise ValueError(f"min_update_ratio must be in [0, 1], got {min_ratio}")
        self.min_update_ratio = min_ratio
        if min_ratio > 0:
            print(f"Min update ratio set to: {min_ratio:.1%} (skip forward when update ratio < {min_ratio:.1%})")

    def _build_static_weight_mask(self, group_size, H, W, device):
        weights_2d = torch.ones(H, W, device=device, dtype=torch.float32)

        for (h1, w1), (h2, w2) in self.static_weight_regions:
            h_start = int(h1 * H)
            h_end = int(h2 * H)
            w_start = int(w1 * W)
            w_end = int(w2 * W)
            weights_2d[h_start:h_end, w_start:w_end] = self.static_weight_value

        weights = weights_2d.unsqueeze(0).expand(group_size, -1, -1).reshape(-1)

        return weights

    def _get_previous_output(self, cond_flag):
        return self.previous_output_even if cond_flag else self.previous_output_odd

    def _get_residual_dict(self, cond_flag):
        return self.previous_residual_even if cond_flag else self.previous_residual_odd

    def _get_previous_e0_g(self, cond_flag):
        return self.previous_e0_g_even if cond_flag else self.previous_e0_g_odd

    def _get_token_accumulated(self, cond_flag):
        return self.token_accumulated_even if cond_flag else self.token_accumulated_odd

    def _get_token_weights(self, cond_flag):
        return self.token_weights_even if cond_flag else self.token_weights_odd

    def _get_token_phase1_counter(self, cond_flag):
        return self.token_phase1_counter_even if cond_flag else self.token_phase1_counter_odd

    def _get_token_phase1_done(self, cond_flag):
        return self.token_phase1_done_even if cond_flag else self.token_phase1_done_odd

    def _smooth_weights_by_grid(self, weights, H, W, grid_size):
        if grid_size <= 1:
            return weights

        pad_h = (grid_size - H % grid_size) % grid_size
        pad_w = (grid_size - W % grid_size) % grid_size

        if pad_h > 0 or pad_w > 0:
            if pad_w > 0:
                right_edge = weights[:, -1:].expand(-1, pad_w)
                weights_padded = torch.cat([weights, right_edge], dim=1)
            else:
                weights_padded = weights
            if pad_h > 0:
                bottom_edge = weights_padded[-1:, :].expand(pad_h, -1)
                weights_padded = torch.cat([weights_padded, bottom_edge], dim=0)
        else:
            weights_padded = weights

        H_padded, W_padded = weights_padded.shape

        weights_reshaped = weights_padded.view(
            H_padded // grid_size, grid_size,
            W_padded // grid_size, grid_size
        )
        grid_means = weights_reshaped.mean(dim=(1, 3))

        smoothed = grid_means.unsqueeze(1).unsqueeze(3).expand(
            -1, grid_size, -1, grid_size
        ).contiguous().reshape(H_padded, W_padded)

        smoothed = smoothed[:H, :W].contiguous()

        return smoothed

    def _compute_weights_from_output(self, output, grid_sizes):
        B, L, C = output.shape
        group_size = grid_sizes[0]
        H, W = grid_sizes[1], grid_sizes[2]
        spatial_size = H * W

        output_reshaped = output.view(B, group_size, spatial_size, C)

        token_importance = torch.abs(output_reshaped).mean(dim=(0, 3))  # [group_size, H*W]

        if self.token_weight_gamma != 1.0:
            mean_imp = token_importance.mean(dim=1, keepdim=True)
            token_importance = mean_imp * torch.pow(token_importance / (mean_imp + 1e-8), self.token_weight_gamma)

        # [group_size, 1]
        frame_mean = token_importance.mean(dim=1, keepdim=True)
        weights = token_importance / (frame_mean + 1e-8)

        weights = weights.clamp(min=self.weight_min, max=self.weight_max)

        if self.weight_smooth_grid_size > 1:
            H_val = H.item() if hasattr(H, 'item') else H
            W_val = W.item() if hasattr(W, 'item') else W
            smoothed_weights = []
            for f in range(group_size):
                frame_weights = weights[f].view(H_val, W_val)
                smoothed_frame = self._smooth_weights_by_grid(
                    frame_weights, H_val, W_val, self.weight_smooth_grid_size
                )
                smoothed_weights.append(smoothed_frame.view(-1))
            weights = torch.stack(smoothed_weights, dim=0)  # [group_size, H*W]

        if self.first_frame_full_weight:
            weights[0, :] = 1.0

        weights = weights.view(L)

        return weights

    def _get_prev_group_last_frame(self, cond_flag):
        return self.prev_group_last_frame_even if cond_flag else self.prev_group_last_frame_odd

    def _get_prev_group_last_frame_input(self, cond_flag):
        return self.prev_group_last_frame_input_even if cond_flag else self.prev_group_last_frame_input_odd

    def _compute_weights_from_frame_diff(self, output, grid_sizes, group_idx, cond_flag):
        B, L, C = output.shape
        group_size = grid_sizes[0]
        H, W = grid_sizes[1], grid_sizes[2]
        spatial_size = H * W

        output_reshaped = output.view(B, group_size, spatial_size, C)

        prev_group_last_frame_dict = self._get_prev_group_last_frame(cond_flag)

        frame_diff = torch.zeros(group_size, spatial_size, device=output.device, dtype=torch.float32)

        for frame_idx in range(group_size):
            current_frame = output_reshaped[:, frame_idx, :, :]  # [B, H*W, C]

            if frame_idx == 0:
                if group_idx == 0:
                    if self.group0_first_frame_mode == "ones":
                        frame_diff[frame_idx] = torch.ones(spatial_size, device=output.device, dtype=torch.float32)
                    else:
                        frame_diff[frame_idx] = torch.zeros(spatial_size, device=output.device, dtype=torch.float32)
                elif group_idx - 1 in prev_group_last_frame_dict:
                    prev_frame = prev_group_last_frame_dict[group_idx - 1]  # [B, H*W, C]
                    diff = (current_frame - prev_frame).abs().mean(dim=(0, 2))  # [H*W]
                    base = prev_frame.abs().mean(dim=(0, 2)) + 1e-8  # [H*W]
                    frame_diff[frame_idx] = diff / base
                else:
                    frame_diff[frame_idx] = torch.ones(spatial_size, device=output.device, dtype=torch.float32)
            else:
                prev_frame = output_reshaped[:, frame_idx - 1, :, :]  # [B, H*W, C]
                diff = (current_frame - prev_frame).abs().mean(dim=(0, 2))  # [H*W]
                base = prev_frame.abs().mean(dim=(0, 2)) + 1e-8  # [H*W]
                frame_diff[frame_idx] = diff / base

        prev_group_last_frame_dict[group_idx] = output_reshaped[:, -1, :, :].clone()  # [B, H*W, C]

        if group_idx == 0 and self.group0_first_frame_mode != "ones":
            if self.group0_first_frame_mode == "second_frame":
                frame_diff[0] = frame_diff[1].clone()
            elif self.group0_first_frame_mode == "group_mean":
                frame_diff[0] = frame_diff[1:].mean(dim=0)

        if self.token_weight_gamma != 1.0:
            mean_diff = frame_diff.mean(dim=1, keepdim=True)
            frame_diff = mean_diff * torch.pow(frame_diff / (mean_diff + 1e-8), self.token_weight_gamma)

        if self.weight_norm_mode == "max":
            frame_max = frame_diff.max(dim=1, keepdim=True)[0]
            weights = frame_diff / (frame_max + 1e-8)
        elif self.weight_norm_mode == "max_rescale":
            frame_min = frame_diff.min(dim=1, keepdim=True)[0]
            frame_max = frame_diff.max(dim=1, keepdim=True)[0]
            normalized = (frame_diff - frame_min) / (frame_max - frame_min + 1e-8)
            weights = self.weight_floor + (1 - self.weight_floor) * normalized
        else:
            frame_mean = frame_diff.mean(dim=1, keepdim=True)
            weights = frame_diff / (frame_mean + 1e-8)

        weights = weights.clamp(min=self.weight_min, max=self.weight_max)

        if group_idx == 0 and self.group0_first_frame_mode == "ones":
            weights[0, :] = 1.0

        if self.weight_smooth_grid_size > 1:
            H_val = H.item() if hasattr(H, 'item') else H
            W_val = W.item() if hasattr(W, 'item') else W
            smoothed_weights = []
            for f in range(group_size):
                frame_weights = weights[f].view(H_val, W_val)
                smoothed_frame = self._smooth_weights_by_grid(
                    frame_weights, H_val, W_val, self.weight_smooth_grid_size
                )
                smoothed_weights.append(smoothed_frame.view(-1))
            weights = torch.stack(smoothed_weights, dim=0)  # [group_size, H*W]

        if self.first_frame_full_weight:
            weights[0, :] = 1.0

        weights = weights.view(L)


        return weights

    def _compute_weights_from_input_diff(self, x_raw, group_idx, cond_flag):
        B, C, F, H_raw, W_raw = x_raw.shape
        group_size = F

        H = H_raw // 2
        W = W_raw // 2
        spatial_size = H * W
        L = group_size * spatial_size

        x_reshaped = x_raw.permute(0, 2, 3, 4, 1)  # [B, F, H_raw, W_raw, C]

        prev_group_last_frame_input_dict = self._get_prev_group_last_frame_input(cond_flag)

        frame_diff_raw = torch.zeros(group_size, H_raw, W_raw, device=x_raw.device, dtype=torch.float32)

        for frame_idx in range(group_size):
            current_frame = x_reshaped[:, frame_idx, :, :, :]  # [B, H_raw, W_raw, C]

            if frame_idx == 0:
                if group_idx == 0:
                    if self.group0_first_frame_mode == "ones":
                        frame_diff_raw[frame_idx] = torch.ones(H_raw, W_raw, device=x_raw.device, dtype=torch.float32)
                    else:
                        frame_diff_raw[frame_idx] = torch.zeros(H_raw, W_raw, device=x_raw.device, dtype=torch.float32)
                elif group_idx - 1 in prev_group_last_frame_input_dict:
                    prev_frame = prev_group_last_frame_input_dict[group_idx - 1]  # [B, H_raw, W_raw, C]
                    diff = (current_frame - prev_frame).abs().mean(dim=(0, 3))  # [H_raw, W_raw]
                    base = prev_frame.abs().mean(dim=(0, 3)) + 1e-8
                    frame_diff_raw[frame_idx] = diff / base
                else:
                    frame_diff_raw[frame_idx] = torch.ones(H_raw, W_raw, device=x_raw.device, dtype=torch.float32)
            else:
                prev_frame = x_reshaped[:, frame_idx - 1, :, :, :]
                diff = (current_frame - prev_frame).abs().mean(dim=(0, 3))
                base = prev_frame.abs().mean(dim=(0, 3)) + 1e-8
                frame_diff_raw[frame_idx] = diff / base

        prev_group_last_frame_input_dict[group_idx] = x_reshaped[:, -1, :, :, :].clone()

        if group_idx == 0 and self.group0_first_frame_mode != "ones":
            if self.group0_first_frame_mode == "second_frame":
                frame_diff_raw[0] = frame_diff_raw[1].clone()
            elif self.group0_first_frame_mode == "group_mean":
                frame_diff_raw[0] = frame_diff_raw[1:].mean(dim=0)

        frame_diff_raw_4d = frame_diff_raw.unsqueeze(1)  # [F, 1, H_raw, W_raw]
        frame_diff = torch.nn.functional.avg_pool2d(frame_diff_raw_4d, kernel_size=2, stride=2)
        frame_diff = frame_diff.squeeze(1)  # [F, H, W]
        frame_diff = frame_diff.view(group_size, spatial_size)  # [F, H*W]

        if self.token_weight_gamma != 1.0:
            mean_diff = frame_diff.mean(dim=1, keepdim=True)
            frame_diff = mean_diff * torch.pow(frame_diff / (mean_diff + 1e-8), self.token_weight_gamma)

        if self.weight_norm_mode == "max":
            frame_max = frame_diff.max(dim=1, keepdim=True)[0]
            weights = frame_diff / (frame_max + 1e-8)
        elif self.weight_norm_mode == "max_rescale":
            frame_min = frame_diff.min(dim=1, keepdim=True)[0]
            frame_max = frame_diff.max(dim=1, keepdim=True)[0]
            normalized = (frame_diff - frame_min) / (frame_max - frame_min + 1e-8)
            weights = self.weight_floor + (1 - self.weight_floor) * normalized
        else:
            frame_mean = frame_diff.mean(dim=1, keepdim=True)
            weights = frame_diff / (frame_mean + 1e-8)

        weights = weights.clamp(min=self.weight_min, max=self.weight_max)

        if group_idx == 0 and self.group0_first_frame_mode == "ones":
            weights[0, :] = 1.0

        if self.weight_smooth_grid_size > 1:
            H_val = H.item() if hasattr(H, 'item') else H
            W_val = W.item() if hasattr(W, 'item') else W
            smoothed_weights = []
            for f in range(group_size):
                frame_weights = weights[f].view(H_val, W_val)
                smoothed_frame = self._smooth_weights_by_grid(
                    frame_weights, H_val, W_val, self.weight_smooth_grid_size
                )
                smoothed_weights.append(smoothed_frame.view(-1))
            weights = torch.stack(smoothed_weights, dim=0)

        if self.first_frame_full_weight:
            weights[0, :] = 1.0

        weights = weights.view(L)

        return weights

    def _visualize_frame_diff(self, weights, group_idx, group_size, grid_sizes, is_cond, step_num):
        if not is_cond:
            return

        import os
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        H, W = grid_sizes[1].item(), grid_sizes[2].item()
        step_num_val = step_num.item() if hasattr(step_num, 'item') else step_num

        viz_dir = os.path.join(self.frame_diff_viz_output_dir, "viz")
        data_dir = os.path.join(self.frame_diff_viz_output_dir, "data")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        colors = plt.cm.jet(np.linspace(0, 1, self.distance_n_classes))

        # Reshape weights: [L] -> [group_size, H*W]
        weights_reshaped = weights.view(group_size, H * W)

        print(f"  Frame diff visualization for group {group_idx}:")

        for frame_idx in range(group_size):
            frame_weights = weights_reshaped[frame_idx].float().cpu().numpy().reshape(H, W)

            min_val = frame_weights.min()
            max_val = frame_weights.max()
            mean_val = frame_weights.mean()
            print(f"    Frame {frame_idx}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

            quantiles = np.linspace(0, 1, self.distance_n_classes + 1)
            thresholds = np.quantile(frame_weights, quantiles)

            class_map = np.zeros_like(frame_weights, dtype=np.int32)
            for i in range(self.distance_n_classes):
                if i == self.distance_n_classes - 1:
                    mask = (frame_weights >= thresholds[i]) & (frame_weights <= thresholds[i + 1])
                else:
                    mask = (frame_weights >= thresholds[i]) & (frame_weights < thresholds[i + 1])
                class_map[mask] = i

            rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
            for i in range(self.distance_n_classes):
                mask = (class_map == i)
                rgb_image[mask] = (colors[i][:3] * 255).astype(np.uint8)

            img = Image.fromarray(rgb_image)
            filename = f"niter_{self.current_niter}-group_{group_idx}-frame_{frame_idx}-step_{step_num_val}_frame_diff.png"
            filepath = os.path.join(viz_dir, filename)
            img.save(filepath)

        data_filename = f"niter_{self.current_niter}-group_{group_idx}-step_{step_num_val}_frame_diff_weights.npy"
        data_filepath = os.path.join(data_dir, data_filename)
        weights_numpy = weights_reshaped.float().cpu().numpy()
        np.save(data_filepath, weights_numpy)

        print(f"  Saved frame diff viz: {group_size} frames for group {group_idx}")

    def _visualize_input_diff(self, weights, group_idx, group_size, grid_sizes, is_cond, step_num):
        if not is_cond:
            return

        import os
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        H, W = grid_sizes[1].item(), grid_sizes[2].item()
        step_num_val = step_num.item() if hasattr(step_num, 'item') else step_num

        viz_dir = os.path.join(self.input_diff_viz_output_dir, "viz")
        data_dir = os.path.join(self.input_diff_viz_output_dir, "data")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        colors = plt.cm.jet(np.linspace(0, 1, self.distance_n_classes))
        weights_reshaped = weights.view(group_size, H * W)

        print(f"  Input diff visualization for group {group_idx}:")

        for frame_idx in range(group_size):
            frame_weights = weights_reshaped[frame_idx].float().cpu().numpy().reshape(H, W)

            min_val = frame_weights.min()
            max_val = frame_weights.max()
            mean_val = frame_weights.mean()
            print(f"    Frame {frame_idx}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

            quantiles = np.linspace(0, 1, self.distance_n_classes + 1)
            thresholds = np.quantile(frame_weights, quantiles)

            class_map = np.zeros_like(frame_weights, dtype=np.int32)
            for i in range(self.distance_n_classes):
                if i == self.distance_n_classes - 1:
                    mask = (frame_weights >= thresholds[i]) & (frame_weights <= thresholds[i + 1])
                else:
                    mask = (frame_weights >= thresholds[i]) & (frame_weights < thresholds[i + 1])
                class_map[mask] = i

            rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
            for i in range(self.distance_n_classes):
                mask = (class_map == i)
                rgb_image[mask] = (colors[i][:3] * 255).astype(np.uint8)

            img = Image.fromarray(rgb_image)
            filename = f"niter_{self.current_niter}-group_{group_idx}-frame_{frame_idx}-step_{step_num_val}_input_diff.png"
            filepath = os.path.join(viz_dir, filename)
            img.save(filepath)

        data_filename = f"niter_{self.current_niter}-group_{group_idx}-step_{step_num_val}_input_diff_weights.npy"
        data_filepath = os.path.join(data_dir, data_filename)
        weights_numpy = weights_reshaped.float().cpu().numpy()
        np.save(data_filepath, weights_numpy)

        print(f"  Saved input diff viz: {group_size} frames for group {group_idx}")

    def _compute_per_token_distance(self, e0_g, prev_e0_g):
        diff = (e0_g - prev_e0_g).abs()

        diff_per_token = diff.mean(dim=(0, 1, 3))  # [L]

        base_per_token = prev_e0_g.abs().mean(dim=(0, 1, 3)) + 1e-8  # [L]

        x = diff_per_token / base_per_token  # [L]

        c = self.teacache_coefficients
        dist_per_token = c[0]*x**4 + c[1]*x**3 + c[2]*x**2 + c[3]*x + c[4]

        return dist_per_token

    def _compute_global_distance(self, e0_g, prev_e0_g):
        diff_mean = (e0_g - prev_e0_g).abs().mean()

        base_mean = prev_e0_g.abs().mean() + 1e-8

        rel_l1 = diff_mean / base_mean  # scalar

        rescale_func = np.poly1d(self.teacache_coefficients)
        global_dist = rescale_func(rel_l1.cpu().item())

        return global_dist  # float

    def _get_global_accumulated(self, cond_flag):
        return self.global_accumulated_even if cond_flag else self.global_accumulated_odd

    def _compute_residual_difference(self, current_residual, prev_residual, grid_sizes):
        diff = current_residual - prev_residual

        metric = getattr(self, 'residual_diff_metric', 'L1')

        if metric == 'L1':
            diff_per_token = diff.abs().mean(dim=(0, 2))  # [L]
        elif metric == 'L2':
            diff_per_token = torch.norm(diff, p=2, dim=2).mean(dim=0)  # [L]
        elif metric == 'cosine':
            cos_sim = torch.nn.functional.cosine_similarity(
                current_residual, prev_residual, dim=2
            ).mean(dim=0)  # [L]
            diff_per_token = 1 - cos_sim  # [L]
        else:
            diff_per_token = diff.abs().mean(dim=(0, 2))  # [L]

        group_size = grid_sizes[0].item() if hasattr(grid_sizes[0], 'item') else grid_sizes[0]
        H = grid_sizes[1].item() if hasattr(grid_sizes[1], 'item') else grid_sizes[1]
        W = grid_sizes[2].item() if hasattr(grid_sizes[2], 'item') else grid_sizes[2]

        diff_map = diff_per_token.view(group_size, H * W)

        return diff_map

    def _visualize_residual_difference(self, diff_map, group_idx, group_size,
                                        grid_sizes, is_cond, step_num):
        if not self.enable_residual_diff_visualization or not is_cond:
            return

        import os
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        h, w = grid_sizes[1].item(), grid_sizes[2].item()

        step_num_val = step_num.item() if hasattr(step_num, 'item') else step_num
        if self.residual_diff_save_steps is not None and step_num_val not in self.residual_diff_save_steps:
            return

        viz_dir = os.path.join(self.residual_diff_output_dir, "viz")
        data_dir = os.path.join(self.residual_diff_output_dir, "data")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        colors = plt.cm.jet(np.linspace(0, 1, self.distance_n_classes))

        print(f"  Residual diff statistics for group {group_idx}:")

        for frame_idx in range(group_size):
            frame_diff_map = diff_map[frame_idx].float().cpu().numpy().reshape(h, w)

            min_val = frame_diff_map.min()
            max_val = frame_diff_map.max()
            mean_val = frame_diff_map.mean()
            std_val = frame_diff_map.std()
            print(f"    Frame {frame_idx}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")

            quantiles = np.linspace(0, 1, self.distance_n_classes + 1)
            thresholds = np.quantile(frame_diff_map, quantiles)

            class_map = np.zeros_like(frame_diff_map, dtype=np.int32)
            for i in range(self.distance_n_classes):
                if i == self.distance_n_classes - 1:
                    mask = (frame_diff_map >= thresholds[i]) & (frame_diff_map <= thresholds[i + 1])
                else:
                    mask = (frame_diff_map >= thresholds[i]) & (frame_diff_map < thresholds[i + 1])
                class_map[mask] = i

            rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(self.distance_n_classes):
                mask = (class_map == i)
                rgb_image[mask] = (colors[i][:3] * 255).astype(np.uint8)

            img = Image.fromarray(rgb_image)
            filename = f"niter_{self.current_niter}-group_{group_idx}-frame_{frame_idx}-step_{step_num_val}_residual_diff.png"
            filepath = os.path.join(viz_dir, filename)
            img.save(filepath)

        data_filename = f"niter_{self.current_niter}-group_{group_idx}-step_{step_num_val}_residual_diff.npy"
        data_filepath = os.path.join(data_dir, data_filename)
        diff_map_numpy = diff_map.float().cpu().numpy() if hasattr(diff_map, 'cpu') else diff_map
        np.save(data_filepath, diff_map_numpy)

        print(f"  Saved residual diff viz: {group_size} frames for group {group_idx}")
        print(f"  Saved residual diff data: {data_filename} (shape: {diff_map_numpy.shape})")

    def _classify_tokens_by_distance(self, distance_map, group_size, grid_sizes):

        h, w = grid_sizes[1].item(), grid_sizes[2].item()
        hw = h * w

        distance_map_np = distance_map.float().cpu().numpy()  # [group_size, H*W]

        quantiles = np.linspace(0, 1, self.distance_n_classes + 1) if self.distance_binning_method in ["quantile", None] else None

        frame_classifications = {}

        for frame_idx in range(group_size):
            dist_flat = distance_map_np[frame_idx]  # [H*W]

            if self.distance_binning_method == "quantile":
                thresholds = np.quantile(dist_flat, quantiles)
            elif self.distance_binning_method == "uniform":
                min_val = dist_flat.min()
                max_val = dist_flat.max()
                thresholds = np.linspace(min_val, max_val, self.distance_n_classes + 1)
            else:
                thresholds = np.quantile(dist_flat, quantiles)

            class_labels = np.digitize(dist_flat, thresholds[1:-1], right=False)

            frame_offset = frame_idx * hw

            class_indices = {}
            for i in range(self.distance_n_classes):
                spatial_indices = np.where(class_labels == i)[0]
                class_indices[i] = (frame_offset + spatial_indices).tolist()

            frame_classifications[frame_idx] = class_indices

        return frame_classifications
    
    def _compute_group_frame_token_distance_for_class_cache(self, x_g, group_idx,
                                                            group_size, grid_sizes,
                                                            is_cond, step_num):
        b, l, c = x_g.shape
        h, w = grid_sizes[1].item(), grid_sizes[2].item()
        
        # reshape: [B, group_size*H*W, C] -> [B, group_size, H, W, C]
        x_frames = x_g.view(b, group_size, h, w, c)

        frame_wise_importance = self._compute_frame_wise_importance(x_frames)

        return frame_wise_importance

    def _compute_frame_wise_importance(self, x_frames):
        b, group_size, h, w, c = x_frames.shape

        frame_importance = torch.abs(x_frames).mean(dim=-1)

        # frame_importance: [B, group_size, H, W]
        frame_max = frame_importance.view(b, group_size, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)  # [B, group_size, 1, 1]
        frame_importance_norm = frame_importance / (frame_max + 1e-6)

        frame_importance_batch = frame_importance_norm[0]  # [group_size, H, W]
        frame_importance_flat = frame_importance_batch.view(group_size, h * w)

        return frame_importance_flat

    def _visualize_class_output_distance(self, distance_map, group_idx, group_size,
                                          grid_sizes, is_cond, step_num):
        if not self.enable_distance_visualization or not is_cond:
            return
        
        import os
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        
        h, w = grid_sizes[1].item(), grid_sizes[2].item()
        
        step_num_val = step_num.item() if hasattr(step_num, 'item') else step_num
        if self.distance_save_steps is not None and step_num_val not in self.distance_save_steps:
            return
        
        os.makedirs(self.distance_output_dir, exist_ok=True)

        viz_dir = os.path.join(self.distance_output_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)

        distance_maps_dir = os.path.join(self.distance_output_dir, "distance_maps")
        os.makedirs(distance_maps_dir, exist_ok=True)

        colors = plt.cm.jet(np.linspace(0, 1, self.distance_n_classes))

        print(f"  Frame-wise distance statistics for group {group_idx}:")

        for frame_idx in range(group_size):
            frame_dist_map = distance_map[frame_idx].float().cpu().numpy().reshape(h, w)

            min_val = frame_dist_map.min()
            max_val = frame_dist_map.max()
            mean_val = frame_dist_map.mean()
            std_val = frame_dist_map.std()
            print(f"    Frame {frame_idx}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")

            if self.distance_binning_method == "quantile":
                quantiles = np.linspace(0, 1, self.distance_n_classes + 1)
                thresholds = np.quantile(frame_dist_map, quantiles)
            elif self.distance_binning_method == "uniform":
                thresholds = np.linspace(min_val, max_val, self.distance_n_classes + 1)
            else:
                quantiles = np.linspace(0, 1, self.distance_n_classes + 1)
                thresholds = np.quantile(frame_dist_map, quantiles)

            class_map = np.zeros_like(frame_dist_map, dtype=np.int32)
            for i in range(self.distance_n_classes):
                if i == self.distance_n_classes - 1:
                    mask = (frame_dist_map >= thresholds[i]) & (frame_dist_map <= thresholds[i + 1])
                else:
                    mask = (frame_dist_map >= thresholds[i]) & (frame_dist_map < thresholds[i + 1])
                class_map[mask] = i

            rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(self.distance_n_classes):
                mask = (class_map == i)
                rgb_image[mask] = (colors[i][:3] * 255).astype(np.uint8)

            img = Image.fromarray(rgb_image)
            filename = f"niter_{self.current_niter}-group_{group_idx}-frame_{frame_idx}-step_{step_num_val}_class_cache.png"

            filepath = os.path.join(viz_dir, filename)
            img.save(filepath)

        distance_map_filename = f"niter_{self.current_niter}-group_{group_idx}-step_{step_num_val}_distance_map.npy"
        distance_map_filepath = os.path.join(distance_maps_dir, distance_map_filename)

        distance_map_numpy = distance_map.float().cpu().numpy() if hasattr(distance_map, 'cpu') else distance_map
        np.save(distance_map_filepath, distance_map_numpy)

        print(f"  Saved class cache output viz: {group_size} frames for group {group_idx} (cond only)")
        print(f"  Saved distance map data: {distance_map_filename} (shape: {distance_map_numpy.shape})")
    
    def clear_class_cache(self):
        self.class_accumulated_dist_even = {}
        self.class_accumulated_dist_odd = {}
        self.current_class_indices_even = {}
        self.current_class_indices_odd = {}
        self.previous_output_even = {}
        self.previous_output_odd = {}
        self.previous_residual_even = {}
        self.previous_residual_odd = {}
        self.class_reclassify_counter_even = {}
        self.class_reclassify_counter_odd = {}

        self.first_update_done_even = {}
        self.first_update_done_odd = {}

        for i in range(self.num_layers):
            self.blocks[i].self_attn.k_cache_even = None
            self.blocks[i].self_attn.v_cache_even = None
            self.blocks[i].self_attn.k_cache_odd  = None
            self.blocks[i].self_attn.v_cache_odd  = None

    def set_ar_attention(self, causal_block_size):
        self.num_frame_per_block = causal_block_size
        self.flag_causal_attention = True
        for block in self.blocks:
            block.set_ar_attention()

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        if self.inject_sample_info:
            nn.init.normal_(self.fps_embedding.weight, std=0.02)

            for m in self.fps_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

            nn.init.zeros_(self.fps_projection[-1].weight)
            nn.init.zeros_(self.fps_projection[-1].bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
