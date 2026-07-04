import math
import numpy as np
import torch
import torch.amp as amp
from torch.backends.cuda import sdp_kernel

from ..modules.attention import flash_attention
from ..modules.transformer import sinusoidal_embedding_1d
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, group_idx):
    """
    x:          [B, L, N, C]. Here L == group_size * H * W for the current group
    grid_sizes: [F, H, W] for the whole chunk; we will temporarily set F=group_size per group
    freqs:      [M, C // 2], complex numbers representing rotary embeddings
    group_idx:  index of current group within the chunk
    """
    n, c = x.size(2), x.size(3) // 2
    bs = x.size(0)

    # split freqs across temporal/spatial components as in transformer_teacache
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    f, h, w = grid_sizes.tolist()
    seq_len = f * h * w

    # compute temporal slice for current group
    start_f = group_idx * f
    end_f = start_f + f

    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, seq_len, n, -1, 2))
    freqs_i = torch.cat(
        [
            freqs[0][start_f:end_f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    x = torch.view_as_real(x * freqs_i).flatten(3)
    return x


def _alloc_kv(self_attn_module, total_tokens, batch_size, device, dtype):
    return torch.zeros(
        batch_size,
        total_tokens,
        self_attn_module.num_heads,
        self_attn_module.head_dim,
        dtype=dtype,
        device=device,
    )


def _update_and_return_kv(self_attn_module, k, v, cond_flag, group_idx, group_size, grid_hw, num_groups, batch_size):
    """
    k, v        : [B, group_size*H*W, nH, d] (already rope-applied for this group)
    cond_flag   : whether current pass is conditional (even) or unconditional (odd)
    group_idx   : 0..num_groups-1
    returns accumulated k_full, v_full up to current group end: shape [B, <=total_tokens, nH, d]
    """
    total_tokens = num_groups * group_size * grid_hw
    token_per_grp = group_size * grid_hw
    start = group_idx * token_per_grp
    end = start + k.size(1)

    buf_k = self_attn_module.k_cache_even if cond_flag else self_attn_module.k_cache_odd
    buf_v = self_attn_module.v_cache_even if cond_flag else self_attn_module.v_cache_odd

    if buf_k is None and buf_v is None:
        buf_k = _alloc_kv(self_attn_module, total_tokens, batch_size, k.device, k.dtype)
        buf_v = _alloc_kv(self_attn_module, total_tokens, batch_size, v.device, v.dtype)

    buf_k[:, start:end] = k.detach()
    buf_v[:, start:end] = v.detach()

    if cond_flag:
        self_attn_module.k_cache_even = buf_k
        self_attn_module.v_cache_even = buf_v
    else:
        self_attn_module.k_cache_odd = buf_k
        self_attn_module.v_cache_odd = buf_v

    k_full = buf_k[:, :end]
    v_full = buf_v[:, :end]
    return k_full, v_full


def usp_attn_forward_teacache(self, x, grid_sizes, freqs, block_mask, group_idx, cond_flag, num_groups):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads] for one group
        grid_sizes(Tensor): Shape [3], contains (F_group, H, W) for current group
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        group_idx(int): which group within the chunk
        cond_flag(bool): even(cond)/odd(uncond)
        num_groups(int): total groups within the chunk
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    x = x.to(self.q.weight.dtype)
    q, k, v = qkv_fn(x)

    # group-aware rope
    q = rope_apply(q, grid_sizes, freqs, group_idx)
    k = rope_apply(k, grid_sizes, freqs, group_idx)

    # accumulate KV up to current group
    group_size = grid_sizes[0]
    grid_hw = grid_sizes[1] * grid_sizes[2]
    k_full, v_full = _update_and_return_kv(
        self_attn_module=self,
        k=k,
        v=v,
        cond_flag=cond_flag,
        group_idx=group_idx,
        group_size=group_size,
        grid_hw=grid_hw,
        num_groups=num_groups,
        batch_size=b,
    )

    x = flash_attention(q=q, k=k_full, v=v_full, window_size=self.window_size)

    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_dit_forward_teacache(self, x, t, context, update_mask_i, clip_fea=None, y=None, fps=None):
    r"""
    TeaCache-style forward with grouped processing and per-group KV-cache.

    Args:
        x:              video tensor [B, C, F, H, W]
        t:              [B] or [B, F*H*W]
        context:        text embeddings [B, L, Ctxt]
        update_mask_i:  1D tensor of length group_size*num_groups or 2D [num_groups, group_size]
        clip_fea, y, fps: as in i2v path
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # group configuration from model
    group_size = self.group_size
    num_groups = self.num_groups

    if y is not None:
        x = torch.cat([x, y], dim=1)

    # embeddings
    x = self.patch_embedding(x)
    grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)

    # cache latent dims
    self.latent_width = grid_sizes[2]
    self.latent_height = grid_sizes[1]

    x = x.flatten(2).transpose(1, 2)

    # Sequence Parallel: split tokens across ranks before grouping
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    # time embeddings
    with amp.autocast("cuda", dtype=torch.float32):
        if t.dim() == 2:
            b, f = t.shape
            _flag_df = True
        else:
            _flag_df = False
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(self.patch_embedding.weight.dtype)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

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
    context = self.text_embedding(context)
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    # Sequence Parallel: split e0 across ranks before grouping
    if e0.ndim == 4:
        e0 = torch.chunk(e0, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]

    # prepare grouping
    x_chunks = torch.chunk(x, num_groups, dim=1)
    e0_chunks = torch.chunk(e0, num_groups, dim=2)

    # cond/uncond flag & counters (expected to be prepared on model)
    cond_flag = (self.cnt % 2 == 0)

    # derive group forward/update masks
    update_mask_per_group = update_mask_i.view(num_groups, group_size).any(dim=1)
    update_mask_per_group_list = [bool(update_mask_per_group[g].item()) for g in range(num_groups)]

    should_forward_group = [False] * num_groups
    last_true = -1
    for idx in range(num_groups - 1, -1, -1):
        if update_mask_per_group_list[idx]:
            last_true = idx
            break
    for j in range(last_true + 1):
        should_forward_group[j] = True

    out_chunks = [torch.zeros_like(x_g) for x_g in x_chunks]

    for g, (x_g, e0_g) in enumerate(zip(x_chunks, e0_chunks)):
        if not should_forward_group[g]:
            continue

        # set per-group temporal size in grid
        grid_sizes_group = grid_sizes.clone()
        grid_sizes_group[0] = group_size

        kwargs = dict(
            e=e0_g,
            grid_sizes=grid_sizes_group,
            freqs=self.freqs,
            context=context,
            block_mask=self.block_mask,
            group_idx=g,
            cond_flag=cond_flag,
            num_groups=num_groups,
        )

        # TeaCache skipping logic
        modulated_inp = e0_g
        cnt_vec = self.cnt_even if cond_flag else self.cnt_odd
        step_cnt = cnt_vec[g]
        if cond_flag:
            acc = getattr(self, "accumulated_rel_l1_distance_even", {})
            prev = getattr(self, "previous_e0_even", {})
            res = getattr(self, "previous_residual_even", {})
        else:
            acc = getattr(self, "accumulated_rel_l1_distance_odd", {})
            prev = getattr(self, "previous_e0_odd", {})
            res = getattr(self, "previous_residual_odd", {})

        if self.enable_teacache and update_mask_per_group_list[g]:
            if step_cnt < self.ret_steps or step_cnt >= self.cutoff_steps:
                should_calc = True
                acc[g] = 0.0
            else:
                prev_feat = prev[g]
                rescale_func = np.poly1d(self.coefficients)
                dist = rescale_func(((modulated_inp - prev_feat).abs().mean() / prev_feat.abs().mean()).cpu().item())
                acc[g] = acc[g] + dist
                should_calc = acc[g] >= self.teacache_thresh
                if should_calc:
                    acc[g] = 0.0
            prev[g] = modulated_inp.clone()
            if cond_flag:
                self.accumulated_rel_l1_distance_even = acc
                self.previous_e0_even = prev
            else:
                self.accumulated_rel_l1_distance_odd = acc
                self.previous_e0_odd = prev
        else:
            should_calc = True

        if not should_calc:
            x_g = x_g + res[g]
        else:
            ori_g = x_g.clone()
            for block in self.blocks:
                # swap self-attn forward to TeaCache variant if not already
                x_g = block(x_g, **kwargs)
            if update_mask_per_group_list[g]:
                res[g] = x_g - ori_g
                if cond_flag:
                    self.previous_residual_even = res
                else:
                    self.previous_residual_odd = res

        if update_mask_per_group_list[g]:
            cnt_vec[g] = cnt_vec[g] + 1
            if cond_flag:
                self.cnt_even = cnt_vec
            else:
                self.cnt_odd = cnt_vec

        out_chunks[g] = x_g

    self.cnt = self.cnt + 1

    x = torch.cat(out_chunks, dim=1)

    # Sequence Parallel: split e for head if needed
    if e.ndim == 3:
        e = torch.chunk(e, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    x = self.head(x, e)

    # Sequence Parallel: gather tokens back across ranks
    x = get_sp_group().all_gather(x, dim=1)

    grid_sizes[2] = self.latent_width
    grid_sizes[1] = self.latent_height
    grid_sizes[0] = group_size * num_groups

    x = self.unpatchify(x, grid_sizes)
    return x.float()
