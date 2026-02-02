import math
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

#################################################################
###################### kv cache utilities #######################
#################################################################

def compute_attention_scores(query_states, key_states_cpu, pooling="max"):
    """
    query_states: [q_len, q_heads, head_dim] on GPU
    key_states_cpu: [kv_len, kv_heads, head_dim] on CPU
    """

    q_len, q_heads, head_dim = query_states.shape
    kv_len, kv_heads, _ = key_states_cpu.shape
    query_group_size = q_heads // kv_heads

    device = query_states.device  # GPU

    # print(f"Before computing attention scores, GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")

    if query_group_size == 1:
        chunk_size = 12150

        attn_weights = torch.empty(kv_heads, q_len, kv_len, device=device, dtype=query_states.dtype)

        for i in range(0, kv_len, chunk_size):
            end_i = min(i + chunk_size, kv_len)
            k_chunk = key_states_cpu[i:end_i].to(device)  # Move small chunk to GPU

            attn_chunk = torch.bmm(
                query_states.transpose(0, 1),  # [kv_heads, q_len, head_dim]
                k_chunk.transpose(1, 2)        # [kv_heads, head_dim, chunk_size]
            ) / math.sqrt(head_dim)            # [kv_heads, q_len, chunk_size]

            attn_weights[:, :, i:end_i] = attn_chunk
            del k_chunk, attn_chunk

        return attn_weights

    else:
        # query_states: [q_len, q_heads, head_dim] -> reshape to group
        # We group by query_group, but still compute key in chunks
        query_states = query_states.view(q_len, kv_heads, query_group_size, head_dim)
        # [q_len, kv_heads, g, head_dim] -> permute to [kv_heads, g, q_len, head_dim]
        query_states = query_states.permute(1, 2, 0, 3).contiguous()  # [kv_heads, g, q_len, head_dim]

        if pooling == "mean":
            attn_weights_sum = None
            count = 0
        elif pooling == "max":
            attn_weights_max = None
        else:
            raise ValueError("Pooling method not supported")

        for g in range(query_group_size):
            q_group = query_states[:, g, :, :]  # [kv_heads, q_len, head_dim]

            chunk_size = 12150
            group_attn = torch.empty(kv_heads, q_len, kv_len, device=device, dtype=query_states.dtype)

            for i in range(0, kv_len, chunk_size):
                end_i = min(i + chunk_size, kv_len)
                k_chunk = key_states_cpu[i:end_i].to(device)  # [chunk_size, kv_heads, head_dim]
                k_chunk = k_chunk.permute(1, 2, 0)  # [kv_heads, head_dim, chunk_size]
                attn_chunk = torch.bmm(q_group, k_chunk) / math.sqrt(head_dim)
                group_attn[:, :, i:end_i] = attn_chunk
                del k_chunk, attn_chunk

            # apply pooling over query_group_size dimension
            if pooling == "mean":
                if attn_weights_sum is None:
                    attn_weights_sum = group_attn
                else:
                    attn_weights_sum += group_attn
                count += 1
            elif pooling == "max":
                if attn_weights_max is None:
                    attn_weights_max = group_attn
                else:
                    attn_weights_max = torch.max(attn_weights_max, group_attn)

            del group_attn

        if pooling == "mean":
            attn_weights = attn_weights_sum / count
            del attn_weights_sum
        elif pooling == "max":
            attn_weights = attn_weights_max
            del attn_weights_max

        return attn_weights

def compute_attention_weights_memory_efficient(
    query_states,           # [q_len, q_heads, head_dim] on GPU
    key_states_cpu,         # [kv_len, kv_heads, head_dim] on CPU
    clean_chunk_tokens=None,
    chunk_size=8192,
    pooling="max"
):
    if clean_chunk_tokens is not None:
        key_states_cpu = key_states_cpu[:clean_chunk_tokens]
    kv_len = key_states_cpu.size(0)

    q_len, q_heads, head_dim = query_states.shape
    kv_len, kv_heads, _ = key_states_cpu.shape
    device = query_states.device
    sqrt_d = math.sqrt(head_dim)

    # Verify GQA grouping
    assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads"
    query_group_size = q_heads // kv_heads

    key_states_gpu = key_states_cpu.to(device)

    if query_group_size == 1:
        # ========== No grouping ==========
        # Step 1: Compute per-query max (M) and denominator (D)
        M = torch.full((kv_heads, q_len), -float('inf'), device=device, dtype=torch.float32)
        D = torch.zeros(kv_heads, q_len, device=device, dtype=torch.float32)

        # First pass: compute M and D
        for i in range(0, kv_len, chunk_size):
            end_i = min(i + chunk_size, kv_len)
            # Slice directly from GPU, no need for .to(device)
            k_chunk = key_states_gpu[i:end_i].permute(1, 2, 0)  # [kv_heads, head_dim, chunk_size]

            q = query_states.transpose(0, 1)  # [kv_heads, q_len, head_dim]
            attn_chunk = torch.bmm(q, k_chunk) / sqrt_d  # [kv_heads, q_len, chunk_size]

            # Update per-query max value
            chunk_max = attn_chunk.max(dim=-1, keepdim=True).values  # [H, Q, 1]
            M = torch.maximum(M, chunk_max.squeeze(-1))  # [H, Q]

            # Compute exp(attn - M) and accumulate denominator
            exp_chunk = torch.exp(attn_chunk - M.unsqueeze(-1))  # [H, Q, chunk]
            D += exp_chunk.sum(dim=-1)  # [H, Q]

            # Explicit release (optional, comment out for speed with large chunks)
            del k_chunk, q, attn_chunk, exp_chunk, chunk_max
            # torch.cuda.empty_cache()  # Usually not needed unless memory is extremely tight

        # Step 2: Compute softmax mean over queries
        result = torch.zeros(kv_heads, kv_len, device=device, dtype=torch.float32)

        for i in range(0, kv_len, chunk_size):
            end_i = min(i + chunk_size, kv_len)
            k_chunk = key_states_gpu[i:end_i].permute(1, 2, 0)  # [H, D, chunk]

            q = query_states.transpose(0, 1)  # [H, Q, D]
            attn_chunk = torch.bmm(q, k_chunk) / sqrt_d  # [H, Q, chunk]

            exp_chunk = torch.exp(attn_chunk - M.unsqueeze(-1))  # [H, Q, chunk]
            softmax_chunk = exp_chunk / D.unsqueeze(-1)          # [H, Q, chunk]
            softmax_mean = softmax_chunk.mean(dim=1)             # [H, chunk]

            result[:, i:end_i] = softmax_mean

            del k_chunk, q, attn_chunk, exp_chunk, softmax_chunk, softmax_mean

        return result.to(query_states.dtype)

    else:
        # ========== GQA case: query_group_size > 1 ==========
        # Reshape query: [q_len, q_heads, head_dim] → [q_len, kv_heads, group_size, head_dim]
        query_states = query_states.view(q_len, kv_heads, query_group_size, head_dim)
        # → [kv_heads, group_size, q_len, head_dim]
        query_states = query_states.permute(1, 2, 0, 3).contiguous()

        # Initialize final result
        final_result = None

        # Compute attention for each group
        for g in range(query_group_size):
            q_group = query_states[:, g, :, :]  # [kv_heads, q_len, head_dim]

            # Step 1: Compute M and D for this group
            M = torch.full((kv_heads, q_len), -float('inf'), device=device, dtype=torch.float32)
            D = torch.zeros(kv_heads, q_len, device=device, dtype=torch.float32)

            # First pass
            for i in range(0, kv_len, chunk_size):
                end_i = min(i + chunk_size, kv_len)
                k_chunk = key_states_gpu[i:end_i].permute(1, 2, 0)  # [H, D, chunk]

                attn_chunk = torch.bmm(q_group, k_chunk) / sqrt_d  # [H, Q, chunk]

                chunk_max = attn_chunk.max(dim=-1, keepdim=True).values
                M = torch.maximum(M, chunk_max.squeeze(-1))

                exp_chunk = torch.exp(attn_chunk - M.unsqueeze(-1))
                D += exp_chunk.sum(dim=-1)

                del k_chunk, attn_chunk, exp_chunk, chunk_max

            # Step 2: Compute softmax mean for this group
            group_result = torch.zeros(kv_heads, kv_len, device=device, dtype=torch.float32)

            for i in range(0, kv_len, chunk_size):
                end_i = min(i + chunk_size, kv_len)
                k_chunk = key_states_gpu[i:end_i].permute(1, 2, 0)

                attn_chunk = torch.bmm(q_group, k_chunk) / sqrt_d
                exp_chunk = torch.exp(attn_chunk - M.unsqueeze(-1))
                softmax_chunk = exp_chunk / D.unsqueeze(-1)
                softmax_mean = softmax_chunk.mean(dim=1)  # [H, chunk]

                group_result[:, i:end_i] = softmax_mean

                del k_chunk, attn_chunk, exp_chunk, softmax_chunk, softmax_mean

            # Pooling over groups
            if pooling == "mean":
                if final_result is None:
                    final_result = group_result
                else:
                    final_result += group_result
            elif pooling == "max":
                if final_result is None:
                    final_result = group_result
                else:
                    final_result = torch.maximum(final_result, group_result)
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")

            del group_result, q_group, M, D

        # Final pooling
        if pooling == "mean":
            final_result = final_result / query_group_size

        return final_result.to(query_states.dtype)


# def cal_similarity(
#     key_states,
# ):
#     # key_states shape: [kv_len, kv_heads, head_dim]
#     start = time.time()
#     k = key_states.permute(1, 0, 2).to('cuda')  # shape: [kv_heads, kv_len, head_dim]
#     num_heads = k.shape[0]

#     k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
#     similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2)).to('cpu')

#     for h in range(num_heads):
#         similarity_cos[h].fill_diagonal_(0.0)

#     end = time.time()
#     return similarity_cos.mean(dim=1).softmax(dim=-1)


def cal_similarity(
    key_states,
):
    # [kv_len, H, D] → [H, kv_len, D]
    k = key_states.permute(1, 0, 2).to('cuda')
    H, L, D = k.shape

    # L2 normalize each key vector per head
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)   # [H, L, D]

    # Step 1: Compute sum of all keys per head → [H, D]
    k_sum = k_norm.sum(dim=1)   # Σ_j k_j

    # Step 2: For each key i, compute k_i ⋅ (Σ_j k_j) → [H, L]
    # That is: (k_norm @ k_sum.T) → use bmm for batch
    # k_norm: [H, L, D], k_sum.unsqueeze(-1): [H, D, 1] → bmm → [H, L, 1]
    dot_with_sum = torch.bmm(k_norm, k_sum.unsqueeze(-1)).squeeze(-1)  # [H, L]

    # Step 3: Apply correction for diagonal (since cos(k_i, k_i) = 1 was included in sum)
    # Original: fill_diagonal_(0) then mean(dim=1) ⇒ (total_sum - 1) / L
    if L == 1:
        mean_sim = torch.zeros(H, 1, device=k.device)  # or handle specially
    else:
        mean_sim = (dot_with_sum - 1.0) / L   # [H, L] ← strictly equivalent to original

    avg_sim = mean_sim

    # Step 5: Softmax → final importance-like distribution
    result = avg_sim.softmax(dim=-1).to('cpu')  # move small result to CPU

    return result


class ChunkKVRangeTracker:
    def __init__(self, total_cache_len: int, clip_token_nums: int, max_batch_size: int):
        self.total_cache_len = total_cache_len
        self.clip_token_nums = clip_token_nums
        self.max_batch_size = max_batch_size
        self.tokens_per_chunk = clip_token_nums * max_batch_size
        self.chunk_ranges: Dict[int, Tuple[int, int]] = {}  # chunk_id -> (start, end)
        self.next_free_idx = 0  # Used for sequential allocation when not compressed
        self.registered_chunks_ordered: List[int] = []  # Maintain registration order for compression and concatenation

    def register_chunks(self, chunk_ids: List[int]):
        """Batch register multiple chunks, allocate original space"""
        for cid in chunk_ids:
            if cid in self.chunk_ranges:
                continue
            start = self.next_free_idx
            end = start + self.tokens_per_chunk
            if end > self.total_cache_len:
                import pdb; pdb.set_trace()
                raise ValueError("KV cache is full")
            self.chunk_ranges[cid] = (start, end)
            self.registered_chunks_ordered.append(cid)
            self.next_free_idx = end

    def get_range(self, chunk_id: int) -> Tuple[int, int]:
        if chunk_id not in self.chunk_ranges:
            raise KeyError(f"Chunk {chunk_id} not registered. Call register_chunks first.")
        return self.chunk_ranges[chunk_id]

    def get_all_ranges_previous(self, current_chunk_ids: List[int]) -> List[Tuple[int, int]]:
        # Get KV ranges of all previous chunks
        ranges = []
        if len(current_chunk_ids) > 0:
            min_chunk_id = min(current_chunk_ids)
            for cid in self.registered_chunks_ordered:
                if cid >= min_chunk_id:
                    continue
                ranges.append(self.chunk_ranges[cid])
        else:
            # To adapt to MAGI-1's original logic, should return ranges of all registered chunks
            for cid in self.registered_chunks_ordered:
                ranges.append(self.chunk_ranges[cid])
        return ranges

    def get_all_chunk_ids(self) -> List[int]:
        return self.registered_chunks_ordered.copy()
    
    def update_ranges_after_compression(self, new_ranges: Dict[int, Tuple[int, int]]):
        """Update each chunk's range based on actual compressed length"""
        # Update chunk_ranges
        for cid, (start, end) in new_ranges.items():
            if cid in self.chunk_ranges:
                self.chunk_ranges[cid] = (start, end)

        # Update next_free_idx to max end
        if new_ranges:
            self.next_free_idx = max(end for start, end in new_ranges.values())
        else:
            self.next_free_idx = 0


#################################################################
################### visualization utilities #####################
#################################################################

# Visualize the token eviction pattern for a given head
def visualize_token_eviction(
    output_token_ids, kept_token_indices, tokenizer, head_idx=0
):
    """
    Visualize which tokens are kept vs evicted for a given head

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices: shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
    """
    from IPython.display import HTML

    # Get the kept indices for the specified head
    kept_indices = set(kept_token_indices[head_idx].tolist())

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Build HTML with different colors for kept vs evicted tokens
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")  # Remove space marker
            .replace("Ċ", "\n")  # Convert newline marker to actual newline
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        if idx in kept_indices:
            # Kept tokens in green with bold
            html_parts.append(
                f'<span style="color: green; font-weight: bold;">{token}</span>'
            )
        else:
            # Evicted tokens in gray and lighter
            html_parts.append(f'<span style="color: #999999;">{token}</span>')

    # Join without spaces (since we're now handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)

# Visualize the token eviction pattern for a given heads at each compression step
def visualize_multistep_token_eviction(
    output_token_ids, kept_token_indices_list, tokenizer, head_idx=0, step_idx=-1
):
    """
    Visualize which tokens are kept at each compression step with different colors.
    Later steps are shown in more vibrant colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
    """
    from IPython.display import HTML

    # Get the kept indices for each step for the specified head
    kept_indices_by_step = [
        set(indices[head_idx].tolist()) for indices in kept_token_indices_list
    ]
    num_steps = len(kept_indices_by_step) if step_idx == -1 else 1

    # Generate colors using a distinct color spectrum
    def get_color(step):
        # Use a color spectrum for better distinction between steps
        if num_steps <= 1:
            return "#3498db"  # Default blue if only one step

        # Define a set of distinct colors
        colors = [
            "#e74c3c",  # Red
            "#3498db",  # Blue
            "#2ecc71",  # Green
            "#f39c12",  # Orange
            "#9b59b6",  # Purple
            "#1abc9c",  # Teal
            "#d35400",  # Dark Orange
            "#2980b9",  # Dark Blue
            "#27ae60",  # Dark Green
            "#8e44ad",  # Dark Purple
        ]

        if num_steps <= len(colors):
            # If we have fewer steps than colors, use the colors directly
            return colors[step % len(colors)]
        else:
            # For more steps than colors, interpolate between colors
            # Map step to a position in the color spectrum
            position = (step / (num_steps - 1)) * (len(colors) - 1)
            idx1 = int(position)
            idx2 = min(idx1 + 1, len(colors) - 1)
            fraction = position - idx1

            # Get the two colors to interpolate between
            color1 = colors[idx1]
            color2 = colors[idx2]

            # Convert hex to RGB
            r1, g1, b1 = (
                int(color1[1:3], 16),
                int(color1[3:5], 16),
                int(color1[5:7], 16),
            )
            r2, g2, b2 = (
                int(color2[1:3], 16),
                int(color2[3:5], 16),
                int(color2[5:7], 16),
            )

            # Interpolate
            r = int(r1 * (1 - fraction) + r2 * fraction)
            g = int(g1 * (1 - fraction) + g2 * fraction)
            b = int(b1 * (1 - fraction) + b2 * fraction)

            return f"#{r:02x}{g:02x}{b:02x}"

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        latest_step = -1
        if step_idx == -1:
            # Find the latest step (if any) where this token was kept
            for step, kept_indices in enumerate(kept_indices_by_step[::-1]):
                if idx in kept_indices:
                    latest_step = num_steps - step
                    break

        elif idx in kept_indices_by_step[step_idx]:
            latest_step = num_steps

        # Color the token based on its latest appearance
        if latest_step >= 0:
            color = get_color(latest_step)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)


# Visualize the token eviction pattern for all heads at each compression step
def visualize_multistep_token_eviction_by_head(
    output_token_ids, kept_token_indices_list, tokenizer, step_idx, aggregate=False
):
    """
    Visualize which tokens are kept by which heads with different colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
        aggregate: when set to False, later heads will cover previous heads. when set to `True`, will compute how many times a token are covered by a head.
    """
    from IPython.display import HTML

    # Generate colors using a distinct color spectrum
    def get_color(idx, aggregate):
        # Define a set of distinct colors
        if not aggregate:
            colors = [
                "#3498db",  # Blue
                "#f39c12",  # Orange
                "#9b59b6",  # Purple
                "#1abc9c",  # Teal
                "#d35400",  # Dark Orange
                "#2980b9",  # Dark Blue
                "#27ae60",  # Dark Green
                "#8e44ad",  # Dark Purple
            ]
        else:
            # colors = [
            #     "#D6EAF8",  # Very Light Blue
            #     "#AED6F1",  # Light Blue
            #     "#85C1E9",  # Medium Light Blue
            #     "#5DADE2",  # Medium Blue
            #     "#3498DB",  # Blue
            #     "#2E86C1",  # Medium Dark Blue
            #     "#2874A6",  # Dark Blue
            #     "#1B4F72",  # Very Dark Blue
            # ]
            colors = [
                "#E65100",  # Dark Orange (starting point)
                "#D84315",  # Orange Red
                "#C62828",  # Red Brown
                "#B71C1C",  # Dark Red Brown
                "#A52A00",  # Dark Orange Brown
                "#8B2500",  # Ocher
                "#7C2000",  # Dark Brown
                "#6B1D00"   # Deepest Orange Brown
            ]
        return colors[idx]

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Get kept token id list
    token_indices_lst = kept_token_indices_list[
        step_idx
    ]  # shape: (kv_head, num_kept_tokens)
    token_indices_dict = {
        i: set(token_indices_lst[i].tolist()) for i in range(token_indices_lst.shape[0])
    }

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        color_idx = -1
        for head_idx, kept_token_set in token_indices_dict.items():
            if idx in kept_token_set:
                if aggregate:
                    color_idx += 1
                else:
                    color_idx = head_idx

        # Color the token based on its latest appearance
        if color_idx >= 0:
            color = get_color(color_idx, aggregate)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)


# Visualize the token eviction score for all heads at each compression step
def visualize_multistep_token_eviction_score_by_head(
    output_token_ids, kept_token_indices_list, score_list, tokenizer, step_idx, head_idx
):
    """
    Visualize which tokens are kept by which heads with different colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
        aggregate: when set to False, later heads will cover previous heads. when set to `True`, will compute how many times a token are covered by a head.
    """
    from IPython.display import HTML

    # Generate colors using the common blue to yellow heatmap color spectrum
    def get_color(score):
        # Define the blue and yellow colors
        colors = [
            "#D6EAF8",  # Very Light Blue
            "#AED6F1",  # Light Blue
            "#85C1E9",  # Medium Light Blue
            "#5DADE2",  # Medium Blue
            "#3498DB",  # Blue
            "#2E86C1",  # Medium Dark Blue
            "#2874A6",  # Dark Blue
            "#1B4F72",  # Very Dark Blue
        ]

        if score <= 0:
            return colors[0]

        # Calculate the position of the step within the range of colors
        position = score * (len(colors) - 1)

        # Determine the indices for interpolation
        idx1 = int(position)
        idx2 = min(idx1 + 1, len(colors) - 1)
        fraction = position - idx1

        # Get the two colors to interpolate between
        color1 = colors[idx1]
        color2 = colors[idx2]

    # Convert hex to RGB for color1
        r1, g1, b1 = (
            int(color1[1:3], 16),
            int(color1[3:5], 16),
            int(color1[5:7], 16),
        )
        # Convert hex to RGB for color2
        r2, g2, b2 = (
            int(color2[1:3], 16),
            int(color2[3:5], 16),
            int(color2[5:7], 16),
        )

        # Interpolate between the two colors
        r = int(r1 * (1 - fraction) + r2 * fraction)
        g = int(g1 * (1 - fraction) + g2 * fraction)
        b = int(b1 * (1 - fraction) + b2 * fraction)

        # Return the interpolated color as a hex code
        return f"#{r:02x}{g:02x}{b:02x}"


    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Get kept token id list
    token_indices_lst = kept_token_indices_list[
        step_idx
    ]  # shape: (kv_head, num_kept_tokens)
    token_indices_dict = {
        i: token_indices_lst[i].tolist() for i in range(token_indices_lst.shape[0])
    }

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<｜begin of sentence｜>", "[BOS]")
            .replace("<｜end of sentence｜>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        score = -1
        # for head_idx, kept_token_set in token_indices_dict.items():
            # if idx in kept_token_set:

        # Locate the index of idx in kept_token_set
        if idx in token_indices_dict[head_idx]:
            index = token_indices_dict[head_idx].index(idx)
        else:
            # Handle the case where idx is not in the list
            index = -1  # or other appropriate value

        if index != -1:
            score = score_list[step_idx][head_idx][index].item()

            # if aggregate:
            #     color_idx += 1
            # else:
            #     color_idx = head_idx

        # Color the token based on its latest appearance
        if score >= 0:
            color = get_color(score)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)


def visualize_tensor_heads(tensor, save_path, figsize=None, title=None):
    """
    Visualize a tensor with shape [head, seq_len] where each head is plotted as a subplot.

    Args:
        tensor: torch.Tensor with shape [head, seq_len]
        save_path: str, path to save the visualization
        figsize: tuple, optional figure size (width, height)
        title: str, optional title for the entire figure
    """
    # Convert tensor to numpy if needed
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().float().numpy()
    else:
        tensor_np = np.array(tensor)

    # Check tensor shape
    if len(tensor_np.shape) != 2:
        raise ValueError(f"Expected tensor shape [head, seq_len], got {tensor_np.shape}")

    num_heads, seq_len = tensor_np.shape

    # Calculate subplot grid layout
    cols = min(4, num_heads)  # Maximum 4 columns
    rows = (num_heads + cols - 1) // cols  # Calculate rows needed

    # Set default figure size if not provided
    if figsize is None:
        figsize = (cols * 4, rows * 3)

    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()

    # Plot each head
    for head_idx in range(num_heads):
        ax = axes_flat[head_idx]

        # Plot the tensor values for this head
        x_axis = range(seq_len)
        y_values = tensor_np[head_idx]

        ax.plot(x_axis, y_values, linewidth=1.5, alpha=0.8)
        ax.set_title(f'Head {head_idx}', fontsize=10)
        ax.set_xlabel('Sequence Length', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_heads, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Tensor Visualization - {num_heads} Heads, {seq_len} Sequence Length',
                    fontsize=14, fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle

    # Save the figure
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {save_path}")