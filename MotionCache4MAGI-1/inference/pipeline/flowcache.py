# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import sys
import gc
import math
import torch
import yaml, json
from dataclasses import replace
from types import MethodType
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt

from inference.pipeline import MagiPipeline
from inference.pipeline.video_generate import SampleTransport, find_dit_model
from inference.common import InferenceParams, ModelMetaArgs, PackedCrossAttnParams

from inference.pipeline.utils import get_tensors_memory_usage
from inference.rkv import replace_magi
from inference.rkv.utils import ChunkKVRangeTracker


def no_reuse_steps_first(cnt, n_step, start=None):
    return cnt < n_step

def no_reuse_steps_mid(cnt, n_step, start):
    return start <= cnt < start + n_step


def in_warmup_phase(cnt, warmup_steps):
    """Phase 1: Warmup phase, no cache is used"""
    return cnt < warmup_steps


def in_chunk_wise_only_phase(cnt, warmup_steps, chunk_wise_only_steps):
    """Phase 2: Chunk-wise Only phase, only use chunk-level cache"""
    return warmup_steps <= cnt < warmup_steps + chunk_wise_only_steps


def in_token_wise_phase(cnt, warmup_steps, chunk_wise_only_steps):
    """Phase 3: Token-wise phase, directly use token-level cache"""
    return cnt >= warmup_steps + chunk_wise_only_steps


def apply_temporal_voting(
    reuse_mask: torch.Tensor,
    chunk_token_nums: int,
    num_frames_per_chunk: int
) -> torch.Tensor:
    """
    Apply temporal voting mechanism: For tokens at the same spatial position across different frames,
    enforce consistent reuse decisions through voting.

    Args:
        reuse_mask: [chunk_token_nums] Boolean tensor, True means reuse, False means forward
        chunk_token_nums: Total number of tokens in the chunk
        num_frames_per_chunk: Number of frames contained in each chunk

    Returns:
        Updated reuse_mask, shape unchanged

    Principle:
        - Tokens in chunk are arranged in (T, H, W) order
        - tokens_per_frame = chunk_token_nums // num_frames_per_chunk
        - For each spatial position spatial_idx (0 <= spatial_idx < tokens_per_frame),
          token indices at that position across different frames are: spatial_idx, spatial_idx + tokens_per_frame, ...
        - For each spatial position, count the number of forwards (False) across all frames
        - If forward count > half of total, force all forward; otherwise keep original
    """
    if chunk_token_nums % num_frames_per_chunk != 0:
        raise ValueError(f"chunk_token_nums ({chunk_token_nums}) must be divisible by num_frames_per_chunk ({num_frames_per_chunk})")

    tokens_per_frame = chunk_token_nums // num_frames_per_chunk

    # Reshape reuse_mask to [num_frames_per_chunk, tokens_per_frame]
    # This way each column corresponds to tokens at the same spatial position across different frames
    mask_2d = reuse_mask.reshape(num_frames_per_chunk, tokens_per_frame)  # [T, S]

    # For each spatial position (column), count the number of forwards (False)
    # sum(dim=0) sums along the frame dimension to get the forward count for each spatial position
    forward_count = (~mask_2d).sum(dim=0).float()  # [tokens_per_frame]
    total_count = num_frames_per_chunk

    # Voting rule: If forward count > certain number, force all forward
    # Otherwise keep original
    force_forward = forward_count > 4

    # Only modify spatial positions that need forced forward, keep others unchanged
    # force_forward_per_token: [num_frames_per_chunk, tokens_per_frame]
    force_forward_per_token = force_forward.unsqueeze(0).expand(num_frames_per_chunk, -1)

    # Use where condition: Set positions needing forced forward to False, keep other positions unchanged
    voted_mask_2d = torch.where(
        force_forward_per_token,
        torch.zeros_like(mask_2d, dtype=torch.bool),  # Force forward (False)
        mask_2d  # Keep original
    )

    # Reshape back to original shape
    voted_mask = voted_mask_2d.reshape(chunk_token_nums)

    return voted_mask


def visualize_temporal_weights_distribution(
    temporal_weights: torch.Tensor,
    frame_index: int,
    save_path: str = None,
    title_prefix: str = "Temporal Weights Distribution"
):
    """
    Visualize the distribution of temporal weights

    Args:
        temporal_weights: [total_T, tokens_per_frame] Tensor containing weights for each token in each frame
        frame_index: Frame index to visualize
        save_path: Path to save the image (PNG format), auto-generated if None
        title_prefix: Prefix for the chart title
    """
    if frame_index >= temporal_weights.shape[0]:
        raise ValueError(f"frame_index {frame_index} out of range [0, {temporal_weights.shape[0]-1}]")

    # Extract weights for specified frame and convert to numpy
    frame_weights = temporal_weights[frame_index].cpu().numpy()  # [tokens_per_frame]

    # Sort by weight value in descending order
    sorted_indices = frame_weights.argsort()[::-1]  # Indices sorted in descending order
    sorted_weights = frame_weights[sorted_indices]

    # Create chart
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sorted_weights)), sorted_weights, linewidth=1.5, color='#2E86AB')
    plt.xlabel('Token Index (sorted by weight value, descending)', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.title(f'{title_prefix} - Frame {frame_index}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add statistics information
    mean_weight = sorted_weights.mean()
    std_weight = sorted_weights.std()
    min_weight = sorted_weights.min()
    max_weight = sorted_weights.max()

    stats_text = f'Mean: {mean_weight:.4f} | Std: {std_weight:.4f} | Min: {min_weight:.4f} | Max: {max_weight:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save image
    if save_path is None:
        save_path = f"temporal_weights_frame_{frame_index}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Visualization] Saved temporal weights distribution to: {save_path}")
    plt.close()


def teacache_forward_velocity(self, infer_idx: int, cur_denoise_step: int) -> torch.Tensor:
        # 1. Get current work status
        x = self.xs[infer_idx]
        transport_input = self.transport_inputs[infer_idx]
        batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)

        if self.compress_kv_cache:
            self.total_cache_len = self.total_cache_chunk_nums * (
                self.chunk_width
                * (self.transport_inputs[infer_idx].latent_size[3] // self.model_config.patch_size)
                * (self.transport_inputs[infer_idx].latent_size[4] // self.model_config.patch_size)
            )
            # Initialize query_states storage
            if not hasattr(self, 'chunk_query_states'):
                self.chunk_query_states = {}

            # Initialize tracker
            if not hasattr(self.inference_params[infer_idx], 'kv_chunk_tracker'):
                self.inference_params[infer_idx].kv_chunk_tracker = ChunkKVRangeTracker(
                    total_cache_len=self.total_cache_len,
                    clip_token_nums=chunk_token_nums,
                    max_batch_size=1
                )

        # Initialize data structures based on reuse mode
        if self.token_wise_reuse:
            # Token-wise reuse: use dict with tensor values for token-level caching
            if not hasattr(self, 'token_accumulated_rel_l1') or self.token_accumulated_rel_l1 is None:
                self.token_accumulated_rel_l1 = {}
            if not hasattr(self, 'token_reuse_masks') or self.token_reuse_masks is None:
                self.token_reuse_masks = {}
            if not hasattr(self, 'previous_residual') or self.previous_residual is None:
                self.previous_residual = {}
            # Initialize chunk-level structures (for backward compatibility)
            if not hasattr(self, 'chunk_accumulated_rel_l1') or self.chunk_accumulated_rel_l1 is None:
                self.chunk_accumulated_rel_l1 = {}
            # Initialize temporal weights storage for token-level reuse
            if not hasattr(self, 'temporal_weights') or self.temporal_weights is None:
                self.temporal_weights = {}
            # Initialize continuous reuse tracking for adaptive refresh
            if getattr(self, 'enable_continuous_reuse_tracking', False):
                if not hasattr(self, 'token_continuous_reuse_count') or self.token_continuous_reuse_count is None:
                    self.token_continuous_reuse_count = {}
            # Reset chunk_reuse_flags for each step
            self.chunk_reuse_flags = {i: False for i in range(transport_input.chunk_num)}
        else:
            # Chunk-level reuse: use list with scalar values
            if self.chunk_accumulated_rel_l1 is None:
                self.chunk_accumulated_rel_l1 = [0.0] * transport_input.chunk_num
            if self.previous_residual is None:
                self.previous_residual = [None] * transport_input.chunk_num
            # Each timemustresetallreusemark
            self.chunk_reuse_flags = {i: False for i in range(transport_input.chunk_num)}

        # 2. Extract prefix video KV cache
        (denoise_step_per_stage, denoise_stage, denoise_idx), (
            chunk_offset,
            chunk_start,
            chunk_end,
            t_start,
            t_end,
        ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)

        model_kwargs = dict(chunk_width=self.chunk_width, fwd_extra_1st_chunk=False, num_steps=transport_input.num_steps)
        if hasattr(self, "debug"):
            model_kwargs["debug"] = self.debug
        model_kwargs.update(
            {"denoise_step_per_stage": denoise_step_per_stage, "denoise_stage": denoise_stage, "denoise_idx": denoise_idx, "chunk_num": transport_input.chunk_num
        })
        
        if self.compress_kv_cache:
            model_kwargs.update(
                {"compress_kv": True, "total_cache_len": self.total_cache_len}
            )
        model_kwargs["save_kvcache_every_forward"] = True
        if chunk_offset > 0 and cur_denoise_step == 0:
            self.extract_prefix_video_feature(
                infer_idx, transport_input.prefix_video, transport_input.y, chunk_offset, model_kwargs
            )

        # 3. Prepare inputs
        x_chunk = x[:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width].clone()
        y_chunk = transport_input.y[:, chunk_start:chunk_end]
        mask_chunk = transport_input.emb_masks[:, chunk_start:chunk_end]
        model_kwargs.update(
            {"slice_point": chunk_start, "range_num": chunk_end, "denoising_range_num": chunk_end - chunk_start}
        )
        model_kwargs["chunk_token_nums"] = chunk_token_nums

        # 4. Forward clean chunk and get clean kv
        fwd_extra_1st_chunk = False     # Since each forward now saves KV cache, fwd_extra_1st_chunk is not needed

        # 5. Prepare inputs
        y_chunk_flatten = y_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        mask_chunk_flatten = mask_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        
        denoise_step_of_each_chunk = self.get_denoise_step_of_each_chunk(
            infer_idx, denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=fwd_extra_1st_chunk
        )
        t = self.get_timestep(
            self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=fwd_extra_1st_chunk
        )
        t = t.unsqueeze(0).repeat(x_chunk.size(0), 1)
        
        kv_range = self.generate_kvrange_for_denoising_video(
            infer_idx=infer_idx,
            slice_point=model_kwargs["slice_point"],
            denoising_range_num=model_kwargs["denoising_range_num"],
            denoise_step_of_each_chunk=denoise_step_of_each_chunk,
        )

        # 6. Padding prefix video
        if transport_input.prefix_video is not None:
            x_chunk, t = self.try_pad_prefix_video(
                infer_idx, x_chunk, t, prefix_video_start=model_kwargs["slice_point"] * self.chunk_width
            )

        # 7. Model forward
        forward_fn = find_dit_model(self.model).forward_dispatcher
        nearly_clean_chunk_t = t[0, int(model_kwargs["fwd_extra_1st_chunk"])].item()
        model_kwargs["distill_nearly_clean_chunk"] = (
            nearly_clean_chunk_t > self.engine_config.distill_nearly_clean_chunk_threshold
        )
        model_kwargs["distill_interval"] = self.time_interval[infer_idx][denoise_idx]
        model_kwargs["total_num_steps"] = self.total_forward_step(infer_idx)

        # ============== monkey patch start ============================
        @torch.no_grad()
        def model_forward(
            model_self,
            x,
            t,
            y,
            caption_dropout_mask=None,
            xattn_mask=None,
            kv_range=None,
            inference_params: InferenceParams = None,
            **kwargs,
        ) -> torch.Tensor:
            raw_x = x.clone()
            from einops import rearrange
            # 1. Calculateinputmetric
            metric_x = x.clone()
            metric_x = metric_x * model_self.model_config.x_rescale_factor
            if model_self.model_config.half_channel_vae:
                assert metric_x.shape[1] == 16
                metric_x = torch.cat([metric_x, metric_x], dim=1)
            metric_x = metric_x.float()
            metric_x = model_self.x_embedder(metric_x)
            metric_x = metric_x.to(model_self.model_config.params_dtype)
            metric_x = rearrange(metric_x, "N C T H W -> (T H W) N C").contiguous()

            self.total_num_steps = kwargs['total_num_steps']
            denoise_step_per_stage = kwargs['denoise_step_per_stage']
            kwargs['cur_denoise_step'] = self.cnt

            if kwargs.get("fwd_extra_1st_chunk", False):
                # firstchunknot needed
                metric_x = metric_x[kwargs["chunk_token_nums"]: , :, :]           # fwd_extra_1st_chunk is always False
            if kwargs.get("distill_nearly_clean_chunk", False):
                # lastchunknot needed
                metric_x = metric_x[:-kwargs["chunk_token_nums"], :, :]
            

            # === divide into each chunk ===
            chunk_token_nums = kwargs["chunk_token_nums"]
            assert metric_x.shape[0] % chunk_token_nums == 0
            num_chunks = metric_x.shape[0] // chunk_token_nums
            # split metric_x and x
            # key is chunk id
            metric_chunks = {}
            x_chunks = {}
            # x in, artifact chunk not added
            offset = kwargs['slice_point']
            for i in range(num_chunks):
                start_t = i * chunk_token_nums
                end_t = start_t + chunk_token_nums
                metric_chunks[offset + i] = metric_x[start_t:end_t]

                chunk_width = kwargs["chunk_width"]
                start_idx = i * chunk_width
                end_idx = start_idx + chunk_width
                # x is [B, C, T, H, W], extract temporal chunk: [B, C, chunk_width, H, W]
                x_chunk = x[:, :, start_idx:end_idx, :, :]
                x_chunks[offset + i] = x_chunk


            model_self.discard_nearly_clean_chunk = self.discard_nearly_clean_chunk
            near_clean_chunk_idx = -1
            if self.discard_nearly_clean_chunk:
                kwargs["distill_nearly_clean_chunk"] = False
            else:
                if kwargs.get("distill_nearly_clean_chunk", False):
                    # Addartifactchunk
                    near_clean_chunk_idx = max(x_chunks.keys()) + 1
                    model_self.near_clean_chunk_idx = near_clean_chunk_idx   # for easysubsequentDetermine
                    x_chunks[near_clean_chunk_idx] = x[:, :, -kwargs["chunk_width"]:, :, :]


            if self.cnt == 0 or self.cnt == self.total_num_steps-1:
                pass
            else:
                threshold = self.rel_l1_thresh
                token_threshold = self.token_rel_l1_thresh
                curr_feats = metric_chunks
                prev_feats = self.prev_metric_chunks

                # Determine directly recalculate all when chunk quantity increases
                if self.whole_calc_when_cross and len(curr_feats) > len(prev_feats):
                    if self.token_wise_reuse:
                        # Token-level: reset all tokens for new chunks
                        for i in range(len(curr_feats)):
                            if i not in self.token_accumulated_rel_l1:
                                chunk_token_nums = curr_feats[i].shape[0]
                                self.token_accumulated_rel_l1[i] = torch.zeros(
                                    chunk_token_nums, dtype=torch.float32, device=curr_feats[i].device
                                )
                                self.token_reuse_masks[i] = torch.zeros(
                                    chunk_token_nums, dtype=torch.bool, device=curr_feats[i].device
                                )
                            else:
                                # Reset existing chunks (don't reset accumulated)
                                self.token_accumulated_rel_l1[i] = torch.zeros_like(self.token_accumulated_rel_l1[i])
                                self.token_reuse_masks[i] = torch.zeros_like(self.token_reuse_masks[i])
                    else:
                        # Chunk-level: original logic
                        for i in range(len(curr_feats)):
                            self.chunk_reuse_flags[i] = False
                            self.chunk_accumulated_rel_l1[i] = 0.0

                else:
                    # Calculatecommonpartialeachchunk relativeL1metric
                    common_keys = set(curr_feats.keys()) & set(prev_feats.keys())

                    if self.token_wise_reuse:
                        # ============= Token-wise Reuse: Token-level with Three Phases =============
                        for i in sorted(common_keys):
                            curr_feat = curr_feats[i]  # [chunk_token_nums, N, C]
                            prev_feat = prev_feats[i]  # [chunk_token_nums, N, C]
                            chunk_token_nums = curr_feat.shape[0]

                            # get current chunk denoise step count
                            chunk_step_cnt = self.chunk_denoise_count[infer_idx][i]

                            # ============ stage1: Warmupstage ============
                            if in_warmup_phase(chunk_step_cnt, self.warmup_steps):
                                # do not use any cache, reset all status
                                self.chunk_reuse_flags[i] = False
                                self.chunk_accumulated_rel_l1[i] = 0.0

                                if i not in self.token_accumulated_rel_l1:
                                    self.token_accumulated_rel_l1[i] = torch.zeros(
                                        chunk_token_nums, dtype=torch.float32, device=curr_feat.device
                                    )

                                if i not in self.token_reuse_masks:
                                    self.token_reuse_masks[i] = torch.zeros(
                                        chunk_token_nums, dtype=torch.bool, device=curr_feat.device
                                    )

                                # Reset continuous reuse count (not tracking in warmup phase)
                                if getattr(self, 'enable_continuous_reuse_tracking', False):
                                    if i not in self.token_continuous_reuse_count:
                                        self.token_continuous_reuse_count[i] = torch.zeros(
                                            chunk_token_nums, dtype=torch.int32, device=curr_feat.device
                                        )
                                    else:
                                        self.token_continuous_reuse_count[i] = torch.zeros_like(self.token_continuous_reuse_count[i])

                            # ============ stage2: Chunk-wise Onlystage ============
                            elif in_chunk_wise_only_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                                # only use chunk-level cache, do not use token-level cache

                                # Step 1: Chunk-level rel_L1 calculation
                                diff = (curr_feat - prev_feat).abs().mean()
                                denom = prev_feat.abs().mean() + 1e-8
                                rel_l1 = diff / denom

                                accumulated = self.chunk_accumulated_rel_l1[i] + rel_l1.item()
                                # print(f"At step {self.cnt} accumulated: {accumulated}")
                                if accumulated < threshold:
                                    self.chunk_reuse_flags[i] = True
                                    self.chunk_accumulated_rel_l1[i] = accumulated
                                else:
                                    self.chunk_reuse_flags[i] = False
                                    self.chunk_accumulated_rel_l1[i] = 0.0

                                # reset token-level status, ensure do not use token-level cache
                                if i not in self.token_accumulated_rel_l1:
                                    self.token_accumulated_rel_l1[i] = torch.zeros(
                                        chunk_token_nums, dtype=torch.float32, device=curr_feat.device
                                    )
                                else:
                                    self.token_accumulated_rel_l1[i] = torch.zeros_like(self.token_accumulated_rel_l1[i])

                                if i not in self.token_reuse_masks:
                                    self.token_reuse_masks[i] = torch.zeros(
                                        chunk_token_nums, dtype=torch.bool, device=curr_feat.device
                                    )
                                else:
                                    self.token_reuse_masks[i] = torch.zeros_like(self.token_reuse_masks[i])

                                # Reset continuous reuse count (not tracking in chunk_wise_only phase)
                                if getattr(self, 'enable_continuous_reuse_tracking', False):
                                    if i not in self.token_continuous_reuse_count:
                                        self.token_continuous_reuse_count[i] = torch.zeros(
                                            chunk_token_nums, dtype=torch.int32, device=curr_feat.device
                                        )
                                    else:
                                        self.token_continuous_reuse_count[i] = torch.zeros_like(self.token_continuous_reuse_count[i])

                            # ============ stage3: Token-wisestage ============
                            elif in_token_wise_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                                # directly use token-level cache, no longer use chunk-level determine

                                # Initialize continuous reuse tracking (only in token-wise phase)
                                if getattr(self, 'enable_continuous_reuse_tracking', False):
                                    if i not in self.token_continuous_reuse_count:
                                        self.token_continuous_reuse_count[i] = torch.zeros(
                                            chunk_token_nums, dtype=torch.int32, device=curr_feat.device
                                        )

                                # chunk-level no longer reuse, always set to False
                                self.chunk_reuse_flags[i] = False
                                self.chunk_accumulated_rel_l1[i] = 0.0

                                # calculate chunk-level rel_l1 (for token-level "chunk" mode use)
                                diff = (curr_feat - prev_feat).abs().mean()
                                denom = prev_feat.abs().mean() + 1e-8
                                rel_l1 = diff / denom

                                # Token-level rel_L1 calculation
                                # Two modes:
                                #   - "chunk": Use chunk-level rel_l1 for all tokens (with optional temporal weights)
                                #   - "token": Compute per-token rel_l1 independently

                                # Initialize or get accumulated values
                                if i not in self.token_accumulated_rel_l1:
                                    self.token_accumulated_rel_l1[i] = torch.zeros(
                                        chunk_token_nums, dtype=torch.float32, device=curr_feat.device
                                    )
                                    self.token_reuse_masks[i] = torch.zeros(
                                        chunk_token_nums, dtype=torch.bool, device=curr_feat.device
                                    )

                                # Get tokenwise_l1_mode with default "chunk"
                                tokenwise_l1_mode = getattr(self, 'tokenwise_l1_mode', 'chunk')

                                if tokenwise_l1_mode == "token":
                                    # ============= Token-wise Mode: Compute per-token rel_l1 =============
                                    # curr_feat and prev_feat have shape [chunk_token_nums, N, C]
                                    # Compute per-token rel_l1 by averaging over N and C dimensions
                                    diff_per_token = (curr_feat - prev_feat).abs().mean(dim=(1, 2))  # [chunk_token_nums]
                                    denom_per_token = prev_feat.abs().mean(dim=(1, 2)) + 1e-8  # [chunk_token_nums]
                                    token_rel_l1 = diff_per_token / denom_per_token  # [chunk_token_nums]

                                    # Apply temporal weights if available
                                    if hasattr(self, 'temporal_weights') and infer_idx in self.temporal_weights:
                                        temporal_weights = self.temporal_weights[infer_idx]
                                        total_frames, tokens_per_frame = temporal_weights.shape
                                        chunk_id = i

                                        # Vectorized temporal weights application
                                        token_indices = torch.arange(chunk_token_nums, device=curr_feat.device)
                                        global_token_indices = chunk_id * chunk_token_nums + token_indices
                                        frame_indices = global_token_indices // tokens_per_frame
                                        token_in_frame_indices = global_token_indices % tokens_per_frame

                                        # Extract weights using advanced indexing
                                        weights = torch.ones(chunk_token_nums, dtype=torch.float32, device=curr_feat.device)
                                        valid_mask = frame_indices < total_frames
                                        weights[valid_mask] = temporal_weights[frame_indices[valid_mask], token_in_frame_indices[valid_mask]]

                                        weighted_rel_l1 = token_rel_l1 * weights
                                        accumulated = self.token_accumulated_rel_l1[i] + weighted_rel_l1
                                    else:
                                        # No temporal weights, use token-level rel_l1 directly
                                        accumulated = self.token_accumulated_rel_l1[i] + token_rel_l1

                                else:
                                    # ============= Chunk Mode: Use chunk-level rel_l1 for all tokens =============
                                    # Apply temporal weights if available
                                    if hasattr(self, 'temporal_weights') and infer_idx in self.temporal_weights:
                                        # Get temporal weights: [total_frames, tokens_per_frame]
                                        temporal_weights = self.temporal_weights[infer_idx]

                                        # Get chunk offset
                                        offset = kwargs['slice_point']
                                        chunk_id = i

                                        # Calculate tokens per frame from temporal weights shape
                                        total_frames, tokens_per_frame = temporal_weights.shape

                                        # Apply temporal weights to chunk-level rel_l1 for each token (VECTORIZED)
                                        token_indices = torch.arange(chunk_token_nums, device=curr_feat.device)
                                        global_token_indices = chunk_id * chunk_token_nums + token_indices
                                        frame_indices = global_token_indices // tokens_per_frame
                                        token_in_frame_indices = global_token_indices % tokens_per_frame

                                        # Extract weights using advanced indexing
                                        weights = torch.ones(chunk_token_nums, dtype=torch.float32, device=curr_feat.device)
                                        valid_mask = frame_indices < total_frames
                                        weights[valid_mask] = temporal_weights[frame_indices[valid_mask], token_in_frame_indices[valid_mask]]

                                        weighted_rel_l1 = rel_l1.item() * weights
                                        accumulated = self.token_accumulated_rel_l1[i] + weighted_rel_l1
                                    else:
                                        # No temporal weights, use unweighted chunk-level rel_l1
                                        weighted_rel_l1 = torch.full((chunk_token_nums,), rel_l1.item(),
                                                                     dtype=torch.float32, device=curr_feat.device)
                                        accumulated = self.token_accumulated_rel_l1[i] + weighted_rel_l1


                                # Calculate dynamic threshold based on continuous reuse count
                                if getattr(self, 'enable_continuous_reuse_tracking', False):
                                    max_count = getattr(self, 'continuous_reuse_max_count', None)

                                    if max_count is not None:
                                        # Mode 2: Force forward after N consecutive reuses
                                        # First, calculate reuse_mask using normal threshold
                                        reuse_mask = accumulated < token_threshold
                                        # Then, force forward tokens that have reached max_count
                                        forced_forward_mask = self.token_continuous_reuse_count[i] >= max_count
                                        reuse_mask = reuse_mask & ~forced_forward_mask
                                    else:
                                        # Mode 1: Dynamic threshold (gradually lower threshold)
                                        decay_mode = getattr(self, 'continuous_reuse_decay_mode', 'exponential')
                                        decay_factor = getattr(self, 'continuous_reuse_decay_factor', 0.1)

                                        if decay_mode == 'exponential':
                                            # Formula A: threshold * decay_factor^count
                                            # decay_factor = 0.1 means multiply by 0.9 each time (10% decay)
                                            decay_rate = 1.0 - decay_factor
                                            dynamic_threshold = token_threshold * (decay_rate ** self.token_continuous_reuse_count[i])
                                        else:  # linear
                                            # Formula B: threshold / (1 + decay_factor * count)
                                            # decay_factor = 0.1 means divide by (1 + 0.1*count)
                                            dynamic_threshold = token_threshold / (1 + decay_factor * self.token_continuous_reuse_count[i])

                                        reuse_mask = accumulated < dynamic_threshold
                                else:
                                    # Original logic (backward compatibility)
                                    reuse_mask = accumulated < token_threshold


                                # ============= Apply token reuse ratio limit =============
                                # Determine which ratio to use: dynamic or fixed
                                if (self.initial_token_reuse_ratio is not None and
                                    self.final_token_reuse_ratio is not None):
                                    # Use dynamic ratio (linear interpolation in token-wise phase)
                                    token_wise_start_step = self.warmup_steps + self.chunk_wise_only_steps
                                    token_wise_total_steps = self.total_num_steps - token_wise_start_step

                                    # Calculate progress within token-wise phase
                                    if token_wise_total_steps > 0:
                                        progress = (chunk_step_cnt - token_wise_start_step) / token_wise_total_steps
                                        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
                                    else:
                                        progress = 1.0

                                    # Linear interpolation
                                    current_ratio = (self.initial_token_reuse_ratio +
                                                    (self.final_token_reuse_ratio - self.initial_token_reuse_ratio) * progress)
                                elif self.max_token_reuse_ratio < 1.0:
                                    # Use fixed ratio (backward compatibility)
                                    current_ratio = self.max_token_reuse_ratio
                                else:
                                    # No ratio limit
                                    current_ratio = 1.0

                                # Apply ratio limit if enabled
                                if current_ratio < 1.0:
                                    num_total_tokens = reuse_mask.numel()
                                    num_reusable_tokens = reuse_mask.sum().item()

                                    # Calculate maximum allowed reusable tokens
                                    max_reusable_tokens = int(num_total_tokens * current_ratio)

                                    # If exceeded, keep only the most stable tokens (lowest accumulated rel_l1)
                                    if num_reusable_tokens > max_reusable_tokens:
                                        # Get accumulated values for reusable tokens only
                                        reusable_accumulated = accumulated[reuse_mask]

                                        # Find the threshold value for the max_reusable_tokens-th smallest accumulated value
                                        # Get the max_reusable_tokens-th smallest value as threshold
                                        threshold_value = torch.topk(reusable_accumulated, max_reusable_tokens, largest=False).values[-1]

                                        # Update reuse_mask: only tokens with accumulated <= threshold_value are reused
                                        reuse_mask = accumulated <= threshold_value

                                # Update: reset tokens that exceed threshold, keep others
                                self.token_accumulated_rel_l1[i] = torch.where(
                                    reuse_mask,
                                    accumulated,
                                    torch.zeros_like(accumulated)
                                )
                                self.token_reuse_masks[i] = reuse_mask

                                # ============= Applytemporalvotingmechanism =============
                                if getattr(self, 'enable_temporal_voting', False):
                                    # calculate frame count for each chunk
                                    # chunk_width is frame count on temporal dimension
                                    # t_patch_size is temporal patch size
                                    t_patch_size = model_self.model_config.t_patch_size
                                    num_frames_per_chunk = kwargs["chunk_width"] // t_patch_size

                                    # Applyvotingmechanism
                                    reuse_mask = apply_temporal_voting(
                                        reuse_mask=reuse_mask,
                                        chunk_token_nums=chunk_token_nums,
                                        num_frames_per_chunk=num_frames_per_chunk
                                    )

                                    # Update reuse_mask
                                    self.token_reuse_masks[i] = reuse_mask

                                # Update continuous reuse count
                                # Reused tokens: count + 1
                                # Forward tokens: count reset to 0
                                if getattr(self, 'enable_continuous_reuse_tracking', False):
                                    self.token_continuous_reuse_count[i] = torch.where(
                                        reuse_mask,
                                        self.token_continuous_reuse_count[i] + 1,
                                        torch.zeros_like(self.token_continuous_reuse_count[i])
                                    )
                            else:
                                # Should not reach here
                                raise ValueError(f"Invalid step count: {chunk_step_cnt}, warmup_steps: {self.warmup_steps}, chunk_wise_only_steps: {self.chunk_wise_only_steps}")

                    else:
                        # ============= Chunk-level Reuse (Original) =============
                        for i in sorted(common_keys):
                            # differentreusemode
                            if self.no_reuse_mode == 'first':
                                no_reuse_steps = no_reuse_steps_first
                            elif self.no_reuse_mode == 'mid':
                                no_reuse_steps = no_reuse_steps_mid
                            else:
                                raise ValueError(f"Unknown no_reuse_mode: {self.no_reuse_mode}")

                            # each chunk do not perform reuse for first n steps
                            if no_reuse_steps(self.chunk_denoise_count[infer_idx][i], self.warmup_steps, 5):
                                self.chunk_reuse_flags[i] = False
                                self.chunk_accumulated_rel_l1[i] = 0.0
                            else:
                                curr_feat = curr_feats[i]
                                prev_feat = prev_feats[i]

                                diff = (curr_feat - prev_feat).abs().mean()
                                denom = prev_feat.abs().mean() + 1e-8
                                rel_l1 = diff / denom

                                accumulated = self.chunk_accumulated_rel_l1[i] + rel_l1.item()
                                if accumulated < threshold:
                                    self.chunk_reuse_flags[i] = True
                                    self.chunk_accumulated_rel_l1[i] = accumulated
                                else:
                                    self.chunk_reuse_flags[i] = False
                                    self.chunk_accumulated_rel_l1[i] = 0.0

            # Nearly clean chunk handling
            if self.token_wise_reuse:
                # Token-wise: chunk-levelno longerreuse，retainnearly clean chunk
                slice_point = kwargs["slice_point"]
                if self.chunk_reuse_flags.get(slice_point, False):
                    # Chunk-level reuse，drop nearly clean chunk
                    x_chunks.pop(near_clean_chunk_idx, None)
            else:
                # Chunk-level: original logic
                if self.chunk_reuse_flags[kwargs["slice_point"]] == True:
                    # nearly clean chunk canreuse，thendirectlydiscardartifactchunk
                    x_chunks.pop(near_clean_chunk_idx, None)

            self.prev_metric_chunks = {i: f.clone().detach() for i, f in metric_chunks.items()}

            # === Only forward chunks/tokens that are not reused ===
            current_infer_outputs = {}

            if self.token_wise_reuse:
                # ============= Token-wise Forward: Three Phases =============
                for i in sorted(x_chunks.keys()):
                    x_i = x_chunks[i]  # [B, C, chunk_width, H, W]
                    chunk_id = i
                    chunk_token_nums = kwargs["chunk_token_nums"]

                    # get current chunk denoise step count
                    chunk_step_cnt = self.chunk_denoise_count[infer_idx][i]

                    # ============ stage1: Warmupstage ============
                    if in_warmup_phase(chunk_step_cnt, self.warmup_steps):
                        # do not use any cache, normal forward entire chunk
                        t_i = t[:, chunk_id-offset:chunk_id-offset+1]
                        y_i = y[chunk_id-offset:chunk_id-offset+1]
                        xattn_mask_i = xattn_mask[chunk_id-offset:chunk_id-offset+1]

                        kwargs["start_chunk_id"] = chunk_id
                        kwargs["end_chunk_id"] = chunk_id + 1
                        kwargs["denoising_range_num"] = 1
                        kwargs["is_sparse_forward"] = False

                        # artifactchunknot neededSavekv cache
                        if chunk_id == near_clean_chunk_idx:
                            kwargs["distill_nearly_clean_chunk"] = True
                        else:
                            kwargs["distill_nearly_clean_chunk"] = False

                        # --- Pre-process ---
                        if self.compress_kv_cache:
                            assert kwargs["total_cache_len"] % chunk_token_nums == 0
                            if self.inference_params[infer_idx].kv_compressed:
                                kv_range = generate_dynamic_kv_range(self, infer_idx, chunk_id, x_chunks.keys(), kwargs, near_clean_chunk_idx)

                        kwargs["near_clean_chunk_idx"] = near_clean_chunk_idx
                        (processed_x, condition, condition_map, y_xattn_flat, rope, meta_args) = model_self.forward_pre_process(
                            x_i, t_i, y_i, caption_dropout_mask, xattn_mask_i, kv_range, **kwargs
                        )

                        if not model_self.pre_process:
                            processed_x = pp_scheduler().recv_prev_data(processed_x.shape, processed_x.dtype)
                            model_self.videodit_blocks.set_input_tensor(processed_x)
                        else:
                            processed_x = processed_x.clone()

                        # --- Transformer Forward ---
                        out = model_self.videodit_blocks.forward(
                            hidden_states=processed_x,
                            condition=condition,
                            condition_map=condition_map,
                            y_xattn_flat=y_xattn_flat,
                            rotary_pos_emb=rope,
                            inference_params=inference_params,
                            meta_args=meta_args,
                        )

                        if self.compress_kv_cache:
                            for layer in model_self.videodit_blocks.layers:
                                layer_num = layer.self_attention.layer_number
                                if hasattr(layer.self_attention, '_last_query'):
                                    self.chunk_query_states[layer_num] = layer.self_attention._last_query

                        if not model_self.post_process:
                            pp_scheduler().isend_next(out)
                        out = model_self.forward_post_process(out, meta_args)

                        current_infer_outputs[i] = out.clone().detach()

                    # ============ stage2: Chunk-wise Onlystage ============
                    elif in_chunk_wise_only_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                        # only check chunk-level reuse
                        if self.chunk_reuse_flags.get(i, False):
                            # entire chunk reuse，skip forward
                            continue

                        # chunk not reuse, normal forward entire chunk
                        t_i = t[:, chunk_id-offset:chunk_id-offset+1]
                        y_i = y[chunk_id-offset:chunk_id-offset+1]
                        xattn_mask_i = xattn_mask[chunk_id-offset:chunk_id-offset+1]

                        kwargs["start_chunk_id"] = chunk_id
                        kwargs["end_chunk_id"] = chunk_id + 1
                        kwargs["denoising_range_num"] = 1
                        kwargs["is_sparse_forward"] = False

                        # artifactchunknot neededSavekv cache
                        if chunk_id == near_clean_chunk_idx:
                            kwargs["distill_nearly_clean_chunk"] = True
                        else:
                            kwargs["distill_nearly_clean_chunk"] = False

                        # --- Pre-process ---
                        if self.compress_kv_cache:
                            assert kwargs["total_cache_len"] % chunk_token_nums == 0
                            if self.inference_params[infer_idx].kv_compressed:
                                kv_range = generate_dynamic_kv_range(self, infer_idx, chunk_id, x_chunks.keys(), kwargs, near_clean_chunk_idx)

                        kwargs["near_clean_chunk_idx"] = near_clean_chunk_idx
                        (processed_x, condition, condition_map, y_xattn_flat, rope, meta_args) = model_self.forward_pre_process(
                            x_i, t_i, y_i, caption_dropout_mask, xattn_mask_i, kv_range, **kwargs
                        )

                        if not model_self.pre_process:
                            processed_x = pp_scheduler().recv_prev_data(processed_x.shape, processed_x.dtype)
                            model_self.videodit_blocks.set_input_tensor(processed_x)
                        else:
                            processed_x = processed_x.clone()

                        # --- Transformer Forward ---
                        out = model_self.videodit_blocks.forward(
                            hidden_states=processed_x,
                            condition=condition,
                            condition_map=condition_map,
                            y_xattn_flat=y_xattn_flat,
                            rotary_pos_emb=rope,
                            inference_params=inference_params,
                            meta_args=meta_args,
                        )

                        if self.compress_kv_cache:
                            for layer in model_self.videodit_blocks.layers:
                                layer_num = layer.self_attention.layer_number
                                if hasattr(layer.self_attention, '_last_query'):
                                    self.chunk_query_states[layer_num] = layer.self_attention._last_query

                        if not model_self.post_process:
                            pp_scheduler().isend_next(out)
                        out = model_self.forward_post_process(out, meta_args)

                        current_infer_outputs[i] = out.clone().detach()

                    # ============ stage3: Token-wisestage ============
                    elif in_token_wise_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                        # directlyChecktoken-levelreuse
                        reuse_mask = self.token_reuse_masks.get(i, None)

                        if reuse_mask is None or not reuse_mask.any():
                            # no tokens can reuse, normal forward entire chunk
                            t_i = t[:, chunk_id-offset:chunk_id-offset+1]
                            y_i = y[chunk_id-offset:chunk_id-offset+1]
                            xattn_mask_i = xattn_mask[chunk_id-offset:chunk_id-offset+1]

                            kwargs["start_chunk_id"] = chunk_id
                            kwargs["end_chunk_id"] = chunk_id + 1
                            kwargs["denoising_range_num"] = 1
                            kwargs["is_sparse_forward"] = False

                            # artifactchunknot neededSavekv cache
                            if chunk_id == near_clean_chunk_idx:
                                kwargs["distill_nearly_clean_chunk"] = True
                            else:
                                kwargs["distill_nearly_clean_chunk"] = False

                            # --- Pre-process ---
                            if self.compress_kv_cache:
                                assert kwargs["total_cache_len"] % chunk_token_nums == 0
                                if self.inference_params[infer_idx].kv_compressed:
                                    kv_range = generate_dynamic_kv_range(self, infer_idx, chunk_id, x_chunks.keys(), kwargs, near_clean_chunk_idx)

                            kwargs["near_clean_chunk_idx"] = near_clean_chunk_idx
                            (processed_x, condition, condition_map, y_xattn_flat, rope, meta_args) = model_self.forward_pre_process(
                                x_i, t_i, y_i, caption_dropout_mask, xattn_mask_i, kv_range, **kwargs
                            )

                            if not model_self.pre_process:
                                processed_x = pp_scheduler().recv_prev_data(processed_x.shape, processed_x.dtype)
                                model_self.videodit_blocks.set_input_tensor(processed_x)
                            else:
                                processed_x = processed_x.clone()

                            # --- Transformer Forward ---
                            out = model_self.videodit_blocks.forward(
                                hidden_states=processed_x,
                                condition=condition,
                                condition_map=condition_map,
                                y_xattn_flat=y_xattn_flat,
                                rotary_pos_emb=rope,
                                inference_params=inference_params,
                                meta_args=meta_args,
                            )

                            if self.compress_kv_cache:
                                for layer in model_self.videodit_blocks.layers:
                                    layer_num = layer.self_attention.layer_number
                                    if hasattr(layer.self_attention, '_last_query'):
                                        self.chunk_query_states[layer_num] = layer.self_attention._last_query

                            if not model_self.post_process:
                                pp_scheduler().isend_next(out)
                            out = model_self.forward_post_process(out, meta_args)

                            current_infer_outputs[i] = out.clone().detach()

                        else:
                            # partial tokens can reuse, perform sparse forward
                            # extract non-reused tokens for sparse forward
                            non_reuse_indices = torch.where(~reuse_mask)[0]  # Original positions within chunk
                            num_non_reuse = non_reuse_indices.numel()

                            if num_non_reuse == 0:
                                continue
                                # need forward tokens quantity is 0, that is although chunk-level determined as forward, but token-level determine entire chunk tokens as reuse

                            # Prepare sparse y and xattn_mask
                            y_i = y[chunk_id-offset:chunk_id-offset+1]  # [1, 1, L, D]
                            xattn_mask_i = xattn_mask[chunk_id-offset:chunk_id-offset+1]

                            kwargs["start_chunk_id"] = chunk_id
                            kwargs["end_chunk_id"] = chunk_id + 1
                            kwargs["denoising_range_num"] = 1
                            kwargs["is_sparse_forward"] = True
                            kwargs["sparse_token_indices"] = non_reuse_indices  # Track original positions

                            # artifactchunknot neededSavekv cache
                            if chunk_id == near_clean_chunk_idx:
                                kwargs["distill_nearly_clean_chunk"] = True
                            else:
                                kwargs["distill_nearly_clean_chunk"] = False

                            # --- Pre-process with sparse input ---
                            if self.compress_kv_cache:
                                assert kwargs["total_cache_len"] % chunk_token_nums == 0
                                if self.inference_params[infer_idx].kv_compressed:
                                    kv_range = generate_dynamic_kv_range(self, infer_idx, chunk_id, x_chunks.keys(), kwargs, near_clean_chunk_idx)

                            kwargs["near_clean_chunk_idx"] = near_clean_chunk_idx
                            kwargs["chunk_token_nums_sparse"] = num_non_reuse

                            # For sparse forward, get the full chunk's embeddings first,
                            # then extract embeddings for sparse token positions only
                            x_i_full = x_chunks[i]  # Full chunk [B, C, chunk_width, H, W]

                            # Call forward_pre_process ONCE to get all embeddings and rope
                            (processed_x_full, condition, condition_map, y_xattn_flat, rope_full, meta_args) = model_self.forward_pre_process(
                                x_i_full, t[:, chunk_id-offset:chunk_id-offset+1], y_i,
                                caption_dropout_mask, xattn_mask_i, kv_range, **kwargs
                            )

                            # Extract embeddings for sparse token positions only
                            # processed_x_full shape: [chunk_token_nums, N, D]
                            # rope_full shape: [chunk_token_nums, head_dim]
                            # condition_map shape: [chunk_token_nums, N]
                            # non_reuse_indices: positions within chunk
                            processed_x_sparse = processed_x_full[non_reuse_indices]  # [num_non_reuse, N, D]
                            rope_sparse = rope_full[non_reuse_indices]              # [num_non_reuse, head_dim]
                            condition_map_sparse = condition_map[non_reuse_indices]  # [num_non_reuse, N]

                            if not model_self.pre_process:
                                processed_x_sparse = pp_scheduler().recv_prev_data(processed_x_sparse.shape, processed_x_sparse.dtype)
                                model_self.videodit_blocks.set_input_tensor(processed_x_sparse)
                            else:
                                processed_x_sparse = processed_x_sparse.clone()

                            # ============= modify cross_attn_params for sparse forward =============
                            from dataclasses import replace

                            # create sparse forward dedicated cross_attn_params
                            num_sparse = processed_x_sparse.shape[0]  # sparse token quantity
                            batch_size = processed_x_sparse.shape[1]

                            # recalculate cu_seqlens_q (sparse input)
                            # format: [0, num_sparse * batch_size]
                            sparse_cu_seqlens_q = torch.tensor(
                                [0, num_sparse * batch_size],
                                dtype=torch.int32,
                                device=processed_x_sparse.device
                            )

                            # update max_seqlen_q to sparse token quantity
                            sparse_max_seqlen_q = num_sparse

                            # create sparse cross_attn_params
                            # q_ranges and kv_ranges keep unchanged (they do not affect flash_attn_varlen_func)
                            # cu_seqlens_kv and max_seqlen_kv keep unchanged (caption completely retained)
                            sparse_cross_attn_params = PackedCrossAttnParams(
                                q_ranges=meta_args.cross_attn_params.q_ranges,
                                kv_ranges=meta_args.cross_attn_params.kv_ranges,
                                cu_seqlens_q=sparse_cu_seqlens_q,
                                cu_seqlens_kv=meta_args.cross_attn_params.cu_seqlens_kv,
                                max_seqlen_q=sparse_max_seqlen_q,
                                max_seqlen_kv=meta_args.cross_attn_params.max_seqlen_kv,
                            )

                            # use replace Createnew meta_args（since meta_args is frozen dataclass）
                            sparse_meta_args = replace(meta_args, cross_attn_params=sparse_cross_attn_params)
                            # ==================================================================

                            # --- Transformer Forward (sparse) ---
                            out_sparse = model_self.videodit_blocks.forward(
                                hidden_states=processed_x_sparse,
                                condition=condition,
                                condition_map=condition_map_sparse,  # Use sparse condition_map
                                y_xattn_flat=y_xattn_flat,
                                rotary_pos_emb=rope_sparse,  # Use rope for original positions of sparse tokens
                                inference_params=inference_params,
                                meta_args=sparse_meta_args,  # use sparse meta_args
                            )

                            # === Reassemble sparse output to full chunk before post_process ===
                            # out_sparse shape: [num_non_reuse, N, D]
                            # Need to expand to full chunk: [chunk_token_nums, N, D]
                            chunk_token_nums = kwargs["chunk_token_nums"]

                            # Use current step's input embeddings as base
                            # Note: For reused tokens, their velocity will be zeroed out later,
                            # so using input embeddings or previous outputs makes no difference
                            out_full = processed_x_full.clone()

                            # Replace non-reused positions with new transformer outputs
                            # Ensure dtype matches (processed_x_full might be bfloat16, out_sparse might be float)
                            out_full[non_reuse_indices] = out_sparse.to(out_full.dtype)

                            # === CRITICAL FIX for Sparse Forward ===
                            # For sparse forward, we should ONLY apply velocity to non-reused tokens!
                            # The velocity at reused token positions is garbage (from previous transformer outputs).
                            # Solution: Zero out velocity at reused token positions after unpatchify.

                            # Convert out_full to latent space via unpatchify
                            if not model_self.post_process:
                                pp_scheduler().isend_next(out_full)
                            out = model_self.forward_post_process(out_full, meta_args)  # [N, C, T, H, W]

                            # CRITICAL FIX: Zero out velocity at REUSED token positions!
                            # Reused tokens should keep their previous values, so their velocity should be zero.
                            # Each token corresponds to patch_size^2 * C values in the latent space.
                            # We need to zero out ALL channels for reused tokens.

                            N, C, T, H, W = out.shape
                            spatial_size = T * H * W

                            # Get patch sizes
                            patch_size = self.model_config.patch_size
                            t_patch_size = self.model_config.t_patch_size

                            # Calculate patch grid dimensions (token space dimensions)
                            T_tokens = T // t_patch_size
                            H_tokens = H // patch_size
                            W_tokens = W // patch_size

                            # Create mask for spatial positions (T*H*W)
                            # non_reuse_indices are token positions, each mapping to a t_patch_size x patch_size x patch_size patch region
                            token_mask = torch.zeros(T_tokens * H_tokens * W_tokens,
                                                    dtype=torch.bool, device=out.device)
                            # Mark non-reused tokens in token space
                            token_mask[non_reuse_indices] = True

                            # Reshape token mask to 3D [T_tokens, H_tokens, W_tokens]
                            token_mask_3d = token_mask.reshape(T_tokens, H_tokens, W_tokens)

                            # Upsample token mask to spatial resolution by repeating each token to its patch region
                            velocity_mask_3d = token_mask_3d.repeat_interleave(t_patch_size, dim=0) \
                                                          .repeat_interleave(patch_size, dim=1) \
                                                          .repeat_interleave(patch_size, dim=2)

                            # Flatten to 1D [T*H*W]
                            velocity_mask = velocity_mask_3d.reshape(spatial_size)

                            # Apply mask to all channels: [N, C, T, H, W] -> mask on spatial dims
                            # Reshape out to [N, C, T*H*W] for easier masking
                            out_reshaped = out.reshape(N, C, spatial_size)
                            velocity_mask = velocity_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, spatial_size]

                            # Zero out velocity at reused token positions (all channels)
                            out_reshaped = torch.where(velocity_mask, out_reshaped, torch.zeros_like(out_reshaped))

                            # Reshape back to [N, C, T, H, W]
                            out = out_reshaped.reshape(N, C, T, H, W)

                            # DEBUG: Check for NaN after masking
                            if torch.isnan(out).any():
                                raise ValueError("NaN found in output after masking")

                            current_infer_outputs[i] = {
                                'output': out.clone().detach(),  # Velocity (zero at reused positions)
                                'sparse_indices': non_reuse_indices,
                                'is_sparse': True,
                                'is_integrated': False  # NOT integrated yet
                            }

            else:
                # ============= Chunk-level Forward (Original) =============
                for i in sorted(x_chunks.keys()):
                    if i in self.chunk_reuse_flags and self.chunk_reuse_flags[i]:
                        continue

                    x_i = x_chunks[i]
                    t_i = t[:, i-offset:i-offset+1]
                    y_i = y[i-offset:i-offset+1]
                    xattn_mask_i = xattn_mask[i-offset:i-offset+1]

                    kwargs["start_chunk_id"] = i
                    kwargs["end_chunk_id"] = i + 1
                    kwargs["denoising_range_num"] = 1

                    # artifactchunknot neededSavekv cache
                    if i == near_clean_chunk_idx:
                        kwargs["distill_nearly_clean_chunk"] = True
                    else:
                        kwargs["distill_nearly_clean_chunk"] = False

                    # --- Pre-process ---
                    if self.compress_kv_cache:
                        assert kwargs["total_cache_len"] % kwargs["chunk_token_nums"] == 0
                        # performed kv cache compression, need update kv range
                        if self.inference_params[infer_idx].kv_compressed:
                            kv_range = generate_dynamic_kv_range(self, infer_idx, i, x_chunks.keys(), kwargs, near_clean_chunk_idx)

                    kwargs["near_clean_chunk_idx"] = near_clean_chunk_idx
                    (processed_x, condition, condition_map, y_xattn_flat, rope, meta_args) = model_self.forward_pre_process(
                        x_i, t_i, y_i, caption_dropout_mask, xattn_mask_i, kv_range, **kwargs
                    )

                    if not model_self.pre_process:
                        processed_x = pp_scheduler().recv_prev_data(processed_x.shape, processed_x.dtype)
                        model_self.videodit_blocks.set_input_tensor(processed_x)
                    else:
                        processed_x = processed_x.clone()

                    # --- Transformer Forward ---
                    out = model_self.videodit_blocks.forward(
                        hidden_states=processed_x,
                        condition=condition,
                        condition_map=condition_map,
                        y_xattn_flat=y_xattn_flat,
                        rotary_pos_emb=rope,
                        inference_params=inference_params,
                        meta_args=meta_args,
                    )

                    if self.compress_kv_cache:
                        # Get and store query for subsequent compression
                        for layer in model_self.videodit_blocks.layers:
                            layer_num = layer.self_attention.layer_number
                            if hasattr(layer.self_attention, '_last_query'):
                                self.chunk_query_states[layer_num] = layer.self_attention._last_query

                    if not model_self.post_process:
                        pp_scheduler().isend_next(out)
                    out = model_self.forward_post_process(out, meta_args)

                    current_infer_outputs[i] = out.clone().detach()

            return current_infer_outputs

        @torch.no_grad()
        def new_get_embedding_and_meta(model_self, x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs):
            ###################################
            #          Part1: Embed x         #
            ###################################
            x = model_self.x_embedder(x)  # [N, C, T, H, W]
            batch_size, _, T, H, W = x.shape


            # ================== Only modified here start =======================
            # Prepare necessary variables
            range_num = kwargs["range_num"]
            denoising_range_num = kwargs["denoising_range_num"]
            slice_point = kwargs.get("slice_point", 0)
            frame_in_range = T // denoising_range_num
            prev_clean_T = frame_in_range * slice_point
            # distill_nearly_clean_chunk is True when there is one more chunk
            T_total = (range_num + kwargs.get("distill_nearly_clean_chunk", False)) * frame_in_range

            ###################################
            #          Part2: rope            #
            ###################################
            # caculate rescale_factor for multi-resolution & multi aspect-ratio training
            # the base_size [16*16] is A predefined size based on data:(256x256)  vae: (8,8,4) patch size: (1,1,2)
            # This definition do not have any relationship with the actual input/model/setting.
            # ref_feat_shape is used to calculate innner rescale factor, so it can be float.
            rescale_factor = math.sqrt((H * W) / (16 * 16))
            rope = model_self.rope.get_embed(shape=[T_total, H, W], ref_feat_shape=[T_total, H / rescale_factor, W / rescale_factor])
            # the shape of rope is (T*H*W, -1) aka (seq_length, head_dim), as T is the first dimension, we can directly cut it.
            rope = rope[kwargs["start_chunk_id"] * frame_in_range * H * W : kwargs["end_chunk_id"] * frame_in_range * H * W]
            if rope.shape[0] == 0:
                raise ValueError("Rope shape is zero, please check the slice_point and range_num settings.")

            # ================== Only modified here end =======================

            ###################################
            #          Part3: Embed t         #
            ###################################
            assert t.shape[0] == batch_size, f"Invalid t shape, got {t.shape[0]} != {batch_size}"  # nolint
            assert t.shape[1] == denoising_range_num, f"Invalid t shape, got {t.shape[1]} != {denoising_range_num}"  # nolint
            t_flat = t.flatten()  # (N * denoising_range_num,)
            t = model_self.t_embedder(t_flat)  # (N, D)

            if model_self.engine_config.distill:
                distill_dt_scalar = 2
                if kwargs["num_steps"] == 12:
                    base_chunk_step = 4
                    distill_dt_factor = base_chunk_step / kwargs["distill_interval"] * distill_dt_scalar
                else:
                    distill_dt_factor = kwargs["num_steps"] / 4 * distill_dt_scalar
                distill_dt = torch.ones_like(t_flat) * distill_dt_factor
                distill_dt_embed = model_self.t_embedder(distill_dt)
                t = t + distill_dt_embed
            t = t.reshape(batch_size, denoising_range_num, -1)  # (N, range_num, D)

            ######################################################
            # Part4: Embed y, prepare condition and y_xattn_flat #
            ######################################################
            # (N * denoising_range_num, 1, L, D)
            y_xattn, y_adaln = model_self.y_embedder(y, model_self.training, caption_dropout_mask)

            assert xattn_mask is not None
            xattn_mask = xattn_mask.squeeze(1).squeeze(1)

            # condition: (N, range_num, D)
            y_adaln = y_adaln.squeeze(1)  # (N, D)
            condition = t + y_adaln.unsqueeze(1)

            assert condition.shape[0] == batch_size
            assert condition.shape[1] == denoising_range_num
            seqlen_per_chunk = (T * H * W) // denoising_range_num
            condition_map = torch.arange(batch_size * denoising_range_num, device=x.device)
            condition_map = torch.repeat_interleave(condition_map, seqlen_per_chunk)
            condition_map = condition_map.reshape(batch_size, -1).transpose(0, 1).contiguous()

            # y_xattn_flat: (total_token, D)
            y_xattn_flat = torch.masked_select(y_xattn.squeeze(1), xattn_mask.unsqueeze(-1).bool()).reshape(-1, y_xattn.shape[-1])
            xattn_mask_for_cuda_graph = None

            ######################################################
            # Part5: Prepare cross_attn_params for cross_atten   #
            ######################################################
            # (N * denoising_range_num, L)
            xattn_mask = xattn_mask.reshape(xattn_mask.shape[0], -1)
            y_index = torch.sum(xattn_mask, dim=-1)
            clip_token_nums = H * W * frame_in_range

            cu_seqlens_q = torch.Tensor([0] + ([clip_token_nums] * denoising_range_num * batch_size)).to(torch.int64).to(x.device)
            cu_seqlens_k = torch.cat([y_index.new_tensor([0]), y_index]).to(torch.int64).to(x.device)
            cu_seqlens_q = cu_seqlens_q.cumsum(-1).to(torch.int32)
            cu_seqlens_k = cu_seqlens_k.cumsum(-1).to(torch.int32)
            assert (
                cu_seqlens_q.shape == cu_seqlens_k.shape
            ), f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, cu_seqlens_k.shape: {cu_seqlens_k.shape}"

            xattn_q_ranges = torch.cat([cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1)
            xattn_k_ranges = torch.cat([cu_seqlens_k[:-1].unsqueeze(1), cu_seqlens_k[1:].unsqueeze(1)], dim=1)
            assert (
                xattn_q_ranges.shape == xattn_k_ranges.shape
            ), f"xattn_q_ranges.shape: {xattn_q_ranges.shape}, xattn_k_ranges.shape: {xattn_k_ranges.shape}"

            cross_attn_params = PackedCrossAttnParams(
                q_ranges=xattn_q_ranges,
                kv_ranges=xattn_k_ranges,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=clip_token_nums,
                max_seqlen_kv=model_self.caption_max_length,
            )

            ##################################################
            #  Part6: Prepare core_atten related q/kv range  #
            ##################################################
            q_range = torch.cat([cu_seqlens_q[:-1].unsqueeze(1), cu_seqlens_q[1:].unsqueeze(1)], dim=1)
            flat_kv = torch.unique(kv_range, sorted=True)
            max_seqlen_k = (flat_kv[-1] - flat_kv[0]).cpu().item()

            ardf_meta = dict(
                clip_token_nums=clip_token_nums,
                slice_point=slice_point,
                range_num=range_num,
                denoising_range_num=denoising_range_num,
                q_range=q_range,
                k_range=kv_range,
                max_seqlen_q=clip_token_nums,
                max_seqlen_k=max_seqlen_k,
            )

            return (x, condition, condition_map, rope, y_xattn_flat, xattn_mask_for_cuda_graph, H, W, ardf_meta, cross_attn_params)

        model  = find_dit_model(self.model)
        model.get_embedding_and_meta = MethodType(new_get_embedding_and_meta, model)
        model.forward = MethodType(model_forward, model)
        # ============== monkey patch end ============================

        # CUDA timing for forward_fn
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        velocity = forward_fn(
            x=x_chunk,
            timestep=t,
            y=y_chunk_flatten,
            mask=mask_chunk_flatten,
            kv_range=kv_range,
            inference_params=self.inference_params[infer_idx],
            **model_kwargs,
        )

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"[Step {self.cnt}] forward_fn execution time: {elapsed_time_ms:.3f} ms")

        self.x_chunks[infer_idx] = x_chunk
        self.velocities[infer_idx] = velocity
        return velocity


def teacache_integrate_velocity(self, infer_idx: int, cur_denoise_step: int):
    transport_input = self.transport_inputs[infer_idx]
    x_chunk = self.x_chunks[infer_idx]
    velocity = self.velocities[infer_idx]
    chunk_denoise_count = self.chunk_denoise_count[infer_idx]

    (denoise_step_per_stage, denoise_stage, denoise_idx), (
        chunk_offset,
        chunk_start,
        chunk_end,
        t_start,
        t_end,
    ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)

    chunk_num = x_chunk.shape[2] // self.chunk_width
    offset = chunk_start
    # Calculate tokens per chunk for statistics
    chunk_token_nums = self.chunk_width * (
        transport_input.latent_size[3] // self.model_config.patch_size
    ) * (
        transport_input.latent_size[4] // self.model_config.patch_size
    )
    ori_x_chunk = x_chunk.clone()
    # divide into each chunk
    x_chunks = {}
    for i in range(chunk_num):
        start_idx = i * self.chunk_width
        end_idx = start_idx + self.chunk_width
        x = x_chunk[:, :, start_idx:end_idx]
        x_chunks[offset + i] = x


    # 9. Walk and integrate
    if self.token_wise_reuse:
        # ============= Token-wise Residual Application: Three Phases =============
        for i in range(chunk_num):
            chunk_id = offset + i

            # get current chunk denoise step count
            chunk_step_cnt = chunk_denoise_count[chunk_id]

            # ============ stage1: Warmupstage ============
            if in_warmup_phase(chunk_step_cnt, self.warmup_steps):
                # do not use any cache, normal integrate entire chunk
                assert chunk_id in velocity
                x_new = self.integrate(
                    x_chunks[chunk_id], velocity[chunk_id], self.ts[infer_idx],
                    denoise_step_per_stage, t_start, t_end, denoise_idx, i
                )
                x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width] = x_new

                # Save residual: entire chunk is new, replace completely
                self.previous_residual[chunk_id] = x_new - ori_x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width]

            # ============ stage2: Chunk-wise Onlystage ============
            elif in_chunk_wise_only_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                # Checkchunk-levelreuse
                if self.chunk_reuse_flags.get(chunk_id, False):
                    # entire chunk reuse，directlyApply residual
                    if chunk_id in self.previous_residual:
                        residual = self.previous_residual[chunk_id]
                        x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width] += residual
                    continue

                # chunk not reuse, normal integrate entire chunk
                assert chunk_id in velocity
                x_new = self.integrate(
                    x_chunks[chunk_id], velocity[chunk_id], self.ts[infer_idx],
                    denoise_step_per_stage, t_start, t_end, denoise_idx, i
                )
                x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width] = x_new

                # Save residual: entire chunk is new, replace completely
                self.previous_residual[chunk_id] = x_new - ori_x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width]

            # ============ stage3: Token-wisestage ============
            elif in_token_wise_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                # directlyChecktoken-levelreuse
                reuse_mask = self.token_reuse_masks.get(chunk_id, None)

                if reuse_mask is None or not reuse_mask.any():
                    # no tokens can reuse, normal integrate entire chunk
                    assert chunk_id in velocity
                    x_new = self.integrate(
                        x_chunks[chunk_id], velocity[chunk_id], self.ts[infer_idx],
                        denoise_step_per_stage, t_start, t_end, denoise_idx, i
                    )
                    x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width] = x_new

                    # Save residual: entire chunk is new, replace completely
                    self.previous_residual[chunk_id] = x_new - ori_x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width]

                else:
                    # partial tokens can reuse (sparse forward)
                    # get velocity output (may be sparse dict or full tensor)
                    vel_output = velocity.get(chunk_id, None)

                    # Check if all tokens are reused (full reuse)
                    if vel_output is None:
                        # full reuse: all tokens are reused, directly use x_old + residual
                        x_old = ori_x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width]
                        residual = self.previous_residual[chunk_id]
                        x_new = x_old + residual
                        x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width] = x_new
                        # residual keep unchanged (fully reuse)

                    # Check if velocity is sparse (dict) or full (tensor)
                    elif isinstance(vel_output, dict) and vel_output.get('is_sparse', False):
                        sparse_velocity = vel_output['output']  # [N, C, T, H, W] after unpatchify
                        sparse_indices = vel_output['sparse_indices']  # non-reused token positions

                        # Get spatial dimensions and patch sizes
                        N, C, T, H, W = x_chunks[chunk_id].shape
                        patch_size = self.model_config.patch_size
                        t_patch_size = self.model_config.t_patch_size

                        # Calculate patch grid dimensions (token space dimensions)
                        T_tokens = T // t_patch_size
                        H_tokens = H // patch_size
                        W_tokens = W // patch_size

                        # Step 1: Call integrate with sparse_velocity
                        # sparse_velocity has values at non-reused positions, zeros at reused positions
                        # So integrate will keep reused positions unchanged (x_old), update non-reused positions
                        x_new = self.integrate(
                            x_chunks[chunk_id], sparse_velocity, self.ts[infer_idx],
                            denoise_step_per_stage, t_start, t_end, denoise_idx, i
                        )

                        # Step 2: For reused spatial positions, apply residual instead
                        # Create mask for reused tokens (inverse of sparse_indices)
                        reused_token_mask = torch.ones(T_tokens * H_tokens * W_tokens,
                                                      dtype=torch.bool, device=sparse_indices.device)
                        reused_token_mask[sparse_indices] = False  # False = non-reused, True = reused

                        # Upsample token mask to spatial resolution
                        reused_token_mask_3d = reused_token_mask.reshape(T_tokens, H_tokens, W_tokens)
                        reused_spatial_mask_3d = reused_token_mask_3d.repeat_interleave(t_patch_size, dim=0) \
                                                                         .repeat_interleave(patch_size, dim=1) \
                                                                         .repeat_interleave(patch_size, dim=2)

                        # Reshape to [N, C, T, H, W] for torch.where
                        reused_spatial_mask = reused_spatial_mask_3d.reshape(T, H, W).unsqueeze(0).unsqueeze(0)

                        # Get original x and residual for reused positions
                        x_old = ori_x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width]
                        residual = self.previous_residual[chunk_id]

                        # For reused positions: use x_old + residual
                        # For non-reused positions: use x_new from integrate
                        x_new = torch.where(reused_spatial_mask, x_old + residual, x_new)

                        x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width] = x_new

                        # Update residual (full chunk residual)
                        new_residual = x_new - x_old
                        self.previous_residual[chunk_id] = new_residual

                    else:
                        # Full chunk was computed (normal forward, not sparse)
                        # Entire chunk is recomputed, replace residual completely
                        x_new = self.integrate(
                            x_chunks[chunk_id], vel_output, self.ts[infer_idx],
                            denoise_step_per_stage, t_start, t_end, denoise_idx, i
                        )
                        x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width] = x_new

                        # Update residual: replace completely (no selective update)
                        new_residual = x_new - ori_x_chunk[:, :, i*self.chunk_width:(i+1)*self.chunk_width]
                        self.previous_residual[chunk_id] = new_residual

    else:
        # ============= Chunk-level Residual Application (Original) =============
        for i in range(chunk_num):
            if self.chunk_reuse_flags[offset + i]:
                # reuse
                x_chunk[:, :, i* self.chunk_width:(i + 1) * self.chunk_width] += self.previous_residual[offset + i]

            else:
                # recalculate
                assert (offset + i) in velocity
                x_new = self.integrate(x_chunks[offset + i], velocity[offset + i], self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx, i)
                x_chunk[:, :, i* self.chunk_width:(i + 1) * self.chunk_width] = x_new
                # save residual
                self.previous_residual[offset + i] = \
                    x_chunk[:, :, i * self.chunk_width : (i + 1) * self.chunk_width] - ori_x_chunk[:, :, i * self.chunk_width : (i + 1) * self.chunk_width]

    # Print token forward statistics if enabled
    if self.print_token_stats:
        if self.token_wise_reuse:
            # Token-wise: Three phases
            print(f"[Step {self.cnt}] Token-wise reuse statistics (Three Phases):")
            for i in range(chunk_num):
                chunk_id = offset + i
                chunk_step_cnt = chunk_denoise_count[chunk_id]

                # ============ stage1: Warmupstage ============
                if in_warmup_phase(chunk_step_cnt, self.warmup_steps):
                    print(f"  Chunk {chunk_id}: [Warmup] FULL FORWARD ({chunk_token_nums} tokens)")

                # ============ stage2: Chunk-wise Onlystage ============
                elif in_chunk_wise_only_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                    if self.chunk_reuse_flags.get(chunk_id, False):
                        print(f"  Chunk {chunk_id}: [Chunk-wise] CHUNK REUSE ({chunk_token_nums} tokens)")
                    else:
                        print(f"  Chunk {chunk_id}: [Chunk-wise] FULL FORWARD ({chunk_token_nums} tokens)")

                # ============ stage3: Token-wisestage ============
                elif in_token_wise_phase(chunk_step_cnt, self.warmup_steps, self.chunk_wise_only_steps):
                    reuse_mask = self.token_reuse_masks.get(chunk_id, None)
                    if reuse_mask is not None and reuse_mask.any():
                        # partial tokens reuse，partial forward
                        reused_tokens = reuse_mask.sum().item()
                        forward_tokens = chunk_token_nums - reused_tokens
                        print(f"  Chunk {chunk_id}: [Token-wise] PARTIAL - {forward_tokens} forward, {reused_tokens} reuse (total: {chunk_token_nums})")
                    else:
                        # no tokens reuse, all tokens forwarded
                        print(f"  Chunk {chunk_id}: [Token-wise] FULL FORWARD ({chunk_token_nums} tokens)")
        else:
            # Chunk-level reuse only
            print(f"[Step {self.cnt}] Chunk-level reuse statistics:")
            for i in range(chunk_num):
                chunk_id = offset + i
                if self.chunk_reuse_flags.get(chunk_id, False):
                    print(f"  Chunk {chunk_id}: REUSE ({chunk_token_nums} tokens)")
                else:
                    print(f"  Chunk {chunk_id}: FORWARD ({chunk_token_nums} tokens)")

    # This step is complete
    self.cnt += 1

    # Monitor current memory usage
    current_memory_mb = get_tensors_memory_usage(self.previous_residual, 'MB')

    # Track peak memory usage
    if not hasattr(self, 'peak_residual_memory_mb'):
        self.peak_residual_memory_mb = 0

    if current_memory_mb > self.peak_residual_memory_mb:
        self.peak_residual_memory_mb = current_memory_mb
        print(f"[NEW PEAK] previous_residual: {self.peak_residual_memory_mb:.2f} MB")

    if self.cnt == self.total_num_steps:
        print(f"Final cache memory usage: {get_tensors_memory_usage(self.previous_residual, 'GB')}")
        print(f"Peak previous_residual memory: {self.peak_residual_memory_mb:.2f} MB")
        self.cnt = 0


    # 10. chunk denoise count
    for chunk_index in range(chunk_start, chunk_end):
        chunk_denoise_count[chunk_index] += 1
    self.xs[infer_idx][:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width] = x_chunk
    self.chunk_denoise_count[infer_idx] = chunk_denoise_count


    # ============= Compute temporal weights for each chunk's first denoise step =============
    if self.token_wise_reuse:
        # Get integrated latent (first half of xs)
        x_integrated = self.xs[infer_idx][:self.xs[infer_idx].shape[0] // 2]  # [N, C, total_T, H, W]
        N, C, total_T, H, W = x_integrated.shape

        # Get patch sizes to calculate token dimensions
        patch_size = self.model_config.patch_size
        t_patch_size = self.model_config.t_patch_size

        # Calculate token grid dimensions
        T_tokens = total_T // t_patch_size
        H_tokens = H // patch_size
        W_tokens = W // patch_size
        tokens_per_frame = T_tokens * H_tokens * W_tokens // total_T  # Tokens per frame

        # Initialize temporal weights if not exists: [total_frames, tokens_per_frame]
        if infer_idx not in self.temporal_weights:
            temporal_weights = torch.ones(total_T, tokens_per_frame, dtype=torch.float32, device=x_integrated.device)
        else:
            temporal_weights = self.temporal_weights[infer_idx]

        # Reshape x to [total_T, C, H, W] for per-frame processing
        x_per_frame = x_integrated.squeeze(0).permute(1, 0, 2, 3)  # [total_T, C, H, W]

        # ============= DEBUG: reusefirstchunkweights =============
        DEBUG_REUSE_FIRST_CHUNK_WEIGHTS = False

        # For each chunk that just completed its first denoise step, compute its temporal weights
        for chunk_index in range(chunk_start, chunk_end):
            # Compute temporal weights based on compute_l1_weights_once setting:
            # - True: only on first denoise step (== 1)
            # - False: whenever chunk is initialized (!= -1)
            if (getattr(self, 'compute_l1_weights_once', False) and chunk_denoise_count[chunk_index] == 1) or \
               (not getattr(self, 'compute_l1_weights_once', False) and chunk_denoise_count[chunk_index] != -1):
                chunk_start_frame = chunk_index * self.chunk_width
                chunk_end_frame = (chunk_index + 1) * self.chunk_width

                # ============= DEBUG: reusefirstchunkweights =============
                if DEBUG_REUSE_FIRST_CHUNK_WEIGHTS and chunk_index > chunk_start and hasattr(self, 'temporal_weights_first_chunk') and chunk_start in self.temporal_weights_first_chunk:
                    # reuse first chunk weights mode (modulo by frame count)
                    print(f"[DEBUG] Reusing first chunk weights for chunk {chunk_index} (frames {chunk_start_frame}-{chunk_end_frame})")
                    for t in range(chunk_start_frame, chunk_end_frame):
                        if t == 0:
                            continue
                        frame_in_first_chunk = t % self.chunk_width
                        temporal_weights[t] = self.temporal_weights_first_chunk[chunk_start][frame_in_first_chunk].clone()
                    continue  # skipcurrentchunkweightsCalculate
                # ============= DEBUG END =============

                # For each frame in this chunk, compute temporal difference with previous frame
                for t in range(chunk_start_frame, chunk_end_frame):
                    if t == 0:
                        # First frame has no previous frame, keep weight as 1.0
                        continue
                    
                    # Get current and previous frames
                    x_curr = x_per_frame[t]  # [C, H, W]
                    x_prev = x_per_frame[t - 1]  # [C, H, W]

                    # Compute absolute difference
                    diff = torch.abs(x_curr - x_prev)  # [C, H, W]

                    # Aggregate over channel dimension to get per-token difference
                    diff_per_token = diff.mean(dim=0)  # [H, W] - average over channels

                    # Now we need to aggregate over each patch region to get one scalar per token
                    # Reshape to [H_tokens, patch_size, W_tokens, patch_size] then average over patch dims
                    diff_per_token = diff_per_token.view(H_tokens, patch_size, W_tokens, patch_size)
                    diff_per_token = diff_per_token.mean(dim=(1, 3))  # [H_tokens, W_tokens]

                    # Flatten to [tokens_per_frame]
                    diff_per_token = diff_per_token.flatten()  # [tokens_per_frame]

                    # Normalize to [floor, 1] within this frame
                    floor = self.temporal_weight_floor
                    min_val = diff_per_token.min()
                    max_val = diff_per_token.max()
                    if max_val > min_val:
                        # First normalize to [0, 1]
                        normalized = (diff_per_token - min_val) / (max_val - min_val)

                        # Apply power transformation if enabled (for convex/concave curves)
                        power = getattr(self, 'temporal_weight_power', None)
                        if power is not None and power != 1.0:
                            # Power transformation: power < 1 makes more tokens close to 1 (convex)
                            #                      power > 1 makes more tokens close to floor (concave)
                            normalized = torch.pow(normalized, power)

                        # Map to [floor, 1]: floor + (1 - floor) * normalized
                        diff_per_token = floor + (1.0 - floor) * normalized
                    else:
                        # All tokens have same difference, set to 1
                        diff_per_token = torch.ones_like(diff_per_token)

                    # Store as weights for this frame
                    temporal_weights[t] = diff_per_token

                # ============= DEBUG: Savefirstchunkweights =============
                if DEBUG_REUSE_FIRST_CHUNK_WEIGHTS and chunk_index == chunk_start:
                    if not hasattr(self, 'temporal_weights_first_chunk'):
                        self.temporal_weights_first_chunk = {}
                    self.temporal_weights_first_chunk[chunk_start] = temporal_weights[chunk_start_frame:chunk_end_frame].clone()
                    print(f"[DEBUG] Saved first chunk ({chunk_start}) weights: shape {temporal_weights[chunk_start_frame:chunk_end_frame].shape}")
                # ============= DEBUG END =============

        self.temporal_weights[infer_idx] = temporal_weights

        # ============= Visualize temporal weights distribution (Optional) =============
        # cancelcommenttoenablevisualization
        # if self.cnt == 45:
        #     visualize_temporal_weights_distribution(
        #         temporal_weights=temporal_weights[:6],
        #         frame_index=3,  # specify frame index to visualize (start from 0)
        #         save_path="temporal_weights_frame_chunk0.png",  # optional: customize save path
        #         title_prefix="Temporal Weights Distribution"
        #     )
        #     visualize_temporal_weights_distribution(
        #         temporal_weights=temporal_weights[6:],
        #         frame_index=3,  # specify frame index to visualize (start from 0)
        #         save_path="temporal_weights_frame_chunk1.png",  # optional: customize save path
        #         title_prefix="Temporal Weights Distribution"
        #     )
        #     import pdb; pdb.set_trace()

        # ============= Save temporal weights for visualization =============
        if self.visualize_temporal_weights:
            # Check if current step is in the list of steps to save
            if self.cnt in self.temporal_weights_steps:
                # Check if this is the last chunk (i.e., all chunks processed for this step)
                is_last_chunk = (chunk_end == transport_input.chunk_num)
                if is_last_chunk:
                    # temporal_weights: [total_T, tokens_per_frame]
                    # Need to reshape to [total_T, H_tokens, W_tokens] for spatial visualization
                    # We already have H_tokens and W_tokens from earlier calculation
                    weights_reshaped = temporal_weights.view(total_T, H_tokens, W_tokens)  # [total_T, H_tokens, W_tokens]

                    # Store for visualization in dict with step as key
                    self.final_temporal_weights_masks[self.cnt] = weights_reshaped.clone().detach()
                    self.final_temporal_weights_latent_sizes[self.cnt] = (N, C, total_T, H, W)
                    self.final_temporal_weights_token_dims[self.cnt] = (H_tokens, W_tokens)

                    # Assign to SampleTransport class attributes for pipeline.py to access
                    SampleTransport.final_temporal_weights_masks = self.final_temporal_weights_masks
                    SampleTransport.final_temporal_weights_latent_sizes = self.final_temporal_weights_latent_sizes
                    SampleTransport.final_temporal_weights_token_dims = self.final_temporal_weights_token_dims


    if self.compress_kv_cache:
        # Check if clean chunk compression should be performed
        compress_clean_chunks_to_make_space(self, infer_idx, chunk_start, transport_input)

    # 11. Save temporal difference mask for visualization (if enabled)
    if self.visualize_temporal_diff:
        # Check if current step is in the list of steps to save
        if self.cnt in self.temporal_diff_steps:
            # Check if this is the last chunk (i.e., all chunks processed for this step)
            is_last_chunk = (chunk_end == transport_input.chunk_num)
            if is_last_chunk:
                # Choose data source based on temporal_diff_mode
                if self.temporal_diff_mode == 'noise':
                    # Use model output (predicted noise/velocity)
                    velocity_dict = self.velocities[infer_idx]

                    # Concatenate all chunks in order to get full velocity tensor
                    sorted_chunk_ids = sorted(velocity_dict.keys())
                    velocity_chunks = [velocity_dict[i] for i in sorted_chunk_ids]

                    # Concatenate along temporal dimension (dim=2)
                    # Each chunk has shape [N, C, chunk_width, H, W]
                    # Result has shape [N, C, total_T, H, W]
                    data_for_diff = torch.cat(velocity_chunks, dim=2)
                else:
                    # Use integrated clean latent (default behavior)
                    # self.xs[infer_idx] shape: [2*N, C, total_T, H, W], we only need the first N
                    x_full = self.xs[infer_idx]
                    N = x_full.shape[0] // 2  # Split point
                    data_for_diff = x_full[:N]  # [N, C, total_T, H, W]

                # Check if we have at least 2 frames to compute difference
                if data_for_diff.shape[2] < 2:
                    print(f"[WARNING] Cannot compute temporal difference: only {data_for_diff.shape[2]} frames available, need at least 2")
                else:
                    # Calculate frame-to-frame differences along temporal dimension
                    # diff[t] = |x[t] - x[t-1]|
                    x_diff = torch.abs(data_for_diff[:, :, 1:] - data_for_diff[:, :, :-1])  # [N, C, T-1, H, W]

                    # Average over temporal dimension to get spatial heatmap
                    # This gives us the average frame-to-frame change at each spatial location
                    temporal_heatmap = x_diff.mean(dim=2)  # [N, C, H, W]

                    # Average over channel dimension (C) to get single scalar per spatial location
                    temporal_heatmap = temporal_heatmap.mean(dim=1)  # [N, H, W]

                    # Store for visualization in dict with step as key
                    self.final_temporal_diff_masks[self.cnt] = temporal_heatmap.clone().detach()
                    self.final_temporal_diff_latent_sizes[self.cnt] = data_for_diff.shape  # Store original shape

                    # Assign to SampleTransport class attributes for pipeline.py to access
                    SampleTransport.final_temporal_diff_masks = self.final_temporal_diff_masks
                    SampleTransport.final_temporal_diff_latent_sizes = self.final_temporal_diff_latent_sizes


    # 12. Return clean chunk
    if chunk_denoise_count[chunk_start] == transport_input.num_steps:
        if transport_input.prefix_video is not None:
            prefix_video_length = transport_input.prefix_video.size(2)
            if (chunk_start + 1) * self.chunk_width <= prefix_video_length:
                return None, None

            real_start = max(chunk_start * self.chunk_width, prefix_video_length)

            # Keep the first 4-frames only for I2V Job
            if chunk_start == 0 and prefix_video_length == 1:
                real_start = 0

            clean_chunk, _ = self.xs[infer_idx][:, :, real_start : (chunk_start + 1) * self.chunk_width].chunk(2, dim=0)
            return clean_chunk, chunk_start - chunk_offset
        else:
            clean_chunk, _ = self.xs[infer_idx][
                :, :, chunk_start * self.chunk_width : (chunk_start + 1) * self.chunk_width
            ].chunk(2, dim=0)
            return clean_chunk, chunk_start - chunk_offset
    return None, None


def generate_dynamic_kv_range(self, infer_idx: int, current_chunk_id: int, x_chunks_keys, kwargs, near_clean_chunk_idx=-1):
    """
    Dynamically generate kv_range based on tracker's actual state and compressed layout

    kv_range meaning: The KV cache range that each chunk can see when performing attention
    - normal chunk: [0, total token count of all previous chunks]
    - nearly clean chunk: (last normal chunk end, last normal chunk end + chunk_token_nums] (placed after normal chunk)
    """
    inference_params = self.inference_params[infer_idx]
    tracker = inference_params.kv_chunk_tracker

    # get all currently processing chunk keys (in order)
    kv_ranges = []

    # first process all normal chunks (excluding near_clean_chunk_idx)
    normal_chunks = [chunk_id for chunk_id in x_chunks_keys if chunk_id != near_clean_chunk_idx]

    # CalculateallnormalchunkKV ranges
    for chunk_id in normal_chunks:
        # normal chunk: need to see itself and all previous chunks KV
        # 1. first get already registered chunks
        # 2. then get current currently forwarding all chunks (newest entered not yet registered)
        # 3. take union of both
        all_chunk_ids = tracker.get_all_chunk_ids() + list(normal_chunks)
        chunks_to_include = [cid for cid in all_chunk_ids if cid <= chunk_id]

        # based ontrackerinactualcompressionafterrangeCalculate
        total_tokens = 0
        for cid in chunks_to_include:
            if cid in tracker.get_all_chunk_ids():
                # usecompressionafteractualrange
                s, e = tracker.get_range(cid)
                total_tokens = max(total_tokens, e)  # take maximum value, since KV is cumulative
            else:
                # just entered chunk not yet registered in tracker, but its size is already known
                total_tokens += kwargs["chunk_token_nums"]

        range_start = 0
        range_end = total_tokens
        kv_ranges.append([range_start, range_end])

    # process near_clean_chunk_idx (if exists), it must be the largest, therefore placed at the end
    if near_clean_chunk_idx != -1:
        # calculate last normal chunk end position
        last_normal_chunk_end = 0
        all_chunk_ids = tracker.get_all_chunk_ids() + normal_chunks
        for cid in all_chunk_ids:
            if cid in tracker.get_all_chunk_ids():
                s, e = tracker.get_range(cid)
                last_normal_chunk_end = max(last_normal_chunk_end, e)
            else:
                # just entered chunk not yet registered in tracker, but its size is already known
                last_normal_chunk_end += kwargs["chunk_token_nums"]

        # near_clean_chunkrangeis（lastnormalchunkend, lastnormalchunkend+chunk_token_nums]
        range_start = last_normal_chunk_end
        range_end = last_normal_chunk_end + kwargs["chunk_token_nums"]
        kv_ranges.append([range_start, range_end])

    return torch.tensor(kv_ranges, device='cuda', dtype=torch.int32)


def compress_clean_chunks_to_make_space(self, infer_idx: int, chunk_start: int, transport_input):
    """
    When cache area is full, convert n clean chunks into n-1 chunks through compression to effectively reduce size and free up space for new chunks

    logic: when KV cache area is full, convert multiple clean chunks through compression algorithm, making total size reduce approximately one chunk size,
    so that can retain most information while freeing up space for new chunks
    """

    # Get necessary parameters and status
    inference_params = self.inference_params[infer_idx]
    tracker = inference_params.kv_chunk_tracker

    cache_size = self.total_cache_len
    total_chunks = transport_input.chunk_num
    tokens_per_chunk = self.get_batch_size_and_chunk_token_nums(infer_idx)[1]

    # get current cached chunks
    cached_chunks = tracker.get_all_chunk_ids()
    cached_count = len(cached_chunks)

    # Two conditions to determine if compression is needed:
    # 1. Cache area is full (next_free_idx > total_cache_len)
    # 2. still have new chunks need to enter (cached_count < total_chunks)
    # 3. New chunk is about to enter (i.e., the last denoising chunk's steps equals num_steps/window_size)
    cache_full = tracker.next_free_idx >= cache_size
    has_more_chunks = cached_count < total_chunks
    last_chunk_id = cached_chunks[-1]
    steps_per_stage = transport_input.num_steps // self.window_size
    next_chunk_will_enter = self.chunk_denoise_count[infer_idx][last_chunk_id] == steps_per_stage

    should_compress = cache_full and has_more_chunks and next_chunk_will_enter

    if not should_compress:
        return

    # Get chunk_offset to distinguish prefix video chunks from generated chunks
    chunk_offset = 0
    if transport_input.prefix_video is not None:
        chunk_offset = transport_input.prefix_video.size(2) // self.chunk_width

    # Only truly clean chunks can be compressed:
    # 1. Prefix video chunk (cid < chunk_offset): Always clean, can be compressed
    # 2. Generated chunk (cid >= chunk_offset): Only those that completed denoising are clean
    clean_chunks = []
    for cid in cached_chunks:
        if cid < chunk_offset:
            # Prefix video chunk, always clean
            clean_chunks.append(cid)
        elif cid <= chunk_start:
            # Generated chunk, need to check if denoising is completed
            if self.chunk_denoise_count[infer_idx][cid] == transport_input.num_steps:
                clean_chunks.append(cid)

    active_chunks = [cid for cid in cached_chunks if cid not in clean_chunks]

    if len(clean_chunks) < 2:
        return  # At least 2 chunks are required for compression

    # Get model to access kv_cluster of each layer
    model = find_dit_model(self.model)

    # Perform compression on each layer
    for layer in model.videodit_blocks.layers:
        kv_cluster = layer.self_attention.kv_cluster

        # 1. Extract KV cache of chunks that need compression
        clean_kv_list = []
        clean_lengths = []
        for cid in clean_chunks:
            s, e = tracker.get_range(cid)
            chunk_kv = inference_params.key_value_memory_dict[layer.self_attention.layer_number][s:e, ...]
            clean_kv_list.append(chunk_kv)
            clean_lengths.append(e - s)

        # Concatenate KV that needs compression
        clean_kv = torch.cat(clean_kv_list, dim=0)
        key_clean, value_clean = torch.chunk(clean_kv, 2, dim=-1)

        # 2. Extract KV cache of active chunks that remain unchanged (currently being denoised)
        active_kv_list = []
        active_lengths = []
        for cid in active_chunks:
            s, e = tracker.get_range(cid)
            chunk_kv = inference_params.key_value_memory_dict[layer.self_attention.layer_number][s:e, ...]
            active_kv_list.append(chunk_kv)
            active_lengths.append(e - s)

        query_states = self.chunk_query_states[layer.self_attention.layer_number]
        total_clean_tokens = sum(clean_lengths)  # Total token count of all clean chunks

        # Simple compression budget: Compress n clean chunks into n-1 chunk equivalent size
        # i.e., total token count minus one chunk's token count
        compress_budget = max(total_clean_tokens - tokens_per_chunk, tokens_per_chunk)
        kv_cluster.budget = compress_budget


        # Get latent size information
        latent_size = transport_input.latent_size
        H = latent_size[3] // self.model_config.patch_size
        W = latent_size[4] // self.model_config.patch_size
        T = tokens_per_chunk // (H * W)
        if not tokens_per_chunk % (H * W) == 0:
            import pdb; pdb.set_trace()
        
        # Only key and value that need compression are passed in
        key_compressed, value_compressed, indices = kv_cluster.update_kv(
            key_clean,
            query_states,
            value_clean,
            clean_chunk_tokens=total_clean_tokens,
            latent_size_t=T,
            latent_size_h=H,
            latent_size_w=W,
        )

        # 4. Reassemble the compressed KV cache order
        final_kv_parts = []
        final_chunk_ids = []
        final_lengths = []

        # 4.1 Add compressed chunk (compressed part)
        compressed_kv = torch.cat([key_compressed, value_compressed], dim=-1)
        final_kv_parts.append(compressed_kv)

        # Calculate the length corresponding to each chunk after compression
        all_lengths_after_compress = []
        start_idx = 0
        # TODO: This part has some issues temporarily, different heads retain different ranges, but in our setup weattend to to all previous chunks' KV cache, so it doesn't matter
        indices_1d = indices[:, 0, 0]  # shape: (num_to_keep,)
        # Iterate through chunks that need compression
        for chunk_id, chunk_len in zip(clean_chunks, clean_lengths):
            if chunk_id in clean_chunks:
                end_idx = start_idx + chunk_len
                # Only count selected tokens within range
                mask = (indices_1d >= start_idx) & (indices_1d < min(end_idx, total_clean_tokens))
                kept_in_chunk = mask.sum().item()
                all_lengths_after_compress.append(kept_in_chunk)
                start_idx = end_idx

        final_chunk_ids.extend(clean_chunks)
        final_lengths.extend(all_lengths_after_compress[:len(clean_chunks)])

        # 4.2 Add unchanged active chunk
        for i, chunk_kv in enumerate(active_kv_list):
            final_kv_parts.append(chunk_kv)
            final_chunk_ids.append(active_chunks[i])
            # for active_chunks, need to get corresponding length from all_lengths_after_compress
            active_chunk_length = active_lengths[i]
            final_lengths.append(active_chunk_length)

        # Concatenate final KV cache
        final_kv = torch.cat(final_kv_parts, dim=0)

        # 5. Update KV cache
        total_kv_len = final_kv.size(0)
        inference_params.key_value_memory_dict[layer.self_attention.layer_number][:total_kv_len, ...] = final_kv
        inference_params.key_value_memory_dict[layer.self_attention.layer_number][total_kv_len:, ...] = 0.0

    # 6. update compressed range, must be outside the loop, since all layer trackers are shared
    current_start = 0
    new_ranges = {}
    for cid, length in zip(final_chunk_ids, final_lengths):
        new_end = current_start + length
        new_ranges[cid] = (current_start, new_end)
        current_start = new_end

    tracker.update_ranges_after_compression(new_ranges)

    # Mark KV cache as compressed
    self.inference_params[infer_idx].kv_compressed = True


def load_config(config_path):
    _, ext = os.path.splitext(config_path)
    with open(config_path, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MagiPipeline with different modes.")
    parser.add_argument('--config_file', type=str, help='Path to the configuration file.')
    parser.add_argument(
        '--mode', type=str, choices=['t2v', 'i2v', 'v2v'], required=True, help='Mode to run: t2v, i2v, or v2v.'
    )
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the pipeline.')
    parser.add_argument('--image_path', type=str, help='Path to the image file (for i2v mode).')
    parser.add_argument('--prefix_video_path', type=str, help='Path to the prefix video file (for v2v mode).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output video.')

    parser.add_argument('--additional_config', type=str, help='Path to additional config file which use teacache and kv cache compression.')
    parser.add_argument('--visualize_reuse_mask', action='store_true', help='Visualize token reuse mask on the output video.')
    parser.add_argument('--temporal_weight_floor', type=float, default=0.0, help='Floor for temporal weight normalization, maps weights to [floor, 1] range.')
    parser.add_argument('--temporal_weight_power', type=float, default=None, help='Power for nonlinear temporal weight normalization (default: None=linear). Values < 1 make more tokens closer to 1 (convex curve), > 1 make more tokens closer to floor.')
    parser.add_argument('--visualize_temporal_diff', action='store_true', help='Visualize temporal difference heatmap on the output video.')
    parser.add_argument('--temporal_diff_step', type=int, nargs='+', default=[0], help='Which denoising step(s) to compute temporal difference mask (0-based). Can be a single step or multiple steps.')
    parser.add_argument('--temporal_diff_mode', type=str, default='clean', choices=['clean', 'noise'], help='Mode for temporal difference calculation: "clean" uses integrated clean latent, "noise" uses model output (predicted noise).')
    parser.add_argument('--visualize_temporal_weights', action='store_true', help='Visualize temporal weights heatmap on the output video.')
    parser.add_argument('--temporal_weights_step', type=int, nargs='+', default=[0], help='Which denoising step(s) to compute temporal weights mask (0-based). Can be a single step or multiple steps.')
    parser.add_argument('--enable_temporal_voting', action='store_true', help='Enable temporal voting: force tokens at the same spatial position across frames to have the same reuse decision via majority voting.')

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if args.additional_config:
        additional_config = load_config(args.additional_config)
        print(f"Loading additional config: {additional_config}")

        for key, value in additional_config.items():
            setattr(args, key, value)
            print(f"Added to args: {key} = {value}")
    else:
        print("No additional config provided.")

    print(f"TeaCache config arguments: {args}")

    if args.print_peak_memory:
        # Check if GPU is available and reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            device = torch.cuda.current_device()
            print(f"Running on GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory before pipeline: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
        else:
            print("CUDA not available, running on CPU")

    # TeaCache
    SampleTransport.rel_l1_thresh = args.rel_l1_thresh
    SampleTransport.token_rel_l1_thresh = getattr(args, 'token_rel_l1_thresh', args.rel_l1_thresh)
    SampleTransport.chunk_accumulated_rel_l1 = 0
    SampleTransport.previous_modulated_input = None
    SampleTransport.previous_residual = None
    SampleTransport.cnt = 0
    SampleTransport.forward_velocity = teacache_forward_velocity
    SampleTransport.integrate_velocity = teacache_integrate_velocity

    SampleTransport.reuse_times = 0
    SampleTransport.warmup_steps = args.warmup_steps
    SampleTransport.chunk_wise_only_steps = getattr(args, 'chunk_wise_only_steps', 0)
    SampleTransport.previous_output = None
    SampleTransport.discard_nearly_clean_chunk = args.discard_nearly_clean_chunk
    SampleTransport.whole_calc_when_cross = args.whole_calc_when_cross
    SampleTransport.no_reuse_mode = args.no_reuse_mode
    SampleTransport.token_wise_reuse = getattr(args, 'token_wise_reuse', False)
    SampleTransport.temporal_weight_floor = getattr(args, 'temporal_weight_floor', 0.0)
    SampleTransport.temporal_weight_power = getattr(args, 'temporal_weight_power', None)
    SampleTransport.tokenwise_l1_mode = getattr(args, 'tokenwise_l1_mode', 'chunk')
    SampleTransport.compute_l1_weights_once = getattr(args, 'compute_l1_weights_once', False)
    SampleTransport.max_token_reuse_ratio = getattr(args, 'max_token_reuse_ratio', 1.0)  # Default: no limit (100%)
    # Dynamic token reuse ratio parameters (for token-wise phase)
    SampleTransport.initial_token_reuse_ratio = getattr(args, 'initial_token_reuse_ratio', None)
    SampleTransport.final_token_reuse_ratio = getattr(args, 'final_token_reuse_ratio', None)
    # Continuous reuse tracking parameters (for adaptive refresh)
    SampleTransport.enable_continuous_reuse_tracking = getattr(args, 'enable_continuous_reuse_tracking', False)
    SampleTransport.continuous_reuse_max_count = getattr(args, 'continuous_reuse_max_count', None)  # Force forward after N consecutive reuses
    SampleTransport.continuous_reuse_decay_mode = getattr(args, 'continuous_reuse_decay_mode', 'exponential')
    SampleTransport.continuous_reuse_decay_factor = getattr(args, 'continuous_reuse_decay_factor', 0.1)
    SampleTransport.enable_temporal_voting = getattr(args, 'enable_temporal_voting', False)
    # --- Token-level reuse state ---
    SampleTransport.token_accumulated_rel_l1 = None           # Dict: token-level accumulated rel L1
    SampleTransport.token_reuse_masks = None                   # Dict: token-level reuse masks (current step)
    # --- Per-chunk state ---
    SampleTransport.chunk_accumulated_rel_l1 = None           # List[float]: Cumulative rel L1 for each chunk
    SampleTransport.prev_chunk_features = None               # List[Tensor]: Features from previous step for each chunk
    SampleTransport.chunk_reuse_flags = None                   # Whether each chunk is reused in current step
    SampleTransport.log = args.log
    SampleTransport.print_token_stats = getattr(args, 'print_token_stats', False)

    # ============= Reuse Mask Visualization =================
    SampleTransport.visualize_reuse_mask = args.visualize_reuse_mask
    SampleTransport.final_reuse_masks = None  # Store final reuse masks for visualization
    SampleTransport.final_chunk_num = None    # Store total chunk count for mask assembly

    # ============= Temporal Difference Visualization =================
    SampleTransport.visualize_temporal_diff = getattr(args, 'visualize_temporal_diff', False)
    temporal_diff_step_arg = getattr(args, 'temporal_diff_step', 0)
    # Support both int and list for temporal_diff_step
    if isinstance(temporal_diff_step_arg, int):
        SampleTransport.temporal_diff_steps = [temporal_diff_step_arg]
    else:
        SampleTransport.temporal_diff_steps = temporal_diff_step_arg
    SampleTransport.temporal_diff_mode = getattr(args, 'temporal_diff_mode', 'clean')  # 'clean' or 'noise'
    SampleTransport.final_temporal_diff_masks = {}  # Dict: {step: mask} for multiple steps
    SampleTransport.final_temporal_diff_latent_sizes = {}  # Dict: {step: latent_size}

    # ============= Temporal Weights Visualization =================
    SampleTransport.visualize_temporal_weights = getattr(args, 'visualize_temporal_weights', False)
    temporal_weights_step_arg = getattr(args, 'temporal_weights_step', 0)
    # Support both int and list for temporal_weights_step
    if isinstance(temporal_weights_step_arg, int):
        SampleTransport.temporal_weights_steps = [temporal_weights_step_arg]
    else:
        SampleTransport.temporal_weights_steps = temporal_weights_step_arg
    SampleTransport.final_temporal_weights_masks = {}  # Dict: {step: weights_tensor}
    SampleTransport.final_temporal_weights_latent_sizes = {}  # Dict: {step: latent_size}
    SampleTransport.final_temporal_weights_token_dims = {}  # Dict: {step: (H_tokens, W_tokens)}

    # ============= KV Cache Compression =================
    SampleTransport.compress_kv_cache = args.compress_kv_cache
    SampleTransport.total_cache_chunk_nums = args.total_cache_chunk_nums

    # Check mutual exclusivity of token_wise_reuse and compress_kv_cache
    if SampleTransport.token_wise_reuse and SampleTransport.compress_kv_cache:
        raise ValueError(
            "token_wise_reuse and compress_kv_cache cannot be enabled simultaneously. "
            "Token-level reuse requires full chunk query states for KV cache reassembly, "
            "which is incompatible with KV cache compression. "
            "Please set only one of these options to True."
        )

    compression_config = {
        "method_config": {
            "compress_strategy": args.compress_strategy,
            "mix_lambda": args.mix_lambda,
            "query_granularity": args.query_granularity,
            "score_weighting_method": args.score_weighting_method,
            "power": args.power,
        },
    }
    replace_magi(compression_config)

    # debug
    SampleTransport.debug = args.debug

    pipeline = MagiPipeline(args.config_file)

    if args.mode == 't2v':
        pipeline.run_text_to_video(prompt=args.prompt, output_path=args.output_path)
    elif args.mode == 'i2v':
        if not args.image_path:
            print("Error: --image_path is required for i2v mode.")
            sys.exit(1)
        pipeline.run_image_to_video(prompt=args.prompt, image_path=args.image_path, output_path=args.output_path)
    elif args.mode == 'v2v':
        if not args.prefix_video_path:
            print("Error: --prefix_video_path is required for v2v mode.")
            sys.exit(1)
        pipeline.run_video_to_video(prompt=args.prompt, prefix_video_path=args.prefix_video_path, output_path=args.output_path)

    if args.print_peak_memory:
        # Print peak memory usage after pipeline completion
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            cached_memory = torch.cuda.memory_reserved(device) / 1024**3
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3

            print("\n" + "="*50)
            print("GPU Memory Usage Summary:")
            print(f"Peak memory allocated: {peak_memory:.2f} GB")
            print(f"Current memory allocated: {current_memory:.2f} GB")
            print(f"Cached memory reserved: {cached_memory:.2f} GB")
            print(f"Total GPU memory: {total_memory:.2f} GB")
            print(f"Peak memory usage: {(peak_memory/total_memory)*100:.1f}%")
            print("="*50)

            # Clear cache and show final memory
            gc.collect()
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"Memory after cache cleanup: {final_memory:.2f} GB")

if __name__ == "__main__":
    main()