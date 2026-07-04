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
import gc
import math
import sys
import torch
from dataclasses import replace
from types import MethodType

from inference.pipeline import MagiPipeline
from inference.pipeline.video_generate import SampleTransport, find_dit_model
from inference.common import InferenceParams, ModelMetaArgs, PackedCrossAttnParams

from inference.pipeline.utils import get_tensors_memory_usage


def teacache_forward_velocity(self, infer_idx: int, cur_denoise_step: int) -> torch.Tensor:
        # 1. Get current work status
        x = self.xs[infer_idx]
        transport_input = self.transport_inputs[infer_idx]

        # 2. Extract prefix video KV cache
        (denoise_step_per_stage, denoise_stage, denoise_idx), (
            chunk_offset,
            chunk_start,
            chunk_end,
            t_start,
            t_end,
        ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)

        model_kwargs = dict(chunk_width=self.chunk_width, fwd_extra_1st_chunk=False, num_steps=transport_input.num_steps)
        model_kwargs.update(
            {"denoise_step_per_stage": denoise_step_per_stage, "denoise_stage": denoise_stage, "denoise_idx": denoise_idx
        })
        batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)
        model_kwargs["chunk_token_nums"] = chunk_token_nums
        model_kwargs["chunk_num"] = transport_input.chunk_num
        model_kwargs["chunk_offset"] = chunk_offset
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

        # 4. Forward clean chunk and get clean kv
        # fwd_extra_1st_chunk = chunk_start > chunk_offset and denoise_idx == 0
        fwd_extra_1st_chunk = False     # Since every forward now saves KV cache, fwd_extra_1st_chunk is not needed

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
            # 1. Calculate input metrics
            metric_x = x.clone()
            metric_x = metric_x * model_self.model_config.x_rescale_factor
            if model_self.model_config.half_channel_vae:
                assert metric_x.shape[1] == 16
                metric_x = torch.cat([metric_x, metric_x], dim=1)
            metric_x = metric_x.float()
            metric_x = model_self.x_embedder(metric_x)
            metric_x = metric_x.to(model_self.model_config.params_dtype)
            metric_x = rearrange(metric_x, "N C T H W -> (T H W) N C").contiguous()

            self.num_steps = kwargs['total_num_steps']
            denoise_step_per_stage = kwargs['denoise_step_per_stage']
            kwargs["start_chunk_id"] = kwargs['slice_point']
            kwargs["end_chunk_id"] = kwargs['range_num']

            if kwargs.get("distill_nearly_clean_chunk", False):
                kwargs["end_chunk_id"] += 1

            kwargs['cur_denoise_step'] = self.cnt
            model_self.cur_denoise_step = self.cnt

            if kwargs.get("fwd_extra_1st_chunk", False):
                # First chunk not needed
                metric_x = metric_x[kwargs["chunk_token_nums"]: , :, :]
            if kwargs.get("distill_nearly_clean_chunk", False):
                # Last chunk not needed
                metric_x = metric_x[:-kwargs["chunk_token_nums"], :, :]

            if self.cnt == 0 or self.cnt == self.num_steps-1 or self.cnt < self.no_reuse_first_n_steps:
                self.should_calc = True
                self.accumulated_rel_l1_distance = 0
                if self.log:
                    print(f"Calculate output at step {self.cnt}")
            else:
                a1 = metric_x.clone()
                a2 = self.previous_modulated_input.clone()

                if self.cnt % denoise_step_per_stage == 0:
                    dim1 = a1.shape[0]
                    dim2 = a2.shape[0]
                    if dim1 > dim2:
                        # Next stage has more chunks than previous stage, take the front part of next stage's chunks
                        a1 = a1[:dim2]
                    else:
                        # dim1 <= dim2
                        # Next stage has fewer chunks than previous stage, take the back part of previous stage's chunks
                        a2 = a2[-dim1:]
                
                self.accumulated_rel_l1_distance += ((a1 - a2).abs().mean() / a2.abs().mean()).cpu().item()

                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    if self.cnt % denoise_step_per_stage == 0 and dim1 > dim2:
                        if self.whole_calc_when_cross:
                            self.should_calc = True
                            self.accumulated_rel_l1_distance = 0
                            if self.log:
                                print(f"Partly reuse output at step {self.cnt}, but we re-calculate")

                        # Can reuse at this point, so no need to forward the nearly clean chunk, only need to forward the newly added chunk
                        else:
                            self.should_calc = True
                            if self.log:
                                print(f"Partly reuse output at step {self.cnt}, only calculate new chunk")
                            range_num = kwargs['range_num']
                            # If there is a prefix video, need to subtract chunk_offset from range_num
                            range_num = range_num - kwargs['chunk_offset']
                            if kwargs.get("distill_nearly_clean_chunk", False):
                                x = x[:, :, (range_num - 2) * kwargs['chunk_width'] : (range_num - 1) * kwargs['chunk_width']]
                                y = y[range_num - 2:range_num - 1]
                                t = t[:, range_num - 2:range_num - 1]
                                xattn_mask = xattn_mask[range_num - 2:range_num - 1]
                                kwargs["start_chunk_id"] = kwargs['range_num'] - 2    # Cannot use range_num because range_num has already subtracted chunk_offset
                                kwargs["end_chunk_id"] = kwargs['range_num'] - 1
                                kwargs["denoising_range_num"] = 1
                                model_self.discard_nearly_clean_chunk = True
                            else:
                                x = x[:, :, (range_num - 1) * kwargs['chunk_width'] : range_num * kwargs['chunk_width']]
                                y = y[range_num - 1:range_num]
                                t = t[:, range_num - 1:range_num]
                                xattn_mask = xattn_mask[range_num - 1:range_num]
                                kwargs["start_chunk_id"] = kwargs['range_num'] - 1
                                kwargs["denoising_range_num"] = 1

                            # Single-chunk inference with all reuse
                            model_self.single_chunk_inference = True

                    else:
                        self.reuse_times += 1
                        if self.log:
                            print(f"Reuse output at step {self.cnt}")
                        self.should_calc = False
                else:
                    if self.log:
                        print(f"Calculate output at step {self.cnt}")
                    self.should_calc = True
                    self.accumulated_rel_l1_distance = 0

            self.previous_modulated_input = metric_x
            # Update external denoising_range_num
            model_self.denoising_range_num = kwargs["denoising_range_num"]


            if self.should_calc:
                (x, condition, condition_map, y_xattn_flat, rope, meta_args) = model_self.forward_pre_process(
                    x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs
                )

                if not model_self.pre_process:
                    x = pp_scheduler().recv_prev_data(x.shape, x.dtype)
                    model_self.videodit_blocks.set_input_tensor(x)
                else:
                    # clone a new tensor to ensure x is not a view of other tensor
                    x = x.clone()

                x = model_self.videodit_blocks.forward(
                    hidden_states=x,
                    condition=condition,
                    condition_map=condition_map,
                    y_xattn_flat=y_xattn_flat,
                    rotary_pos_emb=rope,
                    inference_params=inference_params,
                    meta_args=meta_args,
                )

                if not model_self.post_process:
                    pp_scheduler().isend_next(x)

                return model_self.forward_post_process(x, meta_args)

            # Return in case of reuse, this output is not used, just needs to have the same shape as the original input
            return torch.zeros_like(raw_x)
        
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
            # Add an extra chunk when distill_nearly_clean_chunk is True
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

        velocity = forward_fn(
            x=x_chunk,
            timestep=t,
            y=y_chunk_flatten,
            mask=mask_chunk_flatten,
            kv_range=kv_range,
            inference_params=self.inference_params[infer_idx],
            **model_kwargs,
        )
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

    # 9. Walk and integrate
    if self.should_calc:
        ori_x_chunk = x_chunk.clone()
        if velocity.shape[2] < x_chunk.shape[2]:
            # Take the last t
            t_num = x_chunk.shape[2] // self.chunk_width
            # Take the last newly added chunk
            x_chunk = x_chunk[:, :, -self.chunk_width:]
            x_chunk = self.integrate(x_chunk, velocity, self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx, delta_t_index=t_num - 1)
            # Concatenate partially reused chunks with the new chunk
            x_chunk = torch.cat([self.previous_output, x_chunk], dim=2)
        else:
            # Full recalculation
            x_chunk = self.integrate(x_chunk, velocity, self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx)

        self.previous_residual = x_chunk - ori_x_chunk

        # Monitor previous_residual memory usage
        if hasattr(self.previous_residual, 'is_cuda') and self.previous_residual.is_cuda:
            current_memory_mb = self.previous_residual.element_size() * self.previous_residual.numel() / (1024**2)

            # Track peak memory usage
            if not hasattr(self, 'peak_residual_memory_mb'):
                self.peak_residual_memory_mb = 0

            if current_memory_mb > self.peak_residual_memory_mb:
                self.peak_residual_memory_mb = current_memory_mb
                print(f"[NEW PEAK] teacache_all previous_residual: {self.peak_residual_memory_mb:.2f} MB (shape: {self.previous_residual.shape})")
    else:
        x_chunk = x_chunk + self.previous_residual[:, :, -x_chunk.shape[2]:]  # Add residual to input, note the special case of chunk reduction


    self.cnt += 1
    if self.cnt == self.num_steps:
        print(f"Reuse output account for {self.reuse_times} / {self.num_steps} steps, ratio: {self.reuse_times/self.num_steps:.2%}")

        # Print final memory usage summary
        if hasattr(self, 'peak_residual_memory_mb'):
            print(f"Peak teacache_all previous_residual memory: {self.peak_residual_memory_mb:.2f} MB")

        self.cnt = 0

    if (self.cnt  + 1) % denoise_step_per_stage == 0:
        # Record previous chunks' output for subsequent concatenation
        self.previous_output = x_chunk

    # 10. chunk denoise count
    for chunk_index in range(chunk_start, chunk_end):
        chunk_denoise_count[chunk_index] += 1
    self.xs[infer_idx][:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width] = x_chunk
    self.chunk_denoise_count[infer_idx] = chunk_denoise_count

    # 11. Return clean chunk
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
    parser.add_argument('--use_teacache', action='store_true', help='Whether to use TeaCache.')
    parser.add_argument('--rel_l1_thresh', type=float, default=0.01, help='Relative L1 distance threshold for TeaCache.')
    parser.add_argument('--no_reuse_first_n_steps', type=int, default=0, help='Number of steps to not reuse output.')
    parser.add_argument('--whole_calc_when_cross', action='store_true', help='Whether to perform whole calculation when crossing stage which number of chunks increases.')
    parser.add_argument('--log', action='store_true', help='Whether to log the TeaCache information.')

    parser.add_argument('--print_peak_memory', action='store_true', help='Print peak memory usage after pipeline completion.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.print_peak_memory:
        # Check if GPU is available and reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            device = torch.cuda.current_device()
            print(f"Running on GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory before pipeline: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
        else:
            print("CUDA not available, running on CPU")

    print(f"TeaCache config arguments: {args}")

    # TeaCache
    SampleTransport.enable_teacache = True
    SampleTransport.rel_l1_thresh = args.rel_l1_thresh
    SampleTransport.accumulated_rel_l1_distance = 0
    SampleTransport.previous_modulated_input = None
    SampleTransport.previous_residual = None
    SampleTransport.cnt = 0
    SampleTransport.forward_velocity = teacache_forward_velocity
    SampleTransport.integrate_velocity = teacache_integrate_velocity
    SampleTransport.reuse_times = 0
    SampleTransport.no_reuse_first_n_steps = args.no_reuse_first_n_steps
    SampleTransport.previous_output = None
    SampleTransport.whole_calc_when_cross = args.whole_calc_when_cross
    SampleTransport.log = args.log

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
