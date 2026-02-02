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
import torch
import yaml, json
from dataclasses import replace
from types import MethodType

from inference.pipeline import MagiPipeline
from inference.pipeline.video_generate import SampleTransport, find_dit_model
from inference.pipeline.utils import get_tensors_memory_usage
from inference.rkv import replace_magi
from inference.rkv.utils import ChunkKVRangeTracker



def teacache_forward_velocity(self, infer_idx: int, cur_denoise_step: int) -> torch.Tensor:
        # 1. Get current work status
        x = self.xs[infer_idx]
        transport_input = self.transport_inputs[infer_idx]
        batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)
        self.total_cache_len = self.total_cache_chunk_nums * (
            self.chunk_width
            * (self.transport_inputs[infer_idx].latent_size[3] // self.model_config.patch_size)
            * (self.transport_inputs[infer_idx].latent_size[4] // self.model_config.patch_size)
        )

        self.budget_cache_len = (self.total_cache_chunk_nums - self.window_size) * (
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
            {"denoise_step_per_stage": denoise_step_per_stage, "denoise_stage": denoise_stage, "denoise_idx": denoise_idx
        })
        # Update parameters related to RKV
        model_kwargs.update(
            {"compress_kv": True, "total_cache_len": self.total_cache_len, "chunk_num": transport_input.chunk_num}
        )

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
        model_kwargs["save_kvcache_every_forward"] = True

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

        # monkey patch
        from inference.common import InferenceParams, ModelMetaArgs
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
            # For KV cache storage
            kwargs["start_chunk_id"] = kwargs['slice_point']
            kwargs["end_chunk_id"] = kwargs['range_num']
            if kwargs.get("distill_nearly_clean_chunk", False):
                kwargs["end_chunk_id"] += 1
            
            # Adjust KV cache range
            if self.inference_params[infer_idx].kv_compressed:
                kv_range = generate_dynamic_kv_range(self, infer_idx, kwargs["start_chunk_id"], kwargs["end_chunk_id"], kwargs)

            # Additional parameters
            self.total_num_steps = kwargs['total_num_steps']

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

            # Get and store query for subsequent compression
            for layer in model_self.videodit_blocks.layers:
                layer_num = layer.self_attention.layer_number
                if hasattr(layer.self_attention, '_last_query'):
                    self.chunk_query_states[layer_num] = layer.self_attention._last_query

            if not model_self.post_process:
                pp_scheduler().isend_next(x)

            return model_self.forward_post_process(x, meta_args)
            

        model  = find_dit_model(self.model)
        model.forward = MethodType(model_forward, model)

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
    x_chunk = self.integrate(x_chunk, velocity, self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx)


    # This step is complete
    self.cnt += 1
    if self.cnt == self.total_num_steps:
        self.cnt = 0


    # 10. chunk denoise count
    for chunk_index in range(chunk_start, chunk_end):
        chunk_denoise_count[chunk_index] += 1
    self.xs[infer_idx][:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width] = x_chunk
    self.chunk_denoise_count[infer_idx] = chunk_denoise_count

    # Check if clean chunk compression should be performed
    compress_kv_cache_after_clean_chunk(self, infer_idx, chunk_start, transport_input)

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


def generate_dynamic_kv_range(self, infer_idx: int, start_chunk_id: int, end_chunk_id: int, kwargs):
    """
    Dynamically generate kv_range based on tracker's actual state and compressed layout

    kv_range meaning: The KV cache range that each chunk can see when performing attention

    Args:
        self: SampleTransportinstance
        infer_idx: Inference index
        start_chunk_id: Starting ID of the currently processed chunk
        end_chunk_id: Ending ID of the currently processed chunk
        kwargs: Dictionary containing other necessary parameters, such as distill_nearly_clean_chunk, etc.

    Returns:
        torch.Tensor: kv_range for each chunk, shape is [num_chunks, 2]
    """
    inference_params = self.inference_params[infer_idx]
    tracker = inference_params.kv_chunk_tracker

    # Calculate the number of chunks currently being processed
    num_chunks = end_chunk_id - start_chunk_id
    chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)[1]

    kv_ranges = []

    # First calculate the cumulative KV length of all normal chunks
    total_tokens = 0
    all_registered_chunks = tracker.get_all_chunk_ids() # warning: There is a newly entered chunk that has not been registered yet

    for i in range(num_chunks):
        current_chunk_id = start_chunk_id + i

        # nearly clean chunk is the last one, skip it first
        is_nearly_clean = (current_chunk_id == end_chunk_id - 1 and
                          kwargs.get("distill_nearly_clean_chunk", False))

        if is_nearly_clean:
            continue

        # Normal chunk: Calculate the cumulative token count that needs to beattend toed to
        relevant_chunks = []
        for chunk_id in all_registered_chunks:
            if chunk_id <= current_chunk_id:
                relevant_chunks.append(chunk_id)

        # Update total token count to maximum value
        for chunk_id in sorted(relevant_chunks):
            s, e = tracker.get_range(chunk_id)
            total_tokens = max(total_tokens, e)

        # Newly entered chunk needs special handling
        if current_chunk_id not in all_registered_chunks:
            total_tokens += chunk_token_nums

        # Normal chunk's range: [0, cumulative token count]
        kv_ranges.append([0, total_tokens])

    # Finally handle nearly clean chunk (if exists)
    has_nearly_clean = kwargs.get("distill_nearly_clean_chunk", False)
    if has_nearly_clean:
        # nearly clean chunk's range: [cumulative token count, cumulative token count + chunk_token_nums]
        range_start = total_tokens
        range_end = total_tokens + chunk_token_nums
        kv_ranges.append([range_start, range_end])

    import pdb; pdb.set_trace()
    return torch.tensor(kv_ranges, device='cuda', dtype=torch.int32)


def compress_kv_cache_after_clean_chunk(self, infer_idx: int, chunk_start: int, transport_input):
    """
    When a chunk becomes clean, immediately perform KV cache compression

    Logic: With a total of 5 chunk sizes, when the 2nd chunk becomes clean, compress chunks 1 and 2 into 1 chunk size,
    then move the remaining chunk KV cache area forward to free up one chunk size for the next incoming chunk
    """

    # Get necessary parameters and status
    inference_params = self.inference_params[infer_idx]
    tracker = inference_params.kv_chunk_tracker

    total_cache_len = self.total_cache_len
    budget_cache_len = self.budget_cache_len
    chunk_num = transport_input.chunk_num
    chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)[1]

    # Get the number of registered chunks
    all_chunk_ids = tracker.get_all_chunk_ids()
    registered_chunk_count = len(all_chunk_ids)

    # Two conditions to determine if compression is needed:
    # 1. Cache area is full (next_free_idx > total_cache_len)
    # 2. There are still new chunks to enter (registered_chunk_count < chunk_num)
    # 3. New chunk is about to enter (i.e., the last denoising chunk's steps equals num_steps/window_size)
    cache_full = tracker.next_free_idx >= total_cache_len
    has_more_chunks = registered_chunk_count < chunk_num
    last_chunk_id = all_chunk_ids[-1]
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
    compress_chunk_ids = []
    for cid in all_chunk_ids:
        if cid < chunk_offset:
            # Prefix video chunk, always clean
            compress_chunk_ids.append(cid)
        elif cid <= chunk_start:
            # Generated chunk, need to check if denoising is completed
            if self.chunk_denoise_count[infer_idx][cid] == transport_input.num_steps:
                compress_chunk_ids.append(cid)

    active_chunk_ids = [cid for cid in all_chunk_ids if cid not in compress_chunk_ids]

    if len(compress_chunk_ids) < 2:
        return  # At least 2 chunks are required for compression

    # Get model to access kv_cluster of each layer
    model = find_dit_model(self.model)

    # Perform compression on each layer
    for layer in model.videodit_blocks.layers:
        if hasattr(layer.self_attention, 'kv_cluster'):
            kv_cluster = layer.self_attention.kv_cluster

            # 1. Extract KV cache of chunks that need compression
            compress_chunks_kv = []
            compress_lengths = []
            for cid in compress_chunk_ids:
                s, e = tracker.get_range(cid)
                chunk_kv = inference_params.key_value_memory_dict[layer.self_attention.layer_number][s:e, ...]
                compress_chunks_kv.append(chunk_kv)
                compress_lengths.append(e - s)

            # Concatenate KV that needs compression
            compress_kv = torch.cat(compress_chunks_kv, dim=0)
            key_compress, value_compress = torch.chunk(compress_kv, 2, dim=-1)

            # 2. Extract KV cache of active chunks that remain unchanged (currently being denoised)
            active_chunk_kv = []
            active_lengths = []
            for cid in active_chunk_ids:
                s, e = tracker.get_range(cid)
                chunk_kv = inference_params.key_value_memory_dict[layer.self_attention.layer_number][s:e, ...]
                active_chunk_kv.append(chunk_kv)
                active_lengths.append(e - s)

            query_for_compress = self.chunk_query_states[layer.self_attention.layer_number]
            kv_cluster.budget = budget_cache_len
            clean_chunk_tokens = sum(compress_lengths)  # Total token count of all clean chunks

            # Get latent size information
            latent_size = transport_input.latent_size
            H = latent_size[3] // self.model_config.patch_size
            W = latent_size[4] // self.model_config.patch_size
            T = chunk_token_nums // (H * W)
            if not chunk_token_nums % (H * W) == 0:
                import pdb; pdb.set_trace()

            # Only key and value that need compression are passed in
            key_compressed, value_compressed, indices = kv_cluster.update_kv(
                key_compress,
                query_for_compress,
                value_compress,
                clean_chunk_tokens=clean_chunk_tokens,
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
            for chunk_id, chunk_len in zip(compress_chunk_ids, compress_lengths):
                if chunk_id in compress_chunk_ids:
                    end_idx = start_idx + chunk_len
                    # Only count selected tokens within range
                    mask = (indices_1d >= start_idx) & (indices_1d < min(end_idx, clean_chunk_tokens))
                    kept_in_chunk = mask.sum().item()
                    all_lengths_after_compress.append(kept_in_chunk)
                    start_idx = end_idx

            final_chunk_ids.extend(compress_chunk_ids)
            final_lengths.extend(all_lengths_after_compress[:len(compress_chunk_ids)])

            # 4.2 Add unchanged active chunk
            for i, chunk_kv in enumerate(active_chunk_kv):
                final_kv_parts.append(chunk_kv)
                final_chunk_ids.append(active_chunk_ids[i])
                # For active_chunk_ids, need to get the corresponding length's latter part from all_lengths_after_compress
                active_chunk_length = active_lengths[i]
                final_lengths.append(active_chunk_length)

            # Concatenate final KV cache
            final_kv = torch.cat(final_kv_parts, dim=0)

            # 5. Update KV cache
            total_kv_len = final_kv.size(0)
            inference_params.key_value_memory_dict[layer.self_attention.layer_number][:total_kv_len, ...] = final_kv
            inference_params.key_value_memory_dict[layer.self_attention.layer_number][total_kv_len:, ...] = 0.0

            # 6. Update compressed range
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    raise NotImplementedError("TEACache method is not implemented yet.")
    
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
    SampleTransport.enable_teacache = True
    SampleTransport.rel_l1_thresh = args.rel_l1_thresh
    SampleTransport.chunk_accumulated_rel_l1 = 0
    SampleTransport.previous_modulated_input = None
    SampleTransport.previous_residual = None
    SampleTransport.cnt = 0
    SampleTransport.forward_velocity = teacache_forward_velocity
    SampleTransport.integrate_velocity = teacache_integrate_velocity

    SampleTransport.reuse_times = 0
    SampleTransport.no_reuse_first_n_steps = args.no_reuse_first_n_steps
    SampleTransport.previous_output = None
    SampleTransport.discard_nearly_clean_chunk = args.discard_nearly_clean_chunk
    SampleTransport.whole_calc_when_cross = args.whole_calc_when_cross
    SampleTransport.no_reuse_mode = args.no_reuse_mode
    # --- Per-chunk state ---
    SampleTransport.chunk_accumulated_rel_l1 = None           # List[float]: Cumulative rel L1 for each chunk
    SampleTransport.prev_chunk_features = None               # List[Tensor]: Features from previous step for each chunk
    SampleTransport.chunk_reuse_flags = None                   # Whether each chunk is reused in current step
    SampleTransport.log = args.log
    
    SampleTransport.total_cache_chunk_nums = args.total_cache_chunk_nums

    compression_config = {
        "method_config": {
            "compress_strategy": args.compress_strategy,
            "mix_lambda": args.mix_lambda,
            "query_granularity": args.query_granularity,
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