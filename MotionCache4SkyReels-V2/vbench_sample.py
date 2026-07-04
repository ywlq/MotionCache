#!/usr/bin/env python3
import os
import sys
import argparse
import multiprocessing as mp
import time
import random
from typing import List

import torch
import imageio
from diffusers.utils import load_image

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
from moviepy.editor import VideoFileClip


# Short prompt list, corresponds one-to-one with long prompts
SHORT_PROMPTS: List[str] = []


def get_video_num_frames_moviepy(video_path: str):
    with VideoFileClip(video_path) as clip:
        num_frames = 0
        for _ in clip.iter_frames():
            num_frames += 1
        return clip.size, num_frames


def chunk_by_rank(items: List[str], world_size: int, rank: int, paired: List[str] | None = None):
    if world_size <= 0:
        return ([], []) if paired is not None else []
    per = (len(items) + world_size - 1) // world_size
    start = rank * per
    end = min(start + per, len(items))
    if paired is None:
        return items[start:end]
    paired_slice = paired[start:end]
    return items[start:end], paired_slice


def load_prompts(args: argparse.Namespace) -> List[str]:
    global SHORT_PROMPTS
    if args.prompts_file:
        pf = os.path.abspath(args.prompts_file)
        if not os.path.exists(pf):
            raise FileNotFoundError(f"prompts_file not found: {pf}")
        with open(pf, "r", encoding="utf-8") as f:
            long_prompts = [line.strip() for line in f if line.strip()]

        short_pf = f"/path/VBench/prompts/prompts_per_dimension/{args.dimension}.txt"
        with open(short_pf, "r", encoding="utf-8") as sf:
            shorts = [line.strip() for line in sf if line.strip()]
        SHORT_PROMPTS = shorts
        return long_prompts

    if args.dimension:
        pf = f"/path/VBench/prompts/augmented_prompts/gpt_enhanced_prompts/prompts_per_dimension_longer/{args.dimension}.txt"
        if not os.path.exists(pf):
            raise FileNotFoundError(f"dimension file not found: {pf}")
        with open(pf, "r", encoding="utf-8") as f:
            long_prompts = [line.strip() for line in f if line.strip()]

        base_dim = args.dimension.replace("_longer", "")
        short_pf = f"/path/VBench/prompts/prompts_per_dimension/{base_dim}.txt"
        with open(short_pf, "r", encoding="utf-8") as sf:
            shorts = [line.strip() for line in sf if line.strip()]
        SHORT_PROMPTS = shorts
        return long_prompts

    raise ValueError("Either --prompts_file or --dimension must be provided.")


def build_pipeline(args: argparse.Namespace) -> DiffusionForcingPipeline:
    model_id = download_model(args.model_id)
    print("model_id:", model_id)

    pipe = DiffusionForcingPipeline(
        model_id,
        dit_path=model_id,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=args.use_usp,
        offload=args.offload,
    )

    if args.causal_attention:
        pipe.transformer.set_ar_attention(args.causal_block_size)

    # Configure token cache (new token-wise approach from run_10s.sh)
    if args.enable_token_cache:
        pipe.transformer.set_token_cache_config(
            enable=True,
            threshold=args.token_cache_threshold,
            warmup_steps=args.token_cache_warmup,
            phase1_update_count=args.token_phase1_update_count,
            distance_mode=args.token_distance_mode,
        )

    # Configure frame diff weight mode
    if args.enable_frame_diff_weight:
        pipe.transformer.set_frame_diff_weight_config(
            enable=args.enable_frame_diff_weight,
            viz_enable=False,
            output_dir=None
        )

    # Configure weight normalization mode
    if args.weight_norm_mode != "mean":
        pipe.transformer.set_weight_norm_mode(args.weight_norm_mode, args.weight_floor)

    # Configure temporal consistency constraint
    if args.enable_temporal_consistency:
        pipe.transformer.set_temporal_consistency_config(
            enable=args.enable_temporal_consistency,
            threshold=args.temporal_consistency_threshold,
        )

    # Configure minimum update ratio
    pipe.transformer.set_min_update_ratio(args.min_update_ratio)

    return pipe


def run_generation(pipe: DiffusionForcingPipeline, args: argparse.Namespace, prompt: str, expname: str) -> None:
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    if args.resolution == "540P":
        height = 544
        width = 960
    elif args.resolution == "720P":
        height = 720
        width = 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")

    num_frames = args.num_frames
    fps = args.fps

    if num_frames > args.base_num_frames:
        assert args.overlap_history is not None, 'You are supposed to specify the "overlap_history" to support the long video generation. 17 and 37 are recommanded to set.'
    if args.addnoise_condition > 60:
        print(f'You have set "addnoise_condition" as {args.addnoise_condition}. The value is too large which can cause inconsistency in long video generation. The value is recommanded to set 20.')

    guidance_scale = args.guidance_scale
    shift = args.shift

    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)

    prompt_input = prompt
    if args.prompt_enhancer and args.image is None:
        print(f"init prompt enhancer")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"enhanced prompt: {prompt_input}")
        del prompt_enhancer
        torch.cuda.empty_cache()

    print(f"prompt:{prompt_input}")
    print(f"guidance_scale:{guidance_scale}")

    if os.path.exists(args.video_path):
        (v_width, v_height), input_num_frames = get_video_num_frames_moviepy(args.video_path)
        assert input_num_frames >= args.overlap_history, "The input video is too short."
        if v_height > v_width:
            width, height = height, width
        video_frames = pipe.extend_video(
            prompt=prompt_input,
            negative_prompt=negative_prompt,
            prefix_video_path=args.video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=args.inference_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            overlap_history=args.overlap_history,
            addnoise_condition=args.addnoise_condition,
            base_num_frames=args.base_num_frames,
            ar_step=args.ar_step,
            causal_block_size=args.causal_block_size,
            fps=fps,
        )[0]
    else:
        image = None
        end_image = None
        if args.image:
            args.image = load_image(args.image)
            image_width, image_height = args.image.size
            if image_height > image_width:
                height, width = width, height
            args.image = resizecrop(args.image, height, width)
            image = args.image.convert("RGB")
        if args.end_image:
            args.end_image = load_image(args.end_image)
            args.end_image = resizecrop(args.end_image, height, width)
            end_image = args.end_image.convert("RGB")

        print(f"Saving to: {os.path.join(save_dir, f'{expname}-0.mp4')}")
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                image=image,
                end_image=end_image,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=args.inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                overlap_history=args.overlap_history,
                addnoise_condition=args.addnoise_condition,
                base_num_frames=args.base_num_frames,
                ar_step=args.ar_step,
                causal_block_size=args.causal_block_size,
                fps=fps,
                args=args,
            )[0]

    video_out_file = f"{expname}-0.mp4"
    output_path = os.path.join(save_dir, video_out_file)
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])


def worker(gpu_id: int, rank: int, args: argparse.Namespace, all_prompts: List[str]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(33789 + gpu_id))

    my_prompts, my_short_prompts = chunk_by_rank(all_prompts, args.num_gpus, rank, SHORT_PROMPTS)
    if not my_prompts:
        print(f"[GPU {gpu_id}] No prompts assigned.")
        return

    print(f"[GPU {gpu_id}] Processing {len(my_prompts)} prompts")

    pipe = build_pipeline(args)

    for idx, prompt in enumerate(my_prompts):
        prompt = prompt.strip()
        if not prompt:
            continue

        expname = my_short_prompts[idx].strip() if idx < len(my_short_prompts) else f"exp_{rank}_{idx}"

        # Check if video already exists
        video_out_file = f"{expname}-0.mp4"
        output_path = os.path.join(args.outdir, video_out_file)
        if os.path.exists(output_path):
            print(f"[GPU {gpu_id}] Skipping idx={idx} (already exists): {output_path}")
            continue

        print(f"[GPU {gpu_id}] pair idx={idx} short_expname='{expname}' | long_prompt='{prompt}'")

        try:
            run_generation(pipe, args, prompt, expname)
            print(f"[GPU {gpu_id}] Done: {prompt}")
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed: {prompt} | err={e}")


def main():
    parser = argparse.ArgumentParser()
    # Input source
    parser.add_argument("--prompts_file", type=str, help="Path to a plain text file of prompts (one per line).")
    parser.add_argument("--dimension", type=str, help="VBench dimension name; reads prompts from predefined path.")

    # Multi-GPU
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to use, e.g., '0,1'.")

    # Generation parameters
    parser.add_argument("--outdir", type=str, default="diffusion_forcing")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-DF-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P")
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--end_image", type=str, default=None)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--ar_step", type=int, default=0)
    parser.add_argument("--causal_attention", action="store_true")
    parser.add_argument("--causal_block_size", type=int, default=1)
    parser.add_argument("--base_num_frames", type=int, default=97)
    parser.add_argument("--overlap_history", type=int, default=None)
    parser.add_argument("--addnoise_condition", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--prompt_enhancer", action="store_true")

    # ============================================
    # Token Cache parameters
    # ============================================
    parser.add_argument("--enable_token_cache", action="store_true",
                        help="Enable token-wise caching")
    parser.add_argument("--token_cache_threshold", type=float, default=0.1,
                        help="Token cache threshold")
    parser.add_argument("--token_cache_warmup", type=int, default=4,
                        help="Number of warmup steps before enabling token cache")
    parser.add_argument("--token_phase1_update_count", type=int, default=6,
                        help="Number of updates in phase 1")
    parser.add_argument("--token_distance_mode", type=str, default="global",
                        choices=["token", "global"],
                        help="Distance calculation mode")
    # ============================================
    # Frame difference weight parameters
    # ============================================
    parser.add_argument("--enable_frame_diff_weight", action="store_true",
                        help="Enable frame difference weight mode")
    parser.add_argument("--weight_norm_mode", type=str, default="mean",
                        choices=["mean", "max", "max_rescale"],
                        help="Weight normalization mode")
    parser.add_argument("--weight_floor", type=float, default=0.6,
                        help="Minimum weight value for max_rescale mode")

    # ============================================
    # Temporal consistency parameters
    # ============================================
    parser.add_argument("--enable_temporal_consistency", action="store_true",
                        help="Enable temporal consistency constraint")
    parser.add_argument("--temporal_consistency_threshold", type=float, default=0.5,
                        help="Temporal consistency threshold: when more than this ratio of frames need update, all frames at that position update")
    # ============================================
    # Minimum update ratio parameters
    # ============================================
    parser.add_argument("--min_update_ratio", type=float, default=0.4,
                        help="Minimum update ratio (0.0-1.0): skip forward when update ratio < this value. 0 to disable.")

    args = parser.parse_args()

    # Multi-GPU distribution
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise ValueError("No GPU IDs provided in --gpus")
    args.num_gpus = len(gpu_ids)

    prompts = load_prompts(args)
    print(f"Total prompts: {len(prompts)}")
    os.makedirs(args.outdir, exist_ok=True)

    procs = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=worker, args=(gpu_id, rank, args, prompts))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All jobs finished.")


if __name__ == "__main__":
    main()
