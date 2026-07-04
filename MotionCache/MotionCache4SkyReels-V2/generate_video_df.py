import argparse
import gc
import os
import random
import time

import imageio
import torch
from diffusers.utils import load_image

from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
from moviepy.editor import VideoFileClip
# from moviepy import *


def get_video_num_frames_moviepy(video_path):
    with VideoFileClip(video_path) as clip:
        num_frames = 0
        for _ in clip.iter_frames():
            num_frames += 1
        return clip.size, num_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="diffusion_forcing")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-DF-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"])
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
    parser.add_argument(
        "--prompt",
        type=str,
        default="A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway at sunset, her hair blowing wildly in the wind as the motorcycle kicks up dust, with the golden sun casting long shadows across the barren landscape.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--expname", type=str)
    # Distance visualization arguments
    parser.add_argument("--enable_distance_viz", action="store_true", 
                        help="Enable input/output distance visualization")
    parser.add_argument("--distance_classes", type=int, default=30,
                        help="Number of distance classes for visualization")
    parser.add_argument("--distance_metric", type=str, choices=["L1", "L2"], default="L1",
                        help="Distance metric to use")
    parser.add_argument("--distance_binning", type=str, choices=["quantile", "uniform"], 
                        default="quantile", help="Binning method for distance classes")
    parser.add_argument("--distance_output_dir", type=str, 
                        default="./result/distance_viz",
                        help="Output directory for distance visualizations")
    parser.add_argument("--distance_save_steps", type=str, default=None,
                        help="Only save visualization for specific steps, e.g., '0,1,2,24,25,26,48,49,50'. None to save all steps")
    # Selective token visualization arguments
    parser.add_argument("--enable_selective_viz", action="store_true",
                        help="Enable selective token visualization (red=selected, white=unselected)")
    parser.add_argument("--selective_viz_output_dir", type=str,
                        default="./result/selective_viz",
                        help="Output directory for selective token visualizations")
    # Token-wise cache arguments
    parser.add_argument("--enable_token_cache", action="store_true",
                        help="Enable token-wise weighted distance accumulation cache")
    parser.add_argument("--token_cache_threshold", type=float, default=0.10,
                        help="Unified threshold for token cache (default: 0.10)")
    parser.add_argument("--token_cache_warmup", type=int, default=4,
                        help="Number of warmup steps before enabling token cache (default: 5)")
    parser.add_argument("--token_phase1_update_count", type=int, default=3,
                        help="Number of updates in phase 1 before switching to differentiated weights (default: 3)")
    parser.add_argument("--token_distance_mode", type=str, choices=["token", "global"], default="token",
                        help="Distance calculation mode: 'token' for per-token distance, 'global' for overall L1 distance (default: token)")
    # Static weight mask arguments
    parser.add_argument("--enable_static_weight", action="store_true",
                        help="Enable static weight mask for specified region (uses fixed weight instead of dynamic weights)")
    parser.add_argument("--static_weight_value", type=float, default=0.3,
                        help="Weight value for static region (lower = less updates, more cache). Default: 0.3")
    parser.add_argument("--static_weight_region", type=str, default="0,0,0.5,1",
                        help="Static regions as 'h1,w1,h2,w2;h1,w1,h2,w2;...' (semicolon-separated, ratio 0-1). Default: '0,0,0.5,1' (top half)")
    # Residual difference visualization arguments
    parser.add_argument("--enable_residual_diff_viz", action="store_true",
                        help="Enable residual difference visualization between steps")
    parser.add_argument("--residual_diff_output_dir", type=str,
                        default="./result/residual_diff_viz",
                        help="Output directory for residual diff visualizations")
    parser.add_argument("--residual_diff_metric", type=str, choices=["L1", "L2", "cosine", "L2_sum"],
                        default="L1", help="Metric for residual diff: L1, L2, L2_sum, or cosine")
    parser.add_argument("--residual_diff_save_steps", type=str, default=None,
                        help="Only save residual diff for specific steps, e.g., '1,2,3,4,5'. None to save all steps")

    # Frame diff weight arguments
    parser.add_argument("--enable_frame_diff_weight", action="store_true",
                        help="Enable frame diff weight mode (use inter-frame diff instead of output activation)")
    parser.add_argument("--enable_frame_diff_viz", action="store_true",
                        help="Enable frame diff visualization")
    parser.add_argument("--frame_diff_viz_output_dir", type=str,
                        default="./result/frame_diff_viz",
                        help="Output directory for frame diff visualizations")

    # Input diff weight arguments (use sampler output / transformer input for weight calculation)
    parser.add_argument("--enable_input_diff_weight", action="store_true",
                        help="Enable input diff weight mode (use transformer input instead of output for weight calculation)")
    parser.add_argument("--enable_input_diff_viz", action="store_true",
                        help="Enable input diff visualization")
    parser.add_argument("--input_diff_viz_output_dir", type=str,
                        default="./result/input_diff_viz",
                        help="Output directory for input diff visualizations")

    # Weight normalization mode
    parser.add_argument("--weight_norm_mode", type=str, choices=["mean", "max", "max_rescale"], default="mean",
                        help="Weight normalization mode: 'mean' (default), 'max', or 'max_rescale'. 'max_rescale' maps to [weight_floor, 1]")
    parser.add_argument("--weight_floor", type=float, default=0.3,
                        help="Minimum weight value for max_rescale mode (default: 0.3)")

    # Temporal consistency constraint
    parser.add_argument("--enable_temporal_consistency", action="store_true",
                        help="Enable temporal consistency constraint to reduce noise in static regions")
    parser.add_argument("--temporal_consistency_threshold", type=float, default=0.5,
                        help="Threshold for temporal consistency (default: 0.5)")
    # ============================================
    # Minimum update ratio parameters
    # ============================================
    parser.add_argument("--min_update_ratio", type=float, default=0.4,
                        help="Minimum update ratio (0.0-1.0): skip forward when update ratio < this value. 0 to disable.")

    args = parser.parse_args()

    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"
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
        assert (
            args.overlap_history is not None
        ), 'You are supposed to specify the "overlap_history" to support the long video generation. 17 and 37 are recommanded to set.'
    if args.addnoise_condition > 60:
        print(
            f'You have set "addnoise_condition" as {args.addnoise_condition}. The value is too large which can cause inconsistency in long video generation. The value is recommanded to set 20.'
        )

    guidance_scale = args.guidance_scale
    shift = args.shift
    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    # save_dir = os.path.join("result", args.outdir)
    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)
    local_rank = 0
    if args.use_usp:
        assert not args.prompt_enhancer, "`--prompt_enhancer` is not allowed if using `--use_usp`. We recommend running the skyreels_v2_infer/pipelines/prompt_enhancer.py script first to generate enhanced prompt before enabling the `--use_usp` parameter."
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        import torch.distributed as dist

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())
        device = "cuda"

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )

    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        print(f"init prompt enhancer")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        print(f"enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    pipe = DiffusionForcingPipeline(
        args.model_id,
        dit_path=args.model_id,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=args.use_usp,
        offload=args.offload,
    )

    if args.causal_attention:
        pipe.transformer.set_ar_attention(args.causal_block_size)

    # Configure distance classification parameters (independent of visualization)
    # These parameters affect distance calculation and classification logic, always need to be configured
    pipe.transformer.set_distance_classification_config(
        n_classes=args.distance_classes,
        metric=args.distance_metric,
        binning_method=args.distance_binning
    )

    # Configure distance visualization (optional)
    if args.enable_distance_viz:
        # Parse save_steps parameter
        save_steps = None
        if args.distance_save_steps is not None:
            try:
                save_steps = [int(s.strip()) for s in args.distance_save_steps.split(',')]
            except ValueError:
                print(f"Warning: Invalid distance_save_steps format '{args.distance_save_steps}', saving all steps")
                save_steps = None
        
        pipe.transformer.set_distance_visualization_config(
            enable=args.enable_distance_viz,  # Control whether to save PNG
            output_dir=args.distance_output_dir,
            save_steps=save_steps
        )
    
    # Configure selective token visualization
    if args.enable_selective_viz:
        pipe.transformer.set_selective_viz_config(
            enable=True,
            output_dir=args.selective_viz_output_dir
        )

    # Configure token-wise cache
    if args.enable_token_cache:
        pipe.transformer.set_token_cache_config(
            enable=True,
            threshold=args.token_cache_threshold,
            warmup_steps=args.token_cache_warmup,
            phase1_update_count=args.token_phase1_update_count,
            distance_mode=args.token_distance_mode
        )

    # Configure static weight mask
    if args.enable_static_weight:
        # Parse region string "h1,w1,h2,w2;h1,w1,h2,w2;..." to list of tuples [((h1,w1), (h2,w2)), ...]
        try:
            regions = []
            region_strs = args.static_weight_region.split(';')
            for region_str in region_strs:
                region_str = region_str.strip()
                if not region_str:
                    continue
                coords = [float(x.strip()) for x in region_str.split(',')]
                if len(coords) != 4:
                    raise ValueError(f"Each region must have 4 values, got {len(coords)} in '{region_str}'")
                regions.append(((coords[0], coords[1]), (coords[2], coords[3])))
            if not regions:
                raise ValueError("No valid regions found")
        except Exception as e:
            print(f"Warning: Invalid static_weight_region '{args.static_weight_region}', using default. Error: {e}")
            regions = [((0.0, 0.0), (0.5, 1.0))]

        pipe.transformer.set_static_weight_config(
            enable=True,
            weight_value=args.static_weight_value,
            regions=regions
        )

    # Configure residual difference visualization
    if args.enable_residual_diff_viz:
        # Parse save_steps parameter
        residual_diff_save_steps = None
        if args.residual_diff_save_steps is not None:
            try:
                residual_diff_save_steps = [int(s.strip()) for s in args.residual_diff_save_steps.split(',')]
            except ValueError:
                print(f"Warning: Invalid residual_diff_save_steps format '{args.residual_diff_save_steps}', saving all steps")
                residual_diff_save_steps = None

        pipe.transformer.set_residual_diff_visualization_config(
            enable=True,
            output_dir=args.residual_diff_output_dir,
            metric=args.residual_diff_metric,
            save_steps=residual_diff_save_steps
        )

    # Configure frame diff weight mode
    if args.enable_frame_diff_weight or args.enable_frame_diff_viz:
        pipe.transformer.set_frame_diff_weight_config(
            enable=args.enable_frame_diff_weight,
            viz_enable=args.enable_frame_diff_viz,
            output_dir=args.frame_diff_viz_output_dir
        )

    # Configure input diff weight mode (use transformer input for weight calculation)
    if args.enable_input_diff_weight or args.enable_input_diff_viz:
        pipe.transformer.set_input_diff_weight_config(
            enable=args.enable_input_diff_weight,
            viz_enable=args.enable_input_diff_viz,
            output_dir=args.input_diff_viz_output_dir
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
        if args.image:
            args.image = load_image(args.image)
            image_width, image_height = args.image.size
            if image_height > image_width:
                height, width = width, height
            args.image = resizecrop(args.image, height, width)
            if args.end_image:
                args.end_image = load_image(args.end_image)
                args.end_image = resizecrop(args.end_image, height, width)

        image = args.image.convert("RGB") if args.image else None
        end_image = args.end_image.convert("RGB") if args.end_image else None

        print(f"Saving to: {os.path.join(save_dir, f'{args.expname}-0.mp4')}")
        
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

    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # video_out_file = f"{args.expname}_{args.seed}_{current_time}.mp4"
        video_out_file = f"{args.expname}-0.mp4"
        # video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
