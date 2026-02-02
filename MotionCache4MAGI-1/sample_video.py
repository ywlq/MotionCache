import os
import sys
import argparse
import subprocess
import torch
import multiprocessing as mp
import csv
import re
from pathlib import Path
import logging
from datetime import datetime

def setup_gpu_logger(gpu_id: int, base_log_dir: str, run_timestamp: str):
    """Set up independent log file for each GPU"""
    # If base_log_dir contains /videos/ subpath, extract its parent directory
    if "videos/" in base_log_dir or "/videos/" in base_log_dir:
        # e.g.: path/output/videos/overall_consistency -> path/output
        if "/videos/" in base_log_dir:
            base_log_dir = base_log_dir.split("/videos/")[0]
        else:
            base_log_dir = base_log_dir.split("videos/")[0]

    # Create subfolder by run time
    log_dir = os.path.join(base_log_dir, "log", run_timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Simplified log file name
    log_file = os.path.join(log_dir, f"gpu_{gpu_id}.log")

    # Create logger
    logger = logging.getLogger(f"GPU_{gpu_id}")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler (optional, if you still want to see output in terminal)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        f'[GPU {gpu_id}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    # Also redirect print to log file
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    return logger, log_file


class LoggerWriter:
    """Redirect print output to logger"""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass


def worker_process(gpu_id: int, rank: int, args: argparse.Namespace, all_samples: list, run_timestamp: str):
    """Independent worker running on each GPU"""

    # ========== For pdb debugging in terminal ==========
    sys.stdin = open(0)
    # ==================================================

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29510 + gpu_id)

    # ========== Set up independent log file for each GPU ==========
    logger, log_file = setup_gpu_logger(gpu_id, args.save_path, run_timestamp)
    logger.info(f"GPU {gpu_id} logger initialized. Log file: {log_file}")

    from inference.pipeline.video_generate import SampleTransport
    if args.reuse_strategy == "original":
        pass
    elif args.reuse_strategy == "all":
        from inference.pipeline.teacache_all import teacache_forward_velocity, teacache_integrate_velocity
        # TeaCache
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
    elif args.reuse_strategy == "chunkwise":
        from inference.pipeline.flowcache import teacache_forward_velocity, teacache_integrate_velocity
        if args.compress_kv_cache:
            print("KV cache compression is enabled.")
            # KV cache compression
            SampleTransport.compress_kv_cache = True
            assert args.total_cache_chunk_nums is not None
            compression_config = {
                "method_config": {
                    "compress_strategy": args.compress_strategy,
                    "mix_lambda": args.mix_lambda,
                    "query_granularity": args.query_granularity,
                    "score_weighting_method": args.score_weighting_method,
                    "power": args.power,
                },
            }
            from inference.rkv import replace_magi
            from inference.rkv.utils import ChunkKVRangeTracker
            replace_magi(compression_config)
        else:
            SampleTransport.compress_kv_cache = False

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

        # Token-wise reuse parameters
        SampleTransport.token_wise_reuse = args.token_wise_reuse
        SampleTransport.token_rel_l1_thresh = args.token_rel_l1_thresh if args.token_rel_l1_thresh is not None else args.rel_l1_thresh
        SampleTransport.tokenwise_l1_mode = args.tokenwise_l1_mode

        # Three-phase control parameters
        SampleTransport.warmup_steps = args.warmup_steps
        SampleTransport.chunk_wise_only_steps = args.chunk_wise_only_steps

        # Token reuse ratio control
        SampleTransport.max_token_reuse_ratio = args.max_token_reuse_ratio
        SampleTransport.initial_token_reuse_ratio = args.initial_token_reuse_ratio
        SampleTransport.final_token_reuse_ratio = args.final_token_reuse_ratio

        # Continuous reuse tracking parameters
        SampleTransport.enable_continuous_reuse_tracking = args.enable_continuous_reuse_tracking
        SampleTransport.continuous_reuse_max_count = args.continuous_reuse_max_count
        SampleTransport.continuous_reuse_decay_mode = args.continuous_reuse_decay_mode
        SampleTransport.continuous_reuse_decay_factor = args.continuous_reuse_decay_factor

        # Temporal weight parameters
        SampleTransport.temporal_weight_floor = args.temporal_weight_floor
        SampleTransport.temporal_weight_power = args.temporal_weight_power
        SampleTransport.enable_temporal_voting = args.enable_temporal_voting

        # Debug and visualization parameters
        SampleTransport.print_token_stats = args.print_token_stats
        SampleTransport.visualize_reuse_mask = args.visualize_reuse_mask
        SampleTransport.visualize_temporal_diff = args.visualize_temporal_diff
        # Support both int and list for temporal_diff_step
        temporal_diff_step_arg = args.temporal_diff_step
        if isinstance(temporal_diff_step_arg, int):
            SampleTransport.temporal_diff_steps = [temporal_diff_step_arg]
        else:
            SampleTransport.temporal_diff_steps = temporal_diff_step_arg
        SampleTransport.temporal_diff_mode = args.temporal_diff_mode
        SampleTransport.final_temporal_diff_masks = {}
        SampleTransport.final_temporal_diff_latent_sizes = {}

        SampleTransport.visualize_temporal_weights = args.visualize_temporal_weights
        # Support both int and list for temporal_weights_step
        temporal_weights_step_arg = args.temporal_weights_step
        if isinstance(temporal_weights_step_arg, int):
            SampleTransport.temporal_weights_steps = [temporal_weights_step_arg]
        else:
            SampleTransport.temporal_weights_steps = temporal_weights_step_arg
        SampleTransport.final_temporal_weights_masks = {}
        SampleTransport.final_temporal_weights_latent_sizes = {}
        SampleTransport.final_temporal_weights_token_dims = {}

        # --- Per-chunk state ---
        SampleTransport.chunk_accumulated_rel_l1 = None           # List[float]: Accumulated rel L1 for each chunk
        SampleTransport.prev_chunk_features = None               # List[Tensor]: Features of each chunk from previous step
        SampleTransport.chunk_reuse_flags = None                   # Whether each chunk is reused in current step

        # KV cache compression
        SampleTransport.total_cache_chunk_nums = args.total_cache_chunk_nums

        # Check mutual exclusivity of token_wise_reuse and compress_kv_cache
        if SampleTransport.token_wise_reuse and SampleTransport.compress_kv_cache:
            raise ValueError(
                "token_wise_reuse and compress_kv_cache cannot be enabled simultaneously. "
                "Token-level reuse requires full chunk query states for KV cache reassembly, "
                "which is incompatible with KV cache compression. "
                "Please set only one of these options to True."
            )

        # log
        SampleTransport.log = args.log
    else:
        raise ValueError(f"Unknown reuse strategy: {args.reuse_strategy}")


    try:
        magi_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip()
        os.environ["MAGI_ROOT"] = magi_root
        os.environ["PYTHONPATH"] = f"{magi_root}:{os.environ.get('PYTHONPATH', '')}"
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to set MAGI_ROOT: {e}")
        return

    filtered_samples = []
    for sample in all_samples:
        if args.benchmark == 'vbench':
            prompt = sample
            safe_prompt = prompt
            seed = 0
            cur_save_path = os.path.join(args.save_path, f"{safe_prompt}-{seed}.mp4")
        else:  # physicsiq
            cur_save_path = sample['output_path']

        cur_save_path = os.path.abspath(cur_save_path)
        if not os.path.exists(cur_save_path):
            filtered_samples.append(sample)
        else:
            print(f"[✅ SKIP] Already exists: {cur_save_path}")

    if not filtered_samples:
        print(f"[GPU {gpu_id}] No samples need to be generated.")
        return

    print(f"Totally Processing {len(filtered_samples)} samples.")

    # === Split samples ===
    world_size = args.num_gpus
    samples_per_gpu = (len(filtered_samples) + world_size - 1) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, len(filtered_samples))
    my_samples = filtered_samples[start_idx:end_idx]

    if not my_samples:
        print(f"[GPU {gpu_id}] No samples assigned.")
        return

    print(f"[GPU {gpu_id}] Processing {len(my_samples)} samples: {start_idx} ~ {end_idx-1}")

    from inference.pipeline.entry import MagiPipeline
    from inference.common import set_random_seed

    print(f"[GPU {gpu_id}] Loading model...")
    pipeline = MagiPipeline(args.config_file)
    print(f"[GPU {gpu_id}] Model loaded successfully.")

    for i, sample in enumerate(my_samples):
        if args.benchmark == 'vbench':
            prompt = sample
            seed = 0
            safe_prompt = prompt
            output_path = os.path.join(args.save_path, f"{safe_prompt}-{seed}.mp4")
            output_path = os.path.abspath(output_path)

            if os.path.exists(output_path):
                print(f"[✅ GPU {gpu_id}] Already exists: {output_path}")
                continue

            # Reset seed before each prompt to ensure reproducibility
            set_random_seed(pipeline.config.runtime_config.seed)

            print(f"[🚀 GPU {gpu_id}] Generating T2V: '{prompt}' -> {output_path}")
            pipeline.run_text_to_video(prompt=prompt, output_path=output_path)
            print(f"[✅ GPU {gpu_id}] Saved: {output_path}")

        else:  # physicsiq
            prompt = sample['description']
            prefix_video_path = sample['prefix_video_path']
            output_path = sample['output_path']

            if not os.path.exists(prefix_video_path):
                print(f"[⚠️ GPU {gpu_id}] Conditioning video not found: {prefix_video_path}")
                continue

            if os.path.exists(output_path):
                print(f"[✅ GPU {gpu_id}] Already exists: {output_path}")
                continue

            print(f"[🚀 GPU {gpu_id}] Generating V2V: '{prompt}'")
            print(f"   → Input:  {prefix_video_path}")
            print(f"   → Output: {output_path}")

            # Reset seed before each prompt to ensure reproducibility
            set_random_seed(pipeline.config.runtime_config.seed)

            pipeline.run_video_to_video(
                prompt=prompt,
                prefix_video_path=prefix_video_path,
                output_path=output_path
            )
            print(f"[✅ GPU {gpu_id}] Saved: {output_path}")

    print(f"[GPU {gpu_id}] Done.")

def load_physicsiq_samples(args):
    """Load sample list for PhysicsIQ dataset"""
    DATA_ROOT = args.physicsiq_data_dir
    DESCRIPTIONS_CSV = os.path.join(DATA_ROOT, "descriptions", "descriptions.csv")
    OUTPUT_DIR = args.save_path
    FPS = 24 # MAGI uses 24 FPS

    if not os.path.exists(DESCRIPTIONS_CSV):
        raise FileNotFoundError(f"descriptions.csv not found at {DESCRIPTIONS_CSV}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    samples = []
    with open(DESCRIPTIONS_CSV, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row['scenario'].strip()
            match = re.match(r'^(\d+)_', scenario)
            if not match:
                print(f"Cannot extract ID from scenario: {scenario}")
                continue

            vid_id = match.group(1).zfill(4)
            description = row['description']
            generated_video_name = row['generated_video_name']

            # Construct conditioning video path
            conditioning_dir = os.path.join(DATA_ROOT, "physics-IQ-benchmark", "split-videos", "conditioning", f"{FPS}FPS")
            match_suffix = re.search(r'_(.*)', scenario)
            prefix_video_name = match_suffix.group(1) if match_suffix else ""
            filename = f"{vid_id}_conditioning-videos_{FPS}FPS_{prefix_video_name}"
            prefix_video_path = os.path.join(conditioning_dir, filename)

            output_path = os.path.join(OUTPUT_DIR, generated_video_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            samples.append({
                'vid_id': vid_id,
                'scenario': scenario,
                'description': description,
                'generated_video_name': generated_video_name,
                'prefix_video_path': prefix_video_path,
                'output_path': output_path
            })

    total_samples_num = len(samples)
    unique_samples_num = total_samples_num // 2

    print(f"[✅] Loaded {unique_samples_num} samples.")

    # physicsiq samples twice, we only take the first half
    filtered_samples = samples[:unique_samples_num]

    # Apply start/end slice
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else len(filtered_samples)

        # Ensure index is within valid range
        start = max(0, min(start, len(filtered_samples)))
        end = max(start, min(end, len(filtered_samples)))

        filtered_samples = filtered_samples[start:end]
        print(f"[✂️] Sliced to samples {start} to {end-1}, total {len(filtered_samples)} samples.")

    return filtered_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, choices=['vbench', 'physicsiq'], help="Benchmark name")
    parser.add_argument("--vbench_prompt_dir", type=str, help="Directory containing prompt files")
    parser.add_argument("--dimension", type=str, help="Benchmark dimension")
    parser.add_argument('--physicsiq_data_dir', type=str, default='physics-iq-benchmark', help='Root path of the dataset')
    parser.add_argument("--save_path", type=str, default="./benchmark_videos", help="Output directory")
    parser.add_argument("--config_file", type=str, default="example/4.5B/4.5B_distill_config.json", help="Model config file")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to use")
    parser.add_argument("--reuse_strategy", type=str, choices=['all', 'chunkwise', 'original'], required=True, help="Reuse strategy for video generation")
    parser.add_argument('--rel_l1_thresh', type=float, default=0, help='Relative L1 distance threshold for TeaCache.')
    parser.add_argument('--no_reuse_first_n_steps', type=int, default=0, help='Number of steps to not reuse output.')

    # chunkwise reuse
    parser.add_argument('--discard_nearly_clean_chunk', action='store_true', help='Whether to discard nearly clean chunks.')
    parser.add_argument('--whole_calc_when_cross', action='store_true', help='Whether to perform whole calculation when crossing stage which number of chunks increases.')
    parser.add_argument(
        '--no_reuse_mode', type=str, choices=['first', 'mid', 'none'],
        help='Detemine which part of steps to not reuse output. first: first n steps; mid: mid n steps; none: not set .'
    )
    # KV cache compression
    parser.add_argument('--compress_kv_cache', action='store_true', help='Whether to compress kv cache.')
    parser.add_argument('--total_cache_chunk_nums', type=int, default=None, help='Total length of kv cache.')
    parser.add_argument('--compress_strategy', type=str, default='token', choices=['token', 'frame', 'chunk'], help='KV cache compression strategy.')
    parser.add_argument('--mix_lambda', type=float, default=0.07, help='Mix lambda for token compression.')
    parser.add_argument('--query_granularity', type=str, default='chunk', choices=['chunk', 'frame', 'token'], help='Query granularity for token compression.')
    parser.add_argument('--score_weighting_method', type=str, default='no_weight', choices=['no_weight','hard_code', 'exponential', 'polynomial', 'gaussian', "upper_convex_polynomial"], help='Score weighting method for KV compression.')
    parser.add_argument('--power', type=float, default=3, help='Power parameter for polynomial and upper_convex_polynomial weighting methods.')

    # log
    parser.add_argument('--log', action='store_true', help='Whether to log the TeaCache information.')

    # Token-wise reuse parameters
    parser.add_argument('--token_wise_reuse', action='store_true', help='Enable token-wise reuse strategy.')
    parser.add_argument('--token_rel_l1_thresh', type=float, default=None, help='Token-level relative L1 distance threshold (defaults to rel_l1_thresh if not set).')
    parser.add_argument('--tokenwise_l1_mode', type=str, default='chunk', choices=['chunk', 'token'], help='Mode for token-level rel_l1 calculation: "chunk" uses chunk-level rel_l1 for all tokens, "token" computes per-token rel_l1 independently.')

    # Three-phase control parameters
    parser.add_argument('--warmup_steps', type=int, default=5, help='Number of warmup steps where no caching is used.')
    parser.add_argument('--chunk_wise_only_steps', type=int, default=0, help='Number of steps for chunk-wise only phase (before token-wise phase).')

    # Token reuse ratio control
    parser.add_argument('--max_token_reuse_ratio', type=float, default=1.0, help='Fixed maximum ratio of tokens that can be reused (0~1). Set to 1.0 to allow all tokens to be reused.')
    parser.add_argument('--initial_token_reuse_ratio', type=float, default=None, help='Initial token reuse ratio at the start of hierarchical phase (for dynamic ratio mode).')
    parser.add_argument('--final_token_reuse_ratio', type=float, default=None, help='Final token reuse ratio at the end of hierarchical phase (for dynamic ratio mode).')

    # Continuous reuse tracking (adaptive refresh mechanism)
    parser.add_argument('--enable_continuous_reuse_tracking', action='store_true', help='Enable continuous reuse tracking (adaptive refresh mechanism).')
    parser.add_argument('--continuous_reuse_max_count', type=int, default=None, help='Force forward after N consecutive reuses (set to null for dynamic threshold mode).')
    parser.add_argument('--continuous_reuse_decay_mode', type=str, default='exponential', choices=['exponential', 'linear'], help='Decay mode for continuous reuse tracking.')
    parser.add_argument('--continuous_reuse_decay_factor', type=float, default=0.1, help='Decay factor for continuous reuse tracking.')

    # Temporal weight parameters
    parser.add_argument('--temporal_weight_floor', type=float, default=0.0, help='Floor for temporal weight normalization, maps weights to [floor, 1] range.')
    parser.add_argument('--temporal_weight_power', type=float, default=None, help='Power for nonlinear temporal weight normalization (default: None=linear). Values < 1 make more tokens closer to 1 (convex curve), > 1 make more tokens closer to floor.')
    parser.add_argument('--enable_temporal_voting', action='store_true', help='Enable temporal voting: force tokens at the same spatial position across frames to have the same reuse decision via majority voting.')

    # Debug and visualization
    parser.add_argument('--print_token_stats', action='store_true', help='Print token-wise reuse statistics.')
    parser.add_argument('--visualize_reuse_mask', action='store_true', help='Visualize token reuse mask on the output video.')
    parser.add_argument('--visualize_temporal_diff', action='store_true', help='Visualize temporal difference heatmap on the output video.')
    parser.add_argument('--temporal_diff_step', type=int, nargs='+', default=[0], help='Which denoising step(s) to compute temporal difference mask (0-based). Can be a single step or multiple steps.')
    parser.add_argument('--temporal_diff_mode', type=str, default='clean', choices=['clean', 'noise'], help='Mode for temporal difference calculation: "clean" uses integrated clean latent, "noise" uses model output (predicted noise).')
    parser.add_argument('--visualize_temporal_weights', action='store_true', help='Visualize temporal weights heatmap on the output video.')
    parser.add_argument('--temporal_weights_step', type=int, nargs='+', default=[0], help='Which denoising step(s) to compute temporal weights mask (0-based). Can be a single step or multiple steps.')

    # sampling range control
    parser.add_argument('--start', type=int, default=None, help='Start index of samples to process (inclusive)')
    parser.add_argument('--end', type=int, default=None, help='End index of samples to process (exclusive)')

    args = parser.parse_args()
    print(f"[✅] Parsed arguments: {args}")

    if args.benchmark == 'vbench':
        assert os.path.exists(args.vbench_prompt_dir), f"Prompt directory not found: {args.vbench_prompt_dir}"
        prompt_file = os.path.join(args.vbench_prompt_dir, f"{args.dimension}.txt")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r') as f:
            all_samples = [line.strip() for line in f if line.strip()]

        # Apply start/end slice
        if args.start is not None or args.end is not None:
            start = args.start if args.start is not None else 0
            end = args.end if args.end is not None else len(all_samples)

            # Ensure index is within valid range
            start = max(0, min(start, len(all_samples)))
            end = max(start, min(end, len(all_samples)))

            all_samples = all_samples[start:end]
            print(f"[✂️] Sliced to prompts {start} to {end-1}, total {len(all_samples)} prompts.")

    elif args.benchmark == 'physicsiq':
        assert os.path.exists(args.physicsiq_data_dir), f"Data root directory not found: {args.physicsiq_data_dir}"
        all_samples = load_physicsiq_samples(args)
    
    else:
        raise ValueError(f"Invalid benchmark: {args.benchmark}")
 
    gpu_ids = list(map(int, args.gpus.split(',')))
    args.num_gpus = len(gpu_ids)
    os.makedirs(args.save_path, exist_ok=True)

    print(f"Total prompts: {len(all_samples)}")
    print(f"GPUs: {gpu_ids}")
    print(f"Output: {args.save_path}")
    print(f"Config: {args.config_file}")

    # Notify user of log file location
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_run_dir = os.path.join(args.save_path, "log", run_timestamp)
    print(f"GPU logs will be saved to: {log_run_dir}/")
    print(f"   Each GPU will have its own log file: gpu_<ID>.log")
    print("")

    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        print(f"[Main] Starting worker for GPU {gpu_id} (rank {rank})...")
        p = mp.Process(target=worker_process, args=(gpu_id, rank, args, all_samples, run_timestamp))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()