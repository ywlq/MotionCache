# Copyright 2025 MAGI Authors
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

import os
import cv2
import argparse
import torch
import lpips
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchmetrics.image import StructuralSimilarityIndexMeasure


def load_video_frames(path, resize_to=None):
    """
    Load all frames from a video file as a list of HxWx3 uint8 arrays.
    Optionally resize each frame to `resize_to` (w, h).
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if resize_to is not None:
            img = cv2.resize(img, resize_to)
        frames.append(np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0))
    cap.release()
    return np.concatenate(frames)


def compute_video_metrics(frames_gt, frames_gen,
                          device, ssim_metric, lpips_fn):
    """
    Compute PSNR, SSIM, LPIPS for two lists of frames (uint8 BGR).
    All computations on `device`.
    Returns (psnr, ssim, lpips) scalars.
    """
    # ensure same frame count
    # convert to tensors [N,3,H,W], normalize to [0,1]
    gt_t = torch.from_numpy(frames_gt).float().to(device).permute(0, 3, 1, 2).div_(255).contiguous()
    gen_t = torch.from_numpy(frames_gen).float().to(device).permute(0, 3, 1, 2).div_(255).contiguous()

    # PSNR (data_range=1.0): -10 * log10(mse)
    mse = torch.mean((gt_t - gen_t) ** 2)
    psnr = -10.0 * torch.log10(mse)

    # SSIM: returns average over batch
    ssim_val = ssim_metric(gen_t, gt_t)

    # LPIPS: expects [-1,1]
    with torch.no_grad():
        lpips_val = lpips_fn(gt_t * 2.0 - 1.0, gen_t * 2.0 - 1.0).mean()

    return psnr.item(), ssim_val.item(), lpips_val.item()


def get_video_files(folder):
    """
    Get all video files from a folder (supports .mp4, .avi, .mov, etc.)
    Returns dictionary: {filename: full_path}
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    folder_path = Path(folder)

    video_files = {}
    for ext in video_extensions:
        for f in folder_path.glob(f'*{ext}'):
            video_files[f.name] = str(f)
        for f in folder_path.glob(f'*{ext.upper()}'):
            video_files[f.name] = str(f)

    return video_files


def main():
    parser = argparse.ArgumentParser(
        description="Randomly select n videos from two folders, compute PSNR/SSIM/LPIPS metrics and print averages"
    )
    parser.add_argument("--original_folder", required=True,
                        help="Path to original video folder")
    parser.add_argument("--generated_folder", required=True,
                        help="Path to generated video folder")
    parser.add_argument("--num_videos", type=int, default=10,
                        help="Number of videos to randomly select (default: 10)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device, e.g., 'cuda' or 'cpu'")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"],
                        help="LPIPS backbone network")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Instantiate metric computation modules
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net=args.lpips_net, spatial=True).to(device)

    # Get video files from both folders (returns dict: filename -> path)
    original_videos = get_video_files(args.original_folder)
    generated_videos = get_video_files(args.generated_folder)

    if not original_videos:
        print(f"Error: No video files found in original folder {args.original_folder}")
        return

    if not generated_videos:
        print(f"Error: No video files found in generated folder {args.generated_folder}")
        return

    print(f"Found {len(original_videos)} videos in original folder")
    print(f"Found {len(generated_videos)} videos in generated folder")

    # Randomly select n videos from original folder, then find matching videos in generated folder
    original_filenames = list(original_videos.keys())
    num_to_select = min(args.num_videos, len(original_filenames))

    if num_to_select < args.num_videos:
        print(f"Warning: Requested {args.num_videos} videos, but only {num_to_select} available")

    selected_filenames = random.sample(original_filenames, num_to_select)

    # Filter video pairs that exist in both folders
    video_pairs = []
    for fname in selected_filenames:
        if fname in generated_videos:
            video_pairs.append((original_videos[fname], generated_videos[fname]))
        else:
            print(f"Warning: Matching video not found in generated folder: {fname}, skipping")

    if not video_pairs:
        print("\nError: No matching video pairs found!")
        print("Please ensure both folders have videos with the same filenames.")
        return

    print(f"\nFound {len(video_pairs)} matching video pairs for evaluation\n")

    psnrs, ssims, lpips_vals = [], [], []

    for i, (path_gt, path_gen) in enumerate(tqdm(video_pairs,
                                                  total=len(video_pairs),
                                                  desc="Processing videos")):
        print(f"\nProcessing video pair {i+1}/{len(video_pairs)}:")
        print(f"  Original video: {os.path.basename(path_gt)}")
        print(f"  Generated video: {os.path.basename(path_gen)}")

        try:
            # Load frames; resize generated video to match original video dimensions
            frames_gt = load_video_frames(path_gt)
            frames_gen = load_video_frames(path_gen)

            res = compute_video_metrics(frames_gt, frames_gen,
                                        device, ssim_metric, lpips_fn)
            p, s, l = res
            psnrs.append(p)
            ssims.append(s)
            lpips_vals.append(l)

            print(f"  PSNR: {p:.2f} dB, SSIM: {s:.4f}, LPIPS: {l:.4f}")
        except Exception as e:
            print(f"  Error: Processing failed - {str(e)}")
            continue

    if not psnrs:
        print("\nNo videos were successfully processed.")
        return

    print("\n" + "="*50)
    print(f"Average values based on {len(psnrs)} video pairs:")
    print("="*50)
    print(f"Avg PSNR  : {np.mean(psnrs):.2f} dB (±{np.std(psnrs):.2f})")
    print(f"Avg SSIM  : {np.mean(ssims):.4f} (±{np.std(ssims):.4f})")
    print(f"Avg LPIPS : {np.mean(lpips_vals):.4f} (±{np.std(lpips_vals):.4f})")
    print("="*50)


if __name__ == "__main__":
    main()
