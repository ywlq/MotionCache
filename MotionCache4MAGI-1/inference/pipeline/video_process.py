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

import gc
import os
import tempfile

import ffmpeg
import torch
from einops import rearrange

import inference.infra.distributed.parallel_state as mpu
from inference.common import MagiConfig, magi_logger
from inference.model.vae import AutoModel, DiagonalGaussianDistribution, VideoTokenizerABC


############################################
# VaeHelper
###########################################
class SingletonMeta(type):
    """
    Singleton metaclass
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class VaeHelper(metaclass=SingletonMeta):
    def __init__(self):
        # Initialize cache dict
        if not hasattr(self, "vae_cache_dict"):
            self.vae_cache_dict = {}

    @staticmethod
    def get_vae(vae_ckpt: str) -> VideoTokenizerABC:
        """
        Load a pretrained VAE model.

        Args:
            vae_ckpt (str): Path to the pretrained VAE checkpoint.

        Returns:
            VideoTokenizerABC: Pretrained VAE model.
        """
        vae_helper = VaeHelper()

        if vae_ckpt not in vae_helper.vae_cache_dict:
            vae = AutoModel.from_pretrained(vae_ckpt)
            vae.encode = vae_helper.patch_vae_encode.__get__(vae)
            vae.cuda()
            vae.eval()
            vae.bfloat16()
            if os.environ.get("OFFLOAD_VAE_CACHE") == "true":
                return vae
            vae_helper.vae_cache_dict[vae_ckpt] = vae
        return vae_helper.vae_cache_dict[vae_ckpt]

    @staticmethod
    @torch.no_grad()
    def patch_vae_encode(vae: callable, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input video.

        Args:
            x (torch.Tensor): Input video tensor with shape (N, C, T, H, W).
            sample_posterior (bool): Whether to sample from the posterior.

        Returns:
            torch.Tensor: Encoded tensor with additional information.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input x to be torch.Tensor, but got {type(x)}.")
        if len(x.shape) != 5:
            raise ValueError(f"Expected input tensor x to have shape (N, C, T, H, W), but got {x.shape}.")

        if not hasattr(vae, "encoder") or not callable(vae.encoder):
            raise AttributeError("Encoder is not defined or callable. Please initialize 'self.encoder'.")

        # for setting vae encoding to deterministic
        N, C, T, H, W = x.shape
        if T == 1:
            x = x.expand(-1, -1, 4, -1, -1)
            x = vae.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()

            return z[:, :, :1, :, :].type(x.dtype)
        else:
            x = vae.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()

            return z.type(x.dtype)

    @staticmethod
    def encode(
        video: torch.Tensor,
        vae: VideoTokenizerABC,
        tile_sample_min_length: int = 16,
        tile_sample_min_height: int = 256,
        tile_sample_min_width: int = 256,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        allow_spatial_tiling: bool = True,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """
        Encode the input tensor.
        Args:
            video (torch.Tensor): Input tensor with shape (N, T, C, H, W).
            vae (VideoTokenizerABC): Pretrained VAE model.
            tile_sample_min_length (int): Minimum length of the tile sample.
            tile_sample_min_height (int): Minimum height of the tile sample.
            tile_sample_min_width (int): Minimum width of the tile sample.
            spatial_tile_overlap_factor (float): Spatial tile overlap factor.
            allow_spatial_tiling (bool): Allow spatial tiling.
            parallel_group (ProcessGroup): Distributed encoding group.
        Returns:
            torch.Tensor: Encoded tensor.
        """
        assert video.dim() == 5, f"Expected input tensor to have shape (N, T, C, H, W), but got {video.shape}."
        video = video.cuda()
        video = (video / 127.5) - 1.0
        video = video.bfloat16()
        moments = vae.tiled_encode_3d(
            video,
            tile_sample_min_length=tile_sample_min_length,
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            spatial_tile_overlap_factor=spatial_tile_overlap_factor,
            temporal_tile_overlap_factor=temporal_tile_overlap_factor,
            allow_spatial_tiling=allow_spatial_tiling,
            parallel_group=parallel_group,
        )

        return moments

    @staticmethod
    def decode(
        chunk: torch.Tensor,
        vae: VideoTokenizerABC,
        tile_sample_min_height: int = 256,
        tile_sample_min_width: int = 256,
        spatial_tile_overlap_factor: float = 0.25,
        temporal_tile_overlap_factor: float = 0,
        tile_sample_min_length: int = 16,
        allow_spatial_tiling: bool = True,
        uint8_output: bool = True,
        parallel_group: torch.distributed.ProcessGroup = None,
    ) -> torch.Tensor:
        """
        Decode the input tensor.
        Args:
            chunk (torch.Tensor): Input tensor with shape (N, C, T, H, W).
            vae (VideoTokenizerABC): Pretrained VAE model.
            tile_sample_min_length (int): Minimum length of the tile sample.
            tile_sample_min_height (int): Minimum height of the tile sample.
            tile_sample_min_width (int): Minimum width of the tile sample.
            spatial_tile_overlap_factor (float): Spatial tile overlap factor.
            temporal_tile_overlap_factor (float): Temporal tile overlap factor.
            allow_spatial_tiling (bool): Allow spatial tiling.
            uint8_output (bool): Whether to output uint8 tensor.
            parallel_group (ProcessGroup): Distributed decoding group.
        Returns:
            torch.Tensor: Decoded tensor.
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chunk = vae.tiled_decode_3d(
                chunk,
                tile_sample_min_height=tile_sample_min_height,
                tile_sample_min_width=tile_sample_min_width,
                spatial_tile_overlap_factor=spatial_tile_overlap_factor,
                temporal_tile_overlap_factor=temporal_tile_overlap_factor,
                tile_sample_min_length=tile_sample_min_length,
                allow_spatial_tiling=allow_spatial_tiling,
                parallel_group=parallel_group,
            )
        chunk = rearrange(chunk, "b c t h w -> (b t) c h w")
        if uint8_output:
            chunk = (chunk * 127.5) + 127.5
            chunk = chunk.clamp(0, 255)
            chunk = chunk.type(torch.uint8)
        return chunk


############################################
# Process to get prefix video
###########################################


def ffmpeg_i2v(image_path, w=384, h=224, aspect_policy="fit"):
    r = ffmpeg.input("pipe:0", format="image2pipe")
    if aspect_policy == "crop":
        r = r.filter("scale", w, h, force_original_aspect_ratio="increase").filter("crop", w, h)
    elif aspect_policy == "pad":
        r = r.filter("scale", w, h, force_original_aspect_ratio="decrease").filter(
            "pad", w, h, "(ow-iw)/2", "(oh-ih)/2", color="black"
        )
    elif aspect_policy == "fit":
        r = r.filter("scale", w, h)
    else:
        magi_logger.warning(f"Unknown aspect policy: {aspect_policy}, using fit as fallback")
        r = r.filter("scale", w, h)
    image_byte = open(image_path, "rb").read()
    try:
        out, _ = r.output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=1).run(
            input=image_byte, capture_stdout=True, capture_stderr=True
        )
    except ffmpeg.Error as e:
        print(f"Error occurred: {e.stderr.decode()}")
        raise e

    video = torch.frombuffer(out, dtype=torch.uint8).view(1, h, w, 3)
    return video


def ffmpeg_v2v(video_path, fps, w=384, h=224, prefix_frame=None, prefix_video_max_chunk=5):
    if video_path is None:
        return None
    out, _ = (
        ffmpeg.input(video_path, ss=0, format="mp4")
        .filter("fps", fps=fps)
        .filter("scale", w, h)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", nostdin=None)
        .run(capture_stdout=True, capture_stderr=True)
    )

    video = torch.frombuffer(out, dtype=torch.uint8).view(-1, h, w, 3)

    if prefix_frame is not None:
        return video[:prefix_frame]
    else:
        num_frames_to_read = video.shape[0]
        if num_frames_to_read < fps:
            clip_length = 1
        else:
            PREFIX_VIDEO_MAX_FRAMES = prefix_video_max_chunk * fps
            clip_length = min(num_frames_to_read // fps * fps, PREFIX_VIDEO_MAX_FRAMES)
        return video[-clip_length:]


def save_video_to_disk(video: torch.Tensor, save_path: str, fps: int) -> bytes:
    # TCHW -> THWC
    video = video.permute(0, 2, 3, 1).cpu().numpy()
    _, H, W, _ = video.shape
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video.tobytes())
        temp_file.flush()
        temp_file_path = temp_file.name

    try:
        output, err = (
            ffmpeg
            .input(temp_file_path, format="rawvideo", pix_fmt="rgb24", s=f"{W}x{H}", r=fps)
            .output(save_path, format='mp4', vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("✅ Video saved successfully.")
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf8') if e.stderr else "No stderr output"
        print("❌ FFmpeg Error:")
        print("="*60)
        print(stderr_output)
        print("="*60)
        raise RuntimeError("Failed to encode video with FFmpeg") from e

        os.remove(temp_file_path)
        return output

    os.remove(temp_file_path)
    return output


def encode_prefix_video(prefix_video, fps, vae_ckpt, scale_factor, parallel_group):
    if prefix_video is None:
        return None
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated before vae encode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved before vae encode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    # THWC -> NCTHW
    prefix_video = prefix_video.permute(3, 0, 1, 2).unsqueeze(0)
    magi_logger.debug(f"prefix_video.shape: {prefix_video.shape}")
    vae_model = VaeHelper.get_vae(vae_ckpt)
    tile_sample_min_length = fps // 2
    prefix_video = VaeHelper.encode(
        prefix_video,
        vae_model,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        spatial_tile_overlap_factor=0.25,
        temporal_tile_overlap_factor=0,
        tile_sample_min_length=tile_sample_min_length,
        allow_spatial_tiling=True,
        parallel_group=parallel_group,
    )
    prefix_video = prefix_video * scale_factor
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated after vae encode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved after vae encode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )
    return prefix_video


def process_image(image_path: str, config: MagiConfig) -> torch.Tensor:
    prefix_video = ffmpeg_i2v(image_path, w=config.runtime_config.video_size_w, h=config.runtime_config.video_size_h)
    prefix_video = encode_prefix_video(
        prefix_video,
        config.runtime_config.fps,
        config.runtime_config.vae_pretrained,
        config.runtime_config.scale_factor,
        parallel_group=mpu.get_tp_group(with_context_parallel=True),
    )
    return prefix_video


def process_prefix_video(prefix_video_path: str, config: MagiConfig) -> torch.Tensor:
    prefix_video = ffmpeg_v2v(
        prefix_video_path,
        fps=config.runtime_config.fps,
        prefix_frame=None,    # Modified
        w=config.runtime_config.video_size_w,
        h=config.runtime_config.video_size_h,
    )
    prefix_video = encode_prefix_video(
        prefix_video,
        config.runtime_config.fps,
        config.runtime_config.vae_pretrained,
        config.runtime_config.scale_factor,
        parallel_group=mpu.get_tp_group(with_context_parallel=True),
    )
    return prefix_video


############################################
# Process to get final video
############################################
def decode_chunk(chunk, vae_ckpt, scale_factor, tile_sample_min_length, parallel_group):
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated before vae decode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved before vae decode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    vae_model = VaeHelper.get_vae(vae_ckpt)
    decoded_chunk = VaeHelper.decode(
        chunk / scale_factor,
        vae_model,
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        spatial_tile_overlap_factor=0.25,
        temporal_tile_overlap_factor=0,
        tile_sample_min_length=tile_sample_min_length,
        allow_spatial_tiling=True,
        parallel_group=parallel_group,
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory allocated after vae decode: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    magi_logger.debug(
        f"rank {torch.distributed.get_rank()} memory reserved after vae decode: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )
    return decoded_chunk


def post_chunk_process(chunk: torch.Tensor, config: MagiConfig):
    tile_sample_min_length = config.runtime_config.fps // 2
    chunk = decode_chunk(
        chunk,
        config.runtime_config.vae_pretrained,
        config.runtime_config.scale_factor,
        tile_sample_min_length,
        parallel_group=mpu.get_tp_group(with_context_parallel=True),
    )
    gc.collect()
    torch.cuda.empty_cache()
    return chunk


############################################
# Apply Reuse Mask Visualization
###########################################
def apply_reuse_mask_overlay(
    videos: torch.Tensor,
    reuse_masks: dict,
    model_config,
    latent_size: tuple,
) -> torch.Tensor:
    """
    Apply token reuse mask visualization on decoded videos.

    Args:
        videos: [T, C, H, W] tensor in pixel space (after VAE decode)
        reuse_masks: dict {chunk_id: token_mask} where token_mask is [chunk_token_nums] bool tensor
        model_config: model config containing patch_size and t_patch_size
        latent_size: latent spatial size tuple [B, C, T, H, W]

    Returns:
        videos_with_mask: [T, C, H, W] tensor with semi-transparent color overlay
    """
    import torch.nn.functional as F

    # Hardcoded colors and alpha
    # Colors should be in the appropriate range for the dtype
    # For uint8: [0, 255], for float: [0.0, 1.0]
    if videos.dtype == torch.uint8:
        reuse_color = torch.tensor([0, 255, 0], device=videos.device, dtype=torch.uint8)  # Green for reused
        non_reuse_color = torch.tensor([255, 0, 0], device=videos.device, dtype=torch.uint8)  # Red for non-reused
    else:
        reuse_color = torch.tensor([0.0, 1.0, 0.0], device=videos.device)  # Green for reused
        non_reuse_color = torch.tensor([1.0, 0.0, 0.0], device=videos.device)  # Red for non-reused
    alpha = 0.3  # Semi-transparent

    # Get video dimensions [T, C, H, W]
    T, C, H, W = videos.shape

    # Get patch sizes
    patch_size = model_config.patch_size
    t_patch_size = model_config.t_patch_size

    # Get latent dimensions
    _, _, T_latent, H_latent, W_latent = latent_size

    # Calculate token space dimensions
    T_tokens = T_latent // t_patch_size
    H_tokens = H_latent // patch_size
    W_tokens = W_latent // patch_size

    print(f"[DEBUG] latent_size={latent_size}, T_latent={T_latent}, H_latent={H_latent}, W_latent={W_latent}")
    print(f"[DEBUG] patch_size={patch_size}, t_patch_size={t_patch_size}")
    print(f"[DEBUG] T_tokens={T_tokens}, H_tokens={H_tokens}, W_tokens={W_tokens}")

    # Calculate total tokens per chunk
    # token_reuse_masks are stored per chunk, need to calculate chunk_token_nums
    chunk_token_nums = T_tokens * H_tokens * W_tokens
    print(f"[DEBUG] chunk_token_nums={chunk_token_nums}")

    # Reassemble full token mask from all chunks
    # Sort chunks by ID to ensure correct order
    sorted_chunk_ids = sorted(reuse_masks.keys())
    print(f"[DEBUG] apply_reuse_mask_overlay: sorted_chunk_ids={sorted_chunk_ids}, num_chunks={len(sorted_chunk_ids)}")
    total_tokens = len(sorted_chunk_ids) * chunk_token_nums

    # Create full token mask
    full_token_mask = torch.zeros(total_tokens, dtype=torch.bool, device=videos.device)

    for idx, chunk_id in enumerate(sorted_chunk_ids):
        token_mask = reuse_masks[chunk_id]
        print(f"[DEBUG] apply_reuse_mask_overlay: chunk_id={chunk_id}, token_mask shape={token_mask.shape}, "
              f"num_true={token_mask.sum().item()}, dtype={token_mask.dtype}")
        start_idx = idx * chunk_token_nums
        end_idx = start_idx + chunk_token_nums

        # Handle case where token_mask size might not match exactly
        actual_size = token_mask.shape[0]
        full_token_mask[start_idx:start_idx + actual_size] = token_mask

    # Reshape to 3D token space
    full_token_mask_3d = full_token_mask.reshape(T_tokens, H_tokens, W_tokens)  # [T_tokens, H_tokens, W_tokens]
    print(f"[DEBUG] full_token_mask_3d shape={full_token_mask_3d.shape}, num_true={full_token_mask_3d.sum().item()}")

    # Upsample to latent resolution using repeat_interleave
    latent_mask_3d = full_token_mask_3d.repeat_interleave(t_patch_size, dim=0) \
                                        .repeat_interleave(patch_size, dim=1) \
                                        .repeat_interleave(patch_size, dim=2)
    # latent_mask_3d shape: [T_latent, H_latent, W_latent]
    print(f"[DEBUG] latent_mask_3d shape={latent_mask_3d.shape}, num_true={latent_mask_3d.sum().item()}")

    # Upsample to video resolution using repeat (pixel-perfect, matching VAE decoder)
    # VAE decoder upsampling factors: temporal 4x, spatial 8x
    temporal_upscale = 4
    spatial_upscale = 8

    # Use repeat to precisely match VAE's upsampling
    video_mask_3d = latent_mask_3d.repeat(temporal_upscale, spatial_upscale, spatial_upscale)
    # video_mask_3d shape: [T_latent*4, H_latent*8, W_latent*8]
    print(f"[DEBUG] video_mask_3d shape before crop/pad={video_mask_3d.shape}, num_true={video_mask_3d.sum().item()}")

    # Handle possible boundary misalignment (crop or pad to exact video size)
    T_up, H_up, W_up = video_mask_3d.shape
    if T_up > T or H_up > H or W_up > W:
        # Crop if upscaled size is larger
        video_mask_3d = video_mask_3d[:T, :H, :W]
    elif T_up < T or H_up < H or W_up < W:
        # Pad if upscaled size is smaller (use nearest padding)
        pad_t = T - T_up
        pad_h = H - H_up
        pad_w = W - W_up
        video_mask_3d = F.pad(video_mask_3d, (0, pad_w, 0, pad_h, 0, pad_t), mode='constant', value=0)

    # Convert to bool
    video_mask = video_mask_3d > 0.5  # [T, H, W]
    print(f"[DEBUG] apply_reuse_mask_overlay: video_mask shape={video_mask.shape}, "
          f"num_true={video_mask.sum().item()}, num_false={(~video_mask).sum().item()}, "
          f"videos.dtype={videos.dtype}")

    # Create color overlay [T, C, H, W]
    # Create green mask for reused tokens, red for non-reused
    overlay = torch.zeros(T, C, H, W, device=videos.device, dtype=videos.dtype)

    # Apply colors based on mask
    # For RGB (C=3), apply color per channel
    if C == 3:
        for c in range(3):
            overlay[:, c] = torch.where(video_mask, reuse_color[c], non_reuse_color[c])
    else:
        # For non-RGB, only color first 3 channels
        for c in range(min(3, C)):
            overlay[:, c] = torch.where(video_mask, reuse_color[c], non_reuse_color[c])

    print(f"[DEBUG] apply_reuse_mask_overlay: overlay shape={overlay.shape}, "
          f"min={overlay.min()}, max={overlay.max()}, dtype={overlay.dtype}")
    print(f"[DEBUG] apply_reuse_mask_overlay: reuse_color={reuse_color}, non_reuse_color={non_reuse_color}")

    # Blend with original video
    # videos: [T, C, H, W], overlay: [T, C, H, W]
    # Convert to float for blending if videos is uint8
    original_dtype = videos.dtype
    if original_dtype == torch.uint8:
        videos_float = videos.float() / 255.0
        overlay_float = overlay.float() / 255.0
        videos_with_mask = videos_float * (1 - alpha) + overlay_float * alpha
        videos_with_mask = (videos_with_mask * 255.0).clamp(0, 255).to(original_dtype)
    else:
        videos_with_mask = videos * (1 - alpha) + overlay * alpha

    return videos_with_mask


def apply_temporal_diff_overlay(
    videos: torch.Tensor,
    temporal_diff_mask: torch.Tensor,
    latent_size: tuple,
) -> torch.Tensor:
    """
    Apply temporal difference heatmap visualization on decoded videos.

    Args:
        videos: [T, C, H, W] tensor in pixel space (after VAE decode)
        temporal_diff_mask: [N, H_latent, W_latent] tensor with temporal difference values
        latent_size: latent spatial size tuple [B, C, T, H, W]

    Returns:
        videos_with_overlay: [T, C, H, W] tensor with semi-transparent color overlay
    """
    import torch.nn.functional as F

    # Alpha for blending (higher value = more opaque mask)
    alpha = 0.8  # Strongly visible mask

    # Get video dimensions [T, C, H, W]
    T, C, H, W = videos.shape

    # Get latent dimensions
    _, _, T_latent, H_latent, W_latent = latent_size

    # temporal_diff_mask is [N, H_latent, W_latent]
    # We need to broadcast it to temporal dimension and upscale to video resolution
    N_mask, H_mask, W_mask = temporal_diff_mask.shape

    # Normalize temporal_diff_mask to [0, 1] range for color mapping
    mask_min = temporal_diff_mask.min()
    mask_max = temporal_diff_mask.max()
    if mask_max - mask_min > 1e-8:
        normalized_mask = (temporal_diff_mask - mask_min) / (mask_max - mask_min)
    else:
        normalized_mask = torch.zeros_like(temporal_diff_mask)

    # Upsample mask from latent to video resolution
    # VAE decoder upsampling factors: temporal 4x, spatial 8x
    temporal_upscale = 4
    spatial_upscale = 8

    # First broadcast to temporal dimension (repeat for each frame)
    # normalized_mask: [N, H_latent, W_latent] -> [T_latent, N, H_latent, W_latent]
    mask_4d = normalized_mask.unsqueeze(0).repeat(T_latent, 1, 1, 1)  # [T_latent, N, H_latent, W_latent]

    # Permute to match spatial dimensions for upsampling: [T_latent, N, H_latent, W_latent]
    # Upsample spatial dimensions by 8x using nearest neighbor
    mask_upsampled = F.interpolate(
        mask_4d.view(T_latent * N_mask, 1, H_mask, W_mask),
        scale_factor=spatial_upscale,
        mode='nearest'
    )  # [T_latent * N, 1, H_latent*8, W_latent*8]

    mask_upsampled = mask_upsampled.view(T_latent, N_mask, 1, H_mask * spatial_upscale, W_mask * spatial_upscale)
    mask_upsampled = mask_upsampled.squeeze(2)  # [T_latent, N, H_up, W_up]

    # Now handle temporal upsample
    # We need to go from T_latent to T frames
    # Use repeat to expand (similar to VAE temporal decoder)
    T_up = T_latent * temporal_upscale
    H_up = H_mask * spatial_upscale
    W_up = W_mask * spatial_upscale

    # Repeat temporal dimension
    mask_video = mask_upsampled.repeat_interleave(temporal_upscale, dim=0)  # [T_latent*4, N, H_up, W_up]

    # Average over batch dimension (N) to get single heatmap
    mask_video = mask_video.mean(dim=1)  # [T_up, H_up, W_up]


    # Handle possible boundary misalignment (crop or pad to exact video size)
    T_mask, H_mask_up, W_mask_up = mask_video.shape
    if T_mask > T or H_mask_up > H or W_mask_up > W:
        # Crop if upscaled size is larger
        mask_video = mask_video[:T, :H, :W]
    elif T_mask < T or H_mask_up < H or W_mask_up < W:
        # Pad if upscaled size is smaller
        pad_t = T - T_mask
        pad_h = H - H_mask_up
        pad_w = W - W_mask_up
        mask_video = F.pad(mask_video, (0, pad_w, 0, pad_h, 0, pad_t), mode='constant', value=0)


    # Create color overlay [T, C, H, W]
    # Use blue-to-red gradient: blue (low change) -> red (high change)
    # Blue: [0, 0, 1], Red: [1, 0, 0]
    overlay = torch.zeros(T, C, H, W, device=videos.device, dtype=videos.dtype)

    if videos.dtype == torch.uint8:
        # For uint8, use integer color values
        # Interpolate between blue and red based on mask value
        # R = mask * 255, G = 0, B = (1 - mask) * 255
        mask_float = mask_video.unsqueeze(1)  # [T, 1, H, W]

        if C >= 3:
            overlay[:, 0] = (mask_float * 255).squeeze(1).to(torch.uint8)  # R: high change -> red
            overlay[:, 1] = torch.zeros(T, H, W, device=videos.device, dtype=torch.uint8)  # G: always 0
            overlay[:, 2] = ((1 - mask_float) * 255).squeeze(1).to(torch.uint8)  # B: low change -> blue
        else:
            # For non-RGB, only color first 3 channels
            for c in range(min(3, C)):
                if c == 0:
                    overlay[:, c] = (mask_float * 255).squeeze(1).to(torch.uint8)
                elif c == 1:
                    overlay[:, c] = torch.zeros(T, H, W, device=videos.device, dtype=torch.uint8)
                else:
                    overlay[:, c] = ((1 - mask_float) * 255).squeeze(1).to(torch.uint8)
    else:
        # For float, use float color values in [0, 1]
        mask_float = mask_video.unsqueeze(1)  # [T, 1, H, W]

        if C >= 3:
            overlay[:, 0] = mask_float.squeeze(1)  # R: high change -> red
            overlay[:, 1] = torch.zeros(T, H, W, device=videos.device)  # G: always 0
            overlay[:, 2] = 1 - mask_float.squeeze(1)  # B: low change -> blue
        else:
            # For non-RGB, only color first 3 channels
            for c in range(min(3, C)):
                if c == 0:
                    overlay[:, c] = mask_float.squeeze(1)
                elif c == 1:
                    overlay[:, c] = torch.zeros(T, H, W, device=videos.device)
                else:
                    overlay[:, c] = 1 - mask_float.squeeze(1)


    # Blend with original video
    original_dtype = videos.dtype
    if original_dtype == torch.uint8:
        videos_float = videos.float() / 255.0
        overlay_float = overlay.float() / 255.0
        videos_with_overlay = videos_float * (1 - alpha) + overlay_float * alpha
        videos_with_overlay = (videos_with_overlay * 255.0).clamp(0, 255).to(original_dtype)
    else:
        videos_with_overlay = videos * (1 - alpha) + overlay * alpha

    return videos_with_overlay


def apply_temporal_weights_overlay(
    videos: torch.Tensor,
    temporal_weights: torch.Tensor,
    latent_size: tuple,
    token_dims: tuple,
) -> torch.Tensor:
    """
    Apply temporal weights heatmap visualization on decoded videos.
    Each frame uses its corresponding weights heatmap.

    Args:
        videos: [T, C, H, W] tensor in pixel space (after VAE decode)
        temporal_weights: [T_latent, H_tokens, W_tokens] tensor with per-frame per-token weights
        latent_size: latent spatial size tuple [B, C, T_latent, H_latent, W_latent]
        token_dims: (H_tokens, W_tokens) token grid dimensions

    Returns:
        videos_with_overlay: [T, C, H, W] tensor with semi-transparent color overlay
    """
    import torch.nn.functional as F

    # Alpha for blending (higher value = more opaque mask)
    alpha = 0.8  # Strongly visible mask

    # Get video dimensions [T, C, H, W]
    T, C, H, W = videos.shape

    # Get latent dimensions
    _, _, T_latent, H_latent, W_latent = latent_size
    H_tokens, W_tokens = token_dims

    # temporal_weights is [T_latent, H_tokens, W_tokens]
    # Each frame has its own weights heatmap

    # Normalize temporal_weights to [0, 1] range for color mapping
    weights_min = temporal_weights.min()
    weights_max = temporal_weights.max()
    if weights_max - weights_min > 1e-8:
        normalized_weights = (temporal_weights - weights_min) / (weights_max - weights_min)
    else:
        normalized_weights = torch.zeros_like(temporal_weights)

    # Upsample mask from token grid to video resolution
    # Token -> Latent (unpatchify): 2x spatial upscale
    # Latent -> Pixel (VAE): 8x spatial + 4x temporal upscale
    unpatchify_scale = 2
    vae_spatial_scale = 8
    vae_temporal_scale = 4

    # Step 1: Unpatchify - upscale from token grid to latent resolution (2x spatial)
    weights_latent = F.interpolate(
        normalized_weights.unsqueeze(1),  # [T_latent, 1, H_tokens, W_tokens]
        scale_factor=unpatchify_scale,
        mode='nearest'
    ).squeeze(1)  # [T_latent, H_tokens*2, W_tokens*2] = [T_latent, H_latent, W_latent]

    # Step 2: VAE spatial upscale - from latent to pixel spatial resolution (8x spatial)
    weights_spatial_upsampled = F.interpolate(
        weights_latent.unsqueeze(1),  # [T_latent, 1, H_latent, W_latent]
        scale_factor=vae_spatial_scale,
        mode='nearest'
    ).squeeze(1)  # [T_latent, H_latent*8, W_latent*8]

    # Step 3: VAE temporal upscale - from T_latent to T frames (4x temporal)
    T_up = T_latent * vae_temporal_scale
    H_up = H_tokens * unpatchify_scale * vae_spatial_scale
    W_up = W_tokens * unpatchify_scale * vae_spatial_scale

    # Repeat temporal dimension
    weights_video = weights_spatial_upsampled.repeat_interleave(vae_temporal_scale, dim=0)  # [T_latent*4, H_up, W_up]

    # Handle possible boundary misalignment (crop or pad to exact video size)
    T_mask, H_mask_up, W_mask_up = weights_video.shape
    if T_mask > T or H_mask_up > H or W_mask_up > W:
        # Crop if upscaled size is larger
        weights_video = weights_video[:T, :H, :W]
    elif T_mask < T or H_mask_up < H or W_mask_up < W:
        # Pad if upscaled size is smaller
        pad_t = T - T_mask
        pad_h = H - H_mask_up
        pad_w = W - W_mask_up
        weights_video = F.pad(weights_video, (0, pad_w, 0, pad_h, 0, pad_t), mode='constant', value=0)

    # Create color overlay [T, C, H, W]
    # Use green-to-red gradient: green (low weight) -> red (high weight)
    # Green: [0, 1, 0], Red: [1, 0, 0]
    overlay = torch.zeros(T, C, H, W, device=videos.device, dtype=videos.dtype)

    if videos.dtype == torch.uint8:
        # For uint8, use integer color values
        # Interpolate between green and red based on weight value
        # R = weight * 255, G = (1 - weight) * 255, B = 0
        mask_float = weights_video.unsqueeze(1)  # [T, 1, H, W]

        if C >= 3:
            overlay[:, 0] = (mask_float * 255).squeeze(1).to(torch.uint8)  # R: high weight -> red
            overlay[:, 1] = ((1 - mask_float) * 255).squeeze(1).to(torch.uint8)  # G: low weight -> green
            overlay[:, 2] = torch.zeros(T, H, W, device=videos.device, dtype=torch.uint8)  # B: always 0
        else:
            # For non-RGB, only color first 3 channels
            for c in range(min(3, C)):
                if c == 0:
                    overlay[:, c] = (mask_float * 255).squeeze(1).to(torch.uint8)
                elif c == 1:
                    overlay[:, c] = ((1 - mask_float) * 255).squeeze(1).to(torch.uint8)
                else:
                    overlay[:, c] = torch.zeros(T, H, W, device=videos.device, dtype=torch.uint8)
    else:
        # For float, use float color values in [0, 1]
        mask_float = weights_video.unsqueeze(1)  # [T, 1, H, W]

        if C >= 3:
            overlay[:, 0] = mask_float.squeeze(1)  # R: high weight -> red
            overlay[:, 1] = 1 - mask_float.squeeze(1)  # G: low weight -> green
            overlay[:, 2] = torch.zeros(T, H, W, device=videos.device)  # B: always 0
        else:
            # For non-RGB, only color first 3 channels
            for c in range(min(3, C)):
                if c == 0:
                    overlay[:, c] = mask_float.squeeze(1)
                elif c == 1:
                    overlay[:, c] = 1 - mask_float.squeeze(1)
                else:
                    overlay[:, c] = torch.zeros(T, H, W, device=videos.device)

    # Blend with original video
    original_dtype = videos.dtype
    if original_dtype == torch.uint8:
        videos_float = videos.float() / 255.0
        overlay_float = overlay.float() / 255.0
        videos_with_overlay = videos_float * (1 - alpha) + overlay_float * alpha
        videos_with_overlay = (videos_with_overlay * 255.0).clamp(0, 255).to(original_dtype)
    else:
        videos_with_overlay = videos * (1 - alpha) + overlay * alpha

    return videos_with_overlay
