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

import os

import folder_paths
import node_helpers
import torch
from comfy.comfy_types import IO
from PIL import Image

from inference.common import EngineConfig, MagiConfig, ModelConfig, RuntimeConfig, set_random_seed
from inference.infra.distributed import dist_init
from inference.model.dit import get_dit
from inference.pipeline.prompt_process import get_txt_embeddings
from inference.pipeline.video_generate import generate_per_chunk
from inference.pipeline.video_process import post_chunk_process, process_image, process_prefix_video, save_video_to_disk


class MagiPromptLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load"
    CATEGORY = "Magi"

    def load(self, prompt):
        return (prompt,)


class MagiTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "The text to be encoded."}),
                "t5_pretrained_path": ("STRING", {"default": "path/to/t5/model"}),
                "t5_device": (
                    "COMBO",
                    {
                        "options": ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
                        "default": "cpu",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("text_embeddings",)
    FUNCTION = "encode"
    CATEGORY = "Magi"

    def encode(self, prompt: str, t5_pretrained_path: str, t5_device: str):
        model_config = ModelConfig(model_name="videodit_ardf")
        config = MagiConfig(model_config=model_config, runtime_config=RuntimeConfig(), engine_config=EngineConfig())
        config.runtime_config.t5_pretrained = t5_pretrained_path
        config.runtime_config.t5_device = t5_device
        config.model_config.caption_max_length = 800

        caption_embs, emb_masks = get_txt_embeddings(prompt, config)
        return ([caption_embs, emb_masks],)


class MagiImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required": {"image_path": (sorted(files), {"image_upload": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_path",)
    FUNCTION = "load"
    CATEGORY = "Magi"

    def load(self, image_path):
        image_path = folder_paths.get_annotated_filepath(image_path)
        node_helpers.pillow(Image.open, image_path)
        return (image_path,)


class MagiVideoLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {"required": {"video_path": (sorted(files), {"video_upload": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "load"
    CATEGORY = "Magi"

    def load(self, video_path):
        video_path = folder_paths.get_annotated_filepath(video_path)
        return (video_path,)


class MagiProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task_mode": (
                    "COMBO",
                    {"options": ["text to video", "image to video", "video continuation"], "default": "image to video"},
                ),
                "config_path": ("STRING", {"default": "path/to/config.json"}),
                "image_path": ("STRING", {"default": "path/to/image.jpg"}),
                "text_embeddings": ("CONDITIONING",),
                "magi_seed": ("INT", {"default": 1234, "min": 0, "max": 100000, "step": 1}),
                "video_size_h": ("INT", {"default": 720, "min": 16, "max": 14400, "step": 16}),
                "video_size_w": ("INT", {"default": 720, "min": 16, "max": 14400, "step": 16}),
                "num_frames": ("INT", {"default": 96, "min": 24, "max": 24000, "step": 24}),
                "num_steps": ("INT", {"default": 64, "min": 4, "max": 240, "step": 4}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("video", "fps")
    FUNCTION = "process"
    CATEGORY = "Magi"

    def set_environ_variables(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6009"
        os.environ["GPUS_PER_NODE"] = "1"
        os.environ["NNODES"] = "1"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        os.environ["PAD_HQ"] = "1"
        os.environ["PAD_DURATION"] = "1"

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["OFFLOAD_T5_CACHE"] = "true"
        os.environ["OFFLOAD_VAE_CACHE"] = "true"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9;9.0"

    def process(
        self,
        task_mode: str,
        config_path: str,
        image_path: str,
        text_embeddings: list,
        magi_seed: int,
        video_size_h: int,
        video_size_w: int,
        num_frames: int,
        num_steps: int,
        fps: int,
    ):
        self.set_environ_variables()
        config = MagiConfig.from_json(config_path)
        config.runtime_config.seed = magi_seed
        config.runtime_config.video_size_h = video_size_h
        config.runtime_config.video_size_w = video_size_w
        config.runtime_config.num_frames = num_frames
        config.runtime_config.num_steps = num_steps
        config.runtime_config.fps = fps

        # setup distributed environment
        set_random_seed(config.runtime_config.seed)
        dist_init(config)

        if task_mode == "text to video":
            prefix_video = None
        elif task_mode == "image to video":
            prefix_video = process_image(image_path, config)
        elif task_mode == "video continuation":
            prefix_video = process_prefix_video(image_path, config)
        else:
            raise ValueError(f"Unknown task mode: {task_mode}")

        dit = get_dit(config)

        videos = torch.cat(
            [
                post_chunk_process(chunk, config)
                for chunk in generate_per_chunk(
                    model=dit, prefix_video=prefix_video, caption_embs=text_embeddings[0], emb_masks=text_embeddings[1]
                )
            ],
            dim=0,
        )
        return (videos, fps)


class MagiSaveVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("IMAGE",),
                "output_path": ("STRING", {"default": "path/to/output/video.mp4"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Magi"

    def save(self, video: torch.Tensor, output_path: str, fps: int):
        output = save_video_to_disk(video, output_path, fps)
        return {"output": output}


NODE_CLASS_MAPPINGS = {
    "MagiImageLoader": MagiImageLoader,
    "MagiVideoLoader": MagiVideoLoader,
    "MagiPromptLoader": MagiPromptLoader,
    "MagiTextEncoder": MagiTextEncoder,
    "MagiProcess": MagiProcess,
    "MagiSaveVideo": MagiSaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagiImageLoader": "Load Image",
    "MagiVideoLoader": "Load Video",
    "MagiPromptLoader": "Load Prompt",
    "MagiTextEncoder": "T5 Text Encoder",
    "MagiProcess": "Process with MAGI",
    "MagiSaveVideo": "Save Video",
}
