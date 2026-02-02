# MOTION CACHING FOR AUTOREGRESSIVE VIDEO GENERATION

This repository provides the official implementation of **MotionCache** on **SkyReels-V2** model, a caching-based acceleration method for autoregressive video generation models.


## 🚀 Installation

Please follow the installation instructions provided in the [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2), as this implementation is built on top of SkyReels-V2.

---

## ▶️ Usage

### Video Generation

Run accelerated generation using MotionCache:

```bash
bash run_vbench.sh
```


## ⚙️ Key Parameters

### Basic Generation Parameters

| Parameter | Description | Default |
|----------|-------------|---------|
| `--model_id` | Path to SkyReels-V2 model | Required |
| `--resolution` | Video resolution: `540P` or `720P` | `540P` |
| `--num_frames` | Total number of frames to generate | `177` |
| `--base_num_frames` | Base number of frames for autoregressive generation | `97` |
| `--overlap_history` | Number of overlapping frames between segments | `17` |
| `--ar_step` | Autoregressive step size | `5` |
| `--causal_block_size` | Block size for causal attention | `5` |
| `--inference_steps` | Number of denoising steps | `50` |
| `--guidance_scale` | Classifier-free guidance scale | `6.0` |
| `--shift` | Shift parameter for timestep scheduling | `8.0` |
| `--fps` | Frames per second | `24` |
| `--seed` | Random seed for reproducible generation | `1024` |
| `--addnoise_condition` | Noise condition for long video consistency | `20` |
| `--offload` | Enable model offloading to save GPU memory | `False` |

### Token Cache Parameters (MotionCache Core)

| Parameter | Description | Default |
|----------|-------------|---------|
| `--enable_token_cache` | Enable token-wise caching mechanism | `False` |
| `--token_cache_threshold` | Threshold for cache reuse decision (higher = more aggressive caching) | `0.1` |
| `--token_cache_warmup` | Number of warmup steps before enabling cache | `4` |
| `--token_phase1_update_count` | Number of updates in phase 1 before switching to phase 2 | `6` |

### VBench Evaluation Parameters

| Parameter | Description |
|----------|-------------|
| `--dimension` | VBench dimension to evaluate (e.g., `human_action_longer`, `object_class_longer`) |
| `--gpus` | GPU devices to use (e.g., `"0,1"`) |
| `--outdir` | Output directory for generated videos |

---
