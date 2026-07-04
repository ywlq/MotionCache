# MOTION CACHING FOR AUTOREGRESSIVE VIDEO GENERATION

This repository provides the official implementation of **MotionCache** on **MAGI-1** model, a caching-based acceleration method for autoregressive video generation models.


## 🚀 Installation

Please follow the installation instructions provided in the [MAGI-1](https://github.com/SandAI-org/MAGI-1), as this implementation is built on top of MAGI-1.

---


## ⚙️ Key Parameters

| Parameter | Description |
|----------|-------------|
| `rel_l1_thresh` | Relative L1 distance threshold for cache reuse decision |
| `no_reuse_first_n_steps` | Number of denoising steps where reuse is disabled |
| `no_reuse_mode` | Position of mandatory non-reuse window: `first`, `mid`, or `none` |
| `total_cache_chunk_nums` (`B_total`) | Total number of cache chunks maintained |
| `budget_cache_chunk_nums` (`B_budget`) | Budget number of chunks |
| `compress_strategy` | Granularity for selecting important KV caches: `token`, `frame`, or `chunk` |
| `query_granularity` | Granularity for importance scoring: `token`, `frame`, or `chunk` |
| `mix_lambda` | Weight balancing importance and redundancy (default: `0.07`) |
| `mode` | Generation mode: `t2v` (text-to-video), `i2v` (image-to-video), or `v2v` (video-to-video) |
| `prompt` | Input prompt for conditional generation |
| `output_path` | Path to save generated videos |
| `config_file` | Path to MAGI-1 model configuration |

---
