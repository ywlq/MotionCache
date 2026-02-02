#!/bin/bash

export DEVICES="0,1,2,3,4"

export XDG_CACHE_HOME="/tmp"

export PAD_HQ=1
export PAD_DURATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"


MAGI_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
export MAGI_ROOT="$MAGI_ROOT"

BENCHMARK="physicsiq"
L1_THRESH=0.0
NUM_STEPS=64
REUSE_STRATEGY="chunkwise"
NO_REUSE_MODE="first"
CONFIG_FILE="config/sample/5s_physicsiq.json"   # Remember to modify config


# ================== Parameters to modify =====================
COMPRESS_STRATEGY="token"   # token frame chunk
QUERY_GRANULARITY="frame"   # token frame chunk(cause out of memory)     Token granularity takes 50 tokens
SCORE_WEIGHTING_METHOD="upper_convex_polynomial" # no_weight exponential polynomial gaussian upper_convex_polynomial
TOTAL_CACHE_CHUNK_NUMS=6
POWER=3
PHYSICS_IQ_DATA_DIR="/path/to/physics-IQ-benchmark"
BASE_SAVE_PATH="/path/to/output/physicsiq_sample/noreuse_${TOTAL_CACHE_CHUNK_NUMS}kvbudget_${SCORE_WEIGHTING_METHOD}_power${POWER}/${REUSE_STRATEGY}_${L1_THRESH}_steps${NUM_STEPS}_${NO_REUSE_MODE}_query_${QUERY_GRANULARITY}_key_${COMPRESS_STRATEGY}"

SAMPLE_START=150
SAMPLE_END=200
# ======================================================

mkdir -p "$BASE_SAVE_PATH"

# log
LOG_DIR_PATH="$BASE_SAVE_PATH/log"
mkdir -p "$LOG_DIR_PATH"
LOG_FILE="$LOG_DIR_PATH/sampling_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# =====================================

echo "🚀 Starting multi-GPU benchmark sampling"
echo "🎮 GPUs: $DEVICES"
echo "⚙️  Config: $CONFIG_FILE"


# video save
VIDEO_SAVE_PATH="$BASE_SAVE_PATH/videos"
mkdir -p "$VIDEO_SAVE_PATH"
echo "📁 Output path: $VIDEO_SAVE_PATH"


python sample_video.py \
    --compress_kv_cache \
    --reuse_strategy "$REUSE_STRATEGY" \
    --rel_l1_thresh "$L1_THRESH" \
    --benchmark "$BENCHMARK" \
    --physicsiq_data_dir "$PHYSICS_IQ_DATA_DIR" \
    --save_path "$VIDEO_SAVE_PATH" \
    --config_file "$CONFIG_FILE" \
    --gpus "$DEVICES" \
    --no_reuse_first_n_steps 5 \
    --no_reuse_mode "$NO_REUSE_MODE" \
    --total_cache_chunk_nums "$TOTAL_CACHE_CHUNK_NUMS" \
    --compress_strategy "$COMPRESS_STRATEGY" \
    --query_granularity "$QUERY_GRANULARITY" \
    --mix_lambda 0.07 \
    --score_weighting_method "$SCORE_WEIGHTING_METHOD" \
    --power "$POWER" \
    --start "$SAMPLE_START" \
    --end "$SAMPLE_END" \

echo "---"

echo "🎉 All sampling tasks completed."