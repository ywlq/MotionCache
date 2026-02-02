#!/bin/bash

export DEVICES="0,1,2,3,4,5,6,7"


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
PHYSICS_IQ_DATA_DIR="/path/to/physics-IQ-benchmark"

L1_THRESH=0.0
NUM_STEPS=64
REUSE_STRATEGY="chunkwise"
NO_REUSE_MODE="first"
CONFIG_FILE="config/sample/5s_physicsiq.json"   # Remember to modify config

BASE_SAVE_PATH="/path/to/output/physicsiq_sample/chunkforward_fullkv/${REUSE_STRATEGY}_${L1_THRESH}_steps${NUM_STEPS}_${NO_REUSE_MODE}"
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
    --reuse_strategy "$REUSE_STRATEGY" \
    --rel_l1_thresh "$L1_THRESH" \
    --benchmark "$BENCHMARK" \
    --physicsiq_data_dir "$PHYSICS_IQ_DATA_DIR" \
    --save_path "$VIDEO_SAVE_PATH" \
    --config_file "$CONFIG_FILE" \
    --gpus "$DEVICES" \
    --no_reuse_first_n_steps 5 \
    --no_reuse_mode "$NO_REUSE_MODE" \

if [ $? -eq 0 ]; then
    echo "✅ Completed: $DIMENSION"
else
    echo "❌ Failed: $DIMENSION"
    echo "🛑 Script paused due to error. Fix the issue and rerun."
    exit 1
fi

echo "---"

echo "🎉 All sampling tasks completed."