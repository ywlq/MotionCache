#!/bin/bash

export DEVICES="2,3,4,5"
export TMPDIR="/tmp"

export PAD_HQ=1
export PAD_DURATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"


MAGI_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
export MAGI_ROOT="$MAGI_ROOT"


BENCHMARK="vbench"
NUM_STEPS=64
REUSE_STRATEGY="all"
NO_REUSE_MODE="none"
CONFIG_FILE="config/sample/vbench.json"


# ================== Parameters to modify =====================
L1_THRESH=0.01
PROMPT_FILE_PATH="/path/to/VBench/prompts/prompts_per_dimension"
BASE_SAVE_PATH="/path/to/output/teacache_0.01_correct"
# ======================================================


DIMENSIONS=("overall_consistency" "subject_consistency" "scene")
# DIMENSIONS=("subject_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "appearance_style")


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
echo "🔢 Total dimensions to process: ${#DIMENSIONS[@]}"


for DIMENSION in "${DIMENSIONS[@]}"; do
    VIDEO_SAVE_PATH="$BASE_SAVE_PATH/videos/${DIMENSION}"
    mkdir -p "$VIDEO_SAVE_PATH"

    echo "📌 Processing dimension: $DIMENSION"
    echo "📁 Output path: $VIDEO_SAVE_PATH"

    python sample_video.py \
        --reuse_strategy "$REUSE_STRATEGY" \
        --rel_l1_thresh "$L1_THRESH" \
        --benchmark "$BENCHMARK" \
        --dimension "$DIMENSION" \
        --vbench_prompt_dir "$PROMPT_FILE_PATH" \
        --save_path "$VIDEO_SAVE_PATH" \
        --config_file "$CONFIG_FILE" \
        --gpus "$DEVICES" \
        --no_reuse_first_n_steps 5 \
        --no_reuse_mode "$NO_REUSE_MODE" \

    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $DIMENSION"
    else
        echo "❌ Failed: $DIMENSION"
        echo "🛑 Script paused due to error. Fix the issue and rerun."
        exit 1
    fi

    echo "---"
done

echo "🎉 All sampling tasks completed."