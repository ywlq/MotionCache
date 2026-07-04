#!/bin/bash

export TMPDIR="/tmp"

# ==================== Environment variable setup ====================
export PAD_HQ=1
export PAD_DURATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"


# ==================== Centralized configuration ====================

# ---------- Hardware and environment config ----------
DEVICES="0,1,2,3"
CONFIG_FILE="config/sample/vbench.json"
MOTIONCACHE_CONFIG="addconfig/config.yaml"

# ---------- Basic inference config ----------
BENCHMARK="vbench"
REUSE_STRATEGY="chunkwise"

# ---------- Path configuration ----------
PROMPT_FILE_PATH="/path/to/VBench/prompts/prompts_per_dimension"
BASE_SAVE_PATH="/path/to/output/motioncache_vbench"

# ---------- Test dimension configuration ----------
DIMENSIONS=(
"subject_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "appearance_style"
)

# ==================== End of configuration ====================


# ==================== Initialization ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAGI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$MAGI_ROOT"

export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
export MAGI_ROOT="$MAGI_ROOT"

mkdir -p "$BASE_SAVE_PATH"

LOG_DIR_PATH="$BASE_SAVE_PATH/log"
mkdir -p "$LOG_DIR_PATH"
LOG_FILE="$LOG_DIR_PATH/motioncache_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# =====================================

echo "Starting MotionCache VBench sampling"
echo "GPUs: $DEVICES"
echo "Config: $CONFIG_FILE"
echo "MotionCache config: $MOTIONCACHE_CONFIG"
echo "Prompt dir: $PROMPT_FILE_PATH"
echo "Total dimensions: ${#DIMENSIONS[@]}"
echo "Dimensions: ${DIMENSIONS[*]}"


# ==================== Loop through dimensions ====================
for DIMENSION in "${DIMENSIONS[@]}"; do
    VIDEO_SAVE_PATH="$BASE_SAVE_PATH/videos/${DIMENSION}"
    mkdir -p "$VIDEO_SAVE_PATH"

    echo "Processing dimension: $DIMENSION"
    echo "Output path: $VIDEO_SAVE_PATH"

    ARGS=(
        --benchmark "$BENCHMARK"
        --dimension "$DIMENSION"
        --vbench_prompt_dir "$PROMPT_FILE_PATH"
        --save_path "$VIDEO_SAVE_PATH"
        --config_file "$CONFIG_FILE"
        --additional_config "$MOTIONCACHE_CONFIG"
        --gpus "$DEVICES"
        --reuse_strategy "$REUSE_STRATEGY"
    )

    python sample_video.py "${ARGS[@]}"

    if [ $? -eq 0 ]; then
        echo "Completed: $DIMENSION"
    else
        echo "Failed: $DIMENSION"
        echo "Script paused due to error. Fix the issue and rerun."
        exit 1
    fi

    echo "---"
done

echo "All MotionCache VBench sampling tasks completed."
