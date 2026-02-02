#!/bin/bash

export TMPDIR="/tmp"

# ==================== Environment variable setup ====================
export PAD_HQ=1
export PAD_DURATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export TORCH_CUDA_ARCH_LIST="8.9;9.0"


# ==================== 📋 Centralized configuration ====================

# ---------- Hardware and environment config ----------
DEVICES="0,1,2,3"                      # GPU device list
CONFIG_FILE="config/sample/vbench.json"  # Model config file

# ---------- Basic inference config ----------
BENCHMARK="vbench"                     # Benchmark name
NUM_STEPS=64                           # Inference steps
REUSE_STRATEGY="chunkwise"             # Reuse strategy: chunkwise/all/original
NO_REUSE_MODE="first"                  # No reuse mode: first/mid/none
NO_REUSE_FIRST_N_STEPS=5               # First no-reuse steps

# ---------- TeaCache basic parameters ----------
L1_THRESH=0.0065                         # Relative L1 distance threshold
WARMUP_STEPS=5                         # Warmup steps (no cache)
CHUNK_WISE_ONLY_STEPS=20               # Chunk-wise only reuse steps
DISCARD_NEARLY_CLEAN_CHUNK=true        # Whether to discard nearly clean chunks

# ---------- Token-wise Reuse parameters ----------
TOKEN_WISE_REUSE=true               # Enable token-level reuse
TOKEN_REL_L1_THRESH=0.0065               # Token-level L1 threshold
TOKENWISE_L1_MODE="chunk"              # Token L1 computation mode: chunk/token

# ---------- Token reuse ratio control ----------
MAX_TOKEN_REUSE_RATIO=1.0              # Max token reuse ratio (0~1)
# INITIAL_TOKEN_REUSE_RATIO=0.5        # Dynamic mode: initial ratio (optional)
# FINAL_TOKEN_REUSE_RATIO=0.8          # Dynamic mode: final ratio (optional)

# ---------- Continuous reuse tracking (adaptive refresh) ----------
ENABLE_CONTINUOUS_REUSE_TRACKING=false # Enable continuous reuse tracking
CONTINUOUS_REUSE_MAX_COUNT=5           # Continuous reuse count for forced forward
CONTINUOUS_REUSE_DECAY_MODE="null"     # Decay mode: exponential/linear/null
CONTINUOUS_REUSE_DECAY_FACTOR=0.1      # Decay factor

# ---------- Temporal weight parameters ----------
TEMPORAL_WEIGHT_FLOOR=0.5                # Temporal weight floor
TEMPORAL_WEIGHT_POWER=1.0              # Temporal weight power (null=linear)
ENABLE_TEMPORAL_VOTING=false           # Enable temporal voting

# ---------- KV Cache compression (mutually exclusive with token_wise_reuse) ----------
COMPRESS_KV_CACHE=false                # Enable KV cache compression
TOTAL_CACHE_CHUNK_NUMS=10              # Total KV cache chunk count
COMPRESS_STRATEGY="token"              # Compression strategy: token/frame/chunk
QUERY_GRANULARITY="frame"              # Query granularity: chunk/frame/token
MIX_LAMBDA=0.07                        # Token compression mix factor
SCORE_WEIGHTING_METHOD="no_weight"     # Score weighting method
POWER=3                                # Polynomial weighting power

# ---------- Debug and visualization ----------
LOG=true                               # Enable logging
PRINT_TOKEN_STATS=true                 # Print token statistics
VISUALIZE_REUSE_MASK=false             # Visualize reuse mask
VISUALIZE_TEMPORAL_DIFF=false          # Visualize temporal difference
TEMPORAL_DIFF_STEP="0 1 2 3 4 5 6 7"   # Temporal difference computation steps
TEMPORAL_DIFF_MODE="clean"             # Temporal difference mode: clean/noise
VISUALIZE_TEMPORAL_WEIGHTS=false       # Visualize temporal weights
TEMPORAL_WEIGHTS_STEP="0 1 2 3 4 5 6 7" # Temporal weight computation steps

# ---------- Path configuration ----------
PROMPT_FILE_PATH="/path/to/VBench/prompts/prompts_per_dimension"
BASE_SAVE_PATH="/path/to/output/tokenwise_0.0065_correct"

# ---------- Test dimension configuration ----------
DIMENSIONS=("scene")
# DIMENSIONS=("subject_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "appearance_style")

# ==================== 📋 End of configuration ====================


# ==================== Initialization ====================
MAGI_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$MAGI_ROOT:$PYTHONPATH"
export MAGI_ROOT="$MAGI_ROOT"

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
echo "📌 Total dimensions: ${#DIMENSIONS[@]}"
echo "📋 Dimensions: ${DIMENSIONS[*]}"
echo "📦 Reuse Strategy: $REUSE_STRATEGY"
echo "🎯 L1 Threshold: $L1_THRESH"
echo "🔥 Warmup Steps: $WARMUP_STEPS"
echo "🧩 Chunk-wise Only Steps: $CHUNK_WISE_ONLY_STEPS"


# ==================== Loop through dimensions ====================
for DIMENSION in "${DIMENSIONS[@]}"; do
    VIDEO_SAVE_PATH="$BASE_SAVE_PATH/videos/${DIMENSION}"
    mkdir -p "$VIDEO_SAVE_PATH"

    echo "🔍 Processing dimension: $DIMENSION"
    echo "📁 Output path: $VIDEO_SAVE_PATH"

    # ========== Build argument array ==========
    ARGS=(
        --benchmark "$BENCHMARK"
        --dimension "$DIMENSION"
        --vbench_prompt_dir "$PROMPT_FILE_PATH"
        --save_path "$VIDEO_SAVE_PATH"
        --config_file "$CONFIG_FILE"
        --gpus "$DEVICES"
        --reuse_strategy "$REUSE_STRATEGY"
        --rel_l1_thresh "$L1_THRESH"
        --no_reuse_first_n_steps "$NO_REUSE_FIRST_N_STEPS"
        --no_reuse_mode "$NO_REUSE_MODE"
        --warmup_steps "$WARMUP_STEPS"
        --chunk_wise_only_steps "$CHUNK_WISE_ONLY_STEPS"
        --token_rel_l1_thresh "$TOKEN_REL_L1_THRESH"
        --tokenwise_l1_mode "$TOKENWISE_L1_MODE"
        --temporal_weight_floor "$TEMPORAL_WEIGHT_FLOOR"
        --temporal_weight_power "$TEMPORAL_WEIGHT_POWER"
        --continuous_reuse_max_count "$CONTINUOUS_REUSE_MAX_COUNT"
        # KV cache compression parameters (always needed as score_weighting_method is required)
        --score_weighting_method "$SCORE_WEIGHTING_METHOD"
        --compress_strategy "$COMPRESS_STRATEGY"
        --query_granularity "$QUERY_GRANULARITY"
        --mix_lambda "$MIX_LAMBDA"
        --power "$POWER"
    )

    # Conditionally add optional parameters
    if [ "$DISCARD_NEARLY_CLEAN_CHUNK" = true ]; then
        ARGS+=(--discard_nearly_clean_chunk)
    fi

    if [ "$TOKEN_WISE_REUSE" = true ]; then
        ARGS+=(--token_wise_reuse)
    fi

    if [ "$LOG" = true ]; then
        ARGS+=(--log)
    fi

    if [ "$PRINT_TOKEN_STATS" = true ]; then
        ARGS+=(--print_token_stats)
    fi

    if [ "$ENABLE_TEMPORAL_VOTING" = true ]; then
        ARGS+=(--enable_temporal_voting)
    fi

    if [ "$VISUALIZE_REUSE_MASK" = true ]; then
        ARGS+=(--visualize_reuse_mask)
    fi

    if [ "$VISUALIZE_TEMPORAL_DIFF" = true ]; then
        ARGS+=(--visualize_temporal_diff)
        ARGS+=(--temporal_diff_step ${TEMPORAL_DIFF_STEP})
        ARGS+=(--temporal_diff_mode "$TEMPORAL_DIFF_MODE")
    fi

    if [ "$VISUALIZE_TEMPORAL_WEIGHTS" = true ]; then
        ARGS+=(--visualize_temporal_weights)
        ARGS+=(--temporal_weights_step ${TEMPORAL_WEIGHTS_STEP})
    fi

    # KV cache compression toggle (other parameters already in base array)
    if [ "$COMPRESS_KV_CACHE" = true ]; then
        ARGS+=(--compress_kv_cache)
        ARGS+=(--total_cache_chunk_nums "$TOTAL_CACHE_CHUNK_NUMS")
    fi

    # ========== Execute command ==========
    python sample_video.py "${ARGS[@]}"

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $DIMENSION"
    else
        echo "❌ Failed: $DIMENSION"
        echo "🛑 Script paused due to error. Fix the issue and rerun."
        exit 1
    fi

    echo "---"
done

echo "🎉 All sampling tasks completed."
