#!/usr/bin/env bash


export DEVICES="6,7"
export GPUS_PER_NODE=2

OUTDIR="/path/vbench_samples/enable-compile"
MODEL_ID="/path/SkyReels-V2-DF-1.3B-540P"
RESOLUTION="540P"
NUM_FRAMES=177
BASE_NUM_FRAMES=97
FPS=24
INFERENCE_STEPS=50
GUIDANCE_SCALE=6.0
SHIFT=8.0
SEED=1024
AR_STEP=5
CAUSAL_BLOCK_SIZE=5
OVERLAP_HISTORY=17
OFFLOAD=1
ADDNOISE_CONDITION=20

# ============================================
# Frame diff weight parameters
# ============================================
ENABLE_FRAME_DIFF_WEIGHT=1

WEIGHT_NORM_MODE="max_rescale"
WEIGHT_FLOOR="0.6"

# ============================================
# Temporal consistency parameters
# ============================================
ENABLE_TEMPORAL_CONSISTENCY=1
TEMPORAL_CONSISTENCY_THRESHOLD="0.5"
FIRST_FRAME_FULL_WEIGHT=0
GROUP0_FIRST_FRAME_MODE="second_frame"

# ============================================
# Token cache core parameters
# ============================================
ENABLE_TOKEN_CACHE=1
TOKEN_CACHE_THRESHOLD="0.1"
TOKEN_CACHE_WARMUP=4
TOKEN_PHASE1_UPDATE_COUNT=6
TOKEN_DISTANCE_MODE="global"

# Token weight parameters
TOKEN_WEIGHT_MIN="-100"
TOKEN_WEIGHT_MAX="100"
TOKEN_WEIGHT_GAMMA="1.0"
WEIGHT_SMOOTH_GRID_SIZE="1"

# VBench dimensions
DIMENSIONS=(
"human_action_longer" "object_class_longer" "subject_consistency_longer" "overall_consistency_longer" "multiple_objects_longer" "color_longer" "spatial_relationship_longer" "scene_longer" "temporal_style_longer" "temporal_flickering_longer" "appearance_style_longer"
)

# Experiment name prefix
if [[ "${ENABLE_TOKEN_CACHE}" == "1" ]]; then
  BASE_EXPNAME_PREFIX="token_thr${TOKEN_CACHE_THRESHOLD}_w${TOKEN_CACHE_WARMUP}_p${TOKEN_PHASE1_UPDATE_COUNT}"
  if [[ "${ENABLE_FRAME_DIFF_WEIGHT}" == "1" ]]; then
    BASE_EXPNAME_PREFIX="${BASE_EXPNAME_PREFIX}_framediff_${WEIGHT_NORM_MODE}"
    if [[ "${WEIGHT_NORM_MODE}" == "max_rescale" ]]; then
      BASE_EXPNAME_PREFIX="${BASE_EXPNAME_PREFIX}_floor${WEIGHT_FLOOR}"
    fi
  fi
  if [[ "${ENABLE_TEMPORAL_CONSISTENCY}" == "1" ]]; then
    BASE_EXPNAME_PREFIX="${BASE_EXPNAME_PREFIX}_temporal${TEMPORAL_CONSISTENCY_THRESHOLD}"
  fi
  if [[ "${GROUP0_FIRST_FRAME_MODE}" != "ones" ]]; then
    BASE_EXPNAME_PREFIX="${BASE_EXPNAME_PREFIX}_g0ff${GROUP0_FIRST_FRAME_MODE}"
  fi
  BASE_EXPNAME_PREFIX="${BASE_EXPNAME_PREFIX}_SEED_${SEED}"
else
  BASE_EXPNAME_PREFIX="SEED_${SEED}"
fi

OUTDIR="${OUTDIR}/${BASE_EXPNAME_PREFIX}"
echo "OUTDIR: ${OUTDIR}"

# Python executable and script path
PYBIN="${PYBIN:-python3}"
SCRIPT="/path/vbench_sample.py"

mkdir -p "${OUTDIR}"

# =====================================

echo "Starting DF Token Cache VBench sampling"
echo "GPUs: ${DEVICES} (GPUS_PER_NODE=${GPUS_PER_NODE})"
echo "Model: ${MODEL_ID} | Res: ${RESOLUTION} | Frames: ${NUM_FRAMES} | Steps: ${INFERENCE_STEPS}"
echo "Token Cache: on=${ENABLE_TOKEN_CACHE} threshold=${TOKEN_CACHE_THRESHOLD}"
echo "  warmup=${TOKEN_CACHE_WARMUP} phase1_update=${TOKEN_PHASE1_UPDATE_COUNT}"
echo "Frame Diff Weight: on=${ENABLE_FRAME_DIFF_WEIGHT}"
if [[ "${ENABLE_FRAME_DIFF_WEIGHT}" == "1" ]]; then
  echo "  norm_mode=${WEIGHT_NORM_MODE} floor=${WEIGHT_FLOOR}"
fi
echo "Temporal Consistency: on=${ENABLE_TEMPORAL_CONSISTENCY}"
if [[ "${ENABLE_TEMPORAL_CONSISTENCY}" == "1" ]]; then
  echo "  threshold=${TEMPORAL_CONSISTENCY_THRESHOLD} g0ff_mode=${GROUP0_FIRST_FRAME_MODE}"
fi
echo "Total dimensions: ${#DIMENSIONS[@]}"
echo "Dimensions: ${DIMENSIONS[*]}"
[[ "${OFFLOAD}" == "1" ]] && echo "Offload: on" || echo "Offload: off"

for DIMENSION in "${DIMENSIONS[@]}"; do
  EXPNAME="${BASE_EXPNAME_PREFIX}_${DIMENSION}"

  echo ""
  echo "Processing dimension: ${DIMENSION}"
  echo "Outdir: ${OUTDIR} | Expname: ${EXPNAME}"

  OUTDIR_DIMENSION="${OUTDIR}/${DIMENSION}"
  mkdir -p "${OUTDIR_DIMENSION}"

  CMD=(
    "${PYBIN}" "${SCRIPT}"
    --dimension "${DIMENSION}"
    --gpus "${DEVICES}"
    --outdir "${OUTDIR_DIMENSION}"
    --model_id "${MODEL_ID}"
    --resolution "${RESOLUTION}"
    --num_frames "${NUM_FRAMES}" --base_num_frames "${BASE_NUM_FRAMES}"
    --fps "${FPS}" --inference_steps "${INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --shift "${SHIFT}"
    --ar_step "${AR_STEP}" --causal_block_size "${CAUSAL_BLOCK_SIZE}"
    --overlap_history "${OVERLAP_HISTORY}"
    --addnoise_condition "${ADDNOISE_CONDITION}"
  )

  [[ -n "${SEED}" ]] && CMD+=( --seed "${SEED}" )
  [[ "${OFFLOAD}" == "1" ]] && CMD+=( --offload )

  # Token Cache parameters
  if [[ "${ENABLE_TOKEN_CACHE}" == "1" ]]; then
    CMD+=( --enable_token_cache )
    CMD+=( --token_cache_threshold "${TOKEN_CACHE_THRESHOLD}" )
    CMD+=( --token_cache_warmup "${TOKEN_CACHE_WARMUP}" )
    CMD+=( --token_phase1_update_count "${TOKEN_PHASE1_UPDATE_COUNT}" )
    CMD+=( --token_distance_mode "${TOKEN_DISTANCE_MODE}" )
    CMD+=( --token_weight_min "${TOKEN_WEIGHT_MIN}" )
    CMD+=( --token_weight_max "${TOKEN_WEIGHT_MAX}" )
    CMD+=( --token_weight_gamma "${TOKEN_WEIGHT_GAMMA}" )
    CMD+=( --weight_smooth_grid_size "${WEIGHT_SMOOTH_GRID_SIZE}" )
  fi

  # Frame diff weight parameters
  if [[ "${ENABLE_FRAME_DIFF_WEIGHT}" == "1" ]]; then
    CMD+=( --enable_frame_diff_weight )
    CMD+=( --weight_norm_mode "${WEIGHT_NORM_MODE}" )
    if [[ "${WEIGHT_NORM_MODE}" == "max_rescale" ]]; then
      CMD+=( --weight_floor "${WEIGHT_FLOOR}" )
    fi
  fi

  # Temporal consistency parameters
  if [[ "${ENABLE_TEMPORAL_CONSISTENCY}" == "1" ]]; then
    CMD+=( --enable_temporal_consistency )
    CMD+=( --temporal_consistency_threshold "${TEMPORAL_CONSISTENCY_THRESHOLD}" )
  fi
  if [[ "${FIRST_FRAME_FULL_WEIGHT}" == "1" ]]; then
    CMD+=( --first_frame_full_weight )
  fi
  if [[ "${GROUP0_FIRST_FRAME_MODE}" != "ones" ]]; then
    CMD+=( --group0_first_frame_mode "${GROUP0_FIRST_FRAME_MODE}" )
  fi

  echo "Running:"
  printf ' %q' "${CMD[@]}"; echo

  if "${CMD[@]}"; then
    echo "Completed: ${DIMENSION}"
  else
    echo "Failed: ${DIMENSION}"
  fi

  echo "---"
done

echo "All sampling tasks completed."