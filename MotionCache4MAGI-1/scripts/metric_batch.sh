# Set parameters
ORIGINAL_FOLDER="/path/to/videos/overall_consistency"
GENERATED_FOLDER="/path/to/videos/overall_consistency"
NUM_VIDEOS=10
GPU_ID=6
SEED=42

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Batch Video Quality Evaluation"
echo "=========================================="
echo "Original video folder: $ORIGINAL_FOLDER"
echo "Generated video folder: $GENERATED_FOLDER"
echo "Number of videos to evaluate: $NUM_VIDEOS"
echo "Using GPU ID: $GPU_ID"
echo "Random seed: $SEED"
echo "=========================================="
echo ""

# Run evaluation script
python tools/video_metrics_batch.py \
  --original_folder "$ORIGINAL_FOLDER" \
  --generated_folder "$GENERATED_FOLDER" \
  --num_videos $NUM_VIDEOS \
  --device cuda \
  --seed $SEED
