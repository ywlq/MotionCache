import cv2
import os

def extract_frames(video_path, output_dir, total_frames=240):
    """
    Extract specified number of frames from video and save as images.
    :param video_path: Input video path
    :param output_dir: Output image directory
    :param total_frames: Number of frames to extract (default 240)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file")
        return

    # Get video information (optional, for verification)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"Video information:")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {duration:.2f} seconds")

    frame_id = 0
    saved_count = 0

    while saved_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("Video ended before reaching target frame count.")
            break

        # Save frame as image
        img_name = os.path.join(output_dir, f"frame_{saved_count + 1:04d}.jpg")
        cv2.imwrite(img_name, frame)
        saved_count += 1

        if saved_count % 24 == 0:
            print(f"Saved {saved_count} frames")

    cap.release()
    print(f"Done! Saved {saved_count} images to '{output_dir}'")


# Usage example
if __name__ == "__main__":
    VIDEO_PATH = "/path/to/video.mp4"
    OUTPUT_DIR = "/path/to/output/folder"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extract_frames(VIDEO_PATH, OUTPUT_DIR, total_frames=240)