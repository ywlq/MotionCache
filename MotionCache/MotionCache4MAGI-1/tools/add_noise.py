import cv2
import numpy as np
import os

def add_gaussian_noise(image, mean=0, sigma=30):
    """
    Add Gaussian noise to an image.
    :param image: Input image (numpy array)
    :param mean: Noise mean
    :param sigma: Noise standard deviation (controls noise intensity)
    :return: Noisy image (uint8)
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def generate_noisy_images(input_image_path, output_dir, min_sigma=10, max_sigma=100, num_samples=10):
    """
    Generate specified number of images with different noise intensities and save them.
    :param input_image_path: Original image path
    :param output_dir: Output folder path
    :param min_sigma: Minimum noise standard deviation
    :param max_sigma: Maximum noise standard deviation
    :param num_samples: Number of images to generate
    """
    # Read image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Linear sampling of sigma values
    sigmas = np.linspace(min_sigma, max_sigma, num_samples)

    for i, sigma in enumerate(sigmas):
        sigma = float(sigma)
        noisy_img = add_gaussian_noise(img, sigma=sigma)

        # Generate filename, keep 2 decimal places
        output_filename = f"noisy_sigma_{sigma:.2f}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, noisy_img)
        print(f"Saved: {output_path} (sigma={sigma:.2f})")

    print(f"\nGenerated {num_samples} noisy images, saved to: {output_dir}")

# ==================== Usage Example ====================
if __name__ == "__main__":
    input_path = 'path/to/input/image.jpg'
    output_folder = 'path/to/output/folder'
    os.makedirs(output_folder, exist_ok=True)

    # Parameter settings
    MIN_SIGMA = 1000
    MAX_SIGMA = 2000
    NUM_SAMPLES = 7

    generate_noisy_images(
        input_image_path=input_path,
        output_dir=output_folder,
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA,
        num_samples=NUM_SAMPLES
    )