import cv2
import numpy as np
import os

# ==========================================
# 1. configuration for different degradation types and levels
# ==========================================
def apply_blur(image, intensity_level):
    if intensity_level == 1: kernel = (3, 3)
    elif intensity_level == 2: kernel = (7, 7)
    elif intensity_level == 3: kernel = (11, 11)
    else: return image
    return cv2.GaussianBlur(image, kernel, 0)

def apply_noise(image, intensity_level):
    row, col, ch = image.shape
    if intensity_level == 1: var = 0.001 * 255
    elif intensity_level == 2: var = 0.005 * 255
    elif intensity_level == 3: var = 0.01 * 255
    else: return image
    sigma = var ** 0.5
    gauss = np.random.normal(0, sigma, (row, col, ch))
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_illumination(image, intensity_level):
    alpha = 1.0; beta = 0
    if intensity_level == 1: beta = -70       # Darker
    elif intensity_level == 2: beta = 100     # Brighter
    elif intensity_level == 3: alpha = 0.3    # Low Contrast
    else: return image
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_occlusion(image, intensity_level):
    row, col, ch = image.shape
    occ_img = image.copy()
    if intensity_level == 1:
        cv2.rectangle(occ_img, (int(col*0.7), int(row*0.7)), (int(col*0.9), int(row*0.9)), (0, 0, 0), -1)
    elif intensity_level == 2:
        cv2.rectangle(occ_img, (int(col*0.3), int(row*0.3)), (int(col*0.7), int(row*0.7)), (0, 0, 0), -1)
    elif intensity_level == 3:
        # Simulating extreme damage using high-intensity noise.
        occ_img = apply_noise(occ_img, 3)
    else: return image
    return occ_img

# ==========================================
# 2. Automated batch processing pipeline
# ==========================================
def main():
    # --- Configuration paths ---
    # Assuming your original clean images are all in the dataset/clean folder
    input_dir = os.path.join('dataset', 'clean')
    output_base_dir = os.path.join('dataset', 'degraded')
    
    # Check if the input folder exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input folder '{input_dir}' not found. Please ensure your original images are placed here!")
        return

    # Define the degradation types and intensity levels to generate
    degradations = ['blur', 'noise', 'illum', 'corrupt']
    levels = [1, 2, 3]

    # --- Automatically create output directory tree ---
    print("📁 Checking and creating directory structure...")
    for deg in degradations:
        for lvl in levels:
            dir_path = os.path.join(output_base_dir, f"{deg}_{lvl}")
            os.makedirs(dir_path, exist_ok=True)
            
    # --- Retrieve all pending images. ---
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"⚠️ Warning: No images found in '{input_dir}'!")
        return

    print(f"🚀 Starting batch processing of {total_images} images...\n")

    # --- Start iterating and generating ---
    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  [skipped] unable to load image: {filename}")
            continue

        # 1. Blur degradation
        for lvl in levels:
            res = apply_blur(img, lvl)
            cv2.imwrite(os.path.join(output_base_dir, f"blur_{lvl}", filename), res)
            
        # 2. Noise degradation
        for lvl in levels:
            res = apply_noise(img, lvl)
            cv2.imwrite(os.path.join(output_base_dir, f"noise_{lvl}", filename), res)

        # 3. Illumination degradation
        for lvl in levels:
            res = apply_illumination(img, lvl)
            cv2.imwrite(os.path.join(output_base_dir, f"illum_{lvl}", filename), res)

        # 4. Corruption degradation
        for lvl in levels:
            res = apply_occlusion(img, lvl)
            cv2.imwrite(os.path.join(output_base_dir, f"corrupt_{lvl}", filename), res)

        # Print progress update
        if i % 10 == 0 or i == total_images:
            print(f"  ✅ Progress: {i}/{total_images} images processed")

    print(f"\n🎉 All done! All degraded images have been saved in '{output_base_dir}' directory.")

if __name__ == "__main__":
    main()