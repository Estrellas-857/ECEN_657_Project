import cv2
import numpy as np
import pytesseract
import os
import re
import time

# ⚠️ Ensure your Tesseract path is correct
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 🎯 Agile Test Configuration
# ==========================================
MAX_TEST_IMAGES = 1000
CROP_RATIO = 0.08


# Wiener parameters: Start with a conservative initial value
# The larger K, the more conservative the deconvolution; the smaller K, the more aggressive the sharpening, but also more prone to amplifying artifacts
WIENER_K = 0.03

# The kernel size corresponding to your blur generation
BLUR_KERNEL_MAP = {
    'blur_1': 25,
    'blur_2': 45,
    'blur_3': 65,
}

def get_ground_truth(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r'[^A-Z0-9]', '', name.upper())

def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def calc_character_accuracy(gt, pred):
    if len(gt) == 0 and len(pred) == 0:
        return 1.0
    if len(gt) == 0 or len(pred) == 0:
        return 0.0

    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                ) + 1

    max_len = max(m, n)
    return max(0.0, (max_len - dp[m][n]) / max_len)

def print_progress(iteration, total, prefix='', length=30):
    if total == 0:
        print(f"{prefix} |{'-' * length}| 0.0% (0/0)")
        return

    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total})', end='\r')
    if iteration == total:
        print()

def gaussian_kernel_2d(ksize, sigma=None):
    """
    generate a 2D Gaussian PSF
    If sigma=None, mimic OpenCV's GaussianBlur(..., sigmaX=0) approach
    usually, for a given kernel size ksize, OpenCV's sigma is approximately 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    """
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

def psf2otf(psf, shape):
    """
    utility function to convert PSF to OTF (Optical Transfer Function) for Wiener deconvolution
    """
    otf = np.zeros(shape, dtype=np.float32)
    kh, kw = psf.shape
    otf[:kh, :kw] = psf

    # Move the kernel center to the top-left corner for FFT compatibility
    otf = np.roll(otf, -kh // 2, axis=0)
    otf = np.roll(otf, -kw // 2, axis=1)

    return np.fft.fft2(otf)

def wiener_deconvolution(gray_img, psf, K=0.01):
    """
    utility function to perform Wiener deconvolution on a grayscale image
    input: uint8 gray
    output: uint8 gray
    """
    img = gray_img.astype(np.float32) / 255.0
    H = psf2otf(psf, img.shape)
    G = np.fft.fft2(img)

    H_conj = np.conj(H)
    F_hat = (H_conj / (np.abs(H)**2 + K)) * G

    restored = np.real(np.fft.ifft2(F_hat))
    restored = np.clip(restored, 0, 1)

    return (restored * 255).astype(np.uint8)

def process_image_with_wiener(img_path, folder_name):
    img = cv2.imread(img_path)
    if img is None:
        return ""

    # 1. Fixed 8% crop
    h, w = img.shape[:2]
    crop_start_x = int(w * CROP_RATIO)
    cropped = img[:, crop_start_x:]

    # 2. Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # 3. Select PSF based on blur level
    ksize = BLUR_KERNEL_MAP[folder_name]
    psf = gaussian_kernel_2d(ksize)

    # 4. Wiener deconvolution
    restored = wiener_deconvolution(gray, psf, K=WIENER_K)

    # 5. OCR
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    pred_text = pytesseract.image_to_string(restored, config=custom_config)

    return clean_ocr_text(pred_text)

def main():
    base_dir = 'dataset'
    test_folders = ['blur_1', 'blur_2', 'blur_3']
    results = []

    print(f"🚀 Starting BLUR preprocessing (Wiener Filter) agile testing (per group limit: {MAX_TEST_IMAGES} images)...\n")
    print(f"🧪 Crop Ratio = {CROP_RATIO:.2f}")
    print(f"🧪 Wiener K   = {WIENER_K}\n")

    start_time = time.time()

    for folder in test_folders:
        folder_path = os.path.join(base_dir, 'degraded', folder)
        if not os.path.exists(folder_path):
            continue

        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])[:MAX_TEST_IMAGES]

        if not files:
            continue

        total_images = len(files)
        exact_matches = 0
        total_char_acc = 0.0

        prefix = f"🌀 Wiener processing [{folder:^10}]"
        print_progress(0, total_images, prefix=prefix)

        for i, f in enumerate(files, 1):
            img_path = os.path.join(folder_path, f)
            gt_text = get_ground_truth(f)
            pred_text = process_image_with_wiener(img_path, folder)

            is_exact_match = (gt_text == pred_text)
            char_acc = calc_character_accuracy(gt_text, pred_text)

            if is_exact_match:
                exact_matches += 1
            total_char_acc += char_acc

            print_progress(i, total_images, prefix=prefix)

        results.append({
            'Level': folder,
            'Exact Match (%)': (exact_matches / total_images) * 100,
            'Char Accuracy (%)': (total_char_acc / total_images) * 100
        })

    print("\n" + "=" * 55)
    print("📊 Wiener Filter Agile Test Results")
    print("=" * 55)
    print(f"{'Level':<15} | {'Exact Match (%)':<15} | {'Char Accuracy (%)':<15}")
    print("-" * 55)

    for res in results:
        print(f"{res['Level']:<15} | {res['Exact Match (%)']:.2f}%{'':<9} | {res['Char Accuracy (%)']:.2f}%")

    print(f"\n⏱️ Test Duration: {time.time() - start_time:.2f}s")
    print("💡 Remember to compare with baseline / sharpening / USM.")

if __name__ == "__main__":
    main()