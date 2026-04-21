import cv2
import pytesseract
import os
import re
import time
from collections import Counter

# ⚠️ path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 🎯 Test Configuration
# ==========================================
MAX_TEST_IMAGES = 1000
CROP_RATIO = 0.08
PLATE_LENGTH = 7
TARGET_FOLDERS = ['blur_1', 'blur_2', 'blur_3']

OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# ==========================================
# 1. helper functions for text processing and evaluation
# ==========================================
def get_ground_truth(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r'[^A-Z0-9]', '', name.upper())

def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def fixed_length_text(text, plate_len=7, pad_char='#'):
    """Force padding to 7 characters for Position-wise (per-character) comparison"""
    return text[:plate_len].ljust(plate_len, pad_char)

def calc_character_accuracy(gt, pred):
    if len(gt) == 0 and len(pred) == 0: return 1.0
    if len(gt) == 0 or len(pred) == 0: return 0.0
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    max_len = max(m, n)
    return max(0.0, (max_len - dp[m][n]) / max_len)

# 🌟 New Preprocessing: Super-Resolution Idea (Upscaling + Local Contrast Enhancement)
def preprocess_for_blur(cropped_bgr):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Bicubic Interpolation (Lossless Upscaling) - Double the size, separate pen strokes
    enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 3. CLAHE - Enhance contrast, save gentle grayscale gradients, but don't binarize!
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(enlarged)
    
    return processed

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total})', end='\r')
    if iteration == total: print()

# ==========================================
# 2. Main evaluation loop for BLUR category
# ==========================================
def main():
    base_dir = 'dataset'
    overall_start = time.time()

    print("🚀 starting BLUR (Bicubic+CLAHE) pure memory analysis...")
    print(f"🧪 maximum test images = {MAX_TEST_IMAGES} images/group\n")

    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(base_dir, 'degraded', folder)
        if not os.path.exists(folder_path): continue

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))])[:MAX_TEST_IMAGES]
        if not files: continue

        total_images = len(files)
        exact_matches = 0
        total_char_acc = 0.0

        # 🎯 Position-wise Errors
        position_errors = {i: 0 for i in range(PLATE_LENGTH)}
        confusion_counter = Counter()

        prefix = f"🔍 Analysis in Progress [{folder:^10}]"
        print_progress(0, total_images, prefix=prefix)

        for i, f in enumerate(files, 1):
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path)
            if img is None: continue

            # corpt the left 8% of the image to focus on the license plate area, then apply our new preprocessing for blur
            h, w = img.shape[:2]
            cropped = img[:, int(w * CROP_RATIO):]
            processed = preprocess_for_blur(cropped)

            # OCR
            pred_text = clean_ocr_text(pytesseract.image_to_string(processed, config=OCR_CONFIG))
            gt_text = get_ground_truth(f)

            # results
            is_exact_match = (gt_text == pred_text)
            if is_exact_match: exact_matches += 1
            total_char_acc += calc_character_accuracy(gt_text, pred_text)

            # character-level error analysis
            gt_fixed = fixed_length_text(gt_text)
            pred_fixed = fixed_length_text(pred_text)
            
            for pos in range(PLATE_LENGTH):
                if gt_fixed[pos] != pred_fixed[pos]:
                    position_errors[pos] += 1
                    confusion_counter[(gt_fixed[pos], pred_fixed[pos])] += 1

            print_progress(i, total_images, prefix=prefix)

        # Print the report for the current level
        exact_pct = (exact_matches / total_images) * 100
        char_acc_pct = (total_char_acc / total_images) * 100

        print(f"\n📊 Results Report: {folder.upper()}")
        print(f"  ➜ Exact Match: {exact_pct:.2f}%")
        print(f"  ➜ Char Accuracy: {char_acc_pct:.2f}%")

        print("\n  📍 Position-wise Error Frequencies (which bits are most prone to errors):")
        for pos in range(PLATE_LENGTH):
            err_count = position_errors[pos]
            err_rate = (err_count / total_images) * 100
            # Simple ASCII histogram for intuitive visualization of error-prone positions
            bar = '▓' * int(err_rate / 2) 
            print(f"    Bit {pos + 1} (P{pos+1}): Errors {err_count:>3} | {err_rate:>5.1f}% | {bar}")

        print("\n  🔄 Fatal Confusion Pairs Top 5:")
        for idx, ((gt_c, pred_c), count) in enumerate(confusion_counter.most_common(5), 1):
            print(f"    {idx}. Real '{gt_c}' mistaken for '{pred_c}' -> Occurred {count} times")
        print("-" * 60)

    print(f"\n✅ Analysis complete, total time: {time.time() - overall_start:.2f}s")

if __name__ == "__main__":
    main()