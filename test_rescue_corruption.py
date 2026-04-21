import cv2
import numpy as np
import pytesseract
import os
import re
import time
from collections import Counter

# ⚠️ path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 🎯 config
# ==========================================
MAX_TEST_IMAGES = 1000
CROP_RATIO = 0.08
PLATE_LENGTH = 7
TARGET_FOLDERS = ['corrupt_1', 'corrupt_2', 'corrupt_3']

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
            if gt[i - 1] == pred[j - 1]: dp[i][j] = dp[i - 1][j - 1]
            else: dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    max_len = max(m, n)
    return max(0.0, (max_len - dp[m][n]) / max_len)

def normalize_plate_text_v2(raw_text):
    text = clean_ocr_text(raw_text)
    if len(text) == 8 and text[0] in {'1', 'I', 'O'}: text = text[1:]
    if len(text) != 7: return text
    chars = list(text)
    for i in range(min(2, len(chars))):
        if chars[i] == 'O': chars[i] = '0'
        if chars[i] in {'I', 'L'}: chars[i] = '1'
    if len(chars) > 2:
        if chars[2] == '0': chars[2] = 'O'
        if chars[2] == '1': chars[2] = 'I'
    for i in range(3, min(7, len(chars))):
        if chars[i] == 'O': chars[i] = '0'
        if chars[i] in {'I', 'L'}: chars[i] = '1'
    return ''.join(chars)

# ==========================================
# 🌟 CORRUPT pre-processing
# ==========================================
def preprocess_for_corrupt(cropped_bgr, folder_name):
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Force binary thresholding and invert colors (white text on a black background, to enable the closing operation).
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if folder_name == 'corrupt_3':
        # 🔴 corrupt_3 consists of scattered scratches: it uses a 3x3 closing operation to forcibly connect broken strokes.
        kernel = np.ones((5, 5), np.uint8)
        processed_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
        
    elif folder_name == 'corrupt_2':
        # 🟡 `corrupt_2` consists of a massive central occlusion block: morphological closing cannot resolve this—in fact, it would only lead to adhesions. We will therefore perform only morphological opening to remove noise points.
        kernel = np.ones((3, 3), np.uint8)
        processed_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
        
    else:
        # 🟢 corrupt_1 edge occlusion: also large occlusions, mild processing
        processed_inv = binary_inv

    # 2. Invert colors back (white background, black text) for Tesseract
    final_img = cv2.bitwise_not(processed_inv)
    return final_img

def print_progress(iteration, total, prefix='', length=30):
    if total == 0: return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total})', end='\r')
    if iteration == total: print()

# ==========================================
# 2. Main Workflow
# ==========================================
def main():
    base_dir = 'dataset'
    overall_start = time.time()

    print("🚀 Starting CORRUPT (Morphological Repair + Rule Engine) Agile Analysis...")
    print(f"🧪 Maximum Test Images = {MAX_TEST_IMAGES} / Folder\n")

    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(base_dir, 'degraded', folder)
        if not os.path.exists(folder_path): continue

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))])[:MAX_TEST_IMAGES]
        if not files: continue

        total_images = len(files)
        exact_matches = 0
        total_char_acc = 0.0

        position_errors = {i: 0 for i in range(PLATE_LENGTH)}
        confusion_counter = Counter()

        prefix = f"🔍 Analysis in Progress [{folder:^10}]"
        print_progress(0, total_images, prefix=prefix)

        for i, f in enumerate(files, 1):
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path)
            if img is None: continue

            h, w = img.shape[:2]
            cropped = img[:, int(w * CROP_RATIO):]
            processed = preprocess_for_corrupt(cropped, folder)

            raw_text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
            pred_text = normalize_plate_text_v2(raw_text)
            gt_text = get_ground_truth(f)

            is_exact_match = (gt_text == pred_text)
            if is_exact_match: exact_matches += 1
            total_char_acc += calc_character_accuracy(gt_text, pred_text)

            gt_fixed = fixed_length_text(gt_text)
            pred_fixed = fixed_length_text(pred_text)
            
            for pos in range(PLATE_LENGTH):
                if gt_fixed[pos] != pred_fixed[pos]:
                    position_errors[pos] += 1
                    confusion_counter[(gt_fixed[pos], pred_fixed[pos])] += 1

            print_progress(i, total_images, prefix=prefix)

        exact_pct = (exact_matches / total_images) * 100
        char_acc_pct = (total_char_acc / total_images) * 100

        print(f"\n📊 Results Report: {folder.upper()}")
        print(f"  ➜ Exact Match: {exact_pct:.2f}%")
        print(f"  ➜ Char Accuracy: {char_acc_pct:.2f}%")

        print("\n  📍 Bit-level Error Frequencies (which bits are most prone to errors):")
        for pos in range(PLATE_LENGTH):
            err_count = position_errors[pos]
            err_rate = (err_count / total_images) * 100
            bar = '▓' * int(err_rate / 2) 
            print(f"    Bit {pos + 1} (P{pos+1}): wrong {err_count:>3} times | {err_rate:>5.1f}% | {bar}")

        print("\n  🔄 Fatal Confusion Pair Top 5:")
        for idx, ((gt_c, pred_c), count) in enumerate(confusion_counter.most_common(5), 1):
            print(f"    {idx}. Real '{gt_c}' mistaken for '{pred_c}' -> occurred {count} times")
        print("-" * 60)

    print(f"\n✅ Analysis complete, total time: {time.time() - overall_start:.2f}s")

if __name__ == "__main__":
    main()