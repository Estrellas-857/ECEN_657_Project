import cv2
import pytesseract
import os
import re
import time
import csv
from collections import Counter, defaultdict

# ⚠️ Ensure your Tesseract path is correct
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 🎯 Configuration Zone
# ==========================================
MAX_TEST_IMAGES = 1000
CROP_RATIO = 0.08
PLATE_LENGTH = 7

TARGET_FOLDERS = ['noise_1', 'noise_2', 'noise_3']

OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# ==========================================
# 1. helper functions for text processing and evaluation
# ==========================================
def get_ground_truth(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r'[^A-Z0-9]', '', name.upper())

def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def plate_pattern_score(text):
    """
Score the 7 candidate strings:
P1, P2, P6, and P7 favor digits.
P3 favors letters.
P4 and P5 have no specific preference.
    """
    if len(text) != 7:
        return -999

    score = 0

    # P1, P2: digits
    if text[0].isdigit():
        score += 3
    if text[1].isdigit():
        score += 3

    # P3: letter
    if text[2].isalpha():
        score += 3

    # P4, P5: no strict requirement, give a point if it's alphanumeric
    if text[3].isalnum():
        score += 1
    if text[4].isalnum():
        score += 1

    # P6, P7: digits
    if text[5].isdigit():
        score += 3
    if text[6].isdigit():
        score += 3

    return score


def select_best_7char_window(text):
    """
    If the OCR output exceeds 7 characters, select the most plausible 7-character window.
    """
    if len(text) <= 7:
        return text

    candidates = [text[i:i+7] for i in range(len(text) - 6)]
    return max(candidates, key=plate_pattern_score)


def normalize_plate_text_v3(raw_text):
    """
    Apply light format correction based on license plate rules:
    P1,P2: digits
    P3: letter
    P4,P5: no strict requirement
    P6,P7: digits
    """
    text = clean_ocr_text(raw_text)

    # First, handle overly long output: select the most plausible 7-character window
    text = select_best_7char_window(text)

    if len(text) != 7:
        return text

    chars = list(text)

    # P1, P2: digits
    for i in [0, 1]:
        if chars[i] == 'O':
            chars[i] = '0'
        elif chars[i] in {'I', 'L'}:
            chars[i] = '1'

    # P3: letter
    if chars[2] == '0':
        chars[2] = 'O'
    elif chars[2] == '1':
        chars[2] = 'I'
    elif chars[2] == '5':
        chars[2] = 'S'
    elif chars[2] == '8':
        chars[2] = 'B'
    elif chars[2] == '2':
        chars[2] = 'Z'

    # P4, P5: no strict requirement, do not make rule corrections

    # P6, P7: digits
    for i in [5, 6]:
        if chars[i] == 'O':
            chars[i] = '0'
        elif chars[i] in {'I', 'L'}:
            chars[i] = '1'

    return ''.join(chars)

def levenshtein_distance(gt, pred):
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

    return dp[m][n]

def calc_character_accuracy(gt, pred):
    if len(gt) == 0 and len(pred) == 0:
        return 1.0
    if len(gt) == 0 or len(pred) == 0:
        return 0.0

    dist = levenshtein_distance(gt, pred)
    max_len = max(len(gt), len(pred))
    return max(0.0, (max_len - dist) / max_len)

def fixed_length_text(text, plate_len=7, pad_char='#'):
    return text[:plate_len].ljust(plate_len, pad_char)

def positional_error_count(gt, pred, plate_len=7):
    gt_fixed = fixed_length_text(gt, plate_len)
    pred_fixed = fixed_length_text(pred, plate_len)
    return sum(1 for i in range(plate_len) if gt_fixed[i] != pred_fixed[i])
'''
def preprocess_for_noise_folder(cropped_bgr, folder_name):
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    if folder_name == 'noise_1':
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    elif folder_name == 'noise_2':
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    elif folder_name == 'noise_3':
        denoised = cv2.medianBlur(gray, 3)
    else:
        denoised = gray

    return denoised'''

def preprocess_for_noise_folder(cropped_bgr, folder_name):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    if folder_name == 'noise_1':
        # 🟢 Handling mild Gaussian noise
        # d=5 (pixel neighborhood diameter), sigmaColor=50 (color space filtering strength), sigmaSpace=50 (coordinate space filtering strength)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        # Forced Otsu Binarization
        _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed
        

    elif folder_name == 'noise_2':
        # 🟡 Handling severe Gaussian noise (increase filtering strength)
        # Increase d and sigma parameters to aggressively smooth background snow, while still preserving letter edges
        denoised = cv2.bilateralFilter(gray, 7, 75, 75)
        # Forced Otsu Binarization
        _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Advanced technique: After heavy denoising, letters may become thinner, so use morphological closing to slightly丰满 the strokes (optional)
        # kernel = np.ones((2,2), np.uint8)
        # processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return processed

    elif folder_name == 'noise_3':
        # 🔴 Handling salt-and-pepper noise (maintain median filtering, best效果)
        processed = cv2.medianBlur(gray, 3)
        return processed

    else:
        return gray

def process_image(img_path, folder_name):
    img = cv2.imread(img_path)
    if img is None:
        return "", None

    h, w = img.shape[:2]
    crop_start_x = int(w * CROP_RATIO)
    cropped = img[:, crop_start_x:]

    processed = preprocess_for_noise_folder(cropped, folder_name)

    raw_text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
    pred_text = normalize_plate_text_v3(raw_text)

    return pred_text, processed

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

# ==========================================
# 2. main
# ==========================================
def main():
    base_dir = 'dataset'
    overall_start = time.time()

    print("🚀 starting NOISE error analysis...")
    print(f"🧪 Crop Ratio = {CROP_RATIO:.2f}")
    print(f"🧪 OCR Config = psm7 + whitelist")
    print(f"🧪 MAX_TEST_IMAGES = {MAX_TEST_IMAGES}\n")

    summary_rows = []

    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(base_dir, 'degraded', folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ skipping non-existent folder: {folder_path}")
            continue

        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])[:MAX_TEST_IMAGES]

        if not files:
            print(f"⚠️ folder is empty: {folder_path}")
            continue

        total_images = len(files)
        exact_matches = 0
        total_char_acc = 0.0

        # accuracy distributions and confusion analysis
        error_count_dist = Counter()                
        edit_distance_dist = Counter()              
        confusion_counter = Counter()               
        position_confusions = [Counter() for _ in range(PLATE_LENGTH)]  
        details_rows = []

        folder_start = time.time()
        prefix = f"🔍 analyzing [{folder:^10}]"
        print_progress(0, total_images, prefix=prefix)

        for i, f in enumerate(files, 1):
            img_path = os.path.join(folder_path, f)
            gt_text = get_ground_truth(f)
            pred_text, _ = process_image(img_path, folder)

            is_exact_match = (gt_text == pred_text)
            char_acc = calc_character_accuracy(gt_text, pred_text)
            edit_dist = levenshtein_distance(gt_text, pred_text)
            pos_errs = positional_error_count(gt_text, pred_text, PLATE_LENGTH)

            if is_exact_match:
                exact_matches += 1
            total_char_acc += char_acc

            error_count_dist[pos_errs] += 1
            edit_distance_dist[edit_dist] += 1

            gt_fixed = fixed_length_text(gt_text, PLATE_LENGTH)
            pred_fixed = fixed_length_text(pred_text, PLATE_LENGTH)

            row = {
                'Image': f,
                'Ground Truth': gt_text,
                'OCR Output': pred_text,
                'Exact Match': 'Yes' if is_exact_match else 'No',
                'Edit Distance': edit_dist,
                'Positional Errors': pos_errs,
                'Char Accuracy (%)': f"{char_acc * 100:.2f}"
            }

            for pos in range(PLATE_LENGTH):
                gt_c = gt_fixed[pos]
                pred_c = pred_fixed[pos]
                row[f'GT_P{pos+1}'] = gt_c
                row[f'Pred_P{pos+1}'] = pred_c
                row[f'Match_P{pos+1}'] = 1 if gt_c == pred_c else 0

                if gt_c != pred_c:
                    confusion_counter[(gt_c, pred_c)] += 1
                    position_confusions[pos][(gt_c, pred_c)] += 1

            details_rows.append(row)
            print_progress(i, total_images, prefix=prefix)

        folder_time = time.time() - folder_start
        exact_pct = (exact_matches / total_images) * 100
        char_acc_pct = (total_char_acc / total_images) * 100

        summary_rows.append({
            'Level': folder,
            'Num Images': total_images,
            'Exact Match (%)': f"{exact_pct:.2f}",
            'Char Accuracy (%)': f"{char_acc_pct:.2f}",
            'Time (s)': f"{folder_time:.2f}"
        })

        # ==========================================
        # csv file save
        # ==========================================
        details_csv = f'noise_error_details_{folder}.csv'
        with open(details_csv, 'w', newline='', encoding='utf-8') as f_csv:
            fieldnames = [
                'Image', 'Ground Truth', 'OCR Output', 'Exact Match',
                'Edit Distance', 'Positional Errors', 'Char Accuracy (%)'
            ]
            for pos in range(PLATE_LENGTH):
                fieldnames += [f'GT_P{pos+1}', f'Pred_P{pos+1}', f'Match_P{pos+1}']

            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(details_rows)

        error_dist_csv = f'noise_error_count_distribution_{folder}.csv'
        with open(error_dist_csv, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['Num Wrong Positions', 'Count', 'Percentage (%)'])
            writer.writeheader()
            for k in range(PLATE_LENGTH + 1):
                count = error_count_dist[k]
                writer.writerow({
                    'Num Wrong Positions': k,
                    'Count': count,
                    'Percentage (%)': f"{(count / total_images) * 100:.2f}"
                })

        edit_dist_csv = f'noise_edit_distance_distribution_{folder}.csv'
        with open(edit_dist_csv, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['Edit Distance', 'Count', 'Percentage (%)'])
            writer.writeheader()
            for k in sorted(edit_distance_dist.keys()):
                count = edit_distance_dist[k]
                writer.writerow({
                    'Edit Distance': k,
                    'Count': count,
                    'Percentage (%)': f"{(count / total_images) * 100:.2f}"
                })

        confusion_csv = f'noise_confusions_{folder}.csv'
        with open(confusion_csv, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['GT Char', 'Pred Char', 'Count'])
            writer.writeheader()
            for (gt_c, pred_c), count in confusion_counter.most_common():
                writer.writerow({
                    'GT Char': gt_c,
                    'Pred Char': pred_c,
                    'Count': count
                })

        pos_conf_csv = f'noise_position_confusions_{folder}.csv'
        with open(pos_conf_csv, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['Position', 'GT Char', 'Pred Char', 'Count'])
            writer.writeheader()
            for pos in range(PLATE_LENGTH):
                for (gt_c, pred_c), count in position_confusions[pos].most_common():
                    writer.writerow({
                        'Position': pos + 1,
                        'GT Char': gt_c,
                        'Pred Char': pred_c,
                        'Count': count
                    })

        # ==========================================
        # terminal check and report
        # ==========================================
        print(f"\n📊 [{folder}]")
        print(f"Exact Match: {exact_pct:.2f}%")
        print(f"Char Accuracy: {char_acc_pct:.2f}%")

        print("\nwrong positions per image (compared at fixed 7-digit positions):")
        for k in range(PLATE_LENGTH + 1):
            count = error_count_dist[k]
            pct = (count / total_images) * 100
            print(f"  wrong {k} positions: {count} images ({pct:.2f}%)")

        print("\nmost common confusion pairs (Top 10):")
        for idx, ((gt_c, pred_c), count) in enumerate(confusion_counter.most_common(10), 1):
            print(f"  {idx:>2}. {gt_c} -> {pred_c}: {count}")

        print(f"\n💾 Files saved:")
        print(f"  - {details_csv}")
        print(f"  - {error_dist_csv}")
        print(f"  - {edit_dist_csv}")
        print(f"  - {confusion_csv}")
        print(f"  - {pos_conf_csv}")
        print("-" * 60)

    # summary csv for all levels
    summary_csv = 'noise_error_analysis_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=['Level', 'Num Images', 'Exact Match (%)', 'Char Accuracy (%)', 'Time (s)']
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 70)
    print("✅ All analysis completed")
    print("=" * 70)
    print(f"💾 total summary: {summary_csv}")
    print(f"⏱️ total time cost: {time.time() - overall_start:.2f}s")

if __name__ == "__main__":
    main()