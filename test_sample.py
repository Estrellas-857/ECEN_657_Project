import cv2
import pytesseract
import os
import re
import csv
import time
from collections import Counter

# ==========================================
# ⚠️ Tesseract path
# ==========================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 🎯 experiment set up
# ==========================================
TARGET_TYPES = [
    'clean',
    # 'blur',
]

# try different crop ratios to see how it affects OCR performance
CROP_RATIOS = [0.00, 0.05, 0.07, 0.08, 0.10]

# each folder will only run the first 1000 images
MAX_IMAGES = 1000

PLATE_LENGTH = 7

USE_FORMAT_POSTPROCESS = False

# ==========================================
# 1. helper functions for text processing and evaluation
# ==========================================
def get_ground_truth(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r'[^A-Z0-9]', '', name.upper())

def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def apply_plate_format_postprocess(text):
    """
    Optional: If the license plate format is fixed as DDLLLDD, position correction can be applied
    This is optional and not enabled by default
    """
    chars = list(text[:PLATE_LENGTH])

    for i, ch in enumerate(chars):
        # digit positions: 0,1,5,6 - aggressive correction
        if i in [0, 1, 5, 6]:
            if ch == 'O':
                chars[i] = '0'
            elif ch == 'I':
                chars[i] = '1'
            elif ch == 'Z':
                chars[i] = '2'
            elif ch == 'S':
                chars[i] = '5'
            elif ch == 'B':
                chars[i] = '8'

        # letter positions: 2,3,4 - more conservative correction
        elif i in [2, 3, 4]:
            if ch == '0':
                chars[i] = 'O'
            elif ch == '1':
                chars[i] = 'I'
            elif ch == '2':
                chars[i] = 'Z'
            elif ch == '5':
                chars[i] = 'S'
            elif ch == '8':
                chars[i] = 'B'

    return ''.join(chars)

def calc_character_accuracy(gt, pred):
    """
    Character-level accuracy based on edit distance
    accuracy = 1 - edit_distance / max(len(gt), len(pred))
    """
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

def calc_positionwise_matches(gt, pred, plate_len=7):
    gt_fixed = gt[:plate_len].ljust(plate_len, '#')
    pred_fixed = pred[:plate_len].ljust(plate_len, '#')
    return [1 if gt_fixed[i] == pred_fixed[i] else 0 for i in range(plate_len)]

def process_image(img_path, crop_ratio=0.0):
    """
    Returns:
        pred_text, status, error_msg
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return "", "FAIL", "cv2.imread returned None"

        height, width = img.shape[:2]

        crop_start_x = int(width * crop_ratio)
        crop_start_x = max(0, min(crop_start_x, width - 1))
        cropped_img = img[:, crop_start_x:]

        # Single-line license plate OCR
        custom_config = (
            r'--oem 3 --psm 7 '
            r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        raw_text = pytesseract.image_to_string(cropped_img, config=custom_config)
        pred_text = clean_ocr_text(raw_text)

        if USE_FORMAT_POSTPROCESS:
            pred_text = apply_plate_format_postprocess(pred_text)

        return pred_text, "OK", ""

    except pytesseract.TesseractError as e:
        return "", "FAIL", f"TesseractError: {str(e)}"
    except Exception as e:
        return "", "FAIL", f"Unexpected error: {str(e)}"

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

def format_seconds(seconds):
    return f"{seconds:.2f}s"

def crop_tag(crop_ratio):
    return f"crop{int(round(crop_ratio * 100)):02d}"

def safe_char_at(text, idx):
    if idx < len(text):
        return text[idx]
    return "#"

# ==========================================
# 2. Main test workflow
# ==========================================
def main():
    base_dir = 'dataset'
    overall_start = time.perf_counter()

    print("🚀 Starting OCR crop ratio experiment...\n")
    print(f"🧪 TARGET_TYPES = {TARGET_TYPES}")
    print(f"🧪 CROP_RATIOS = {CROP_RATIOS}")
    print(f"🧪 MAX_IMAGES per folder = {MAX_IMAGES}\n")

    for target in TARGET_TYPES:
        print("\n" + "=" * 70)
        print(f"🎯 doing major test: [{target.upper()}]")
        print("=" * 70)

        if target == 'clean':
            folders = ['clean']
        else:
            folders = [f"{target}_1", f"{target}_2", f"{target}_3"]

        for crop_ratio in CROP_RATIOS:
            crop_name = crop_tag(crop_ratio)
            target_start = time.perf_counter()

            print("\n" + "-" * 70)
            print(f"✂️ Current crop ratio: {crop_ratio:.0%} ({crop_name})")
            print("-" * 70)

            summary_results = []
            position_results = []

            sum_csv = f'baseline_summary_{target}_{crop_name}.csv'
            det_csv = f'baseline_details_{target}_{crop_name}.csv'
            pos_csv = f'baseline_position_accuracy_{target}_{crop_name}.csv'
            p1_conf_csv = f'baseline_p1_confusion_{target}_{crop_name}.csv'

            # First character confusion statistics: GT -> Pred
            p1_confusions = Counter()

            with open(det_csv, 'w', newline='', encoding='utf-8') as f_det:
                det_writer = csv.DictWriter(
                    f_det,
                    fieldnames=[
                        'Image', 'Level', 'Crop Ratio',
                        'Ground Truth', 'OCR Output',
                        'Exact Match', 'Char Accuracy',
                        'P1_GT', 'P1_Pred', 'P1_Match',
                        'Pos1', 'Pos2', 'Pos3', 'Pos4', 'Pos5', 'Pos6', 'Pos7',
                        'Status', 'Error'
                    ]
                )
                det_writer.writeheader()

                for folder in folders:
                    folder_path = os.path.join(
                        base_dir,
                        'degraded' if target != 'clean' else '',
                        folder
                    )

                    if not os.path.exists(folder_path):
                        print(f"⚠️ Skipping non-existent folder: {folder_path}")
                        continue

                    files = sorted([
                        f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ])[:MAX_IMAGES]

                    if not files:
                        print(f"⚠️ Folder is empty or contains no images: {folder_path}")
                        continue

                    total_images = len(files)
                    exact_matches = 0
                    total_char_acc = 0.0
                    failed_images = 0

                    pos_correct_counts = [0] * PLATE_LENGTH

                    folder_start = time.perf_counter()

                    prefix = f"📂 [{folder:^10}] [{crop_name}]"
                    print_progress(0, total_images, prefix=prefix)

                    for i, f in enumerate(files, 1):
                        gt_text = get_ground_truth(f)
                        img_path = os.path.join(folder_path, f)

                        pred_text, status, error_msg = process_image(img_path, crop_ratio=crop_ratio)

                        is_exact_match = (gt_text == pred_text)
                        char_acc = calc_character_accuracy(gt_text, pred_text)
                        pos_matches = calc_positionwise_matches(gt_text, pred_text, plate_len=PLATE_LENGTH)

                        if status != "OK":
                            failed_images += 1

                        if is_exact_match:
                            exact_matches += 1

                        total_char_acc += char_acc

                        for idx in range(PLATE_LENGTH):
                            pos_correct_counts[idx] += pos_matches[idx]

                        # First character confusion statistics
                        p1_gt = safe_char_at(gt_text, 0)
                        p1_pred = safe_char_at(pred_text, 0)
                        p1_confusions[(folder, p1_gt, p1_pred)] += 1

                        det_writer.writerow({
                            'Image': f,
                            'Level': folder,
                            'Crop Ratio': f"{crop_ratio:.2f}",
                            'Ground Truth': gt_text,
                            'OCR Output': pred_text,
                            'Exact Match': 'Yes' if is_exact_match else 'No',
                            'Char Accuracy': f"{char_acc * 100:.1f}%",
                            'P1_GT': p1_gt,
                            'P1_Pred': p1_pred,
                            'P1_Match': 1 if p1_gt == p1_pred else 0,
                            'Pos1': pos_matches[0],
                            'Pos2': pos_matches[1],
                            'Pos3': pos_matches[2],
                            'Pos4': pos_matches[3],
                            'Pos5': pos_matches[4],
                            'Pos6': pos_matches[5],
                            'Pos7': pos_matches[6],
                            'Status': status,
                            'Error': error_msg
                        })

                        f_det.flush()
                        print_progress(i, total_images, prefix=prefix)

                    folder_time = time.perf_counter() - folder_start

                    summary_row = {
                        'Level': folder,
                        'Crop Ratio': f"{crop_ratio:.2f}",
                        'Num Images': total_images,
                        'Exact Match (%)': (exact_matches / total_images) * 100,
                        'Char Accuracy (%)': (total_char_acc / total_images) * 100,
                        'P1 Accuracy (%)': (pos_correct_counts[0] / total_images) * 100,
                        'Failed Images': failed_images,
                        'Time (s)': f"{folder_time:.2f}"
                    }
                    summary_results.append(summary_row)

                    pos_row = {
                        'Level': folder,
                        'Crop Ratio': f"{crop_ratio:.2f}"
                    }
                    for idx in range(PLATE_LENGTH):
                        pos_acc = (pos_correct_counts[idx] / total_images) * 100
                        pos_row[f'Position {idx + 1} (%)'] = f"{pos_acc:.2f}"
                    position_results.append(pos_row)

            # save summary
            if summary_results:
                with open(sum_csv, 'w', newline='', encoding='utf-8') as f_sum:
                    sum_writer = csv.DictWriter(
                        f_sum,
                        fieldnames=[
                            'Level',
                            'Crop Ratio',
                            'Num Images',
                            'Exact Match (%)',
                            'Char Accuracy (%)',
                            'P1 Accuracy (%)',
                            'Failed Images',
                            'Time (s)'
                        ]
                    )
                    sum_writer.writeheader()
                    sum_writer.writerows(summary_results)

                print(f"\n📊 [{target.upper()} - {crop_name}] summary:")
                print(f"{'Level':<12} | {'Crop':<6} | {'N':<6} | {'Exact':<10} | {'CharAcc':<10} | {'P1Acc':<10} | {'Fail':<6} | {'Time':<8}")
                print("-" * 95)
                for res in summary_results:
                    print(
                        f"{res['Level']:<12} | "
                        f"{res['Crop Ratio']:<6} | "
                        f"{res['Num Images']:<6} | "
                        f"{res['Exact Match (%)']:.2f}%{'':<3} | "
                        f"{res['Char Accuracy (%)']:.2f}%{'':<3} | "
                        f"{res['P1 Accuracy (%)']:.2f}%{'':<3} | "
                        f"{res['Failed Images']:<6} | "
                        f"{res['Time (s)']:<8}"
                    )

                print(f"\n💾 Summary data saved: {sum_csv}")

            # save position accuracy
            if position_results:
                with open(pos_csv, 'w', newline='', encoding='utf-8') as f_pos:
                    pos_writer = csv.DictWriter(
                        f_pos,
                        fieldnames=[
                            'Level',
                            'Crop Ratio',
                            'Position 1 (%)',
                            'Position 2 (%)',
                            'Position 3 (%)',
                            'Position 4 (%)',
                            'Position 5 (%)',
                            'Position 6 (%)',
                            'Position 7 (%)'
                        ]
                    )
                    pos_writer.writeheader()
                    pos_writer.writerows(position_results)

                print(f"💾 Position-wise data saved: {pos_csv}")

            # save P1 confusion
            with open(p1_conf_csv, 'w', newline='', encoding='utf-8') as f_conf:
                conf_writer = csv.DictWriter(
                    f_conf,
                    fieldnames=['Level', 'P1_GT', 'P1_Pred', 'Count']
                )
                conf_writer.writeheader()

                # order by count descending
                for (level, gt_c, pred_c), count in sorted(
                    p1_confusions.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    conf_writer.writerow({
                        'Level': level,
                        'P1_GT': gt_c,
                        'P1_Pred': pred_c,
                        'Count': count
                    })

            print(f"💾 First character confusion statistics saved: {p1_conf_csv}")
            print(f"💾 Detailed data saved: {det_csv}")

            target_time = time.perf_counter() - target_start
            print(f"⏱️ Current crop ratio [{crop_name}] total time: {format_seconds(target_time)}")

    overall_time = time.perf_counter() - overall_start
    print("\n" + "=" * 70)
    print(f"✅ All experiments completed, total time: {format_seconds(overall_time)}")
    print("=" * 70)

if __name__ == "__main__":
    main()