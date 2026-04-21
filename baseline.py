import cv2
import pytesseract
import os
import re
import csv
import time

# ⚠️ Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==========================================
# 🎯 configuration
# ==========================================
TARGET_TYPES = [
    #'clean',
    #'blur',
    #'noise',
    #'illum',
     'corrupt'
]

PLATE_LENGTH = 7

# ==========================================
# 1. helper functions
# ==========================================
def get_ground_truth(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r'[^A-Z0-9]', '', name.upper())

def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def calc_character_accuracy(gt, pred):
    """
    Character-level Accuracy Based on Edit Distance
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
    """
    Return whether each digit was recognized correctly:
    [1, 0, 1, 1, 0, 1, 1]
    """
    gt_fixed = gt[:plate_len].ljust(plate_len, '#')
    pred_fixed = pred[:plate_len].ljust(plate_len, '#')

    return [1 if gt_fixed[i] == pred_fixed[i] else 0 for i in range(plate_len)]

def process_image(img_path):
    """
    return:
        pred_text, status, error_msg
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return "", "FAIL", "cv2.imread returned None"

        height, width = img.shape[:2]
        crop_ratio = 0.07
        crop_start_x = int(width * crop_ratio)
        cropped_img = img[:, crop_start_x:]

        custom_config = (
            r'--oem 3 --psm 7 '
            r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        raw_text = pytesseract.image_to_string(cropped_img, config=custom_config)
        pred_text = clean_ocr_text(raw_text)

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

# ==========================================
# 2. main evaluation loop
# ==========================================
def main():
    base_dir = 'dataset'
    overall_start = time.perf_counter()

    print("🚀 start Baseline OCR test...\n")

    for target in TARGET_TYPES:
        target_start = time.perf_counter()

        print("\n" + "=" * 60)
        print(f"🎯 Executing major category tests: [{target.upper()}]")
        print("=" * 60)

        if target == 'clean':
            folders = ['clean']
        else:
            folders = [f"{target}_1", f"{target}_2", f"{target}_3"]

        summary_results = []
        position_results = []

        sum_csv = f'baseline_summary_{target}.csv'
        det_csv = f'baseline_details_{target}.csv'
        pos_csv = f'baseline_position_accuracy_{target}.csv'

        with open(det_csv, 'w', newline='', encoding='utf-8') as f_det:
            det_writer = csv.DictWriter(
                f_det,
                fieldnames=[
                    'Image', 'Level', 'Ground Truth', 'OCR Output',
                    'Exact Match', 'Char Accuracy',
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
                ])

                if not files:
                    print(f"⚠️ Folder is empty or contains no images: {folder_path}")
                    continue

                total_images = len(files)
                exact_matches = 0
                total_char_acc = 0.0
                failed_images = 0

                # The cumulative count of correct answers for each bit
                pos_correct_counts = [0] * PLATE_LENGTH

                folder_start = time.perf_counter()

                prefix = f"📂 Processing [{folder:^10}]"
                print_progress(0, total_images, prefix=prefix)

                for i, f in enumerate(files, 1):
                    gt_text = get_ground_truth(f)
                    img_path = os.path.join(folder_path, f)

                    pred_text, status, error_msg = process_image(img_path)

                    is_exact_match = (gt_text == pred_text)
                    char_acc = calc_character_accuracy(gt_text, pred_text)
                    pos_matches = calc_positionwise_matches(
                        gt_text, pred_text, plate_len=PLATE_LENGTH
                    )

                    if status != "OK":
                        failed_images += 1

                    if is_exact_match:
                        exact_matches += 1

                    total_char_acc += char_acc

                    for idx in range(PLATE_LENGTH):
                        pos_correct_counts[idx] += pos_matches[idx]

                    det_writer.writerow({
                        'Image': f,
                        'Level': folder,
                        'Ground Truth': gt_text,
                        'OCR Output': pred_text,
                        'Exact Match': 'Yes' if is_exact_match else 'No',
                        'Char Accuracy': f"{char_acc * 100:.1f}%",
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

                summary_results.append({
                    'Level': folder,
                    'Exact Match (%)': (exact_matches / total_images) * 100,
                    'Char Accuracy (%)': (total_char_acc / total_images) * 100,
                    'Failed Images': failed_images,
                    'Time (s)': f"{folder_time:.2f}"
                })

                pos_row = {'Level': folder}
                for idx in range(PLATE_LENGTH):
                    pos_acc = (pos_correct_counts[idx] / total_images) * 100
                    pos_row[f'Position {idx + 1} (%)'] = f"{pos_acc:.2f}"
                position_results.append(pos_row)

        # ==========================================
        # 3. Output Summary Table
        # ==========================================
        if summary_results:
            print(f"\n📊 [{target.upper()}] Summary Report:")
            print(f"{'Level':<15} | {'Exact Match (%)':<16} | {'Char Accuracy (%)':<18} | {'Failed':<8} | {'Time':<8}")
            print("-" * 90)

            for res in summary_results:
                print(
                    f"{res['Level']:<15} | "
                    f"{res['Exact Match (%)']:.2f}%{'':<9} | "
                    f"{res['Char Accuracy (%)']:.2f}%{'':<9} | "
                    f"{res['Failed Images']:<8} | "
                    f"{res['Time (s)']:<8}"
                )

            with open(sum_csv, 'w', newline='', encoding='utf-8') as f_sum:
                sum_writer = csv.DictWriter(
                    f_sum,
                    fieldnames=[
                        'Level',
                        'Exact Match (%)',
                        'Char Accuracy (%)',
                        'Failed Images',
                        'Time (s)'
                    ]
                )
                sum_writer.writeheader()
                sum_writer.writerows(summary_results)

            print(f"\n💾 Summary data saved: {sum_csv}")

        # ==========================================
        # 4. Output position-wise accuracy
        # ==========================================
        if position_results:
            with open(pos_csv, 'w', newline='', encoding='utf-8') as f_pos:
                pos_writer = csv.DictWriter(
                    f_pos,
                    fieldnames=[
                        'Level',
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

            print(f"\n📍 [{target.upper()}] Position-wise Accuracy:")
            print(f"{'Level':<15} | {'P1':<8} | {'P2':<8} | {'P3':<8} | {'P4':<8} | {'P5':<8} | {'P6':<8} | {'P7':<8}")
            print("-" * 95)
            for row in position_results:
                print(
                    f"{row['Level']:<15} | "
                    f"{row['Position 1 (%)'] + '%':<8} | "
                    f"{row['Position 2 (%)'] + '%':<8} | "
                    f"{row['Position 3 (%)'] + '%':<8} | "
                    f"{row['Position 4 (%)'] + '%':<8} | "
                    f"{row['Position 5 (%)'] + '%':<8} | "
                    f"{row['Position 6 (%)'] + '%':<8} | "
                    f"{row['Position 7 (%)'] + '%':<8}"
                )

        print(f"💾 Detailed data saved in real-time: {det_csv}")

        target_time = time.perf_counter() - target_start
        print(f"\n⏱️ Major Category [{target.upper()}] Total Time: {format_seconds(target_time)}")

    overall_time = time.perf_counter() - overall_start
    print("\n" + "=" * 60)
    print(f"✅ All tests completed, Total Time: {format_seconds(overall_time)}")
    print("=" * 60)

if __name__ == "__main__":
    main()