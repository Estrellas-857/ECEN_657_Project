import cv2
import numpy as np
import pytesseract
import os
import re
import time
import csv
from collections import Counter

# ==========================================
# 🎯 Ultimate Full-Volume Scoring Configuration
# ==========================================
# ⚠️ Ensure your Tesseract path is correct
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🌟 Switch the category you want to run! (Options: 'BLUR', 'NOISE', 'ILLUM', 'CORRUPT')
CURRENT_TASK = 'CORRUPT'  

MAX_TEST_IMAGES = 100000  # Set to 100000 for full-volume testing
CROP_RATIO = 0.08
PLATE_LENGTH = 7
OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Automatically map folders
TASK_FOLDERS = {
    'BLUR': ['blur_1', 'blur_2', 'blur_3'],
    'NOISE': ['noise_1', 'noise_2', 'noise_3'],
    'ILLUM': ['illum_1', 'illum_2', 'illum_3'],
    'CORRUPT': ['corrupt_1', 'corrupt_2', 'corrupt_3']
}
TARGET_FOLDERS = TASK_FOLDERS[CURRENT_TASK]

# ==========================================
# 1. Core Text Rule Engine and Helper Functions
# ==========================================
def get_ground_truth(filename):
    return re.sub(r'[^A-Z0-9]', '', os.path.splitext(filename)[0].upper())

def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def fixed_length_text(text, plate_len=7, pad_char='#'):
    return text[:plate_len].ljust(plate_len, pad_char)

def levenshtein_distance(gt, pred):
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]: dp[i][j] = dp[i - 1][j - 1]
            else: dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]

def calc_character_accuracy(gt, pred):
    if len(gt) == 0 and len(pred) == 0: return 1.0
    if len(gt) == 0 or len(pred) == 0: return 0.0
    dist = levenshtein_distance(gt, pred)
    max_len = max(len(gt), len(pred))
    return max(0.0, (max_len - dist) / max_len)

def positional_error_count(gt, pred, plate_len=7):
    gt_fixed = fixed_length_text(gt, plate_len)
    pred_fixed = fixed_length_text(pred, plate_len)
    return sum(1 for i in range(plate_len) if gt_fixed[i] != pred_fixed[i])

def plate_pattern_score(text):
    if len(text) != 7:
        return -999

    score = 0
    # P1, P2 应该是数字
    if text[0].isdigit():
        score += 2
    if text[1].isdigit():
        score += 2

    # P3 应该是字母
    if text[2].isalpha():
        score += 2

    # P4, P5 不强制（字母或数字都行）
    if text[3].isalnum():
        score += 1
    if text[4].isalnum():
        score += 1

    # P6, P7 应该是数字
    if text[5].isdigit():
        score += 2
    if text[6].isdigit():
        score += 2

    return score

def normalize_plate_text_v3(raw_text):
    text = clean_ocr_text(raw_text)

    # 如果 OCR 输出太长，选最符合牌照规则的 7 位窗口
    if len(text) > 7:
        candidates = [text[i:i+7] for i in range(len(text) - 6)]
        text = max(candidates, key=plate_pattern_score)

    if len(text) != 7:
        return text

    chars = list(text)

    # P1, P2: 数字位
    for i in [0, 1]:
        if chars[i] == 'O':
            chars[i] = '0'
        elif chars[i] in {'I', 'L'}:
            chars[i] = '1'

    # P3: 字母位
    if chars[2] == '0':
        chars[2] = 'O'
    elif chars[2] == '1':
        chars[2] = 'I'

    # P4, P5: 不强制，不改

    # P6, P7: 数字位
    for i in [5, 6]:
        if chars[i] == 'O':
            chars[i] = '0'
        elif chars[i] in {'I', 'L'}:
            chars[i] = '1'

    return ''.join(chars)

# ==========================================
# 2. 🌟 终极预处理路由引擎
# ==========================================
def apply_preprocessing(cropped_bgr, task, folder_name):
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    
    if task == 'BLUR':
        # 放大 + CLAHE (超分辨率思想)
        enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(enlarged)
        
    elif task == 'NOISE':
        if folder_name == 'noise_1':
            denoised = cv2.bilateralFilter(gray, 5, 50, 50)
            _, processed = cv2.threshold(
                denoised, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return processed

        elif folder_name == 'noise_2':
            denoised = cv2.bilateralFilter(gray, 7, 75, 75)
            _, processed = cv2.threshold(
                denoised, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return processed

        else:  # noise_3
            return cv2.medianBlur(gray, 3)
            
    elif task == 'ILLUM':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        return cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        
    elif task == 'CORRUPT':
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if folder_name == 'corrupt_3':
            kernel = np.ones((5, 5), np.uint8)
            processed_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
        elif folder_name == 'corrupt_2':
            kernel = np.ones((3, 3), np.uint8)
            processed_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
        else:
            processed_inv = binary_inv
        return cv2.bitwise_not(processed_inv)
        
    return gray

def print_progress(iteration, total, prefix='', length=40):
    if total == 0: return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total})', end='\r')
    if iteration == total: print()

# ==========================================
# 3. 主流程与 CSV 落地
# ==========================================
def main():
    base_dir = 'dataset'
    overall_start = time.time()

    print(f"🚀 开始执行 [{CURRENT_TASK}] 大类 10万级全量跑分...")
    print(f"🧪 最大测试数量 = {MAX_TEST_IMAGES} 张/组\n")

    summary_rows = []

    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(base_dir, 'degraded', folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ 跳过不存在的文件夹: {folder_path}")
            continue

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))])[:MAX_TEST_IMAGES]
        if not files: continue

        total_images = len(files)
        exact_matches = 0
        total_char_acc = 0.0

        error_count_dist = Counter()
        edit_distance_dist = Counter()
        confusion_counter = Counter()
        position_confusions = [Counter() for _ in range(PLATE_LENGTH)]
        details_rows = []

        folder_start = time.time()
        prefix = f"🔥 狂飙中 [{folder:^10}]"
        print_progress(0, total_images, prefix=prefix)

        for i, f in enumerate(files, 1):
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path)
            if img is None: continue

            # 1. 裁切与预处理
            h, w = img.shape[:2]
            cropped = img[:, int(w * CROP_RATIO):]
            processed = apply_preprocessing(cropped, CURRENT_TASK, folder)

            # 2. OCR 与文本规则后处理
            raw_text = pytesseract.image_to_string(processed, config=OCR_CONFIG)
            pred_text = normalize_plate_text_v3(raw_text)
            gt_text = get_ground_truth(f)

            # 3. 成绩计算
            is_exact_match = (gt_text == pred_text)
            char_acc = calc_character_accuracy(gt_text, pred_text)
            edit_dist = levenshtein_distance(gt_text, pred_text)
            pos_errs = positional_error_count(gt_text, pred_text, PLATE_LENGTH)

            if is_exact_match: exact_matches += 1
            total_char_acc += char_acc
            error_count_dist[pos_errs] += 1
            edit_distance_dist[edit_dist] += 1

            # 4. 位级与详情记录
            gt_fixed = fixed_length_text(gt_text, PLATE_LENGTH)
            pred_fixed = fixed_length_text(pred_text, PLATE_LENGTH)

            row = {
                'Image': f, 'Ground Truth': gt_text, 'OCR Output': pred_text,
                'Exact Match': 'Yes' if is_exact_match else 'No',
                'Edit Distance': edit_dist, 'Positional Errors': pos_errs,
                'Char Accuracy (%)': f"{char_acc * 100:.2f}"
            }

            for pos in range(PLATE_LENGTH):
                gt_c, pred_c = gt_fixed[pos], pred_fixed[pos]
                row[f'GT_P{pos+1}'], row[f'Pred_P{pos+1}'], row[f'Match_P{pos+1}'] = gt_c, pred_c, 1 if gt_c == pred_c else 0
                if gt_c != pred_c:
                    confusion_counter[(gt_c, pred_c)] += 1
                    position_confusions[pos][(gt_c, pred_c)] += 1

            details_rows.append(row)
            # 每 100 张更新一下进度条，降低终端刷新带来的性能损耗
            if i % 100 == 0 or i == total_images:
                print_progress(i, total_images, prefix=prefix)

        # ==========================================
        # 💾 CSV 文件落地保存
        # ==========================================
        folder_time = time.time() - folder_start
        exact_pct = (exact_matches / total_images) * 100
        char_acc_pct = (total_char_acc / total_images) * 100

        summary_rows.append({
            'Task': CURRENT_TASK, 'Level': folder, 'Num Images': total_images,
            'Exact Match (%)': f"{exact_pct:.2f}", 'Char Accuracy (%)': f"{char_acc_pct:.2f}",
            'Time (s)': f"{folder_time:.2f}"
        })

        # 写入 5 个维度的 CSV
        details_csv = f'final_eval_{folder}_details.csv'
        with open(details_csv, 'w', newline='', encoding='utf-8') as f_csv:
            fieldnames = ['Image', 'Ground Truth', 'OCR Output', 'Exact Match', 'Edit Distance', 'Positional Errors', 'Char Accuracy (%)']
            for pos in range(PLATE_LENGTH): fieldnames += [f'GT_P{pos+1}', f'Pred_P{pos+1}', f'Match_P{pos+1}']
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(details_rows)

        with open(f'final_eval_{folder}_error_dist.csv', 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['Num Wrong Positions', 'Count', 'Percentage (%)'])
            writer.writeheader()
            for k in range(PLATE_LENGTH + 1):
                writer.writerow({'Num Wrong Positions': k, 'Count': error_count_dist[k], 'Percentage (%)': f"{(error_count_dist[k] / total_images) * 100:.2f}"})

        with open(f'final_eval_{folder}_edit_dist.csv', 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['Edit Distance', 'Count', 'Percentage (%)'])
            writer.writeheader()
            for k in sorted(edit_distance_dist.keys()):
                writer.writerow({'Edit Distance': k, 'Count': edit_distance_dist[k], 'Percentage (%)': f"{(edit_distance_dist[k] / total_images) * 100:.2f}"})

        with open(f'final_eval_{folder}_confusions.csv', 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['GT Char', 'Pred Char', 'Count'])
            writer.writeheader()
            for (gt_c, pred_c), count in confusion_counter.most_common():
                writer.writerow({'GT Char': gt_c, 'Pred Char': pred_c, 'Count': count})

        with open(f'final_eval_{folder}_pos_confusions.csv', 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['Position', 'GT Char', 'Pred Char', 'Count'])
            writer.writeheader()
            for pos in range(PLATE_LENGTH):
                for (gt_c, pred_c), count in position_confusions[pos].most_common():
                    writer.writerow({'Position': pos + 1, 'GT Char': gt_c, 'Pred Char': pred_c, 'Count': count})

        print(f"📊 {folder} 跑分完毕! Exact Match: {exact_pct:.2f}% | 耗时: {folder_time:.2f}s")
        print(f"💾 数据已存入 final_eval_{folder}_*.csv 系列文件\n")

    # 大类总汇总 CSV
    summary_csv = f'final_eval_{CURRENT_TASK}_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=['Task', 'Level', 'Num Images', 'Exact Match (%)', 'Char Accuracy (%)', 'Time (s)'])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 70)
    print(f"✅ [{CURRENT_TASK}] 全量分析完美落幕！汇总表: {summary_csv}")
    print(f"⏱️ 总耗时: {time.time() - overall_start:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()