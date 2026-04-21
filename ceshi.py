import cv2
import pytesseract
import os
import re

# ========= Configuration Area =========
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

TARGET_FOLDER = 'noise_2'     # Can be changed to noise_2 / noise_3
TARGET_IMAGE = None           # None = automatically take the first image in the folder; can also be written as '80-TNP-64.png'
CROP_RATIO = 0.08
EXTRA_LEFT_TRIM = 0          # try 0 / 5 / 10 / 15
OUTPUT_DIR = 'debug_noise_single'
# ==========================

OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def normalize_plate_text(text):
    text = clean_ocr_text(text)

    # Common case: reading an extra '1' / 'I' / 'O' at the beginning
    if len(text) == 8 and text[0] in {'1', 'I', 'O'}:
        text = text[1:]

    # If it's still too long, keep only the last 7 characters
    if len(text) > 7:
        text = text[-7:]

    # First two characters are usually digits, do some light correction
    chars = list(text)
    for i in range(min(2, len(chars))):
        if chars[i] == 'O':
            chars[i] = '0'
        elif chars[i] in {'I', 'L'}:
            chars[i] = '1'

    return ''.join(chars)


def preprocess_for_noise_folder(cropped_bgr, folder_name):
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    if folder_name == 'noise_1':
        processed = cv2.GaussianBlur(gray, (3, 3), 0)
    elif folder_name == 'noise_2':
        processed = cv2.GaussianBlur(gray, (5, 5), 0)
    elif folder_name == 'noise_3':
        processed = cv2.medianBlur(gray, 3)
    else:
        processed = gray

    return gray, processed


def main():
    input_dir = os.path.join('dataset', 'degraded', TARGET_FOLDER)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    if not files:
        print(f'❌ Folder is empty: {input_dir}')
        return

    if TARGET_IMAGE is not None:
        if TARGET_IMAGE not in files:
            print(f'❌ Specified image does not exist: {TARGET_IMAGE}')
            return
        filename = TARGET_IMAGE
    else:
        filename = files[0]

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f'❌ Failed to read image: {img_path}')
        return

    h, w = img.shape[:2]

    # 1. one left crop
    crop_start_x = int(w * CROP_RATIO)
    cropped = img[:, crop_start_x:]

    # 2. a little extra left trim (simulate plate not fully in the frame)
    if EXTRA_LEFT_TRIM > 0:
        cropped_trimmed = cropped[:, EXTRA_LEFT_TRIM:]
    else:
        cropped_trimmed = cropped.copy()

    # 3. pre-pocessing
    gray, processed = preprocess_for_noise_folder(cropped_trimmed, TARGET_FOLDER)

    # 4. OCR
    raw_ocr = pytesseract.image_to_string(processed, config=OCR_CONFIG)
    cleaned_ocr = clean_ocr_text(raw_ocr)
    normalized_ocr = normalize_plate_text(raw_ocr)

    # 5. keep the middle result
    base_name = os.path.splitext(filename)[0]

    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_01_original.png'), img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_02_crop7.png'), cropped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_03_crop7_trim.png'), cropped_trimmed)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_04_gray.png'), gray)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_05_processed.png'), processed)

    print('=' * 60)
    print(f'Image: {filename}')
    print(f'Folder: {TARGET_FOLDER}')
    print(f'Original size: {w} x {h}')
    print(f'Crop ratio: {CROP_RATIO}')
    print(f'crop_start_x: {crop_start_x}')
    print(f'Extra left trim: {EXTRA_LEFT_TRIM}')
    print(f'Cropped size: {cropped.shape[1]} x {cropped.shape[0]}')
    print(f'Trimmed size: {cropped_trimmed.shape[1]} x {cropped_trimmed.shape[0]}')
    print('-' * 60)
    print(f'Raw OCR       : {repr(raw_ocr)}')
    print(f'Cleaned OCR   : {cleaned_ocr}')
    print(f'Normalized OCR: {normalized_ocr}')
    print('-' * 60)
    print(f'saved to: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()