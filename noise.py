import cv2
import numpy as np
import os

# ============ Configuration Section ============
TEST_MODE = False  # 【Switch】True: Test only one image; False: Run all
TEST_IMAGE = '80-TNP-64.png' # The image name you want to test
# ===============================

def apply_salt_and_pepper(image, amount):
    out = np.copy(image)
    # Salt noise (white points)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    # Pepper noise (black points)
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out

def process_noise():
    input_dir = os.path.join('dataset', 'clean')
    base_out = os.path.join('dataset', 'degraded')
    for lvl in [1, 2, 3]: os.makedirs(os.path.join(base_out, f"noise_{lvl}"), exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    if TEST_MODE:
        files = [TEST_IMAGE] if TEST_IMAGE in files else files[:1]
        print(f"🛠️ [Test Mode] Processing only image: {files[0]}")
    
    total = len(files)
    for i, f in enumerate(files, 1):
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None: continue
        row, col, ch = img.shape
        
        # Level 1 & 2: High-intensity Gaussian noise (from 50 to 100)
        n1 = np.clip(img + np.random.normal(0, 50, (row, col, ch)), 0, 255).astype(np.uint8)
        n2 = np.clip(img + np.random.normal(0, 100, (row, col, ch)), 0, 255).astype(np.uint8)
        # Level 3: Salt-and-Pepper Noise (bad pixel ratio 0.1)
        n3 = apply_salt_and_pepper(img, 0.1)
        
        cv2.imwrite(os.path.join(base_out, "noise_1", f), n1)
        cv2.imwrite(os.path.join(base_out, "noise_2", f), n2)
        cv2.imwrite(os.path.join(base_out, "noise_3", f), n3)
        
        if i % 10 == 0 or i == total:
            print(f" ✅ Progress: {i}/{total}")

if __name__ == "__main__":
    process_noise()