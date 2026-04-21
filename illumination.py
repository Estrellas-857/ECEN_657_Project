import cv2
import numpy as np
import os

# ============ Configuration ============
TEST_MODE = False  # [switch] True: test single image; False: run full-volume
TEST_IMAGE = '80-TNP-64.png'
# ===============================

def process_illum():
    input_dir = os.path.join('dataset', 'clean')
    base_out = os.path.join('dataset', 'degraded')
    for lvl in [1, 2, 3]: os.makedirs(os.path.join(base_out, f"illum_{lvl}"), exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    if TEST_MODE: files = [TEST_IMAGE] if TEST_IMAGE in files else files[:1]
    
    total = len(files)
    for i, f in enumerate(files, 1):
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None: continue
        r, c, _ = img.shape

        # Level 1: Gradient Shadow (simulate uneven lighting)
        mask1 = np.tile(np.linspace(0.2, 1.0, c), (r, 1))[..., np.newaxis]
        i1 = np.clip(img * mask1, 0, 255).astype(np.uint8)
        # Level 2: Wave Pattern (Stripe Shadow)
        wave = (np.sin(np.linspace(0, c/15, c)) + 1) / 2
        mask2 = np.tile(0.4 + 0.6 * wave, (r, 1))[..., np.newaxis]
        i2 = np.clip(img * mask2, 0, 255).astype(np.uint8)
        # Level 3: Extreme Low Contrast (Entirely Gray Image)
        i3 = cv2.convertScaleAbs(img, alpha=0.25, beta=110)
        
        cv2.imwrite(os.path.join(base_out, "illum_1", f), i1)
        cv2.imwrite(os.path.join(base_out, "illum_2", f), i2)
        cv2.imwrite(os.path.join(base_out, "illum_3", f), i3)
        
        if i % 10 == 0 or i == total:
            print(f" ✅ Progress: {i}/{total}")

if __name__ == "__main__":
    process_illum()