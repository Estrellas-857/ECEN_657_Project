import cv2
import numpy as np
import os
import random

# ============ Configuration Area ============
TEST_MODE = False  # [switch] True: test single image; False: process all images
TEST_IMAGE = '80-TNP-64.png'
# ===============================

def process_corrupt():
    input_dir = os.path.join('dataset', 'clean')
    base_out = os.path.join('dataset', 'degraded')
    for lvl in [1, 2, 3]: os.makedirs(os.path.join(base_out, f"corrupt_{lvl}"), exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    if TEST_MODE: files = [TEST_IMAGE] if TEST_IMAGE in files else files[:1]
    
    total = len(files)
    for i, f in enumerate(files, 1):
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None: continue
        r, c, _ = img.shape
        
        c1, c2, c3 = img.copy(), img.copy(), img.copy()
        # Level 1: edge corruption (simulate partial occlusion)
        cv2.rectangle(c1, (int(c*0.8), int(r*0.7)), (c, r), (0,0,0), -1)
        # Level 2: letter occlusion (black blocks at key positions)
        cv2.rectangle(c2, (int(c*0.4), int(r*0.3)), (int(c*0.5), int(r*0.6)), (0,0,0), -1)
        # Level 3: random scratches + dirt spots (simulate scratched license plate)
        for _ in range(12):
            p1 = (random.randint(0, c), random.randint(0, r))
            p2 = (random.randint(0, c), random.randint(0, r))
            cv2.line(c3, p1, p2, (0,0,0), random.randint(2, 5))
        for _ in range(20):
            cv2.circle(c3, (random.randint(0, c), random.randint(0, r)), random.randint(2, 8), (0,0,0), -1)
        
        cv2.imwrite(os.path.join(base_out, "corrupt_1", f), c1)
        cv2.imwrite(os.path.join(base_out, "corrupt_2", f), c2)
        cv2.imwrite(os.path.join(base_out, "corrupt_3", f), c3)
        
        if i % 10 == 0 or i == total:
            print(f" ✅ Progress: {i}/{total}")

if __name__ == "__main__":
    process_corrupt()