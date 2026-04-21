import cv2
import os

def process_blur():
    input_dir = os.path.join('dataset', 'clean')
    base_out = os.path.join('dataset', 'degraded')
    levels = [1, 2, 3]
    
    for lvl in levels:
        os.makedirs(os.path.join(base_out, f"blur_{lvl}"), exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    # 🌟 get the total image number
    total_images = len(files)
    print(f"🌀 Starting to generate Blur degradation ({total_images} images)...")

    # 🌟 use enumerate，i is the current number (starting from 1)，f is the folder name
    for i, f in enumerate(files, 1):
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None: continue

        # Level 1: light blur (simulate slight defocus)
        cv2.imwrite(os.path.join(base_out, "blur_1", f), cv2.GaussianBlur(img, (25, 25), 0))
        # Level 2: medium blur (simulate motion blur/unfocused)
        cv2.imwrite(os.path.join(base_out, "blur_2", f), cv2.GaussianBlur(img, (45, 45), 0))
        # Level 3: heavy blur (complete edge loss)
        cv2.imwrite(os.path.join(base_out, "blur_3", f), cv2.GaussianBlur(img, (65, 65), 0))

        # 🌟 use the number i to perform mathematical calculations
        if i % 10 == 0 or i == total_images:
            print(f"  ✅ Progress: {i}/{total_images} images processed")

    print("✅ Blur degradation completed!\n")

if __name__ == "__main__":
    process_blur()