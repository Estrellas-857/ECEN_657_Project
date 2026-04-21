import cv2
import pytesseract
import os
import re

# ⚠️ path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_ocr_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def main():
    # 🎯 pick randomly
    #test_image_name = '80-TNP-64.png' 
    test_image_name = '01-A-2808.png'  
    img_path = os.path.join('dataset', 'degraded', 'noise_3', test_image_name)
    
    if not os.path.exists(img_path):
        print(f"cannot find image: {img_path}，please change test_image_name")
        return
        
    print(f"testing the image: {img_path}")
    
    # 1. crop the left 8% of the image to focus on the license plate area, then apply our new preprocessing for noise
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    crop_start_x = int(width * 0.08)
    cropped_img = img[:, crop_start_x:]
    
    # 2. directly OCR for comparison before preprocessing
    config = r'--oem 3 --psm 6'
    raw_text = clean_ocr_text(pytesseract.image_to_string(cropped_img, config=config))
    print(f"❌ Before Preprocessing (Bare OCR) Result: [{raw_text}]")
    
    # 3. apply magic: median blur (Median Blur, Kernel size = 5)
    # the principle of median blur is: replace each pixel with the "median" of its surrounding pixels, which can perfectly remove isolated black and white noise points!
    denoised_img = cv2.medianBlur(cropped_img, 5)
    
    # 4. OCR again after preprocessing
    rescued_text = clean_ocr_text(pytesseract.image_to_string(denoised_img, config=config))
    print(f"✅ After Preprocessing (Median Blur) Result: [{rescued_text}]")
    
    # 5. display image comparison to see the effect
    cv2.imshow("1. Degraded (Noise 3)", cropped_img)
    cv2.imshow("2. Rescued (Median Blur)", denoised_img)
    print("\nPlease press any key in the pop-up image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()