import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_ocr(image_path):
    # 1. read the image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        print("Image not found, please check the path!")
        return
        
    # 2. call Tesseract for recognition
    # --oem 3 represents using the default LSTM engine (exactly matching your friend's suggestion for Step 1)
    custom_config = r'--oem 3 --psm 6' 
    text = pytesseract.image_to_string(img, config=custom_config)
    
    print("=== OCR Recognition Result ===")
    print(text.strip())
    print("====================")

# Run the test
test_ocr('80-TNP-64.png')