# ECEN_657_Project: Enhancing License Plate OCR Under Challenging Imaging Conditions

## Overview

This project studies how different image degradations affect license plate OCR, and whether task-specific preprocessing pipelines can improve recognition performance.

The core question is simple:

**Does classical image preprocessing always help OCR, or does its usefulness depend on the degradation type?**

To answer this, I evaluated a Tesseract LSTM-based OCR pipeline on a large degraded license plate dataset and compared:

1. **OCR baseline without preprocessing**
2. **Task-specific rescue pipelines with preprocessing**
3. **Rule-based postprocessing based on plate-format knowledge**

The project shows that there is **no universal preprocessing silver bullet**. Some degradations can be rescued effectively, while others are already handled well by the OCR baseline or remain difficult even after preprocessing.

---

## Main Takeaway

The main conclusion of this project is:

- **Preprocessing is not universally beneficial**
- **Its effectiveness strongly depends on the degradation family**
- **Rule-based correction is critical for structured OCR confusions**
- **Severe physical corruption remains difficult for single-frame classical methods**

In other words, a one-size-fits-all preprocessing pipeline is not ideal. A more practical ALPR workflow should use:

- degradation diagnosis
- selective routing
- family-specific rescue methods
- format-aware postprocessing

---

## Dataset and Degradation Families

The dataset contains synthetic license plate images with multiple degradation families.  
Each family contains three subtypes / levels.

### Degradation families

- **Blur**
  - `blur_1`
  - `blur_2`
  - `blur_3`

- **Noise**
  - `noise_1`
  - `noise_2`
  - `noise_3`

- **Illumination**
  - `illum_1`
  - `illum_2`
  - `illum_3`

- **Corruption**
  - `corrupt_1`
  - `corrupt_2`
  - `corrupt_3`

---

## Plate Format Assumption

The project also uses lightweight plate-format rules for postprocessing.

The assumed 7-character plate pattern is:

- **P1, P2:** digits
- **P3:** letter
- **P4, P5:** alphanumeric
- **P6, P7:** digits

This rule is used only in the rule-based postprocessing stage, not in the raw OCR baseline.

---

## OCR Baseline

The OCR baseline uses:

- **Tesseract OCR**
- **LSTM-based recognition backend**
- `--oem 3`
- `--psm 7`
- alphanumeric whitelist
- fixed left crop to remove the blue strip region

The baseline already shows nontrivial robustness on some degradations, which is why preprocessing sometimes helps, sometimes hurts.

---

## Rescue Pipelines

This project evaluates **task-specific rescue pipelines**, not a universal preprocessing method.

### Blur rescue
- bicubic enlargement
- CLAHE contrast enhancement

### Noise rescue
- bilateral filtering for Gaussian-like noise
- median filtering for impulse-like noise
- rule-based postprocessing for OCR confusions such as `0/O` and `1/I`

### Illumination rescue
- CLAHE
- adaptive thresholding

### Corruption rescue
- morphology-based repair attempts
- OCR + rule-based correction
- severe corruption remains challenging

---

## Key Findings

### 1. Blur
Blur rescue is **mixed**.

It helps substantially for some blur subtypes, especially moderate and stronger blur, but it is not always beneficial for mild blur. This supports the idea that preprocessing can become **over-processing** when the OCR baseline is already strong.

### 2. Noise
Noise rescue is one of the strongest parts of the project.

Classical denoising alone is not the whole story. The best improvements come from combining:

- denoising
- OCR
- plate-format-aware rule correction

This means the dominant failure mode under noise is often **structured symbol confusion**, not total recognition collapse.

### 3. Illumination
Illumination rescue is **pattern-dependent**.

Some illumination patterns benefit from adaptive thresholding, while others do not. This again shows that preprocessing should be selective rather than universal.

### 4. Corruption
Corruption is the hardest family.

Light corruption may still be partially recoverable, but severe physical damage or heavy occlusion remains difficult for single-frame classical vision methods.

---

## Evaluation Metrics

The project uses multiple complementary metrics:

- **Exact Match Accuracy**
  - the full 7-character plate must be correct

- **Character Accuracy**
  - based on normalized edit distance

- **Edit Distance Distribution**
  - measures how far predictions are from the ground truth

- **Positional Error Analysis**
  - measures how many character positions are wrong

- **Confusion Analysis**
  - tracks frequent OCR confusions such as `0 -> O`, `I -> 1`, etc.

Using both exact-match and character-level metrics is important, because some rescue pipelines reduce catastrophic OCR failures into near-miss errors.

---


## How to Run

### 1. Configure Tesseract path

Update the Tesseract executable path in the script:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

### 2. Select the task to evaluate

In final_eval.py, set:

```python
CURRENT_TASK = 'BLUR'

Available options:

BLUR
NOISE
ILLUM
CORRUPT

### 3. Run evaluation
```python
python final_eval.py

### 4. Outputs

The script generates:

summary CSV
detail CSV
edit distance distribution
positional error distribution
confusion statistics

These outputs are used for both quantitative analysis and slide/report generation.

## Current Project Message

This project is not trying to find one preprocessing method that works for everything.

Instead, it argues that:

some images should be passed directly to OCR
some images benefit from task-specific rescue
some OCR errors are best corrected by rule-based postprocessing
some severe corruptions remain beyond the capability of simple classical preprocessing

This is the main contribution of the project.

## Limitations
The OCR engine is still based on Tesseract, not a modern end-to-end deep ALPR model
The dataset is synthetic rather than fully real-world
Some rescue pipelines are evaluated at the degradation-family level rather than per-instance adaptive routing
Severe corruption is still difficult to handle with single-frame classical methods

## Future Work

Possible future directions include:

degradation diagnosis before routing
OCR engine comparison
per-instance adaptive rescue selection
multi-frame temporal fusion
generative restoration / inpainting for severe corruption
comparison against end-to-end ALPR models
