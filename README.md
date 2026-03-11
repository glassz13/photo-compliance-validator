# Face Compliance Validator

> Deep learning pipeline for automated ID and passport photo compliance checking.
> Trained on CelebA (15,000 images) using transfer learning with MobileNetV2.
> Deployed as an interactive Streamlit app.

Live App: [paste your URL here](#)

---

## Demo

![App Demo](assets/demo.gif)

**Compliant Photo**
![Pass](assets/pass.png)

**Non-Compliant Photo**
![Fail](assets/fail.png)

---

## Project Structure
```
face-validator/
├── train.py                 ← training pipeline (run on Google Colab)
├── validator.py             ← model loading and inference
├── app.py                   ← Streamlit app
├── face_attributes.pt       ← trained model weights
├── requirements.txt
├── sample_photos/           ← test photos to try the app
│   ├── compliant.jpg
│   ├── glasses.jpg
│   ├── hat.jpg
│   └── smiling.jpg
└── assets/
    ├── demo.gif
    ├── pass.png
    └── fail.png
```

---

## What It Checks

| Check | Compliance Rule | Accuracy |
|-------|----------------|----------|
| Eyeglasses | No glasses allowed | 97% |
| Headwear | No hat or headwear | 97% |
| Eyes Open | Eyes must be fully open | 70% |
| Expression | Neutral expression required | 75% |
| Gender | Detected as metadata | 90% |

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | CelebA — Large-scale Face Attributes Dataset |
| Total available | 202,599 face images |
| Used for training | 15,000 images (sampled) |
| Split | 12,000 train / 1,500 val / 1,500 test |
| Attributes available | 40 binary labels |
| Attributes used | 5 |

CelebA attribute values are -1/1 — converted to 0/1 before training. Class imbalance handled via `pos_weight` in BCEWithLogitsLoss.

---

## Model

**Architecture:** MobileNetV2 (pretrained on ImageNet) + custom classification head
```
MobileNetV2 base (frozen)
    → Dropout(0.3)
    → Linear(1280, 128)
    → ReLU
    → Dropout(0.2)
    → Linear(128, 5)
    → Sigmoid (per attribute)
```

**Why MobileNetV2:**
Lightweight, fast inference, strong ImageNet features that transfer well to face attribute detection. Freezing the base and training only the head keeps training fast and prevents overfitting on 15k images.

**Loss:** BCEWithLogitsLoss with pos_weight for imbalanced attributes
**Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
**Scheduler:** ReduceLROnPlateau (patience=2, factor=0.5)
**Epochs:** 15 | **Batch size:** 64 | **Image size:** 128×128
**Trainable parameters:** 164,613

---

## Results

### Per-Attribute Performance

| Attribute | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| Eyeglasses | 97% | 0.85 | 0.64 | 0.73 |
| Wearing_Hat | 97% | 0.68 | 0.82 | 0.74 |
| Smiling | 75% | 0.75 | 0.75 | 0.75 |
| Male | 90% | 0.90 | 0.90 | 0.90 |
| Narrow_Eyes | 70% | 0.23 | 0.67 | 0.34 |

Overall test accuracy: **85.79%**

---

## Model Analysis

Eyeglasses and Wearing_Hat both hit 97% accuracy — strong results driven by visually distinct features that MobileNetV2's pretrained filters detect reliably. Smiling at 75% and Male at 90% perform well given balanced class distributions in the dataset.

Narrow_Eyes is the weakest attribute at 70% accuracy with low precision. Two factors explain this — only 11.5% of CelebA images are labeled narrow eyes creating a 9:1 class imbalance, and the annotation itself is subjective in CelebA where annotators disagreed on borderline cases. Applying pos_weight=8.0 improved recall from 1% to 67% but precision remains low at 0.23. In practice this means the model catches most narrow-eye cases but with false positives. For a production system this attribute would benefit from a dedicated eye-openness detector or a cleaner labeled dataset.

---

## Limitations

- Trained on celebrity face images — CelebA skews toward certain demographics and lighting conditions. Performance may vary on more diverse real-world photos.
- Narrow_Eyes precision is low due to class imbalance and label noise in CelebA — see model analysis above.
- No geometric checks — head tilt, face centering, and background uniformity are not covered by this model. These are better handled by rule-based geometry checks.
- 15k training samples is sufficient for transfer learning but a larger sample would improve generalization, especially for minority classes.

---

## Extending to a Full Production Pipeline

This project covers the DL attribute classification component. A complete ID photo compliance system would combine this with:

- **Face detection** — MediaPipe or RetinaFace to confirm exactly one face is present and extract the face region before passing to this model
- **Geometric checks** — OpenCV for head tilt angle, face centering, and image size validation
- **Background check** — a simple CNN classifier trained on plain vs busy backgrounds
- **Blur and brightness** — Laplacian variance and mean pixel intensity checks via OpenCV
- **Domain-specific fine-tuning** — for specific use cases like UPSC forms or passport applications, collecting labeled compliant/non-compliant photos from that domain and fine-tuning this model would significantly improve accuracy over the general CelebA-trained version

The modular design of `validator.py` makes it straightforward to extend — additional checks return the same dictionary structure and plug directly into the app.

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the app:**
```bash
streamlit run app.py
```

**Try with sample photos:**
```
sample_photos/compliant.jpg    ← should pass all checks
sample_photos/glasses.jpg      ← should fail eyeglasses check
sample_photos/hat.jpg          ← should fail headwear check
sample_photos/smiling.jpg      ← should fail expression check
```

**To retrain the model:**
Open `train.py` in Google Colab, connect T4 GPU, update the dataset path and run.

---

## Tech Stack

- Python 3.x
- PyTorch + torchvision
- Streamlit
- Pillow
- scikit-learn

---

## requirements.txt
```
torch
torchvision
streamlit
Pillow
scikit-learn
```
