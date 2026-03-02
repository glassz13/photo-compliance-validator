# Photo Compliance Validator

A computer vision system that checks whether a photograph meets standard requirements before submission to official portals, government applications, or any process that demands a specific photo format.

🔗 **Live Demo** → [https://photo-compliance-validator.streamlit.app](https://photo-compliance-validator.streamlit.app)

---

## The Problem

Anyone who has filled an online application knows the frustration of getting a photo rejected. The reasons are almost always the same — blurry image, face too small, background not plain, photo too dark, head tilted. These rejections waste time and in many cases cost money when re-submissions involve fees or missed deadlines.

This project automates those checks. Upload your photo and get an instant report telling you exactly what passes and what does not — before you submit anywhere.

---

## Where This Is Useful

This is not limited to government exams. Any process that requires a compliant photograph can benefit from this:

- **Competitive exam registrations** — UPSC, SSC, JEE, NEET, GATE, IELTS, and similar portals
- **Passport and visa applications** — strict requirements around face size, background colour, and alignment
- **Job and internship applications** — company portals and HR systems requiring a professional headshot
- **University and college admissions** — both domestic and international institutions
- **Bank account opening and KYC** — photo submissions for identity verification
- **Driving licence and government ID applications** — transport and civil service portals
- **Employee onboarding** — verifying that submitted photos meet internal HR standards
- **Online exam proctoring** — checking the candidate photo before a session begins
- **Scholarship and fellowship forms** — most require a passport-size photo with specific formatting

Basically if a form asks for a photo, this tool can tell you whether yours will be accepted.

---

## Checks Performed

| Check | Method | Fails When |
|---|---|---|
| Image size | Dimension comparison | Too small or wrong aspect ratio |
| Blur | Laplacian variance | Image not sharp enough |
| Brightness | Grayscale mean | Too dark or overexposed |
| Background | Corner region std-dev | Busy or non-plain background |
| Face detection | MediaPipe | No face or multiple faces found |
| Face ratio | MediaPipe bounding box | Face too small or too large in frame |
| Head tilt | Eye landmark geometry | Tilt beyond allowed degrees |
| Obstruction | Facial landmark geometry | Mask, sunglasses, or cap detected |

---

## Tech Stack

- **OpenCV** — image loading and classical CV operations
- **MediaPipe** — face detection, face mesh, and landmark extraction
- **Streamlit** — web interface
- **Pillow** — image handling in the UI

---

## Setup

Python 3.11 is required. MediaPipe does not support Python 3.12 or 3.13 yet.
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure
```
photo_compliance/
├── validator.py      # all checks and core validation logic
├── app.py            # streamlit web interface
├── requirements.txt
├── README.md
└── resources/
```

---

## Tuning

All threshold values are at the top of `validator.py`. If a check is too strict or too lenient, change the number there — nothing else needs to be touched.

For example if good photos are being rejected for background, raise `BACKGROUND_STD_MAX` from `35` to `50`.

---

## Limitations

- Obstruction detection uses facial landmark geometry, not a trained classifier. It may not catch very subtle obstructions like a thin scarf or partial coverage.
- Background check uses corner region sampling, so it works best for standard portrait photos with the subject centred.
- The system is designed for validation only — it does not correct or edit photos.

---

## Future Scope

This project can be extended and applied in several interesting directions:

**Attendance verification**
The same face compliance checks can be adapted for automated attendance systems — verifying that a live photo taken at login matches a stored compliant photo and is not a printout or screen capture.

**API service**
Wrap the validator in a FastAPI backend so any application portal can call it programmatically before accepting a photo upload — rejecting non-compliant images at the source.

**Real-time camera validation**
Use a webcam feed instead of file upload. Give the user live feedback — "move closer", "too dark", "tilt your head" — before they even take the photo.

**Mobile app**
Port the logic to a mobile application using TensorFlow Lite or MediaPipe's mobile SDK so users can validate directly from their phone camera.

**Fine-tuned obstruction classifier**
Replace the current geometric obstruction check with a proper MobileNetV2 classifier trained on labelled data — mask, sunglasses, cap, clear — for more reliable and accurate detection.

**Document photo extraction**
Automatically crop and extract the photo region from a scanned ID card or form, then run compliance checks on the extracted photo.

**Multi-standard support**
Different portals have different rules. Add preset profiles — UPSC standard, passport standard, visa standard — so users can validate against the specific requirements of the portal they are submitting to.
