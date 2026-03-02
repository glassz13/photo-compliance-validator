import cv2
import math
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False


# tweak these values if checks are too strict or too lenient
BLUR_THRESHOLD     = 60.0   # laplacian variance, below this = blurry
BRIGHTNESS_MIN     = 80     # too dark below this
BRIGHTNESS_MAX     = 220    # too bright/washed out above this
FACE_RATIO_MIN     = 0.15   # face should cover at least 15% of image
FACE_RATIO_MAX     = 0.80   # face shouldn't fill the entire frame
HEAD_TILT_MAX      = 10.0   # max allowed tilt in degrees
BACKGROUND_STD_MAX = 35.0   # std-dev of corner pixels, high = busy background
IMAGE_MIN_SIZE     = 150    # minimum width or height in pixels
FACE_CONFIDENCE    = 0.6    # mediapipe detection confidence threshold


def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1)
    passed = score >= BLUR_THRESHOLD
    return {
        "name": "Blur",
        "passed": passed,
        "score": str(score),
        "message": "Image is sharp." if passed else
                   f"Image is too blurry (score: {score}). Retake with steady hands in good lighting."
    }


def check_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = round(float(np.mean(gray)), 1)
    if score < BRIGHTNESS_MIN:
        return {"name": "Brightness", "passed": False, "score": str(score),
                "message": f"Photo is too dark (brightness: {score}). Take in better lighting."}
    if score > BRIGHTNESS_MAX:
        return {"name": "Brightness", "passed": False, "score": str(score),
                "message": f"Photo is overexposed (brightness: {score}). Avoid direct flash."}
    return {"name": "Brightness", "passed": True, "score": str(score),
            "message": "Brightness is acceptable."}


def check_size(image):
    h, w = image.shape[:2]
    ratio = round(w / h, 2)
    if w < IMAGE_MIN_SIZE or h < IMAGE_MIN_SIZE:
        return {"name": "Size", "passed": False, "score": f"{w}x{h}",
                "message": f"Image too small ({w}x{h}px). Minimum is {IMAGE_MIN_SIZE}x{IMAGE_MIN_SIZE}px."}
    if not (0.65 <= ratio <= 1.10):
        return {"name": "Size", "passed": False, "score": f"{w}x{h} ratio={ratio}",
                "message": f"Unusual aspect ratio ({ratio}). Photo looks too wide or too narrow."}
    return {"name": "Size", "passed": True, "score": f"{w}x{h}",
            "message": "Image dimensions are acceptable."}


def check_background(image):
    # sample the four corners of the image — those should be the background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ch, cw = max(1, int(h * 0.15)), max(1, int(w * 0.15))
    corners = np.concatenate([
        gray[:ch, :cw].flatten(),
        gray[:ch, -cw:].flatten(),
        gray[-ch:, :cw].flatten(),
        gray[-ch:, -cw:].flatten(),
    ])
    score = round(float(np.std(corners)), 1)
    passed = score <= BACKGROUND_STD_MAX
    return {
        "name": "Background",
        "passed": passed,
        "score": str(score),
        "message": "Background appears uniform." if passed else
                   f"Background is not plain (variation: {score}). Use a white or light solid background."
    }


def check_face(image):
    if not MEDIAPIPE_OK:
        return [{"name": "Face", "passed": False, "score": "N/A",
                 "message": "MediaPipe not installed. Run: pip install mediapipe==0.10.11"}]

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = []

    # face detection
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=FACE_CONFIDENCE
    ) as fd:
        det = fd.process(rgb)

    faces = det.detections if det.detections else []

    if len(faces) == 0:
        results.append({"name": "Face Detection", "passed": False, "score": "0 faces",
                        "message": "No face detected. Make sure your face is clearly visible and well-lit."})
        for name in ["Face Ratio", "Head Tilt", "Obstruction"]:
            results.append({"name": name, "passed": False, "score": "N/A",
                            "message": "Skipped — no face found."})
        return results

    if len(faces) > 1:
        results.append({"name": "Face Detection", "passed": False, "score": f"{len(faces)} faces",
                        "message": f"{len(faces)} faces detected. Photo must contain exactly one face."})
        return results

    results.append({"name": "Face Detection", "passed": True, "score": "1 face",
                    "message": "Exactly one face detected."})

    # face ratio check using bounding box area
    bb = faces[0].location_data.relative_bounding_box
    face_area = round(bb.width * bb.height, 4)
    if face_area < FACE_RATIO_MIN:
        results.append({"name": "Face Ratio", "passed": False, "score": f"{face_area*100:.1f}%",
                        "message": f"Face too small ({face_area*100:.1f}% of image). Move closer to the camera."})
    elif face_area > FACE_RATIO_MAX:
        results.append({"name": "Face Ratio", "passed": False, "score": f"{face_area*100:.1f}%",
                        "message": f"Face too large ({face_area*100:.1f}% of image). Move back from the camera."})
    else:
        results.append({"name": "Face Ratio", "passed": True, "score": f"{face_area*100:.1f}%",
                        "message": "Face size is acceptable."})

    # face mesh for tilt and obstruction
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=FACE_CONFIDENCE
    ) as fm:
        mesh = fm.process(rgb)

    if not mesh.multi_face_landmarks:
        results.append({"name": "Head Tilt", "passed": True, "score": "N/A",
                        "message": "Could not measure head tilt."})
        results.append({"name": "Obstruction", "passed": True, "score": "N/A",
                        "message": "Could not check for obstruction."})
        return results

    lm = mesh.multi_face_landmarks[0].landmark

    # head tilt — angle of the line connecting outer eye corners
    lx, ly = int(lm[33].x * w), int(lm[33].y * h)
    rx, ry = int(lm[263].x * w), int(lm[263].y * h)
    tilt = round(abs(math.degrees(math.atan2(ry - ly, rx - lx))), 1)
    if tilt > HEAD_TILT_MAX:
        results.append({"name": "Head Tilt", "passed": False, "score": f"{tilt}°",
                        "message": f"Head is tilted {tilt}° (max allowed: {HEAD_TILT_MAX}°). Keep your head straight."})
    else:
        results.append({"name": "Head Tilt", "passed": True, "score": f"{tilt}°",
                        "message": "Head alignment is acceptable."})

    # obstruction — checks where the mouth sits relative to nose and chin
    # on a clear face this ratio is consistently between 0.2 and 0.95
    nose_y = lm[1].y
    chin_y = lm[152].y
    mouth_y = lm[13].y
    ratio = round((mouth_y - nose_y) / (chin_y - nose_y), 2) if chin_y != nose_y else 0.0

    if 0.2 <= ratio <= 0.95:
        results.append({"name": "Obstruction", "passed": True, "score": str(ratio),
                        "message": "No facial obstruction detected."})
    else:
        results.append({"name": "Obstruction", "passed": False, "score": str(ratio),
                        "message": "Face appears partially covered. Remove mask, sunglasses, or cap."})

    return results


def validate(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        return {"overall": "FAIL", "checks": [],
                "reasons": [f"Could not load image: {image_path}"]}

    checks = []
    checks.append(check_size(image))
    checks.append(check_blur(image))
    checks.append(check_brightness(image))
    checks.append(check_background(image))
    checks.extend(check_face(image))

    reasons = [c["message"] for c in checks if not c["passed"]]
    return {
        "overall": "PASS" if not reasons else "FAIL",
        "checks": checks,
        "reasons": reasons,
    }
