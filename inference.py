"""
BrailleVision — Inference Script
---------------------------------
Usage:
    python inference.py --image path/to/braille.jpg

Example:
    python inference.py --image test.png
"""

import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model

# ── Config ──────────────────────────────────────────
MODEL_PATH = "model.keras"
IMG_SIZE   = 64
CLASSES    = [
    'A','B','C','D','E','F','G','H','I',
    'J','K','L','M','N','O','P','Q','R',
    'S','T','U','V','W','X','Y','Z',' '
]
CONFIDENCE_THRESHOLD = 0.85
# ────────────────────────────────────────────────────


def load_braille_model():
    print(f"[*] Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("[✓] Model loaded successfully")
    return model


def predict_character(model, gray_img):
    img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(img, verbose=0)
    idx  = np.argmax(pred)
    conf = float(np.max(pred))
    return CLASSES[idx], conf


def run_inference(image_path, model):
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[✗] Could not open image: {image_path}")
        return

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    # Preprocess
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    # Find dots
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 15:
            dots.append((x, y, w, h))

    if not dots:
        print("[!] No Braille dots detected. Try better lighting or a clearer image.")
        return

    # Group dots into character cells
    avg_dot_w     = int(np.mean([w for x, y, w, h in dots]))
    gap_threshold = avg_dot_w * 2
    dots          = sorted(dots, key=lambda d: d[0])

    cells = [[dots[0]]]
    for i in range(1, len(dots)):
        prev_right = cells[-1][-1][0] + cells[-1][-1][2]
        curr_left  = dots[i][0]
        if curr_left - prev_right > gap_threshold:
            cells.append([dots[i]])
        else:
            cells[-1].append(dots[i])

    # Predict each cell
    CELL_W   = int(img_h * 0.6)
    result   = ""
    print(f"\n[*] Detected {len(cells)} character(s):\n")

    for i, cell_dots in enumerate(cells):
        cx  = int(np.mean([d[0] + d[2] // 2 for d in cell_dots]))
        x1  = max(0, cx - CELL_W // 2)
        x2  = min(img_w, cx + CELL_W // 2)
        roi = gray[0:img_h, x1:x2]

        letter, conf = predict_character(model, roi)

        if conf >= CONFIDENCE_THRESHOLD:
            result += letter
            print(f"  Cell {i+1}: {letter}  (confidence: {conf*100:.1f}%)")
        else:
            print(f"  Cell {i+1}: ? — low confidence ({conf*100:.1f}%), skipped")

    print(f"\n{'─'*40}")
    print(f"  Detected Text : {result if result else 'Nothing detected'}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrailleVision Inference")
    parser.add_argument("--image", required=True, help="Path to Braille image")
    args = parser.parse_args()

    model = load_braille_model()
    run_inference(args.image, model)
