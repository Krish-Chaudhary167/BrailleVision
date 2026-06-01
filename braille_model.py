import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

IMG_SIZE = 64
CLASSES = [
    'a','b','c','d','e','f','g','h','i',
    'j','k','l','m','n','o','p','q','r',
    's','t','u','v','w','x','y','z',' '
]

_model = None

def get_model():
    global _model
    if _model is None:
        _model = keras_load_model('model.keras')
        print("[✓] CNN model loaded")
    return _model

def run_inference(frame_bgr):
    """
    Takes BGR image, returns list of dicts same format as YOLO:
    [{label, conf, box: (x1,y1,x2,y2)}, ...]
    """
    model = get_model()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 15:
            dots.append((x, y, w, h))

    if not dots:
        return []

    avg_dot_w = int(np.mean([w for x,y,w,h in dots]))
    gap_threshold = avg_dot_w * 2
    dots = sorted(dots, key=lambda d: d[0])

    cells = [[dots[0]]]
    for i in range(1, len(dots)):
        prev_right = cells[-1][-1][0] + cells[-1][-1][2]
        curr_left = dots[i][0]
        if curr_left - prev_right > gap_threshold:
            cells.append([dots[i]])
        else:
            cells[-1].append(dots[i])

    CELL_W = int(img_h * 0.6)
    detections = []

    for cell_dots in cells:
        cx = int(np.mean([d[0] + d[2]//2 for d in cell_dots]))
        x1 = max(0, cx - CELL_W//2)
        x2 = min(img_w, cx + CELL_W//2)
        y1 = 0
        y2 = img_h

        char_img = gray[y1:y2, x1:x2]
        char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE))
        char_img = char_img / 255.0
        char_img = char_img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        pred = model.predict(char_img, verbose=0)
        idx = np.argmax(pred)
        conf = float(np.max(pred))
        if conf < 0.85:
           continue

        detections.append({
            "label": CLASSES[idx],
            "conf": conf,
            "box": (x1, y1, x2, y2)
        })

    return detections