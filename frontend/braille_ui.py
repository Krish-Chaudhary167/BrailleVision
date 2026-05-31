"""
Braille AI - Real-Time Handwritten Braille Detection & Speech
=============================================================
Requirements:
    pip install gradio ultralytics opencv-python gtts numpy Pillow

Usage:
    python braille_ui.py

Replace MODEL_PATH with your trained YOLO model path.
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import os
import tempfile

# ─────────────────────────────────────────────
# CONFIG — change these to match your setup
# ─────────────────────────────────────────────
MODEL_PATH = "best.pt"          # Path to your trained YOLOv8 weights
CONFIDENCE_THRESHOLD = 0.45     # Min confidence to accept a detection
TTS_LANG = "en"                 # Language for gTTS

# Colour palette per class (matches your reference image style)
CLASS_COLORS = {
    "a": (255, 100, 100),   "b": (100, 200, 255),
    "c": (255, 80,  80),    "d": (255, 180,  50),
    "e": (80,  255, 150),   "f": (200, 100, 255),
    "g": (255, 255,  80),   "h": (100, 255, 200),
    "i": (80,  255, 255),   "j": (255, 150, 255),
    "k": (150, 255,  80),   "l": (255, 200, 100),
    "m": (100, 100, 255),   "n": (80,  200, 100),
    "o": (255, 130,  50),   "p": (200, 255, 100),
    "q": (255,  80, 200),   "r": (255, 255,  50),
    "s": (150,  80, 255),   "t": (50,  200, 255),
    "u": (255, 100, 180),   "v": (100, 255, 130),
    "w": (255, 200,  50),   "x": (180, 255,  80),
    "y": (255,  50, 150),   "z": (50,  255, 200),
    " ": (180, 180, 180),
}
DEFAULT_COLOR = (220, 220, 220)

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"[✓] Model loaded from {MODEL_PATH}")
        else:
            print(f"[!] Model not found at {MODEL_PATH}. Running in DEMO mode.")
    except ImportError:
        print("[!] ultralytics not installed. Running in DEMO mode.")

load_model()

# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────

def run_inference(frame_bgr):
    """
    Run YOLO inference on a BGR frame.
    Returns list of dicts: [{label, conf, box: (x1,y1,x2,y2)}, ...]
    Sorted left-to-right by x1.
    """
    if model is None:
        return _demo_detections(frame_bgr)

    results = model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2)})

    detections.sort(key=lambda d: d["box"][0])   # left → right order
    return detections


def _demo_detections(frame_bgr):
    """Fake detections for demo / model-not-found mode."""
    h, w = frame_bgr.shape[:2]
    chars  = list("braille")
    confs  = [0.82, 0.71, 0.88, 0.65, 0.79, 0.84, 0.76]
    gap    = w // (len(chars) + 1)
    cy     = h // 2
    bh, bw = 70, 45
    dets = []
    for i, (ch, cf) in enumerate(zip(chars, confs)):
        cx = gap * (i + 1)
        dets.append({
            "label": ch,
            "conf":  cf,
            "box":   (cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2)
        })
    return dets


def draw_detections(frame_bgr, detections):
    """
    Draw YOLO-style coloured bounding boxes with label + confidence.
    Mimics the colourful style in the reference image.
    """
    out = frame_bgr.copy()
    for det in detections:
        lbl   = det["label"]
        conf  = det["conf"]
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(lbl.lower(), DEFAULT_COLOR)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label banner above box
        tag    = f"{lbl}  {int(conf*100)}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        banner_y = max(y1 - th - 8, 0)
        cv2.rectangle(out,
                      (x1, banner_y),
                      (x1 + tw + 6, banner_y + th + 6),
                      color, -1)
        cv2.putText(out, tag,
                    (x1 + 3, banner_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out


def detections_to_text(detections):
    return "".join(d["label"] for d in detections)


def text_to_speech(text):
    if not text.strip():
        return None
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=TTS_LANG)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None


# ─────────────────────────────────────────────
# GRADIO CALLBACK
# ─────────────────────────────────────────────

def process_frame(frame):
    """Called on every webcam frame by Gradio streaming."""
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank, "", 0, ""

    # Gradio gives numpy RGB → convert to BGR for OpenCV/YOLO
    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    detections = run_inference(frame_bgr)
    annotated_bgr = draw_detections(frame_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    text       = detections_to_text(detections)
    char_count = len(detections)

    # Build a pretty stats string
    stats = "  |  ".join(
        f"{d['label'].upper()} {int(d['conf']*100)}%"
        for d in detections
    ) or "No braille detected"

    return annotated_rgb, text, char_count, stats


def speak_text(text):
    """TTS button handler."""
    audio_path = text_to_speech(text)
    return audio_path


# ─────────────────────────────────────────────
# CUSTOM CSS  — dark, high-contrast, techy
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;500;700&display=swap');

:root {
    --bg:      #0d0f14;
    --panel:   #13161e;
    --border:  #1e2330;
    --accent:  #00e5a0;
    --accent2: #5b8fff;
    --warn:    #ffcc44;
    --text:    #e8eaf0;
    --muted:   #6b7280;
    --radius:  12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
.braille-header {
    background: linear-gradient(135deg, #0d0f14 0%, #13161e 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 32px 20px;
    margin-bottom: 24px;
}
.braille-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: var(--accent);
    margin: 0 0 4px 0;
    letter-spacing: -1px;
}
.braille-header p {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: var(--accent);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    margin-left: 10px;
    vertical-align: middle;
    letter-spacing: 1px;
}

/* Panels */
.panel-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Gradio component overrides */
.gradio-container label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
}
.gradio-container input, .gradio-container textarea {
    background: #1a1d27 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.2rem !important;
    letter-spacing: 3px !important;
}
button.primary {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    letter-spacing: 1px !important;
}
button.secondary {
    background: transparent !important;
    border: 1px solid var(--accent2) !important;
    color: var(--accent2) !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 8px !important;
}
button:hover { opacity: 0.85 !important; }

/* Stats bar */
#stats-bar {
    background: #0a0c12;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--warn);
    letter-spacing: 1px;
    min-height: 36px;
}

/* Character count */
#char-count {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: var(--accent2);
    text-align: center;
}

/* Image panel */
.webcam-panel {
    border: 2px solid var(--accent);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: 0 0 24px rgba(0, 229, 160, 0.08);
}

/* Audio */
audio {
    width: 100%;
    margin-top: 8px;
    filter: invert(1) hue-rotate(180deg);
}

/* Footer */
.footer-bar {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    padding: 20px;
    border-top: 1px solid var(--border);
    margin-top: 24px;
    letter-spacing: 1px;
}
"""

# ─────────────────────────────────────────────
# BUILD UI
# ─────────────────────────────────────────────

with gr.Blocks(css=CSS, title="Braille AI — Detect & Speak") as app:

    # ── Header ──────────────────────────────
    gr.HTML("""
    <div class="braille-header">
        <h1>⠃⠗⠁⠊⠇⠇⠑  AI <span class="badge">LIVE</span></h1>
        <p>Real-time handwritten braille detection → text → speech &nbsp;|&nbsp; YOLOv8 + OpenCV + gTTS</p>
    </div>
    """)

    # ── Main layout ─────────────────────────
    with gr.Row(equal_height=True):

        # Left — webcam + annotated output
        with gr.Column(scale=3):
            gr.HTML('<div class="section-label">📷 Live Camera Feed</div>')
            webcam_in = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Point camera at braille",
                elem_classes=["webcam-panel"],
                height=360,
                show_label=False,
            )
            gr.HTML('<div class="section-label" style="margin-top:16px">🔍 Detection Output</div>')
            annotated_out = gr.Image(
                label="Annotated Frame",
                show_label=False,
                height=360,
                elem_classes=["webcam-panel"],
            )

        # Right — results panel
        with gr.Column(scale=2):
            gr.HTML('<div class="section-label">📊 Detection Stats</div>')
            stats_out = gr.Textbox(
                label="",
                elem_id="stats-bar",
                interactive=False,
                show_label=False,
                placeholder="Waiting for detections…",
            )

            gr.HTML('<div class="section-label" style="margin-top:20px">🔤 Detected Characters</div>')
            char_count_out = gr.Number(
                label="Characters Found",
                value=0,
                interactive=False,
                elem_id="char-count",
            )

            gr.HTML('<div class="section-label" style="margin-top:20px">📝 Decoded Text</div>')
            text_out = gr.Textbox(
                label="",
                placeholder="Decoded braille will appear here…",
                lines=3,
                interactive=True,   # allow manual correction
                show_label=False,
            )

            gr.HTML('<div class="section-label" style="margin-top:20px">🔊 Text-to-Speech</div>')
            with gr.Row():
                speak_btn = gr.Button("▶  Speak Text", variant="primary", scale=2)
                clear_btn = gr.Button("✕  Clear", variant="secondary", scale=1)

            audio_out = gr.Audio(
                label="",
                autoplay=True,
                show_label=False,
                type="filepath",
            )

            gr.HTML("""
            <div style="margin-top:16px; padding:12px; background:#0a0c12;
                        border:1px solid #1e2330; border-radius:8px;
                        font-family:'Space Mono',monospace; font-size:0.7rem;
                        color:#6b7280; line-height:1.8;">
                <span style="color:#00e5a0">TIP:</span> Hold braille card steady<br>
                <span style="color:#00e5a0">TIP:</span> Good lighting improves accuracy<br>
                <span style="color:#00e5a0">TIP:</span> Edit text before speaking if needed
            </div>
            """)

    # ── Braille Reference Card ───────────────
    with gr.Accordion("📖 Braille Alphabet Reference", open=False):
        gr.HTML("""
        <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
                    color:#6b7280; padding:16px; line-height:2.2;">
            Standard Grade-1 English Braille — each cell is a 2×3 dot grid (dots 1-6).<br><br>
            <span style="color:#00e5a0">a</span>=⠁  <span style="color:#5b8fff">b</span>=⠃  <span style="color:#ff6464">c</span>=⠉  <span style="color:#ffcc44">d</span>=⠙
            <span style="color:#00e5a0">e</span>=⠑  <span style="color:#5b8fff">f</span>=⠋  <span style="color:#ff6464">g</span>=⠛  <span style="color:#ffcc44">h</span>=⠓
            <span style="color:#00e5a0">i</span>=⠊  <span style="color:#5b8fff">j</span>=⠚  <span style="color:#ff6464">k</span>=⠅  <span style="color:#ffcc44">l</span>=⠇<br>
            <span style="color:#00e5a0">m</span>=⠍  <span style="color:#5b8fff">n</span>=⠝  <span style="color:#ff6464">o</span>=⠕  <span style="color:#ffcc44">p</span>=⠏
            <span style="color:#00e5a0">q</span>=⠟  <span style="color:#5b8fff">r</span>=⠗  <span style="color:#ff6464">s</span>=⠎  <span style="color:#ffcc44">t</span>=⠞
            <span style="color:#00e5a0">u</span>=⠥  <span style="color:#5b8fff">v</span>=⠧  <span style="color:#ff6464">w</span>=⠺  <span style="color:#ffcc44">x</span>=⠭
            <span style="color:#00e5a0">y</span>=⠽  <span style="color:#5b8fff">z</span>=⠵
        </div>
        """)

    # ── Footer ──────────────────────────────
    gr.HTML("""
    <div class="footer-bar">
        BRAILLE AI &nbsp;·&nbsp; YOLOV8 + OPENCV + GTTS &nbsp;·&nbsp; HACKATHON BUILD
    </div>
    """)

    # ─────────────────────────────────────────
    # EVENTS
    # ─────────────────────────────────────────

    # Stream: webcam frame → inference → outputs
    webcam_in.stream(
        fn=process_frame,
        inputs=[webcam_in],
        outputs=[annotated_out, text_out, char_count_out, stats_out],
        time_limit=60,
        stream_every=0.15,   # ~7 FPS inference — tune as needed
    )

    # TTS button
    speak_btn.click(
        fn=speak_text,
        inputs=[text_out],
        outputs=[audio_out],
    )

    # Clear button
    clear_btn.click(
        fn=lambda: ("", 0, "", None),
        inputs=[],
        outputs=[text_out, char_count_out, stats_out, audio_out],
    )


# ─────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,           # set True to get a public gradio.live link
        show_error=True,
    )
