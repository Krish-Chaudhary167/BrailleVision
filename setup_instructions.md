# Setup Instructions — BrailleVision AI

## Requirements

- Python 3.11
- Webcam
- `model.keras` file (trained weights)

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/Krish-Chaudhary167/BrailleVision.git
cd BrailleVision
```

---

## Step 2 — Create virtual environment

```bash
py -3.11 -m venv venv
```

Activate it:

**Windows:**
```bash
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

## Step 3 — Install dependencies

```bash
pip install gradio gtts opencv-python numpy tensorflow
```

---

## Step 4 — Add model file

Place `model.keras` in the root project folder (same folder as `braille_ui.py`).

---

## Step 5 — Run

```bash
python braille_ui.py
```

Open your browser at **http://localhost:7860**

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `tensorflow` not installing | Make sure you're using Python 3.11, not 3.12+ |
| `model.keras` not found | Place it in the same folder as `braille_ui.py` |
| Webcam not working | Allow camera access in your browser |
| Random detections with no Braille | Improve lighting and hold the page flat |
