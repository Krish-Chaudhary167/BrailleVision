# BrailleVision
Real time braille reader that converts handwritten/embossed braille into english text and speech.

## Features
- Real-time Braille detection using a webcam
- Recognition of handwritten and embossed Braille
- Conversion of Braille characters into English text
- Text-to-Speech (TTS) output for audio feedback
- Interactive web interface built with Gradio
- YOLOv8-based object detection for accurate character recognition

## Tech Stack
- Python 3.11.9
- YOLOv8
- OpenCV
- Gradio
- NumPy
- gTTS (Google Text-to-Speech)

## How It Works
1. Capture Braille input through a webcam.
2. Preprocess frames using OpenCV.
3. Detect Braille characters using a trained YOLOv8 model.
4. Convert detected characters into readable English text.
5. Generate speech output from the recognized text.

## Applications
- Assistive technology for visually impaired users
- Braille learning and education
- Accessibility tools for public and educational environments
- Real-time Braille translation systems
