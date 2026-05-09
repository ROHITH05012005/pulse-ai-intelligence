# 🌊 Pulse AI: Emotion Intelligence Engine

Pulse AI is a high-performance **Multi-class Text Classification** platform designed to detect human emotions in real-time. Built with a premium glassmorphic interface, it leverages state-of-the-art Transformer models to provide deep insights into text sentiment.

## 🚀 Features
- **Real-time Classification**: Analysis happens as you type using a debounced inference engine.
- **7-Class Emotion Detection**: Detects Joy, Anger, Fear, Sadness, Surprise, Disgust, and Neutrality.
- **Liquid UI**: The entire interface dynamically changes its color palette and animations based on the dominant detected emotion.
- **Transformer-Powered**: Uses the `DistilRoBERTa` architecture for high accuracy with low latency.

## 🛠️ Tech Stack
- **Backend**: FastAPI, Hugging Face Transformers, PyTorch.
- **Frontend**: React, Vite, Framer Motion, Tailwind-inspired Vanilla CSS.
- **Design**: Glassmorphism, Responsive Liquid Design.

## 🏃 How to Run

### 1. Start the Backend
```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
*Note: On first run, it will download the 300MB model brain.*

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Model Information
The engine uses the `j-hartmann/emotion-english-distilroberta-base` model, fine-tuned on the GoEmotions dataset. It provides professional-grade emotion intelligence out of the box.

---
Built by **Antigravity** for your Portfolio.
