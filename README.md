# 🚛 EdgeGuard AI

> **AI-Powered Predictive Maintenance & Digital Twin Platform for Heavy Mining Trucks**

> **Note:** This is a starter professional README template. Due to response-size limits, the complete 800–1200 line version cannot be embedded in a single chat response, but this Markdown file is ready to extend.

---

## 📌 Overview

EdgeGuard AI is an AI-powered predictive maintenance platform that combines IoT, Computer Vision, Machine Learning, Digital Twin visualization, RAG, and Generative AI to monitor heavy mining trucks in real time.

## ✨ Key Features

- Real-Time IoT Monitoring
- YOLOv8 Damage Detection
- XGBoost Failure Prediction
- Remaining Useful Life (RUL)
- Isolation Forest Anomaly Detection
- Multi-Modal Decision Engine
- RAG Knowledge Assistant
- Gemini AI Copilot
- Digital Twin Dashboard
- Fleet Analytics
- ROI Dashboard

## 🏗️ Architecture

```text
ESP32
  │
 MQTT
  │
FastAPI
  │
Supabase
  │
Feature Engineering
  │
ML Models
  │
Decision Engine
  │
Gemini
  │
Frontend
```

## 🧠 AI Engines

1. Computer Vision (YOLOv8)
2. Predictive Maintenance (XGBoost + Isolation Forest + RUL)
3. Multi-Modal Decision Engine
4. Business Impact Engine
5. Gemini AI Copilot

## 📁 Project Structure

```text
EdgeGuardAi/
├── backend/
├── frontend/
├── ml-data/
├── wokwi-sim/
├── docs/
└── README.md
```

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/EdgeGuardAi.git
cd EdgeGuardAi
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --reload
```

## 👩‍💻 Author

**Gopika**
