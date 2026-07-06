# 🚛 EdgeGuard AI

## AI-Powered Predictive Maintenance & Digital Twin Platform for Heavy Mining Trucks

> **An intelligent predictive maintenance platform for heavy-duty tipper trucks using IoT, Computer Vision, Machine Learning, Digital Twin, Retrieval-Augmented Generation (RAG), and Large Language Models.**

---

# 📌 Overview

EdgeGuard AI is a next-generation industrial predictive maintenance platform developed to monitor the health of heavy mining trucks in real time.

The platform continuously collects IoT sensor data, detects equipment anomalies, predicts component failures, estimates **Remaining Useful Life (RUL)**, provides maintenance recommendations, calculates financial impact, and visualizes the entire vehicle through an interactive **Digital Twin Dashboard**.

Unlike conventional monitoring systems that only display sensor values, EdgeGuard AI combines multiple AI engines to deliver intelligent decision support for maintenance teams.

---

# ✨ Key Features

- 🚛 Real-Time Truck Health Monitoring
- 🌡️ IoT Sensor Integration (ESP32 + MQTT)
- 📡 Live Telemetry Dashboard
- 🤖 Predictive Maintenance using XGBoost
- 📈 Remaining Useful Life (RUL) Prediction
- 🔍 Isolation Forest Anomaly Detection
- 👁️ Computer Vision Damage Detection (YOLOv8)
- 🧠 Multi-Modal AI Decision Engine
- 📚 RAG-based SOP Recommendation System
- 💬 Gemini AI Maintenance Copilot
- 💰 ROI & Business Impact Analysis
- 🛰️ Interactive Digital Twin
- 📊 Fleet Health Monitoring
- ⚠️ Risk Prioritization
- 📉 Failure Trend Analysis
- 🔄 Demo Mode Simulation
- 📱 Modern Responsive Dashboard

---

# 🧠 AI Architecture

EdgeGuard AI consists of **five specialized AI engines**, each responsible for a critical part of the predictive maintenance pipeline.

---

## 👁️ AI Engine 1 — Computer Vision

Uses **YOLOv8** to inspect truck images and automatically detect equipment defects.

### Detects

- Oil Leakage
- Tyre Wear
- Structural Crack
- Rust

### Outputs

- Bounding Boxes
- Confidence Scores
- Damage Severity

---

## 🤖 AI Engine 2 — Predictive Maintenance

Processes real-time sensor data to predict equipment failures before they occur.

### Machine Learning Models

- XGBoost Classifier
- Remaining Useful Life (RUL) Regressor
- Isolation Forest

### Outputs

- Failure Probability
- Health Score
- Remaining Useful Life
- Anomaly Score

---

## 🧠 AI Engine 3 — Multi-Modal Decision Engine

Combines multiple AI outputs for intelligent decision-making.

### Inputs

- Computer Vision Results
- IoT Sensor Data
- Historical Maintenance Records

### Outputs

- Final Maintenance Decision
- Overall Risk Score
- Maintenance Priority

---

## 💰 AI Engine 4 — Business Impact Engine

Converts technical predictions into business insights.

### Calculates

- Estimated Downtime
- Maintenance Cost
- Revenue Loss
- Return on Investment (ROI)
- Savings from Early Failure Detection

---

## 💬 AI Engine 5 — AI Maintenance Copilot

Powered by **Google Gemini**.

Provides intelligent maintenance assistance including:

- Prediction Explanation
- Root Cause Analysis
- Recommended Repair Steps
- Maintenance Guidance

---

# 📊 Dashboard Modules

## 🏠 Command Center

Displays:

- Live Truck Status
- Active Alerts
- Health Score
- AI Predictions
- Copilot Recommendations

---

## 🚛 Digital Twin

Interactive visualization of the truck including:

- Component Health
- Live Sensor Values
- Color-Coded Failure Indicators
- Animated Vehicle Status

---

## 🚚 Fleet Dashboard

Fleet-wide analytics including:

- Active Vehicles
- Healthy Vehicles
- High-Risk Vehicles
- Maintenance Schedule

---

## 💰 ROI Dashboard

Business analytics including:

- Cost Savings
- Downtime Reduction
- Maintenance ROI
- Failure Prevention Statistics

---

## 🔧 Maintenance Dashboard

Displays:

- Pending Repairs
- Recommended Actions
- Remaining Useful Life
- Maintenance History

---

# 📚 RAG Knowledge Assistant

EdgeGuard AI integrates a **Retrieval-Augmented Generation (RAG)** system trained on industrial **Standard Operating Procedures (SOPs)**.

The system retrieves the most relevant maintenance documentation based on predicted faults and provides contextual recommendations to maintenance engineers.

---

# 📡 IoT Pipeline

```text
ESP32 Sensors
      │
      ▼
 MQTT Broker
      │
      ▼
 FastAPI Backend
      │
      ▼
 Supabase Database
      │
      ▼
 Machine Learning Engine
      │
      ▼
 Decision Engine
      │
      ▼
 Frontend Dashboard
```

---

# 🤖 Machine Learning Pipeline

The predictive maintenance workflow consists of:

1. Synthetic Dataset Generation
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Real-Time Inference
6. Remaining Useful Life Prediction
7. Failure Classification
8. Business Impact Analysis

---

# 🖥️ Technology Stack

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- FastAPI
- MQTT
- REST API

## Machine Learning

- Python
- XGBoost
- Isolation Forest
- Scikit-learn
- Pandas
- NumPy

## Computer Vision

- YOLOv8
- OpenCV
- Ultralytics

## Database

- Supabase
- PostgreSQL

## Artificial Intelligence

- Google Gemini API
- TF-IDF
- Retrieval-Augmented Generation (RAG)

## IoT

- ESP32
- MQTT
- Wokwi Simulator

---

# 📁 Project Structure

```text
EdgeGuardAi/
│
├── backend/
│
├── frontend/
│
├── ml-data/
│   ├── engine1_vision/
│   ├── engine3_multimodal/
│   ├── engine4_business/
│   ├── copilot/
│   └── models/
│
├── wokwi-sim/
│
├── docs/
│
├── README.md
│
└── RUN.md
```

---

# 🚀 Getting Started

## Clone the Repository

```bash
git clone https://github.com/yourusername/EdgeGuardAi.git

cd EdgeGuardAi
```

---

## Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

## Start the Backend

```bash
cd backend

python -m uvicorn main:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

---

## Launch the Frontend

```bash
cd frontend

python -m http.server
```

---

# 📈 Project Highlights

- 🚛 Real-Time Predictive Maintenance
- 🌐 End-to-End IoT Pipeline
- 🛰️ AI-Powered Digital Twin
- 👁️ Computer Vision Inspection
- 📈 Remaining Useful Life Prediction
- 💬 Explainable AI Recommendations
- 💰 Business ROI Analytics
- 📊 Interactive Dashboard
- 🧠 Multi-Modal AI Decision Making
- 📚 Industrial RAG Knowledge Base

---

# 🔮 Future Enhancements

- Edge AI Deployment
- Mobile Application
- Multi-Vehicle Fleet Scaling
- Cloud Deployment
- Predictive Parts Inventory
- Voice-Based AI Assistant
- Automatic Maintenance Scheduling

---

# 👩‍💻 Author

**Gopika**

**AI & Machine Learning Developer**

---

<div align="center">

### ⭐ If you found this project interesting, consider giving it a star!

**Built with ❤️ using AI, IoT, Machine Learning, Computer Vision, and Modern Web Technologies.**

</div>
