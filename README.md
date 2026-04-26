# ⚽ VelocityVision — AI Football Match Analyzer

🚀 An end-to-end **Computer Vision system** that analyzes football match clips to extract player-level and match-level insights using deep learning and video analytics.

---

## 🎯 Overview

VelocityVision processes short football clips (15–30 seconds) and performs:

* Player & ball detection
* Multi-object tracking with persistent IDs
* Real-time speed estimation (km/h)
* Ball possession tracking
* Pass detection between players
* Annotated video generation

All analysis is performed **directly from raw video input**, without any sensors or manual annotations.

---

## 🧠 Core Pipeline

```
Video Input → YOLO Detection → Tracking → Speed Estimation → 
Ball Possession → Pass Detection → Visualization → Output Video
```

---

## 🚀 Features

* 🎯 **YOLOv8 Detection** — players and ball in real-time
* 🧠 **Multi-Object Tracking** — consistent player IDs across frames
* ⚡ **Speed Estimation** — pixel displacement → real-world km/h
* 🔁 **Pass Detection** — based on ball possession transitions
* 📊 **Analytics Summary** — avg speed, max speed, pass count
* 🎥 **Annotated Output Video** — overlays with trajectories, IDs, speeds
* 🖥 **Streamlit Dashboard UI** — interactive and easy to use

---

## 🛠 Tech Stack

* **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
* **Data Processing:** NumPy
* **Visualization:** Matplotlib
* **Frontend/UI:** Streamlit
* **Language:** Python

---

## 📊 Sample Output

* Player tracking with unique IDs
* Speed labels (km/h)
* Motion trajectories
* Ball tracking & pass highlights

📎 *(Add screenshots and output video here)*

---

## ▶️ Run Locally

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/velocityvision-ai.git
cd velocityvision-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run application

```bash
streamlit run football_analysis_final_pro.py
```

---

## 📁 Project Structure

```
velocityvision-ai/
│
├── football_analysis_final_pro.py   # Main application
├── requirements.txt
├── README.md
└── outputs/                         # Generated videos (ignored in git)
```

---

## 📈 Performance

* Real-time inference on short clips
* Stable tracking with minimal ID switching
* Speed estimation accuracy: ~80–90% (approx.)
* Pass detection based on temporal ball ownership

---

## 🔮 Future Improvements

* Homography-based field calibration (accurate real-world mapping)
* Team identification via jersey color clustering
* Heatmaps & tactical analysis
* Next-pass prediction using ML models
* Real-time live match analysis

---

## 🎯 Use Cases

* Sports analytics & coaching
* Player performance evaluation
* Tactical match analysis
* AI in sports research

---

## 👨‍💻 Author

**Sparsh Modi**
CSE (AI Engineering) Student

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
