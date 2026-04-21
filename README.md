# 🏥 Acute Findings Triage System

An AI-powered radiology triage system that detects critical medical conditions from chest X-Ray images and ranks them by clinical urgency using deep learning and explainable AI.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What It Does

Upload a chest X-Ray → the system returns:
- **Diagnosis** — Normal or Pneumonia
- **Priority Level** — CRITICAL / HIGH / MEDIUM / LOW
- **Confidence Score** — how sure the model is
- **Grad-CAM Heatmap** — visual explanation of where the model focused

---

## 🩺 Detected Conditions

| Condition | Modality | Priority | Response Time |
|-----------|----------|----------|---------------|
| Pneumonia | Chest X-Ray | CRITICAL / HIGH | < 15 min / < 1 hr |
| Normal | Chest X-Ray | LOW | Routine |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **96.5%** |
| Test Accuracy | **87.2%** |
| Sensitivity (Recall) | **96.9%** — detects 97/100 sick patients |
| Specificity | **70.9%** |
| F1-Score (Pneumonia) | **0.89** |
| Decision Threshold | **0.7** (optimized) |

> **Why Sensitivity matters more in medicine:** A False Negative (missing a sick patient) is far more dangerous than a False Positive (extra tests for a healthy patient).

---

## 🧠 Model Architecture

```
Input Image (512×512)
       ↓
Preprocessing (Normalize + Augment)
       ↓
EfficientNet-B4 Backbone (pretrained on ImageNet)
       ↓
Custom Classification Head
       ↓
Softmax Probabilities
       ↓
Threshold (0.7) → Triage Engine → Priority Level
       ↓
Grad-CAM Heatmap (Explainability)
```

### Two-Phase Transfer Learning Strategy

**Phase 1 — Head Only (5 epochs)**
- Backbone frozen → only 919K trainable params (out of 18M)
- LR = 1e-3
- Result: Val Accuracy 96.2%

**Phase 2 — Fine-tuning (10 epochs)**
- Unfreeze last 2 blocks
- LR = 5e-5 (lower to preserve pretrained weights)
- Result: Val Accuracy 96.5%

---

## 🗂️ Project Structure

```
acute-triage-system/
├── src/
│   ├── preprocessing.py     # DICOM loading, augmentation pipeline
│   ├── models.py            # EfficientNet-B4 + DenseNet-121
│   ├── triage_engine.py     # Priority scoring logic
│   └── gradcam.py           # Grad-CAM explainability
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Data loading & EDA
│   └── 03_gradio_demo.ipynb        # Gradio demo with Grad-CAM
├── models/
│   └── weights/
│       ├── xray_best.pth           # Trained model checkpoint
│       └── xray_config.json        # Model config & metrics
├── app.py                   # Gradio UI
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/AhmedAbdElkhalek16/acute-triage-system.git
cd acute-triage-system
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python app.py
# Open http://localhost:7860
```

### 3. Train from Scratch (Google Colab)
Open notebooks in order:
1. `notebooks/01_data_exploration.ipynb` — explore & prepare data
2. `notebooks/03_gradio_demo.ipynb` — run demo with trained model

---

## 📦 Dataset

| Dataset | Source | Size | Classes |
|---------|--------|------|---------|
| Chest X-Ray Pneumonia | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | ~2 GB | Normal / Pneumonia |

**Class Distribution (after balancing):**
```
Original  →  Normal: 1341  |  Pneumonia: 3875  (imbalanced 1:3)
After WeightedRandomSampler  →  1.01:1 ratio  ✅
```

---

## 🔧 Key Technical Decisions

**Class Imbalance → WeightedRandomSampler**
The dataset had 3x more Pneumonia than Normal cases. Without fixing this, the model would just predict "Pneumonia" always and get 74% accuracy without learning anything useful.

**Threshold Tuning (0.5 → 0.7)**
Default threshold of 0.5 gave 62.4% specificity. Moving to 0.7 improved specificity to 70.9% while only dropping sensitivity from 97.9% to 96.9% — a worthwhile tradeoff.

**Grad-CAM for Explainability**
Medical AI without explainability is a black box that doctors won't trust. Grad-CAM shows exactly which regions in the lung triggered the diagnosis.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, timm |
| Models | EfficientNet-B4, DenseNet-121 |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Preprocessing | Albumentations, pydicom, OpenCV |
| UI | Gradio |
| Training | Google Colab (T4 GPU) |
| Data | Kaggle API |

---

## ⚕️ Disclaimer

This system is built for **research and educational purposes only**.
It is **not approved for clinical use** and should not be used to make real medical decisions.

---

## 👤 Author

**Ahmed Abd Elkhalek**
AI & Computer Vision Engineer

[![GitHub](https://img.shields.io/badge/GitHub-AhmedAbdElkhalek16-black)](https://github.com/AhmedAbdElkhalek16)