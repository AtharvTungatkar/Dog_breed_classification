# 🐶 Dog Breed Classification using Deep Learning (VGG16)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-success)
![Flask](https://img.shields.io/badge/Flask-Deployment-black)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

This project implements an **end-to-end deep learning pipeline** to classify dog breeds from images using **Transfer Learning with VGG16** (pre-trained on ImageNet).  
The solution spans the **entire ML lifecycle** — from data preprocessing and model optimization to **real-time deployment via a Flask web application**.

The model is trained on **119 unique dog breeds** and designed with **production-readiness, modularity, and scalability** in mind.

---

## 🎯 Key Objectives

- Build a **high-accuracy CNN-based image classifier**
- Leverage **Transfer Learning** to improve performance on limited labeled data
- Design a **modular, object-oriented ML pipeline**
- Deploy the trained model via a **Flask REST API with a user-friendly UI**
- Enable **real-time image inference**

---

## 🗂️ Dataset & Preprocessing

- **Dataset Size:** Images across **119 dog breeds**
- **Automated Folder Structuring:** Python scripts to organize class-wise directories
- **Train–Validation Split:** Reproducible and stratified
- **Image Preprocessing:**
  - Resizing & normalization
  - Data augmentation:
    - Rotation
    - Zoom
    - Horizontal flips
    - Width/height shifts
- **Class Imbalance Handling:** Dataset balancing strategies to reduce bias

> These steps significantly improved generalization and robustness during validation.

---

## 🧠 Model Architecture

### 🔹 Base Model
- **VGG16** pre-trained on **ImageNet**
- Convolutional layers frozen for feature extraction

### 🔹 Custom Head
- Fully Connected Dense layers
- Batch Normalization
- Dropout regularization

### 🔹 Training Framework
- **TensorFlow & Keras**
- Adam Optimizer
- Categorical Cross-Entropy Loss

---

## 📊 Model Performance

| Metric | Score |
|------|------|
| Training Accuracy | **94.29%** |
| Validation Accuracy | **81.37%** |

> The gap highlights real-world generalization challenges and validates proper regularization.

---

## 🏗️ Object-Oriented ML Pipeline

The project follows **OOP principles** to ensure:

- Modularity
- Code reusability
- Clean separation of concerns

### Pipeline Components:
- `DataLoader` – image loading & augmentation
- `ModelBuilder` – VGG16 architecture setup
- `Trainer` – training & evaluation logic
- `Predictor` – inference pipeline
- `AppController` – deployment interface

This structure makes the project **easy to extend**, retrain, or migrate to cloud infrastructure.

---

## 🌐 Deployment & Application

### 🔥 Flask Web Application
- REST API for predictions
- HTML/CSS frontend for:
  - Image upload
  - Instant breed prediction
- End-to-end inference pipeline:
  1. Image upload
  2. Preprocessing
  3. Model inference
  4. Result rendering

### 🧩 Production Considerations
- Clean API endpoints
- Scalable architecture
- Ready for cloud deployment (AWS / GCP / Azure)

---

## 🚀 How to Run Locally

```bash
# Clone repository
git clone https://github.com/AtharvTungatkar/Dog_breed_classification.git
cd Dog_breed_classification

# Create environment
conda create -n dogbreed python=3.9
conda activate dogbreed

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

