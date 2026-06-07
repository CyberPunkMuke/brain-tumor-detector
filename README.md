# Brain Tumor Detection Using VGG16 and XGBoost

## Overview

This project is a web-based Brain Tumor Detection System that analyzes MRI brain scan images and predicts the presence of a brain tumor using Deep Learning and Machine Learning techniques.

The system uses VGG16 as a feature extractor and XGBoost as the classification model. A Flask web application provides an easy-to-use interface for uploading MRI images and viewing prediction results.

---

## Features

* MRI Brain Tumor Detection
* Deep Feature Extraction using VGG16
* Classification using XGBoost
* User-friendly Flask Web Interface
* MRI Image Upload Functionality
* Grad-CAM Heatmap Visualization Support
* Fast and Accurate Prediction

---

## Technology Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* VGG16

### Machine Learning

* XGBoost
* Scikit-learn

### Web Framework

* Flask

### Image Processing

* OpenCV
* Pillow
* NumPy

### Frontend

* HTML
* CSS

---

## Project Structure

```text
brain-tumor-detector/
│
├── app.py
├── train_model.py
├── feature_extractor.py
├── gradcam.py
├── requirements.txt
│
├── model/
│   ├── tumor_model.pkl
│
├── dataset/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── uploads/
```

---

## Workflow

1. Upload MRI Brain Scan Image
2. Image Preprocessing
3. Feature Extraction using VGG16
4. Classification using XGBoost
5. Display Prediction Result
6. Generate Heatmap Visualization (Optional)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/CyberPunkMuke/brain-tumor-detector.git
cd brain-tumor-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Dataset

The model is trained using MRI brain scan images containing:

* Tumor Images
* Non-Tumor Images

The dataset is used for feature extraction and classification training.

---

## Model Architecture

### Feature Extraction

* VGG16 (Pretrained on ImageNet)
* Top layers removed
* Deep features extracted from MRI images

### Classification

* XGBoost Classifier
* Trained on extracted VGG16 features

---

## Results

The model predicts whether an MRI image contains:

* Tumor
* No Tumor

Prediction results are displayed through the Flask web application.

---

## Future Improvements

* Multi-class Brain Tumor Classification
* Improved Accuracy with Advanced Architectures
* Cloud Deployment
* User Authentication
* Real-time MRI Analysis
* Enhanced Grad-CAM Visualization

---

## Author

**Mukesh Kanna**

GitHub: https://github.com/CyberPunkMuke

---

## License

This project is developed for educational and research purposes.
