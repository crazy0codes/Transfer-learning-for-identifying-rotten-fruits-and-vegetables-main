# SMART SORTING: TRANSFER LEARNING FOR IDENTIFYING ROTTEN FRUITS AND VEGETABLES

---

## 1. Ideation Phase

### 1.1 Project Title

Smart Sorting: Transfer Learning for Identifying Rotten Fruits and Vegetables

### 1.2 Domain

Artificial Intelligence – Deep Learning – Computer Vision

### 1.3 Problem Statement

Manual sorting of fruits and vegetables in food industries, supermarkets, and households is time-consuming, prone to errors, and inefficient. Human inspection cannot consistently maintain accuracy at large scales. There is a need for an automated system that can accurately identify rotten and healthy produce.

### 1.4 Proposed Solution

Develop an AI-based image classification system using Transfer Learning (VGG16) that can classify fruits and vegetables into healthy and rotten categories. The system will be integrated with a Flask web application to allow real-time predictions.

### 1.5 Objectives

* Automate fruit and vegetable freshness detection.
* Reduce food wastage.
* Improve efficiency in quality control processes.
* Deploy a real-time prediction web application.

---

## 2. Requirement Analysis

### 2.1 Functional Requirements

* User should be able to upload an image.
* System should preprocess the image.
* Model should predict the class (healthy/rotten).
* Display prediction result with confidence.

### 2.2 Non-Functional Requirements

* High prediction accuracy.
* Fast response time.
* User-friendly interface.

### 2.3 Hardware Requirements

* System with minimum 8GB RAM.
* Optional GPU for faster training.

### 2.4 Software Requirements

* Python 3.x
* TensorFlow / Keras
* Flask
* NumPy
* OpenCV / PIL
* HTML, CSS
* VS Code / Jupyter Notebook

---

## 3. Project Planning Phase

### 3.1 Timeline

1. Dataset Collection
2. Data Preprocessing
3. Model Building
4. Model Training
5. Model Evaluation
6. Application Development
7. Testing and Deployment

### 3.2 Tools Used

* Kaggle (Dataset Source)
* TensorFlow/Keras
* Flask
* GitHub

---

## 4. Project Design Phase

### 4.1 System Architecture

User → Web Interface → Flask Backend → Preprocessing → VGG16 Model → Prediction → UI Display

### 4.2 Dataset Description

* Total Classes: 28
* Example Classes:

  * Apple_healthy
  * Apple_rotten
  * Banana_healthy
  * Banana_rotten
  * Strawberry_healthy
  * Strawberry_rotten
  * Cucumber_healthy
  * Cucumber_rotten

### 4.3 Data Preprocessing

* Image resizing to 224x224
* Normalization (0–1 scaling)
* Train-validation split

---

## 5. Project Development Phase

### 5.1 Model Building

* Pre-trained VGG16 used (include_top=False)
* Base layers frozen
* Added Flatten layer
* Added Dense layer (Softmax, 28 classes)

### 5.2 Model Compilation

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Metrics: Accuracy

### 5.3 Model Training

* Epochs: 15
* EarlyStopping used
* Validation monitoring

### 5.4 Model Saving

Model saved as: healthy_vs_rotten.h5

### 5.5 Model Performance

* Training Accuracy ≈ 88%
* Validation Accuracy ≈ 75–78%

---

## 6. Application Development

### 6.1 Backend (Flask)

* Image upload handling
* Model loading
* Image preprocessing
* Prediction generation
* Result display

### 6.2 Frontend

Three HTML files:

* index.html (Home Page)
* about.html (Project Details)
* predict.html (Image Upload & Prediction)

### 6.3 Project Folder Structure

SmartSorting/
│
├── app.py
├── healthy_vs_rotten.h5
│
├── static/
│   └── uploads/
│
└── templates/
├── index.html
├── about.html
└── predict.html

---

## 7. Testing Phase

### 7.1 Unit Testing

* Tested with Apple_healthy images
* Tested with Strawberry_healthy images
* Tested with Cucumber_rotten images
* Tested with Strawberry_rotten images

### 7.2 Results

Model successfully predicted the correct class for sample test images.

---

## 8. Project Documentation Phase

### 8.1 Applications

* Food Processing Industries
* Supermarkets
* Smart Refrigerators

### 8.2 Advantages

* Reduces manual labor
* Minimizes food waste
* Improves accuracy
* Scalable solution

### 8.3 Limitations

* Depends on dataset quality
* Lighting conditions affect predictions

### 8.4 Future Enhancements

* Deploy on cloud
* Add live camera detection
* Improve accuracy with EfficientNet
* Mobile app integration

---

## 9. Conclusion

The Smart Sorting system demonstrates the power of Transfer Learning in solving real-world agricultural and food quality problems. By integrating VGG16 with a Flask web application, we successfully developed an automated fruit and vegetable freshness detection system. This project reduces food waste and improves operational efficiency.

---

## 10. References

* Kaggle Dataset
* TensorFlow Documentation
* Keras Documentation
* SmartBridge Internship Portal

---

## 11. Learning Outcomes

* Understanding Transfer Learning
* Image Classification using CNN
* Model Evaluation Techniques
* Flask Web Deployment
* End-to-End AI Application Development

---
Domain: AI and ML
