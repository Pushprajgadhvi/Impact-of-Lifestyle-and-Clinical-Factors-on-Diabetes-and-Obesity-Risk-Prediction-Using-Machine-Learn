# 🩺 Impact of Lifestyle and Clinical Factors on Diabetes and Obesity Risk Prediction

## 📌 Project Overview

This project focuses on predicting **Diabetes** and **Obesity** risk using Machine Learning techniques and healthcare indicators from the **CDC Diabetes Health Indicators Dataset**.

The study investigates how lifestyle, demographic, and clinical factors such as BMI, Age, General Health, Physical Health, Mental Health, Education, and Income influence the likelihood of developing diabetes and obesity.

The project follows a complete Machine Learning pipeline including data preprocessing, exploratory data analysis (EDA), class balancing using SMOTE, model training, evaluation, and performance comparison.

---

## 🎯 Objectives

* Predict Diabetes Risk using health indicators.
* Predict Obesity Risk using BMI-based classification.
* Analyze the impact of lifestyle and clinical factors on health outcomes.
* Handle class imbalance using SMOTE.
* Compare multiple Machine Learning algorithms.
* Evaluate models using comprehensive performance metrics.

---

## 📊 Dataset

**Source:** CDC Diabetes Health Indicators Dataset

The dataset contains demographic, lifestyle, and clinical information including:

* BMI
* Age
* General Health
* Physical Health
* Mental Health
* High Blood Pressure
* High Cholesterol
* Smoking Habits
* Physical Activity
* Education
* Income
* Diabetes Status

### Additional Feature Engineering

An obesity target variable was created using:

```python
Obesity = BMI >= 30
```

Where:

* 0 = Non-Obese
* 1 = Obese

---

## 🔍 Exploratory Data Analysis (EDA)

The project includes:

### Correlation Analysis

* Correlation Heatmaps
* Feature Relationship Analysis

### Distribution Analysis

* Diabetes Distribution
* Obesity Distribution
* BMI Distribution
* Feature Density Plots

### Class Imbalance Visualization

* Before SMOTE
* After SMOTE

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

### Data Cleaning

* Missing value inspection
* Feature selection

### Feature Scaling

StandardScaler was applied to numerical features:

```python
from sklearn.preprocessing import StandardScaler
```

### Class Balancing

SMOTE (Synthetic Minority Oversampling Technique) was used to balance minority classes.

```python
from imblearn.over_sampling import SMOTE
```

Benefits:

* Reduces model bias
* Improves minority class prediction
* Enhances Recall and F1 Score

---

## 🤖 Machine Learning Models

The following algorithms were trained and evaluated:

| Model                     | Purpose                       |
| ------------------------- | ----------------------------- |
| Logistic Regression       | Baseline Classification       |
| Decision Tree             | Rule-Based Prediction         |
| Random Forest             | Ensemble Learning             |
| Gradient Boosting         | Boosting-Based Classification |
| Extra Trees Classifier    | Randomized Ensemble Learning  |
| K-Nearest Neighbors (KNN) | Distance-Based Classification |
| Naive Bayes               | Probabilistic Classification  |

---

## 📈 Model Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve

---

## 🔄 Project Workflow

```text
CDC Health Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Scaling
        │
        ▼
SMOTE Balancing
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Diabetes & Obesity Risk Prediction
```

---

## 📌 Key Findings

* Class imbalance significantly affected prediction performance.
* SMOTE improved minority class representation.
* Ensemble models outperformed traditional models.
* Lifestyle and health indicators showed strong correlations with diabetes and obesity risk.
* Feature engineering enhanced predictive performance.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

```text
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
Imbalanced-Learn (SMOTE)
SciPy
```

### Development Environment

* Jupyter Notebook
* VS Code

---

## 📂 Project Structure

```text
📦 Diabetes-Obesity-Risk-Prediction
│
├── data/
├── notebooks/
├── images/
├── models/
├── results/
├── Diabetes_Obesity_Risk_Analysis.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

* Deep Learning (ANN) Implementation
* Explainable AI using SHAP
* Health Risk Scoring System
* Real-Time Risk Prediction Web Application
* Deployment using Flask/FastAPI
* Cloud Deployment and MLOps Integration

---
