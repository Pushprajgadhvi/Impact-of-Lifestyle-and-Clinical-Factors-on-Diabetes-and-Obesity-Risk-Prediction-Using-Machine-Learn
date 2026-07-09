# 🩺 Impact of Lifestyle and Clinical Factors on Diabetes and Obesity Risk

## 📌 Project Overview

This project investigates the impact of lifestyle habits and clinical health indicators on the risk of developing **Diabetes** and **Obesity**. By combining statistical analysis, exploratory data analysis (EDA), machine learning techniques, and risk prediction models, the project aims to identify important factors that contribute to these chronic health conditions.

The study utilizes healthcare data from publicly available sources and applies data science methodologies to analyze relationships between lifestyle behaviors, clinical measurements, and disease outcomes.

---

## 🎯 Objectives

* Analyze the influence of lifestyle factors on Diabetes and Obesity risk.
* Identify significant clinical indicators associated with disease occurrence.
* Perform comprehensive Exploratory Data Analysis (EDA).
* Compare multiple Machine Learning models for disease prediction.
* Generate personalized risk predictions based on user-provided health information.
* Visualize patterns and trends within healthcare datasets.

---

## 📊 Dataset Information

This project primarily utilizes health-related datasets containing demographic, lifestyle, and clinical features such as:

* Age
* Gender
* Body Mass Index (BMI)
* Blood Pressure
* Physical Activity
* Smoking Habits
* Alcohol Consumption
* General Health Status
* Mental Health Indicators
* Sleep Patterns
* Diabetes Status
* Obesity Status

### Data Sources

* CDC Diabetes Health Indicators Dataset
* Public Healthcare Datasets from UCI Machine Learning Repository
* Obesity-related Health Indicators Dataset

---

## 🔬 Methodology

### 1. Data Collection

Datasets are fetched automatically using the `ucimlrepo` library, eliminating manual downloads and ensuring reproducibility.

### 2. Data Preprocessing

* Missing value handling
* Feature selection
* Label encoding
* Data cleaning
* Feature scaling using StandardScaler
* Target variable generation

### 3. Feature Engineering

An obesity classification is generated using:

BMI ≥ 30 → Obese

This allows simultaneous analysis of:

* Diabetes Risk
* Obesity Risk
* Combined Diabetes & Obesity Risk

### 4. Exploratory Data Analysis (EDA)

The project includes:

* Correlation Heatmaps
* Feature Distribution Analysis
* Risk Factor Visualization
* Class Distribution Analysis
* BMI Impact Analysis
* Physical Activity Impact Analysis
* Age-Based Risk Analysis

### 5. Data Standardization

Visualization of feature distributions:

* Before Scaling
* After Scaling

This demonstrates the effect of preprocessing on model performance.

---

## 🤖 Machine Learning Models

The following Machine Learning models are implemented and evaluated:

### Classification Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Gradient Boosting
* XGBoost (Optional)

### Deep Learning Models

* Artificial Neural Network (ANN)

### Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

## 📈 Statistical Insights

The notebook calculates:

* Percentage of individuals with Diabetes
* Percentage of individuals with Obesity
* Percentage of individuals with both conditions
* Feature correlations with disease outcomes
* Population-level risk distributions

---

## 🧠 Personalized Risk Prediction System

A personalized risk calculator has been implemented that allows users to enter their own clinical and lifestyle information.

The trained models estimate:

✅ Diabetes Risk (%)

✅ Obesity Risk (%)

✅ Combined Diabetes & Obesity Risk (%)

This demonstrates how machine learning can support health-risk assessment.

---

## 📂 Project Structure

```text
├── Diabetes_Obesity_Risk_Analysis.ipynb
├── datasets/
├── images/
├── models/
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Impact-of-Lifestyle-and-Clinical-Factors-on-Diabetes-and-Obesity-Risk.git
```

```bash
cd Impact-of-Lifestyle-and-Clinical-Factors-on-Diabetes-and-Obesity-Risk
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ Running the Project

1. Open the Jupyter Notebook.
2. Run all cells sequentially.
3. Explore EDA visualizations.
4. Train machine learning models.
5. Test the personalized risk calculator using custom input data.

Supported Platforms:

* Jupyter Notebook
* Jupyter Lab
* VS Code
* Google Colab

---

## 📷 Key Visualizations

The project includes:

* Correlation Heatmaps
* Risk Distribution Charts
* BMI Analysis Graphs
* Physical Activity Comparisons
* Feature Importance Graphs
* Before vs After Scaling Distributions
* Model Performance Comparisons

---

## 🏥 Real-World Applications

This project can be extended for:

* Healthcare Analytics
* Disease Risk Assessment Systems
* Preventive Healthcare Solutions
* Clinical Decision Support Systems
* Public Health Research
* AI-Powered Health Monitoring Platforms

---

## ⚠️ Disclaimer

This project is intended solely for:

* Academic Research
* Educational Purposes
* Data Science Learning
* Machine Learning Demonstration

The generated predictions should **NOT** be considered medical advice, diagnosis, or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.

---

## 📝 Reference Note

**Important:**

This repository has been developed as a research and educational project to study the relationship between lifestyle factors, clinical indicators, Diabetes risk, and Obesity risk.

The findings, visualizations, and predictive outputs generated by this project are intended for academic reference and demonstration purposes only. Results may vary depending on dataset quality, preprocessing techniques, and model configurations.

Researchers, students, and practitioners are encouraged to use this work as a reference, extend the methodology, and validate findings using additional datasets and clinical studies.

---

## 👨‍💻 Author

**Pushpraj Gadhvi**

Data Science | Machine Learning | Deep Learning | Healthcare Analytics

---

## ⭐ Future Enhancements

* Advanced Deep Learning Models
* Explainable AI (SHAP/LIME)
* Real-Time Risk Prediction Web Application
* Integration with Electronic Health Records (EHR)
* Multi-Disease Risk Prediction Framework
* Cloud Deployment and MLOps Pipeline
