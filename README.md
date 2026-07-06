# Impact of Lifestyle and Clinical Factors on Diabetes and Obesity Risk

This project explores how various lifestyle and clinical factors contribute to the risk of developing **Diabetes** and **Obesity**. It includes a comprehensive Jupyter Notebook that analyzes real-world health data, visualizes trends, and trains machine learning models to predict personalized risk percentages.

## 📊 Project Overview

The core of this repository is a Jupyter Notebook (`Diabetes_Obesity_Risk_Analysis.ipynb`) that acts as an end-to-end data science pipeline. It covers:
- **Automated Data Retrieval:** Uses the `ucimlrepo` library to fetch the official **CDC Diabetes Health Indicators dataset** directly into the environment, avoiding the need for manual CSV downloads.
- **Data Engineering:** Combines features and targets, and dynamically creates an `Obesity` classification based on Body Mass Index (BMI > 30).
- **Statistical Analysis:** Calculates the baseline prevalence of individuals with Diabetes, individuals with Obesity, and individuals suffering from **both** simultaneously.
- **Exploratory Data Analysis (EDA):** Features correlation heatmaps and distribution plots showing the impact of variables like Age and Physical Activity on health outcomes.
- **Preprocessing Visualization:** Demonstrates the impact of standard scaling on numerical features using "Before vs. After" distribution graphs.
- **Predictive Modeling:** Trains Random Forest Classifiers to learn from the data.
- **Personalized Risk Calculator:** Provides an interactive function where you can input custom test data to calculate the exact percentage probability of having Diabetes, Obesity, and both.

## 🚀 Getting Started

### Prerequisites

You need Python 3 installed on your system. It is recommended to use a virtual environment or run this in Google Colab.

### Installation

The notebook automatically handles the installation of required packages in the first cell, but if you want to install them manually, run:

```bash
pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Clone this repository or download the `Diabetes_Obesity_Risk_Analysis.ipynb` file.
2. Open the file in Jupyter Notebook, JupyterLab, VS Code, or upload it to Google Colab.
3. Run the cells sequentially from top to bottom.

## 🧪 Testing Your Own Data

At the very bottom of the notebook, you will find the `calculate_my_risk(user_input_dict)` function. 

You can modify the `my_test_data` dictionary to reflect your own clinical and lifestyle data (or hypothetical test data). When you run the cell, the trained models will output:
- **Risk of Diabetes (%)**
- **Risk of Obesity (%)**
- **Risk of having BOTH (%)**

*Disclaimer: This tool is built for educational and data science demonstration purposes and should not be used as a substitute for professional medical diagnosis.*
