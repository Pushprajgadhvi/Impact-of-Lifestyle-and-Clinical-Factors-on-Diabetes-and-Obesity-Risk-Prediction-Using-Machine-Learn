# AI-Powered Diabetes & Obesity Risk Prediction Platform

An end-to-end web application developed for your **Major Project (Sem 7)** on AI/ML in healthcare. This platform predicts future diabetes and obesity risks based on clinical history, biometrics, and lifestyle habits, providing personalized precautions, dietary plans, exercise prescriptions, and clinical guidance.

---

## 🚀 Project Architecture

1. **Machine Learning Core**:
   - Trained on the CDC Health Indicators dataset (UCI ID: 891).
   - Utilizes **Random Forest Classifier** ensembles for both Diabetes prediction and Obesity prediction (derived from BMI $\ge 30$).
   - Standardizes numerical attributes (`BMI`, `Age`, `GenHlth`, `MentHlth`, `PhysHlth`, `Education`, `Income`) using `StandardScaler`.
   - Saved artifacts: `diabetes_model.pkl`, `obesity_model.pkl`, `scaler.pkl`, `feature_names.json`.

2. **Backend (FastAPI)**:
   - Built with high-performance Python FastAPI (`main.py`).
   - Exposes REST API endpoints (`/api/predict`, `/api/health`).
   - Dynamically calculates joint risk probabilities and generates evidence-based preventive prescriptions (diet, exercise, clinical follow-ups, and precautions).

3. **Frontend (Modern Web UI)**:
   - Single-page dashboard built with **Tailwind CSS**, **Alpine.js**, and **FontAwesome**.
   - Features an interactive patient assessment form with preset sample profiles, live probability progress bars, risk level indicators (Low, Moderate, High), and categorized medical recommendations.

---

## 🛠️ How to Run Locally

### Prerequisites
Make sure Python 3.10+ and pip are installed.

### Installation & Execution
1. Install dependencies:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy ucimlrepo joblib
   ```
2. Navigate to the project directory:
   ```bash
   cd /home/ubuntu/health_app
   ```
3. Run the application:
   ```bash
   bash run.sh
   ```
4. Open your browser and go to:
   **`http://localhost:8000`**

---

## 🌐 How to Publish / Deploy

You can easily deploy this web application on cloud platforms such as **Render**, **Railway**, **Hugging Face Spaces**, or **Heroku**:

1. **Dockerfile / Procfile setup**:
   Create a `requirements.txt`:
   ```txt
   fastapi
   uvicorn
   scikit-learn
   pandas
   numpy
   joblib
   pydantic
   ```
2. **Start Command**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
