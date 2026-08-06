import os
import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any

app = FastAPI(title="Health Risk Prediction & Prevention API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global artifacts
diabetes_model = None
obesity_model = None
scaler = None
feature_names = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]
numerical_cols = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth', 'Education', 'Income']

try:
    d_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
    o_path = os.path.join(BASE_DIR, "obesity_model.pkl")
    s_path = os.path.join(BASE_DIR, "scaler.pkl")
    m_path = os.path.join(BASE_DIR, "feature_names.json")

    if os.path.exists(d_path):
        with open(d_path, "rb") as f:
            diabetes_model = pickle.load(f)
    if os.path.exists(o_path):
        with open(o_path, "rb") as f:
            obesity_model = pickle.load(f)
    if os.path.exists(s_path):
        with open(s_path, "rb") as f:
            scaler = pickle.load(f)
    if os.path.exists(m_path):
        with open(m_path, "r") as f:
            data_meta = json.load(f)
            feature_names = data_meta.get("feature_names", feature_names)
            numerical_cols = data_meta.get("numerical_cols", numerical_cols)
    print("Model artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")

class HealthInput(BaseModel):
    HighBP: int = Field(..., description="High Blood Pressure (0: No, 1: Yes)")
    HighChol: int = Field(..., description="High Cholesterol (0: No, 1: Yes)")
    CholCheck: int = Field(..., description="Cholesterol check in past 5 years (0: No, 1: Yes)")
    BMI: float = Field(..., description="Body Mass Index")
    Smoker: int = Field(..., description="Smoked at least 100 cigarettes in life (0: No, 1: Yes)")
    Stroke: int = Field(..., description="Ever had a stroke (0: No, 1: Yes)")
    HeartDiseaseorAttack: int = Field(..., description="Coronary heart disease or myocardial infarction (0: No, 1: Yes)")
    PhysActivity: int = Field(..., description="Physical activity in past 30 days - not job (0: No, 1: Yes)")
    Fruits: int = Field(..., description="Consume fruit 1+ times per day (0: No, 1: Yes)")
    Veggies: int = Field(..., description="Consume vegetables 1+ times per day (0: No, 1: Yes)")
    HvyAlcoholConsump: int = Field(..., description="Heavy drinkers (adult men >=14 drinks/week, adult women >=7 drinks/week) (0: No, 1: Yes)")
    AnyHealthcare: int = Field(..., description="Have any kind of health care coverage (0: No, 1: Yes)")
    NoDocbcCost: int = Field(..., description="Was there a time when you needed to see a doctor but could not because of cost (0: No, 1: Yes)")
    GenHlth: int = Field(..., description="General health scale (1: Excellent, 2: Very good, 3: Good, 4: Fair, 5: Poor)")
    MentHlth: int = Field(..., description="Days of poor mental health scale 1-30 days")
    PhysHlth: int = Field(..., description="Physical illness or injury days in past 30 days 1-30")
    DiffWalk: int = Field(..., description="Do you have serious difficulty walking or climbing stairs (0: No, 1: Yes)")
    Sex: int = Field(..., description="Sex (0: Female, 1: Male)")
    Age: int = Field(..., description="Age category (1: 18-24, 9: 60-64, 13: 80+)")
    Education: int = Field(..., description="Education level (1: Never attended up to 6: College 4+ years)")
    Income: int = Field(..., description="Income scale (1: <$10k up to 8: $75k+)")

def generate_recommendations(prob_diabetes: float, prob_obesity: float, bmi: float, high_bp: int, high_chol: int, phys_activity: int) -> dict:
    precautions = []
    diet = []
    exercise = []
    clinical = []

    # Obesity / BMI recommendations
    if bmi >= 30 or prob_obesity > 50:
        precautions.append("High risk of obesity detected. Focus on sustained caloric deficit and metabolic health monitoring.")
        diet.append("Adopt a whole-food, plant-forward diet rich in fiber (vegetables, legumes, whole grains) and lean proteins.")
        diet.append("Eliminate sugary beverages, ultra-processed snacks, and refined carbohydrates.")
        exercise.append("Engage in at least 150-300 minutes of moderate-intensity aerobic physical activity per week (e.g., brisk walking, swimming).")
        exercise.append("Incorporate resistance training 2-3 times per week to improve insulin sensitivity and lean muscle mass.")
    elif bmi >= 25:
        precautions.append("Overweight range (Pre-obesity). Proactive lifestyle modifications can prevent transition to clinical obesity.")
        diet.append("Portion control and mindful eating. Reduce intake of saturated fats and added sugars.")
        exercise.append("Aim for 150 minutes of moderate aerobic activity weekly combined with daily step goals (7,000-10,000 steps).")
    else:
        precautions.append("Healthy weight maintained. Maintain current balanced lifestyle to preserve metabolic vitality.")
        diet.append("Maintain balanced macronutrients with emphasis on micronutrient density and hydration.")
        exercise.append("Maintain regular physical activity (minimum 150 mins/week).")

    # Diabetes recommendations
    if prob_diabetes > 60:
        clinical.append("Consult an endocrinologist or primary care physician for HbA1c and fasting blood glucose lab panels.")
        clinical.append("Regular blood glucose self-monitoring (glycemic tracking) recommended.")
        precautions.append("High diabetes risk profile. Early intervention drastically reduces long-term microvascular and macrovascular complications.")
        diet.append("Prioritize low glycemic index (GI) foods to stabilize postprandial blood glucose spikes.")
    elif prob_diabetes > 30:
        clinical.append("Annual metabolic screening recommended with HbA1c testing.")
        precautions.append("Moderate diabetes risk. Consistent physical activity and weight management are key protective factors.")
    else:
        clinical.append("Routine annual health check-ups.")
        precautions.append("Low diabetes risk. Continue healthy habits.")

    # Blood Pressure & Cholesterol
    if high_bp == 1:
        clinical.append("Monitor blood pressure regularly. Discuss cardiovascular risk management with your physician.")
        diet.append("Adopt the DASH (Dietary Approaches to Stop Hypertension) diet, restricting sodium intake to < 2,300 mg/day.")
    if high_chol == 1:
        clinical.append("Schedule a lipid profile blood test (HDL, LDL, Triglycerides).")
        diet.append("Increase soluble fiber intake (oats, beans, psyllium) and healthy omega-3 fatty acids (fish, flaxseeds, walnuts).")
    if phys_activity == 0:
        exercise.append("Gradually introduce daily physical activity starting with 20-minute daily walks, building up to 45 minutes.")

    return {
        "precautions": precautions,
        "dietary_plan": diet,
        "exercise_plan": exercise,
        "clinical_advice": clinical
    }

@app.post("/api/predict")
def predict_risk(data: HealthInput):
    global diabetes_model, obesity_model, scaler, feature_names, numerical_cols
    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # Ensure correct column order
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Apply scaling on numerical columns if scaler exists
        if scaler is not None:
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict probabilities
        if diabetes_model is not None and obesity_model is not None:
            prob_diabetes = float(diabetes_model.predict_proba(input_df)[0][1] * 100)
            prob_obesity = float(obesity_model.predict_proba(input_df)[0][1] * 100)
        else:
            # Fallback heuristic if models haven't been loaded
            prob_diabetes = float(min(95.0, max(5.0, data.BMI * 1.5 + (data.Age * 2) + (data.HighBP * 20))))
            prob_obesity = float(min(98.0, max(5.0, (data.BMI - 18.5) * 4.5)))

        prob_combined = float((prob_diabetes / 100) * (prob_obesity / 100) * 100)

        # Risk levels
        d_level = "High" if prob_diabetes > 60 else "Moderate" if prob_diabetes > 30 else "Low"
        o_level = "High" if prob_obesity > 60 else "Moderate" if prob_obesity > 30 else "Low"
        c_level = "High" if prob_combined > 40 else "Moderate" if prob_combined > 15 else "Low"

        recommendations = generate_recommendations(
            prob_diabetes=prob_diabetes,
            prob_obesity=prob_obesity,
            bmi=data.BMI,
            high_bp=data.HighBP,
            high_chol=data.HighChol,
            phys_activity=data.PhysActivity
        )

        return {
            "success": True,
            "predictions": {
                "diabetes_risk_percentage": round(prob_diabetes, 2),
                "diabetes_risk_level": d_level,
                "obesity_risk_percentage": round(prob_obesity, 2),
                "obesity_risk_level": o_level,
                "combined_risk_percentage": round(prob_combined, 2),
                "combined_risk_level": c_level
            },
            "recommendations": recommendations,
            "patient_summary": {
                "bmi": data.BMI,
                "age_category": data.Age,
                "general_health": data.GenHlth
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "diabetes_model_loaded": diabetes_model is not None,
        "obesity_model_loaded": obesity_model is not None,
        "scaler_loaded": scaler is not None
    }

# Serve frontend static files
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Backend API is running. Please add index.html to static folder."}
