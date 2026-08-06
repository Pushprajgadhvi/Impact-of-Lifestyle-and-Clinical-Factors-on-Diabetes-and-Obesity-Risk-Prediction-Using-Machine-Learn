import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

print("Fetching dataset...")
cdc_data = fetch_ucirepo(id=891)
X_raw = cdc_data.data.features
y_raw = cdc_data.data.targets

df_master = pd.concat([X_raw, y_raw], axis=1)
df_master['Obesity'] = (df_master['BMI'] >= 30).astype(int)

# To ensure fast execution and robust training, let's take a representative sample or full dataset if feasible.
# The dataset has 253680 records. Training on 40,000 balanced records or full dataset. Let's use 50,000 samples for fast reliable training.
df_sample = df_master.sample(n=50000, random_state=42)

numerical_cols = ['BMI', 'Age', 'GenHlth', 'MentHlth', 'PhysHlth', 'Education', 'Income']
feature_names = [col for col in df_sample.columns if col not in ['Diabetes_binary', 'Obesity']]

X = df_sample[feature_names]
y_diabetes = df_sample['Diabetes_binary']
y_obesity = df_sample['Obesity']

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_d_train, y_d_test = train_test_split(X, y_diabetes, test_size=0.2, random_state=42)
_, _, y_o_train, y_o_test = train_test_split(X, y_obesity, test_size=0.2, random_state=42)

print("Training Diabetes Random Forest Model...")
diabetes_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
diabetes_model.fit(X_train, y_d_train)

print("Training Obesity Random Forest Model...")
obesity_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
obesity_model.fit(X_train, y_o_train)

print("Saving model artifacts...")
with open('/home/ubuntu/diabetes_model.pkl', 'wb') as f:
    pickle.dump(diabetes_model, f)

with open('/home/ubuntu/obesity_model.pkl', 'wb') as f:
    pickle.dump(obesity_model, f)

with open('/home/ubuntu/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('/home/ubuntu/feature_names.json', 'w') as f:
    json.dump({
        "feature_names": feature_names,
        "numerical_cols": numerical_cols
    }, f)

print("Model training and artifact saving complete successfully!")
