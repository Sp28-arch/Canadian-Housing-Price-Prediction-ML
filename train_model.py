# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# CONFIG
# -----------------------------
data_file = "Project/house_data_processed_scaled.csv"
model_file = "Project/linear_regression_model.joblib"
scaler_file = "Project/scaler.joblib"

# -----------------------------
# LOAD DATA
# -----------------------------
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    raise SystemExit(f"Error: CSV file '{data_file}' not found!")

# -----------------------------
# ENCODE CITIES
# -----------------------------
cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']
for c in cities:
    if f'City_{c}' not in data.columns:
        if 'City' in data.columns:
            data[f'City_{c}'] = (data['City'].str.capitalize() == c).astype(int)
        else:
            data[f'City_{c}'] = 0

# -----------------------------
# FEATURES & TARGET
# -----------------------------
numeric_features = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km',
                    'Population_Density','Year_to_build']
feature_cols = numeric_features + [f'City_{c}' for c in cities]

X = data[feature_cols].copy()
y = data['Price']

# -----------------------------
# SCALE NUMERIC FEATURES
# -----------------------------
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# -----------------------------
# SPLIT DATASET
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# SAVE MODEL & SCALER
# -----------------------------
joblib.dump(model, model_file)
joblib.dump(scaler, scaler_file)

print("✅ Model and scaler saved successfully!")

# %%

# %%
