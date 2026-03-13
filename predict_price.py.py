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
# predict_price_simple.py
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL AND SCALER
# -----------------------------
try:
    model = joblib.load("linear_regression_model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    raise SystemExit("Error: Model or scaler not found. Train the model first!")

# -----------------------------
# LOAD CSV (for feature info)
# -----------------------------
try:
    data = pd.read_csv("house_data_processed_scaled.csv")
except FileNotFoundError:
    raise SystemExit("Error: CSV file not found!")

# -----------------------------
# FEATURES
# -----------------------------
cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']
numeric_features = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km','Population_Density','Year_to_build']
all_features = numeric_features + [f"City_{c}" for c in cities]

# -----------------------------
# USER INPUT
# -----------------------------
print("--- House Price Prediction ---")
print("Press ENTER to skip a value.")

user_input = {}
for f in numeric_features:
    val = input(f"{f.replace('_',' ')}: ").strip()
    if val != "":
        user_input[f] = float(val)

city_input = input(f"City ({', '.join(cities)}): ").strip().capitalize()
for c in cities:
    user_input[f"City_{c}"] = 1 if city_input == c else 0

# -----------------------------
# CREATE TEMPLATE ROW
# -----------------------------
template_row = pd.DataFrame([user_input], columns=all_features)

# SCALE NUMERIC FEATURES
features_to_scale = [f for f in numeric_features if f in template_row.columns]
if features_to_scale:
    template_row.loc[:, features_to_scale] = scaler.transform(template_row[features_to_scale])

# -----------------------------
# PREDICT PRICE
# -----------------------------
prediction = model.predict(template_row)[0]
prediction = max(prediction, 0)  # ensure positive

print(f"\nPredicted House Price: $ {prediction:,.2f}")

# %%
