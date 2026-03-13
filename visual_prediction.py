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

# %%
# visual_prediction_fixed.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
scaled_data_file = "Project/house_data_processed_scaled.csv"
original_data_file = "Project/house_data_processed_full.csv"  # original values for plots
model_file = "Project/linear_regression_model.joblib"
scaler_file = "Project/scaler.joblib"
cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']
numeric_features = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km','Population_Density','Year_to_build']

# -----------------------------
# LOAD DATA
# -----------------------------
try:
    data_scaled = pd.read_csv(scaled_data_file)
    data_original = pd.read_csv(original_data_file)
except FileNotFoundError:
    raise SystemExit("Error: CSV files not found. Make sure you have both scaled and original CSVs!")

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
try:
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
except FileNotFoundError:
    raise SystemExit("Error: Model or scaler not found. Train the model first!")

# -----------------------------
# ENSURE CITY COLUMNS
# -----------------------------
for c in cities:
    if f'City_{c}' not in data_scaled.columns:
        if 'City' in data_scaled.columns:
            data_scaled[f'City_{c}'] = (data_scaled['City'].str.capitalize() == c).astype(int)
        else:
            data_scaled[f'City_{c}'] = 0

all_features = numeric_features + [f"City_{c}" for c in cities]

# -----------------------------
# SCALE NUMERIC FEATURES (if not already)
# -----------------------------
data_scaled.loc[:, numeric_features] = scaler.transform(data_scaled[numeric_features])

# -----------------------------
# PREDICT
# -----------------------------
data_scaled['Predicted_Price'] = model.predict(data_scaled[all_features])
data_scaled['Predicted_Price'] = data_scaled['Predicted_Price'].clip(lower=0)

# -----------------------------
# PLOTS: Combined Features vs Price
# -----------------------------
feature_groups = {
    "Total_Area_sqft + Bedrooms": ['Total_Area_sqft','Bedrooms'],
    "Distance_to_CityCenter_km + Population_Density": ['Distance_to_CityCenter_km','Population_Density'],
    "Total_Area_sqft + Bedrooms + Distance_to_CityCenter_km": ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km']
}

for group_name, features in feature_groups.items():
    # Use original numeric values for x-axis
    combined_feature = data_original[features].sum(axis=1)

    plt.figure(figsize=(8,6))
    plt.scatter(combined_feature, data_original['Price'], alpha=0.5, label='Actual Price')
    plt.scatter(combined_feature, data_scaled['Predicted_Price'], alpha=0.5, label='Predicted Price')
    plt.xlabel(" + ".join(features))
    plt.ylabel("Price ($)")
    plt.title(f"{group_name} vs House Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{group_name.replace(' ', '_')}_vs_price.png")
    plt.show()

    # -----------------------------
# PLOT: Predicted Price vs Actual Price
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(data_original['Price'], data_scaled['Predicted_Price'], alpha=0.5, color='purple')
plt.plot([data_original['Price'].min(), data_original['Price'].max()],
         [data_original['Price'].min(), data_original['Price'].max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Predicted Price vs Actual Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Predicted_vs_Actual_Price.png")
plt.show()

# %%

# %%

# %%
