import pandas as pd
import joblib
import numpy as np

model_file = "/home/jupyter/linear_regression_model.joblib"
scaler_file = "/home/jupyter/scaler.joblib"
full_data_file = "/home/jupyter/house_data_processed_full.csv"

# LOAD MODEL, SCALER, AND ORIGINAL DATA
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
full_data = pd.read_csv(full_data_file)

cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']

numeric_features = [
    'Total_Area_sqft',
    'Bedrooms',
    'Distance_to_CityCenter_km',
    'Population_Density',
    'Year_to_build'
]

all_features = numeric_features + [f'City_{c}' for c in cities]

# USE DATASET AVERAGES IF USER SKIPS A VALUE
defaults = {
    feature: float(full_data[feature].mean())
    for feature in numeric_features
}

print("--- House Price Prediction ---")
print("Press ENTER to use the average value.\n")

user_input = {}

for feature in numeric_features:
    value = input(f"{feature.replace('_', ' ')}: ").strip()
    if value == "":
        user_input[feature] = defaults[feature]
    else:
        user_input[feature] = float(value)

city_input = input(f"City ({', '.join(cities)}): ").strip().capitalize()

for city in cities:
    user_input[f"City_{city}"] = 1 if city_input == city else 0

# CREATE INPUT ROW
template_row = pd.DataFrame([user_input], columns=all_features)

# SCALE NUMERIC FEATURES IN THE EXACT ORDER THE SCALER EXPECTS
scaled_numeric = scaler.transform(template_row[numeric_features].values)
template_row.loc[:, numeric_features] = scaled_numeric

# PREDICT PRICE
prediction = model.predict(template_row)[0]
prediction = max(prediction, 0)

print(f"\nPredicted House Price: $ {prediction:,.2f}")
