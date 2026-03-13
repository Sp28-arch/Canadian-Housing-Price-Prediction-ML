import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model, scaler, data
try:
    model = joblib.load("linear_regression_model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Train the model first!")
    exit()

data_file = "house_data_processed_scaled.csv"
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: {data_file} not found!")
    data = None  

# Feature setup
cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']
numeric_features = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km','Population_Density','Year_to_build']
all_features = numeric_features + [f"City_{c}" for c in cities]

# User input
print("--- House Price Prediction ---")
print("Press ENTER to skip a value.")

user_input = {}
used_features = []  
for f in numeric_features:
    val = input(f"{f.replace('_',' ')}: ")
    if val.strip() != "":
        user_input[f] = float(val)
        used_features.append(f)

city_input = input(f"City ({', '.join(cities)}): ").strip().capitalize()
for c in cities:
    user_input[f"City_{c}"] = 1 if city_input == c else 0

# Create template row
template_row = pd.DataFrame([user_input], columns=[f for f in all_features if f in user_input or f.startswith('City_')])

# Scale numeric features that were provided
features_to_scale = [f for f in numeric_features if f in template_row.columns]
if features_to_scale:
    template_row[features_to_scale] = scaler.transform(template_row[features_to_scale])

# Predict price
prediction = model.predict(template_row.reindex(columns=all_features, fill_value=0))[0]
prediction = max(prediction, 0)  # ensure positive
print(f"\nPredicted House Price: $ {prediction:,.2f}")



        # Actual vs Predicted Price
plt.figure(figsize=(8,6))
plt.scatter(data_copy['Price'], data_copy['Predicted_Price'], alpha=0.6)
plt.plot([data_copy['Price'].min(), data_copy['Price'].max()],
                 [data_copy['Price'].min(), data_copy['Price'].max()],
                 'r--', label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_price.png")
plt.show()

        # Feature vs Price plots (only numeric)
for feature in numeric_features:
            plt.figure(figsize=(8,6))
            plt.scatter(data_copy[feature], data_copy['Price'], alpha=0.5, label='Actual Price')
            plt.scatter(data_copy[feature], data_copy['Predicted_Price'], alpha=0.5, label='Predicted Price')
            plt.xlabel(feature.replace("_"," "))
            plt.ylabel('Price ($)')
            plt.title(f'{feature.replace("_"," ")} vs House Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{feature}_vs_price.png')
            plt.show()