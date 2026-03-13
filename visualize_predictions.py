import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
data = pd.read_csv("house_data_processed_full.csv")
model_file = "linear_regression_model.joblib"
scaler_file = "scaler.joblib"
price_unit = 1000  # for labeling the axis

# -----------------------------
# LOAD DATA
# -----------------------------
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: CSV file '{data_file}' not found.")
    exit()

# Encode city columns
cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']
for c in cities:
    data[f'City_{c}'] = (data['City'].str.capitalize() == c).astype(int)

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
try:
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
except FileNotFoundError:
    print("Error: Model or scaler not found. Make sure linear_regression_model.joblib and scaler.joblib exist.")
    exit()

# -----------------------------
# PREPARE DATA FOR PREDICTION
# -----------------------------
feature_cols = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km',
                'Population_Density','Year_to_build'] + [f'City_{c}' for c in cities]

numeric_features = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km',
                    'Population_Density','Year_to_build']

X = data[feature_cols].copy()
X[numeric_features] = scaler.transform(X[numeric_features])

# -----------------------------
# PREDICT
# -----------------------------
data['Predicted_Price'] = model.predict(X)
data['Predicted_Price'] = data['Predicted_Price'].clip(lower=0)  # ensure no negatives

# -----------------------------
# ACTUAL VS PREDICTED PRICE
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(data['Price'], data['Predicted_Price'], alpha=0.6)
plt.plot([data['Price'].min(), data['Price'].max()],
         [data['Price'].min(), data['Price'].max()],
         'r--', label='Perfect Prediction')
plt.xlabel(f'Actual Price ($, 1 unit = ${price_unit})')
plt.ylabel(f'Predicted Price ($, 1 unit = ${price_unit})')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_price.png")
plt.show()

# -----------------------------
# MULTI-FEATURE SUM GRAPHS
# -----------------------------
feature_groups = {
    "Total_Area_sqft + Bedrooms": ['Total_Area_sqft','Bedrooms'],
    "Distance_to_CityCenter_km + Population_Density": ['Distance_to_CityCenter_km','Population_Density'],
    "Total_Area_sqft + Bedrooms + Distance_to_CityCenter_km": ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km']
}

for group_name, features in feature_groups.items():
    # Skip if all features are missing
    if data[features].isnull().all().all():
        continue

    combined_feature = data[features].sum(axis=1)

    plt.figure(figsize=(8,6))
    plt.scatter(combined_feature, data['Price'], alpha=0.5, label='Actual Price')
    plt.scatter(combined_feature, data['Predicted_Price'], alpha=0.5, label='Predicted Price')
    plt.xlabel(" + ".join(features))
    plt.ylabel(f"Price ($, 1 unit = ${price_unit})")
    plt.title(f"{group_name} vs House Price")

    # Add min/max scales in the plot
    feature_scale_text = "\n".join([f"{f}: min={data[f].min()}, max={data[f].max()}" for f in features])
    price_scale_text = f"Price: min={data['Price'].min():.2f}, max={data['Price'].max():.2f} $"
    plt.text(0.95, 0.02,
             f"{feature_scale_text}\n{price_scale_text}",
             transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{group_name.replace(' ', '_')}_vs_price.png")
    plt.show()