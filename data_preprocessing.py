import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load original dataset
data = pd.read_csv("house_cost_1M v2.0 - modified.csv")

# Drop irrelevant columns
data = data.drop(columns=['ID', 'Country'])

# Drop missing values
data = data.dropna()

# Keep a copy of original numeric features for plots
original_numeric_features = ['Total_Area_sqft', 'Bedrooms', 'Distance_to_CityCenter_km',
                             'Year_to_build', 'Population_Density']
data_original = data[original_numeric_features + ['Price']].copy()  # Keep Price too

# One-hot encode City
data = pd.get_dummies(data, columns=['City'])

# Standardize numeric features for modeling only
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[original_numeric_features] = scaler.fit_transform(data[original_numeric_features])

# Save both for later use
data_scaled.to_csv("house_data_processed_scaled.csv", index=False)
data_original.to_csv("house_data_processed_full.csv", index=False)  # original numbers for plots

print(f"Processed dataset saved. Scaled for model: {data_scaled.shape}, Original for plots: {data_original.shape}")