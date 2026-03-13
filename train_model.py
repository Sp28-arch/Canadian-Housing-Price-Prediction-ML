# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv("house_data_processed_scaled.csv")

# Encode cities
cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']
for c in cities:
    data[f'City_{c}'] = (data['City'].str.capitalize() == c)

# Features and target
feature_cols = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km','Population_Density','Year_to_build'] + [f'City_{c}' for c in cities]
X = data[feature_cols]
y = data['Price']

# Standardize numeric features
numeric_cols = ['Total_Area_sqft','Bedrooms','Distance_to_CityCenter_km','Population_Density','Year_to_build']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, "linear_regression_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully!")