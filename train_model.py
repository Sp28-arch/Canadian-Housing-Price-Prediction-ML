import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

data_file = "/home/jupyter/house_data_processed_scaled.csv"
model_file = "/home/jupyter/linear_regression_model.joblib"
prediction_file = "/home/jupyter/house_price_predictions.csv"

# LOAD DATA
data = pd.read_csv(data_file)

# ENCODE CITIES IF NEEDED
cities = ['Calgary', 'Montreal', 'Ottawa', 'Toronto', 'Vancouver']
for c in cities:
    if f'City_{c}' not in data.columns:
        if 'City' in data.columns:
            data[f'City_{c}'] = (data['City'].str.capitalize() == c).astype(int)
        else:
            data[f'City_{c}'] = 0

# FEATURES & TARGET
numeric_features = [
    'Total_Area_sqft',
    'Bedrooms',
    'Distance_to_CityCenter_km',
    'Population_Density',
    'Year_to_build'
]

feature_cols = numeric_features + [f'City_{c}' for c in cities]

X = data[feature_cols].copy()
y = data['Price']

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# MAKE PREDICTIONS
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

# EVALUATE MODEL
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# SAVE PREDICTIONS WITH ORIGINAL INDEX
predictions_df = pd.DataFrame({
    "Original_Index": X_test.index,
    "Actual_Price": y_test.values,
    "Predicted_Price": y_pred
})

predictions_df.to_csv(prediction_file, index=False)
print("\nSaved predictions to:", prediction_file)

# SAVE MODEL
joblib.dump(model, model_file)
print("Model saved successfully!")
