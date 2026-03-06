# week7_model_training.py
# Week 7: Model Training & Evaluation for House Price Prediction
# Team: Shrey Patel, Zarlasht Miakhail, Yuzhe Lin

# ---------------------------------------------------------
# Step 1: Import Libraries
# ---------------------------------------------------------
# Assigned to: All

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------
# Step 2: Load Preprocessed Data
# ---------------------------------------------------------
# Assigned to: All

# Load the preprocessed dataset created in Week 6
data = pd.read_csv("house_cost_1M v2.0 - modified_preprocessed.csv")

# Define features (X) and target (y)
X = data.drop(columns=["Price"])
y = data["Price"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Data loaded and split into training/testing sets.")


# ---------------------------------------------------------
# Step 3: Initialize Linear Regression Model
# ---------------------------------------------------------
# Assigned to: Shrey

baseline_model = LinearRegression()
print("Linear Regression model initialized.")


# ---------------------------------------------------------
# Step 4: Train Model on Training Data
# ---------------------------------------------------------
# Assigned to: Yuzhe

baseline_model.fit(X_train, y_train)
print("Model training complete.")


# ---------------------------------------------------------
# Step 5: Make Predictions
# ---------------------------------------------------------
# Assigned to: Yuzhe

y_pred = baseline_model.predict(X_test)
print("Predictions generated on test set.")


# ---------------------------------------------------------
# Step 6: Evaluate Model Performance
# ---------------------------------------------------------
# Assigned to: Zarlasht

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R² Score
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# ---------------------------------------------------------
# Step 7: Save Predictions (Optional)
# ---------------------------------------------------------
# Assigned to: All

save_output = True  # Change to False if you don't want to save

if save_output:
    results = pd.DataFrame({
        "Actual_Price": y_test.values,
        "Predicted_Price": y_pred,
    })

    results.to_csv("week7_predictions_output.csv", index=False)
    print("\nPredictions saved to week7_predictions_output.csv")

print("\nWeek 7 model training & evaluation complete.")
