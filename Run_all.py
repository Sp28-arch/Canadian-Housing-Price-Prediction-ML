# Run_all.py
import subprocess
import sys
import os

# ------------------------------
# 1. Preprocess the data
# ------------------------------
print("\n=== Step 1: Preprocessing Data ===\n")
subprocess.run([sys.executable, "data_preprocessing.py"])

# ------------------------------
# 2. Train the Linear Regression model
# ------------------------------
print("\n=== Step 2: Training Model ===\n")
subprocess.run([sys.executable, "train_model.py"])

# ------------------------------
# 3. Predict house price (with user input)
# ------------------------------
print("\n=== Step 3: Predict House Price ===\n")
subprocess.run([sys.executable, "predict_price.py"])

# ------------------------------
# 4. Visualize predictions
# ------------------------------
print("\n=== Step 4: Visualizing Predictions ===\n")
# Make sure the CSV names match the preprocessing step
if os.path.exists("house_data_processed_full.csv") and os.path.exists("house_predictions.csv"):
    subprocess.run([sys.executable, "visualize_predictions.py"])
else:
    print("Warning: Processed data or predictions not found. Skipping visualization.")