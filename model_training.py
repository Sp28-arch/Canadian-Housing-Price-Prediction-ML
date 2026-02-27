# week7_model_training.py
# Week 7: Model Training & Evaluation for House Price Prediction
# Team: Shrey Patel, Zarlasht Miakhail, Yuzhe Lin






# -------------------------------
# Step 1: Import Libraries
# -------------------------------
# Assigned to: All
# TODO:
# - Import necessary libraries for modeling and evaluation
#   (e.g., LinearRegression, train_test_split, mean_squared_error, r2_score, numpy, pandas)




# -------------------------------
# Step 2: Load Preprocessed Data
# -------------------------------
# Assigned to: All
# TODO:
# - Load the preprocessed dataset from Week 6
# - Define features X and target y
# - Split the data into training and testing sets (train_test_split)

# -------------------------------
# Step 3: Initialize Linear Regression Model
# -------------------------------
# Assigned to: Shrey
# TODO:
# - Create an instance of LinearRegression()
# - This is the baseline model for predictions

# -------------------------------
# Step 4: Train Model on Training Data
# -------------------------------
# Assigned to: Yuzhe
# TODO:
# - Fit the Linear Regression model to the training data (X_train, y_train)

# -------------------------------
# Step 5: Make Predictions
# -------------------------------
# Assigned to: Yuzhe
# TODO:
# - Use the trained model to predict house prices on the test data (X_test)
# - Store predictions in a variable (y_pred)





# -------------------------------
# Step 6: Evaluate Model Performance
# -------------------------------
# Assigned to: Zarlasht
# TODO:
# - Calculate evaluation metrics:
#   - Mean Squared Error (MSE)
#   - Root Mean Squared Error (RMSE)
#   - R² score
# - Print evaluation results to console




# -------------------------------
# Step 7: Save Predictions (Optional)
# -------------------------------
# Assigned to: All
# TODO:
# - Optionally, save a CSV with test features, actual prices, and predicted prices
# - This file will be used for visualization in Week 8