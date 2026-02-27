


import pandas as pd
import sklearn


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Step 2: Load Dataset into Python
# Assigned to: Shrey
# The CSV file is already in the same folder as this script
data = pd.read_csv("house_cost_1M v2.0 - modified.csv")

# Step 3: Explore Dataset Structure
# Assigned to: Shrey
# Print the first 5 rows to see sample data
print("First 5 rows of dataset:")
print(data.head())

# Print info about column types and missing values
print("\nDataset info (column types, missing values):")
print(data.info())

# Print summary statistics of numerical columns
print("\nSummary statistics of numeric columns:")
print(data.describe())

# Step 4: Handle Missing Values / Outliers
# Assigned to: Shrey
# Check missing values per column
print("\nMissing values per column:")
print(data.isnull().sum())

# Drop rows with missing values (simple approach)
data_clean = data.dropna()

# Optional: You can also add outlier handling here if needed

# Step 5: Encode Categorical Features
# Assigned to: Yuzhe
# One-hot encode categorical columns like 'City'
data_encoded = pd.get_dummies(data_clean, columns=['City'])

# Step 6: Normalize / Standardize Numerical Features
# Assigned to: Yuzhe
# Standardize features like Total Area, Bedrooms, etc.
numeric_features = ['Total_Area_sqft', 'Bedrooms', 'Distance_to_CityCenter_km', 'Year_to_build']
scaler = StandardScaler()
data_encoded[numeric_features] = scaler.fit_transform(data_encoded[numeric_features])

# Step 7: Define Features (X) & Target (y)
# Assigned to: All
# Drop the target and any non-informative columns to get X
X = data_encoded.drop(columns=['Price', 'Country'])
# Set the target variable y
y = data_encoded['Price']

# Step 8: Baseline Model Selection
# Assigned to: All
# Select the Linear Regression model as baseline
baseline_model = LinearRegression()

print("\nData preprocessing complete. Features and target are ready for modeling.")
print("Baseline model selected: LinearRegression")