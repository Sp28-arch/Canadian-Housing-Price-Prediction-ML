import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Step 2: Load Dataset into Python
# Assigned to: Shrey
# ---------------------------------------------------------

# Load CSV (file must be in the same directory)
data = pd.read_csv("house_cost_1M v2.0 - modified.csv")

# ---------------------------------------------------------
# Step 3: Explore Dataset Structure
# Assigned to: Shrey
# ---------------------------------------------------------

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset info (column types, missing values):")
print(data.info())

print("\nSummary statistics of numeric columns:")
print(data.describe())

# ---------------------------------------------------------
# Step 4: Handle Missing Values / Outliers
# Assigned to: Shrey
# ---------------------------------------------------------

print("\nMissing values per column:")
print(data.isnull().sum())

# Simple missing-value handling: drop rows with any missing values
data_clean = data.dropna()

# (Optional) Outlier handling could be added here

# ---------------------------------------------------------
# Step 5: Encode Categorical Features
# Assigned to: Yuzhe
# ---------------------------------------------------------

# One-hot encode categorical column(s)
data_encoded = pd.get_dummies(data_clean, columns=['City'])

# ---------------------------------------------------------
# Step 6: Normalize / Standardize Numerical Features
# Assigned to: Yuzhe
# ---------------------------------------------------------

numeric_features = [
    'Total_Area_sqft',
    'Bedrooms',
    'Distance_to_CityCenter_km',
    'Year_to_build'
]

scaler = StandardScaler()
data_encoded[numeric_features] = scaler.fit_transform(data_encoded[numeric_features])

# ---------------------------------------------------------
# Step 7: Define Features (X) & Target (y)
# Assigned to: All
# ---------------------------------------------------------

# Remove target + non-informative columns
X = data_encoded.drop(columns=['Price', 'Country'])

# Target variable
y = data_encoded['Price']

# ---------------------------------------------------------
# Step 8: Baseline Model Selection
# Assigned to: All
# ---------------------------------------------------------

baseline_model = LinearRegression()

print("\nData preprocessing complete. Features and target are ready for modeling.")
print("Baseline model selected: LinearRegression")
