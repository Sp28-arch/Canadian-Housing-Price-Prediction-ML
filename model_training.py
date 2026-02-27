import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

# -------------------------------
# Step 2: Load Dataset from Online Source
# -------------------------------
# Assigned to: Shrey
# Load the full dataset from Hugging Face
dataset = load_dataset("bdstar/house-cost-prediction-multivariances", split="train")

# Convert to Pandas DataFrame
data = pd.DataFrame(dataset)

# Filter for Canada
data = data[data['Country'] == 'Canada']

# -------------------------------
# Step 3: Explore Dataset Structure
# -------------------------------
# Assigned to: Shrey
print("First 5 rows of dataset:")
print(data.head())

print("\nDataset info (column types, missing values):")
print(data.info())

print("\nSummary statistics of numeric columns:")
print(data.describe())

# -------------------------------
# Step 4: Handle Missing Values / Outliers
# -------------------------------
# Assigned to: Shrey
# Check missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Drop rows with missing values
data_clean = data.dropna()

# Optional: outlier handling could be added here (not implemented for simplicity)

# -------------------------------
# Step 5: Encode Categorical Features
# -------------------------------
# Assigned to: Yuzhe
# One-hot encode the 'City' column
data_encoded = pd.get_dummies(data_clean, columns=['City'])

# -------------------------------
# Step 6: Normalize / Standardize Numerical Features
# -------------------------------
# Assigned to: Yuzhe
numeric_features = ['Total_Area_sqft', 'Bedrooms', 'Distance_to_CityCenter_km', 'Year_to_build']
scaler = StandardScaler()
data_encoded[numeric_features] = scaler.fit_transform(data_encoded[numeric_features])

# -------------------------------
# Step 7: Define Features (X) & Target (y)
# -------------------------------
# Assigned to: All
X = data_encoded.drop(columns=['Price', 'Country'])  # Drop target and non-informative columns
y = data_encoded['Price']  # Target variable

# -------------------------------
# Step 8: Baseline Model Selection
# -------------------------------
# Assigned to: All
from sklearn.linear_model import LinearRegression
baseline_model = LinearRegression()

print("\nData preprocessing complete. Features and target are ready for modeling.")
print("Baseline model selected: LinearRegression")