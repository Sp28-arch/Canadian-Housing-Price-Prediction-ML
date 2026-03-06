

import pandas as pd
from sklearn.preprocessing import StandardScaler

#Load Dataset


data = pd.read_csv("house_cost_1M v2.0 - modified.csv")
print(f"Original dataset has {data.shape[0]} rows and {data.shape[1]} columns.\n")




#  Drop Irrelevant Columns

data_clean = data.drop(columns=['ID', 'Country'])
print(f"After dropping irrelevant columns, dataset has {data_clean.shape[0]} rows and {data_clean.shape[1]} columns.\n")





# Handle Missing Values (none in this dataset)
data_clean = data_clean.dropna()
print(f"After dropping missing values, dataset has {data_clean.shape[0]} rows.\n")





#  Select Numeric Features
numeric_features = ['Total_Area_sqft', 'Bedrooms', 'Distance_to_CityCenter_km', 'Year_to_build', 'Population_Density']





# Encode Categorical Feature (City)
data_encoded = pd.get_dummies(data_clean, columns=['City'])
print(f"After encoding City, dataset has {data_encoded.shape[1]} columns.\n")

# numeric + encoded categorical features
feature_columns = numeric_features + [col for col in data_encoded.columns if 'City_' in col]


X = data_encoded[feature_columns]
y = data_encoded['Price']
print(f"Features ready for modeling: {len(feature_columns)} columns\n")

 
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])
print("Numeric features standardized.\n")


# Save Processed Dataset
processed_data = X.copy()
processed_data['Price'] = y
processed_data.to_csv("house_data_processed_full.csv", index=False)
print(f"Processed dataset saved as 'house_data_processed_full.csv' with {processed_data.shape[0]} rows and {processed_data.shape[1]} columns.\n")




print("Preview of processed dataset:")
print(processed_data.head())