import os

# List of generated files to delete
files_to_delete = [
    "house_data_processed_scaled.csv",
    "house_data_processed_full.csv",
    "linear_regression_model.joblib",
    "scaler.joblib",
    "house_predictions.csv",
    "actual_vs_predicted_price.png",
    "Total_Area_sqft_+_Bedrooms_vs_price.png",
    "Distance_to_CityCenter_km_+_Population_Density_vs_price.png",
    "Total_Area_sqft_+_Bedrooms_+_Distance_to_CityCenter_km_vs_price.png",
    "Total_Area_sqft_vs_price.png",
    "Bedrooms_vs_price.png",
    "Distance_to_CityCenter_km_vs_price.png",
    "Year_to_build_vs_price.png",
    "Population_Density_vs_price.png"
]

deleted_files = []
for file in files_to_delete:
    if os.path.exists(file):
        try:
            os.remove(file)
            deleted_files.append(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

if deleted_files:
    print("Deleted files:")
    for f in deleted_files:
        print(f"  - {f}")
else:
    print("No generated files were found to delete.")