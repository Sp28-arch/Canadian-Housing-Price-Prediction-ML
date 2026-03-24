import pandas as pd
import matplotlib.pyplot as plt

full_file = "/home/jupyter/house_data_processed_full.csv"
prediction_file = "/home/jupyter/house_price_predictions.csv"
price_unit = 1000

# LOAD ORIGINAL DATA AND PREDICTIONS
full_data = pd.read_csv(full_file)
pred_data = pd.read_csv(prediction_file)

# MATCH TEST PREDICTIONS BACK TO ORIGINAL ROWS
plot_data = full_data.loc[pred_data["Original_Index"]].copy()
plot_data["Actual_Price"] = pred_data["Actual_Price"].values
plot_data["Predicted_Price"] = pred_data["Predicted_Price"].values

# 1. ACTUAL VS PREDICTED PRICE
plt.figure(figsize=(8, 6))
plt.scatter(plot_data["Actual_Price"], plot_data["Predicted_Price"], alpha=0.6)
plt.plot(
    [plot_data["Actual_Price"].min(), plot_data["Actual_Price"].max()],
    [plot_data["Actual_Price"].min(), plot_data["Actual_Price"].max()],
    "r--",
    label="Perfect Prediction"
)
plt.xlabel(f"Actual Price ($, 1 unit = ${price_unit})")
plt.ylabel(f"Predicted Price ($, 1 unit = ${price_unit})")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. TWO-FEATURE GRAPHS
feature_pairs = {
    "Square Footage and Bedrooms": ["Total_Area_sqft", "Bedrooms"],
    "Distance to City Center and Population Density": ["Distance_to_CityCenter_km", "Population_Density"],
    "Square Footage and Year Built": ["Total_Area_sqft", "Year_to_build"],
    "Bedrooms and Population Density": ["Bedrooms", "Population_Density"],
    "Square Footage and Distance to City Center": ["Total_Area_sqft", "Distance_to_CityCenter_km"]
}

for graph_name, features in feature_pairs.items():
    x_feature = features[0]
    y_feature = features[1]

    # Actual price color graph
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        plot_data[x_feature],
        plot_data[y_feature],
        c=plot_data["Actual_Price"],
        cmap="viridis",
        alpha=0.7
    )
    plt.colorbar(scatter, label=f"Actual Price ($, 1 unit = ${price_unit})")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"{graph_name} vs Actual Price")

    scale_text = (
        f"{x_feature}: min={plot_data[x_feature].min():.2f}, max={plot_data[x_feature].max():.2f}\n"
        f"{y_feature}: min={plot_data[y_feature].min():.2f}, max={plot_data[y_feature].max():.2f}\n"
        f"Actual Price: min={plot_data['Actual_Price'].min():.2f}, max={plot_data['Actual_Price'].max():.2f}"
    )

    plt.text(
        0.95, 0.02, scale_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Predicted price color graph
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        plot_data[x_feature],
        plot_data[y_feature],
        c=plot_data["Predicted_Price"],
        cmap="plasma",
        alpha=0.7
    )
    plt.colorbar(scatter, label=f"Predicted Price ($, 1 unit = ${price_unit})")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f"{graph_name} vs Predicted Price")

    scale_text = (
        f"{x_feature}: min={plot_data[x_feature].min():.2f}, max={plot_data[x_feature].max():.2f}\n"
        f"{y_feature}: min={plot_data[y_feature].min():.2f}, max={plot_data[y_feature].max():.2f}\n"
        f"Predicted Price: min={plot_data['Predicted_Price'].min():.2f}, max={plot_data['Predicted_Price'].max():.2f}"
    )

    plt.text(
        0.95, 0.02, scale_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. SINGLE-FEATURE GRAPHS
single_features = [
    "Total_Area_sqft",
    "Bedrooms",
    "Distance_to_CityCenter_km",
    "Population_Density",
    "Year_to_build"
]

for feature in single_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_data[feature], plot_data["Actual_Price"], alpha=0.5, label="Actual Price")
    plt.scatter(plot_data[feature], plot_data["Predicted_Price"], alpha=0.5, label="Predicted Price")

    plt.xlabel(feature)
    plt.ylabel(f"Price ($, 1 unit = ${price_unit})")
    plt.title(f"{feature} vs House Price")

    scale_text = (
        f"{feature}: min={plot_data[feature].min():.2f}, max={plot_data[feature].max():.2f}\n"
        f"Actual Price: min={plot_data['Actual_Price'].min():.2f}, max={plot_data['Actual_Price'].max():.2f}\n"
        f"Predicted Price: min={plot_data['Predicted_Price'].min():.2f}, max={plot_data['Predicted_Price'].max():.2f}"
    )

    plt.text(
        0.95, 0.02, scale_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
