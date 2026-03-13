# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# clean_up.py
import os
import glob

# List of files to delete
files_to_remove = [
    "linear_regression_model.joblib",
    "scaler.joblib",
    "actual_vs_predicted_price.png"
]

# Also delete any numeric feature vs price plots
feature_plot_files = glob.glob("*_vs_price.png")

files_to_remove.extend(feature_plot_files)

# Delete files
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted: {file}")
    else:
        print(f"Not found, skipped: {file}")

print("\nCleanup complete. You can now start fresh!")

# %%

# %%

# %%

# %%

# %%
