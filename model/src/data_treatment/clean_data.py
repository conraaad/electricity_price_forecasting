import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../../data/def_data/training_dataset_2024.csv')

# Basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check for infinite values
print("\nInfinite values per column:")
for col in df.columns:
    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"{col}: {inf_count}")

# Assuming your target column is the last column or named 'target', 'label', 'y', etc.
# Adjust the column name as needed
target_col = df.columns[-1]  # or specify the exact name like 'target'

print(f"Target column: {target_col}")
print(f"Target column dtype: {df[target_col].dtype}")
print(f"Target column stats:")
print(df[target_col].describe())

# Check for problematic values in target column
print(f"\nNaN values in target: {df[target_col].isnull().sum()}")
print(f"Infinite values in target: {np.isinf(df[target_col]).sum()}")
print(f"Maximum value in target: {df[target_col].max()}")
print(f"Minimum value in target: {df[target_col].min()}")

# Check for extremely large values
large_threshold = 1e10
print(f"Values > {large_threshold}: {(df[target_col] > large_threshold).sum()}")
print(f"Values < -{large_threshold}: {(df[target_col] < -large_threshold).sum()}")