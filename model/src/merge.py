import pandas as pd


# Load both CSV files
dataset_1_df = pd.read_csv('../data/results/interchange_data.csv')
dataset_2_df = pd.read_csv('../data/results/demand_data.csv')

# Make sure datetime columns are in the same format
dataset_1_df['datetime_iso'] = pd.to_datetime(dataset_1_df['datetime_iso'])
dataset_2_df['datetime_iso'] = pd.to_datetime(dataset_2_df['datetime_iso'])

# Merge the dataframes on datetime columns
merged_df = pd.merge(
    dataset_2_df,
    dataset_1_df[['datetime_iso', 'interchange_balance']],  # Select both datetime and value columns
    on='datetime_iso',  # Join on the datetime column
    how='left'  # Keep all rows from dataset_2_df
)

print(f"Merged DataFrame has {len(merged_df)} rows and {len(merged_df.columns)} columns")
print(f"Columns: {merged_df.columns.tolist()}")

print(merged_df.head())

# Save the result to a new CSV file
merged_df.to_csv('../data/results/def_dataset.csv', index=False)

