import pandas as pd


export_path = '../data/results/def_data/merged_def_dataset_solar.csv'
# Load both CSV files
dataset_1_df = pd.read_csv('../data/results/def_data/merged_def_dataset_solar.csv')
dataset_2_df = pd.read_csv('../data/results/temperatura_data.csv')

# Make sure datetime columns are in the same format
dataset_1_df['datetime_iso'] = pd.to_datetime(dataset_1_df['datetime_iso'])
dataset_2_df['datetime_iso'] = pd.to_datetime(dataset_2_df['datetime_iso'])

# Merge the dataframes on datetime columns
merged_df = pd.merge(
    dataset_1_df,
    dataset_2_df,  #! Canviar aixo
    on='datetime_iso',  # Join on the datetime column
    how='left'  # Keep all rows from dataset_2_df
)

print(f"Merged DataFrame has {len(merged_df)} rows and {len(merged_df.columns)} columns")
print(f"Columns: {merged_df.columns.tolist()}")

print(merged_df.head())

# Save the result to a new CSV file
merged_df.to_csv(export_path, index=False)

