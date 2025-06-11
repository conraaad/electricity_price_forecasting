
import json
import pandas as pd

csv_path = '../data/results/renda_imp_france.csv'

with open('../data/source/renda_import_france.json', 'r', encoding='utf-8') as file:
  intercambios = json.load(file)

# Extract interchange data from the JSON
datetimes = []
interchange_data = []
for entry in intercambios['indicator']['values']:
    if 'datetime_utc' in entry and 'value' in entry:
        datetimes.append(entry['datetime_utc'])
        interchange_data.append(entry['value'])

interchange_df = pd.DataFrame({
    'datetime_iso': datetimes,
    'renda_fr_import': interchange_data
})

interchange_df['datetime_iso'] = pd.to_datetime(interchange_df['datetime_iso'])
interchange_df['datetime_iso'] = interchange_df['datetime_iso'].dt.strftime('%Y-%m-%d %H:%M:%S')


# Print some information about the interchange data
print(f"Loaded {len(interchange_df)} interchange balance records")

# Save the updated DataFrame back to CSV if needed
interchange_df.to_csv(csv_path, index=False)
print(f"Updated DataFrame saved to {csv_path}")

print(interchange_df.head())