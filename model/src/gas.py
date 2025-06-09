

import json
import pandas as pd

csv_path = '../data/results/gas_data.csv'

with open('../data/source/generacio_gas.json', 'r', encoding='utf-8') as file:
  solar_json = json.load(file)

# gas_df = pd.read_csv(csv_path)


# Extract interchange data from the JSON

datetimes = []
gen_value = []
for entry in solar_json['indicator']['values']:
    if 'datetime_utc' in entry and 'value' in entry:
        datetimes.append(entry['datetime_utc'])
        gen_value.append(entry['value'])

gas_df = pd.DataFrame({
    'datetime_iso': datetimes,
    'gas_generation': gen_value
})



gas_df['datetime_iso'] = pd.to_datetime(gas_df['datetime_iso'])
gas_df['datetime_iso'] = gas_df['datetime_iso'].dt.strftime('%Y-%m-%d %H:%M:%S')


# Save the updated DataFrame back to CSV if needed
gas_df.to_csv(csv_path, index=False)
print(f"Updated DataFrame saved to {csv_path}")

print(gas_df.head())