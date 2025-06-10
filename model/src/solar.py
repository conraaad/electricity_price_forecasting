
import json
import pandas as pd

csv_path = '../data/results/def_data/merged_def_dataset_solar.csv'

# with open('../data/source/generacio_gas.json', 'r', encoding='utf-8') as file:
#   solar_json = json.load(file)

df = pd.read_csv(csv_path)

# df['solar'] = round(df['solar'], 4)
# df['wind'] = round(df['wind'], 4)

# df['residual_demand'] = round(df['demand'] - (df['solar'] + df['wind']), 4)

df['solar_share_demand'] = round(df['solar'] / df['demand'], 4)
df['wind_share_demand'] = round(df['wind'] / df['demand'], 4)



# Extract interchange data from the JSON

# datetimes = []
# gen_value = []
# for entry in solar_json['indicator']['values']:
#     if 'datetime_utc' in entry and 'value' in entry:
#         # datetimes.append(entry['datetime_utc'])
#         gen_value.append(entry['value'])

# df = pd.DataFrame({
#     'datetime_iso': datetimes,
#     'solar': gen_value
# })

# df['gas_generation'] = gen_value

# df = df.rename(columns={'interchange_balance': 'solar'})

# df['datetime_iso'] = pd.to_datetime(df['datetime_iso'])
# df['datetime_iso'] = df['datetime_iso'].dt.strftime('%Y-%m-%d %H:%M:%S')


# Save the updated DataFrame back to CSV if needed
df.to_csv(csv_path, index=False)
print(f"Updated DataFrame saved to {csv_path}")

print(df.head())