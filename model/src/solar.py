
import json
import pandas as pd

csv_path = '../data/results/solar_data.csv'

with open('../data/source/generacio_eolica.json', 'r', encoding='utf-8') as file:
  solar_json = json.load(file)

solar_df = pd.read_csv(csv_path)


# Extract interchange data from the JSON

# datetimes = []
# gen_value = []
# for entry in solar_json['indicator']['values']:
#     if 'datetime_utc' in entry and 'value' in entry:
#         # datetimes.append(entry['datetime_utc'])
#         gen_value.append(entry['value'])

# solar_df['wind'] = gen_value

# solar_df = pd.DataFrame({
#     'datetime_iso': datetimes,
#     'solar': gen_value
# })

# solar_df = solar_df.rename(columns={'interchange_balance': 'solar'})

# solar_df['datetime_iso'] = pd.to_datetime(solar_df['datetime_iso'])
# solar_df['datetime_iso'] = solar_df['datetime_iso'].dt.strftime('%Y-%m-%d %H:%M:%S')


# Save the updated DataFrame back to CSV if needed
solar_df.to_csv(csv_path, index=False)
print(f"Updated DataFrame saved to {csv_path}")

print(solar_df.head())