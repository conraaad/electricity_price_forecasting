import json
import pandas as pd

def parse_from_json(csv_path):
    
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


def clean_gas_data(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Clean the gas_price column (remove trailing semicolons)
    df['gas_price'] = df['gas_price'].str.replace(';', '').astype(float)
    
    # Convert datetime_iso from European format (DD/MM/YYYY) to datetime objects
    df['datetime_iso'] = pd.to_datetime(df['datetime_iso'], format='%d/%m/%Y')
    
    # Sort by date in ascending order
    df = df.sort_values('datetime_iso', ascending=True)
    
    # Convert datetime to string format YYYY-MM-DD HH:MM:SS for consistency
    # df['datetime_iso'] = df['datetime_iso'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save the sorted DataFrame back to CSV
    sorted_csv_path = csv_path.replace('.csv', '_sorted.csv')
    df.to_csv(sorted_csv_path, index=False)
    print(f"Sorted DataFrame saved to {sorted_csv_path}")
    
    print("First few rows after sorting:")
    print(df.head())
    
    return df

paths = [
  '../data/source/gas/MIBGAS_Data_2021.csv',
  '../data/source/gas/MIBGAS_Data_2022.csv',
  '../data/source/gas/MIBGAS_Data_2023.csv',
  '../data/source/gas/MIBGAS_Data_2024.csv'
]

for path in paths:
    clean_gas_data(path)


