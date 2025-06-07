
# import requests
import pandas as pd
import json
from datetime import datetime

# TOKEN = "5195fe57d8bdd3cd242451da3de471e9e9eaf6abf08f01f2bd412f6578b03842"
# START_DATE = "2019-01-01T00:00:00Z"
# END_DATE = "2024-12-31T23:00:00Z"

# Load the JSON data
file_path = 'data/demanda_real.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create lists to store the data
datetimes = []
demands = []

# Extract data from the JSON
# Note: This assumes the JSON has 'values' with datetime and value fields
# You may need to adjust this based on the actual structure of your complete JSON
for entry in data['indicator']['values']:
    if 'datetime_utc' in entry and 'value' in entry:
        datetimes.append(entry['datetime_utc'])
        demands.append(entry['value'])

# Create the DataFrame
df_demand = pd.DataFrame({
    'datetime': datetimes,
    'demand': demands
})

# Convert datetime string to datetime object
df_demand['datetime'] = pd.to_datetime(df_demand['datetime'])

# Extract year, month, day, hour
df_demand['year'] = df_demand['datetime'].dt.year
df_demand['month'] = df_demand['datetime'].dt.month
df_demand['day'] = df_demand['datetime'].dt.day
df_demand['hour'] = df_demand['datetime'].dt.hour

# Convert datetime to ISO format string if needed
df_demand['datetime_iso'] = df_demand['datetime'].dt.strftime('%Y-%m-%d %H:%M')

# Reorder columns to match the requested order
df_demand = df_demand[['datetime_iso', 'year', 'month', 'day', 'hour', 'demand']]

# Preview the DataFrame
print(df_demand.head())

# Save DataFrame to CSV
df_demand.to_csv('data/demand_data.csv', index=False)