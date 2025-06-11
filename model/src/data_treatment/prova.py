# import requests

import json
import pandas as pd
# import json
from datetime import datetime

import os
import numpy as np
import math

from temporal_handling import *


# TOKEN = "5195fe57d8bdd3cd242451da3de471e9e9eaf6abf08f01f2bd412f6578b03842"




def get_holiday(date):
    espanya_festius = holidays.ES()
    print(date in espanya_festius)

def main():

    # Define the path to your CSV file

    csv_path = '../data/demand_data.csv'
    # csv_path_export = '../data/interchange_data.csv'

    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
    else:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
        
        # Convert datetime string to datetime object if it's not already
        if 'datetime_iso' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime_iso']):
            print("Converting 'datetime_iso' column to datetime objects...")
            df['datetime_iso'] = pd.to_datetime(df['datetime_iso'])
        
        print(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
        
        # # Create one-hot encoding for months
        # month_names = [
        #     'january', 'february', 'march', 'april', 'may', 'june',
        #     'july', 'august', 'september', 'october', 'november', 'december'
        # ]
        
        # # Create one-hot encoded columns for months
        # for i, month_name in enumerate(month_names, 1):  # months are 1-indexed
        #     df[f'month_{i}'] = (df['month'] == i).astype(int)
            
        #     # Delete old numeric month columns if they exist
        #     if f'month_{month_name}' in df.columns:
        #         df = df.drop(f'month_{month_name}', axis=1)


        # # Extract day of week from datetime (0 = Monday, 6 = Sunday)
        # if 'datetime_iso' in df.columns:
        #     print("Extracting day of week from 'datetime_iso' column...")
        #     df['weekday'] = df['datetime_iso'].dt.dayofweek
        
        # # # Create one-hot encoding for weekdays
        # week_days = ['mond', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']
        
        # # # Create one-hot encoded columns for days of the week
        # for i, week_day in enumerate(week_days):  # weekday is 0-indexed (0 = Monday)
        #     df[f'is_{week_day}'] = (df['weekday'] == i).astype(int)

        
        # # Calculate day features (using max days=31 as in your formula)
        # df['day_sin'] = round(np.sin((2 * np.pi * df['day']) / 31), 6)
        # df['day_cos'] = round(np.cos((2 * np.pi * df['day']) / 31), 6)
        
        # # Calculate hour features
        # df['hour_sin'] = round(np.sin((2 * np.pi * df['hour']) / 24), 6)
        # df['hour_cos'] = round(np.cos((2 * np.pi * df['hour']) / 24), 6)


        # # Type of day classification
        # df['type_of_day'] = df['datetime_iso'].dt.date.apply(get_type_of_day)

        # # Type of day one hot encoding
        # type_of_days_list = ["workday", "sat", "sun", "holiday"]

        # for i, type_of_day in enumerate(type_of_days_list):  # type of day is 0-indexed (0 = workday)
        #     df[f'type_day_{type_of_day}'] = (df['type_of_day'] == i).astype(int)

        # # Calculate holiday coefficient

        # df['holiday_coef'] = df['datetime_iso'].dt.date.apply(calcula_coeficient_festiu, json_data=festius_json)
        # print(df['datetime_iso'].dt.date.apply(calcula_coeficient_festiu, json_data=festius_json))

        # Create code here for interchange_balance
        

        
        # Show the first few rows with the new column
        print("\nDataFrame with interchange balance column:")
        print(df[['datetime_iso', 'demand', 'interchange_balance']].head())

        # Save the updated DataFrame back to CSV if needed
        df.to_csv(csv_path, index=False) #TODO canviar el path
        print(f"Updated DataFrame saved to {csv_path}")

        # Show the first few rows with the new columns
        print(df.head())
        
        




with open("../data/holiday_2021-2024.json", "r", encoding="utf-8") as f:
    festius_json = json.load(f)

main()



# dt = pd.to_datetime("2021-09-11")

# print(calcula_coeficient_festiu(dt, festius_json))

# print(get_type_of_day(dt))  # → 3 (festiu a Catalunya)

