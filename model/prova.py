# import requests
import pandas as pd
import json
from datetime import datetime
import holidays
import os


# TOKEN = "5195fe57d8bdd3cd242451da3de471e9e9eaf6abf08f01f2bd412f6578b03842"


def get_day_of_week(date):
    """
    Returns the day of the week for a given date.
    """
    print(date.strftime("%A"))


def get_holiday(date):
    espanya_festius = holidays.ES()
    print(date in espanya_festius)


# Define the path to your CSV file
csv_path = 'data/demand_data.csv'

# Check if the file exists
if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
else:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Convert datetime string to datetime object if it's not already
    if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create one-hot encoding for months
    month_names = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    
    # Create one-hot encoded columns for months
    for i, month_name in enumerate(month_names, 1):  # months are 1-indexed
        df[f'month_{month_name}'] = (df['month'] == i).astype(int)
    
    # Show the first few rows with the new columns
    print("\nDataFrame with one-hot encoded month columns:")
    print(df.head())
    
    # Save the updated DataFrame back to CSV if needed
    df.to_csv(csv_path, index=False)
    print(f"Updated DataFrame saved to {csv_path}")
    
    # Display some basic statistics
    print("\nSummary statistics for demand:")
    print(df.head())