import pandas as pd
import joblib
import numpy as np
from datetime import datetime


def get_day_features(date_string: str):
    """
    Given a model object and a date string, loads the test data and returns
    features for all 24 hours of that day with all engineered features.
    
    Args:
        model: The trained model object
        date_string (str): Date in format 'yyyy-mm-dd'
    
    Returns:
        dict: Dictionary with hour as key and list of features as value
              Format: {0: [feature1, feature2, ...], 1: [...], ..., 23: [...]}
    """
    # Load the test data
    df = pd.read_csv("../../../data/analysis/test_data.csv")
    
    # Apply the same feature engineering as in training
    df['gas_generation_share'] = round(df['gas_generation'] / df['demand'], 4)
    df['target_price'] = abs(df['target_price'])  # Evitar preus negatius
    df['renewable_ratio'] = round((df['solar'] + df['wind']) / df['demand'], 4)
    df['is_sunday_or_holiday'] = ((df['is_sun'] == 1) | (df['type_day_holiday'] == 1)).astype(int)
    df['high_renewable_ratio'] = (df['renewable_ratio'] > 0.8).astype(int)
    df['low_demand'] = (df['demand'] < df['demand'].quantile(0.2)).astype(int)
    df['renewables_to_gas'] = (df['solar_share_demand'] + df['wind_share_demand']) / (df['gas_generation_share'] + 1e-5)
    df['demand_per_gas'] = df['demand'] / (df['gas_generation_share'] + 1e-5)
    df["price_rolling_3h"] = df["target_price"].rolling(window=3).mean().shift(1)
    df["gas_price_lag1"] = df["gas_price"].shift(24)
    df['is_zero'] = (df['target_price'] == 0).astype(int)
    
    # Use the same features as in training
    features_reduced = [
        'is_mond', 'is_tues','is_wed','is_thurs','is_fri','is_sat','is_sun',
        'is_sunday_or_holiday',
        'hour_sin', 'hour_cos',
        'type_day_workday','type_day_sat','type_day_sun','type_day_holiday',
        'holiday_coef',
        'demand', 'low_demand',
        'solar_share_demand', 'wind_share_demand',
        'gas_generation_share', 'gas_price',
        'residual_demand',
        'interchange_balance',
        'renewable_ratio',
        'high_renewable_ratio',
        'temp_dev',
        'price_es_24h',
        'renewables_to_gas',
        'demand_per_gas',
        'price_rolling_3h',
        'gas_price_lag1'
    ]
    
    # Convert date_string to datetime for comparison
    target_date = datetime.strptime(date_string, "%Y-%m-%d").date()
    
    # Convert datetime_iso column to datetime if it's not already
    if 'datetime_iso' in df.columns:
        df['datetime_iso'] = pd.to_datetime(df['datetime_iso'])
        df['date'] = df['datetime_iso'].dt.date
    else:
        # If no datetime_iso, try to construct from year, month, day, hour columns
        df['datetime_iso'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df['date'] = df['datetime_iso'].dt.date
    
    # Filter rows for the given date
    day_data = df[df['date'] == target_date].copy()
    
    if day_data.empty:
        raise ValueError(f"No data found for date {date_string}")
    
    # Sort by hour to ensure correct order
    day_data = day_data.sort_values('hour')
    
    # Create the result dictionary using the exact same features as training
    result = {}
    
    for _, row in day_data.iterrows():
        hour = int(row['hour'])
        features = {
            'features' : row[features_reduced].tolist(),
            'target_price' : row['target_price']
        }
        result[hour] = features
    
    # Verify we have all 24 hours
    if len(result) != 24:
        missing_hours = set(range(24)) - set(result.keys())
        print(f"Warning: Missing data for hours {sorted(missing_hours)} on {date_string}")
    
    return result


def predict_from_model_and_date(model_path, date: str):
    pass

# try:
#   print(get_day_features("2023-01-02"))
# except Exception as e:
#     print(f"Error: {e}")