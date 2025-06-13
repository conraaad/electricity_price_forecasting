import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime


def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100


def get_day_features(date_string: str):
    """
    Given a model object and a date string, loads
    the test data and returns features for all 24 hours of that day with all engineered features.

    Args:
        model: The trained model object
        date_string (str): Date in format 'yyyy-mm-dd'

    Returns:
        dict: Dictionary with hour as key and list of features as value
              Format: {0: [feature1, feature2, ...], 1: [...], ..., 23: [...]}
    """
    # Load the test data
    df = pd.read_csv("../../../data/analysis/test_data.csv")
    # df = pd.read_csv("../../../data/datasets/def_dataset.csv")

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
        'is_mond', 'is_tues', 'is_wed', 'is_thurs', 'is_fri', 'is_sat', 'is_sun',
        'is_sunday_or_holiday',
        'hour_sin', 'hour_cos',
        'type_day_workday', 'type_day_sat', 'type_day_sun', 'type_day_holiday',
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
            'features': row[features_reduced].tolist(),
            'target_price': row['target_price']
        }
        result[hour] = features

    # Verify we have all 24 hours
    if len(result) != 24:
        missing_hours = set(range(24)) - set(result.keys())
        print(f"Warning: Missing data for hours {sorted(missing_hours)} on {date_string}")

    return result, features_reduced


def predict_from_model_and_date(model_path, date_string):
    """
    Load model and predict electricity prices for all 24 hours of a given date.

    Args:
        model_path (str): Path to the saved model file
        date_string (str): Date in format 'yyyy-mm-dd'

    Returns:
        dict: Dictionary with structure:
        {
          'name_model': str,  # Name of the model file
          'date': str,  # Date string
          'hour_predictions': {
              0: {'predicted_price': float, 'mae': float, 'rmse': float, 'smape': float},
              1: {'predicted_price': float, 'mae': float, 'rmse': float, 'smape': float},
              ...
              23: {'predicted_price': float, 'mae': float, 'rmse': float, 'smape': float}
          },
          'daily_mean': {
              'mae': float,
              'rmse': float,
              'smape': float
          }
        }
    """

    # Load the trained model
    model = joblib.load(model_path)

    # Get features and actual prices for the day
    day_data, features_names = get_day_features(date_string)  # This now returns dict with features and target_price

    # Initialize result structure
    result = {
        "name_model": model_path.split('/')[-1].split('.')[0],
        "date": date_string,
        'hour_predictions': {}
    }

    # Lists to collect all metrics for daily mean calculation
    all_mae = []
    all_rmse = []
    all_smape = []

    # Make predictions for each hour
    for hour in range(24):
        if hour in day_data:
            # Get features and actual price for this hour
            features = day_data[hour]['features']
            actual_price = day_data[hour]['target_price']

            X = pd.DataFrame([features], columns=features_names)

            # Make prediction
            predicted_price_log = model.predict(X)[0]
            predicted_price = np.expm1(predicted_price_log)

            # Calculate metrics (comparing single prediction vs actual)
            mae = abs(predicted_price - actual_price)
            rmse = np.sqrt((predicted_price - actual_price) ** 2)
            smape_value = smape(np.array([actual_price]), np.array([predicted_price]))

            # Store results for this hour
            result['hour_predictions'][hour] = {
                'predicted_price': float(predicted_price),
                'mae': float(mae),
                'rmse': float(rmse),
                'smape': float(smape_value)
            }

            # Collect metrics for daily mean
            all_mae.append(mae)
            all_rmse.append(rmse)
            all_smape.append(smape_value)

        else:
            # Handle missing hour data
            result['hour_predictions'][hour] = {
                'predicted_price': None,
                'mae': None,
                'rmse': None,
                'smape': None
            }

    # Calculate daily means
    if all_mae:  # If we have at least one valid prediction
        result['daily_mean'] = {
            'mae': float(np.mean(all_mae)),
            'rmse': float(np.mean(all_rmse)),  # Mean of individual RMSE values
            'smape': float(np.mean(all_smape))
        }
    else:
        result['daily_mean'] = {
            'mae': None,
            'rmse': None,
            'smape': None
        }

    return result


# Enhanced utility function to print results with daily means
def print_predictions_summary(predictions, date_string):
    """
    Pretty print the prediction results including daily means.
    """
    print(f"\nElectricity Price Predictions for {date_string}")
    print("=" * 70)
    print(f"{'Hour':>4} {'Predicted':>10} {'Actual':>8} {'MAE':>8} {'RMSE':>8} {'SMAPE':>8}")
    print("-" * 70)

    hour_preds = predictions['hour_predictions']

    # Print hourly predictions
    for hour in range(24):
        if hour in hour_preds and hour_preds[hour]['predicted_price'] is not None:
            pred = hour_preds[hour]
            # Note: We don't have actual price in the display, so we calculate it from MAE and predicted
            actual = pred['predicted_price'] - pred['mae'] if pred['mae'] > 0 else pred['predicted_price'] + pred['mae']
            print(f"{hour:4d} {pred['predicted_price']:10.2f} {actual:8.2f} {pred['mae']:8.2f} {pred['rmse']:8.2f} {pred['smape']:8.1f}%")
        else:
            print(f"{hour:4d} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

    # Print daily means
    print("=" * 70)
    daily_mean = predictions['daily_mean']
    if daily_mean['mae'] is not None:
        print(f"{'DAILY MEAN':>4} {'':<10} {'':<8} {daily_mean['mae']:8.2f} {daily_mean['rmse']:8.2f} {daily_mean['smape']:8.1f}%")
    else:
        print(f"{'DAILY MEAN':>4} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

    print("=" * 70)


# Alternative version that also returns summary statistics
def predict_from_model_and_date_with_summary(model_path, date_string):
    """
    Same as predict_from_model_and_date but includes daily summary statistics.
    """

    # Get hourly predictions
    result = predict_from_model_and_date(model_path, date_string)

    # Calculate daily summary statistics
    valid_hours = [h for h in result['hour_predictions'].values() if h['predicted_price'] is not None]

    if valid_hours:
        daily_mae = np.mean([h['mae'] for h in valid_hours])
        daily_rmse = np.sqrt(np.mean([h['rmse']**2 for h in valid_hours]))
        daily_smape = np.mean([h['smape'] for h in valid_hours])

        # Add summary to result
        result['daily_summary'] = {
            'average_mae': float(daily_mae),
            'average_rmse': float(daily_rmse),
            'average_smape': float(daily_smape),
            'hours_predicted': len(valid_hours)
        }
    else:
        result['daily_summary'] = {
            'average_mae': None,
            'average_rmse': None,
            'average_smape': None,
            'hours_predicted': 0
        }

    return result


# Example usage
if __name__ == "__main__":
    try:
        # Basic prediction
        model_path = "../models/random_forest_model.joblib"
        date = "2023-10-04"

        predictions = predict_from_model_and_date(model_path, date)

        print(predictions)

        # Print results
        print_predictions_summary(predictions, date)

        # Access specific hour prediction
        hour_12_prediction = predictions['hour_predictions'][12]
        print(f"\nHour 12 prediction: {hour_12_prediction['predicted_price']:.2f} €/MWh")
        print(f"Hour 12 MAE: {hour_12_prediction['mae']:.2f}")

    except Exception as e:
        print(f"Error: {e}")