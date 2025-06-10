import pandas as pd

def add_lag_variable(input_file, output_file=None):
    # Load the data
    df = pd.read_csv(input_file)

    df['datetime_iso'] = pd.to_datetime(df['datetime_iso'])

    # Create the lag variable (price from 24 hours ago)
    df['price_es_24h'] = df['target_price'].shift(24)

    df.to_csv(output_file, index=False)
    # Display the first few rows to confirm
    print(df.head(30))

input_file = '../data/results/target_price_data.csv'
output_file = '../data/results/target_price_data_with_lag.csv'

add_lag_variable(input_file, output_file)