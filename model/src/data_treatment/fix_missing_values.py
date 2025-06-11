import pandas as pd

def create_full_range_df_interpolate(path):
    df = pd.read_csv(path)
    df['datetime_iso'] = pd.to_datetime(df['datetime_iso'])

    # Aquí es corregeix l'ús de min i max
    full_range = pd.date_range(start=df['datetime_iso'].min(), end=df['datetime_iso'].max(), freq='H')
    df_full = pd.DataFrame({'datetime_iso': full_range})

    df_merged = df_full.merge(df, on='datetime_iso', how='left')
    df_merged['gas_generation'] = df_merged['gas_generation'].interpolate(method='linear')

    df_merged.to_csv('../data/results/merged_gas_data.csv', index=False)

    print(f"Full range DataFrame created with {len(df_merged)} rows and {len(df_merged.columns)} columns")

def expand_daily_to_hourly_forward_fill(input_path, output_path):
    
    print(f"Expanding daily gas price data from {input_path} to hourly resolution...")
    
    # Read the daily gas price data
    df = pd.read_csv(input_path)
    
    # Convert datetime_iso to datetime
    df['datetime_iso'] = pd.to_datetime(df['datetime_iso'])
    
    # Create a list to store hourly data
    hourly_data = []
    
    # For each day in the dataset
    for _, row in df.iterrows():
        day_datetime = row['datetime_iso']
        gas_price = row['gas_price']
        
        # Create 24 hourly entries for the current day
        for hour in range(24):
            hour_datetime = day_datetime.replace(hour=hour)
            
            hourly_data.append({
                'datetime_iso': hour_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'gas_price': gas_price
            })
    
    # Create the hourly DataFrame
    hourly_df = pd.DataFrame(hourly_data)
    
    # Save to CSV
    hourly_df.to_csv(output_path, index=False)
    print(f"Saved hourly gas price data to {output_path}")
    print(f"Generated {len(hourly_df)} hourly records from {len(df)} daily records")
    
    # Display sample
    print("\nSample of hourly data:")
    print(hourly_df.head(24))
    
    return hourly_df

# create_full_range_df_interpolate('../data/results/gas_data.csv')

expand_daily_to_hourly_forward_fill('../data/results/gas_price.csv', '../data/results/gas_price_def.csv')