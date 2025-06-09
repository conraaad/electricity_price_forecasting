import pandas as pd

def add_avg_temperature(input_file, output_file=None):
    print(f"Processing {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if the required columns exist
    if 'tmax' not in df.columns or 'tmin' not in df.columns:
        print(f"Error: {input_file} does not contain both 'tmax' and 'tmin' columns")
        return None
    
    # Calculate the average temperature
    df['temp'] = (df['tmax'] + df['tmin']) / 2
    
    # Round to 1 decimal place for consistency with the input data
    df['temp'] = df['temp'].round(1)
    
    # Save to output file or overwrite input file
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved modified data to {output_file}")
    else:
        df.to_csv(input_file, index=False)
        print(f"Updated {input_file} with average temperature column")
    
    return df

def build_temperature_file():
    barna_df = pd.read_csv(barna_file)
    madrid_df = pd.read_csv(madrid_file)
    sevilla_df = pd.read_csv(sevilla_file)

    temp_df = pd.DataFrame(barna_df['datetime_iso'])

    temp_df['temp'] = ((madrid_df['temp'] + barna_df['temp'] + sevilla_df['temp']) / 3).round(2)

    temp_df.to_csv('../data/results/temperatura_data.csv', index=False)
    print(temp_df.head())


def expand_daily_to_hourly(input_file, output_file):
    """
    Expands daily temperature data to hourly resolution using linear interpolation
    """
    print(f"Expanding daily temperature data from {input_file} to hourly resolution...")
    
    # Read the daily temperature data
    daily_df = pd.read_csv(input_file)
    
    # Convert the datetime column to proper datetime objects
    daily_df['datetime'] = pd.to_datetime(daily_df['datetime_iso'])
    
    # Create a list to store hourly data
    hourly_data = []
    
    # For each day in the dataset
    for i in range(len(daily_df)):
        day_data = daily_df.iloc[i]
        current_date = day_data['datetime']
        temp = day_data['temp']
        
        # If not the last day, we can interpolate between current and next day
        if i < len(daily_df) - 1:
            next_day_data = daily_df.iloc[i + 1]
            next_date = next_day_data['datetime']
            next_temp = next_day_data['temp']
            
            # Calculate temperature gradient per hour
            hours_diff = 24
            temp_diff = next_temp - temp
            hourly_gradient = temp_diff / hours_diff
            
            # Create 24 hourly entries for the current day
            for hour in range(24):
                hour_datetime = current_date.replace(hour=hour)
                hour_temp = temp + (hourly_gradient * hour)
                
                hourly_data.append({
                    'datetime_iso': hour_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'temp': round(hour_temp, 2)
                })
        else:
            # For the last day, just use the same temperature for all hours
            for hour in range(24):
                hour_datetime = current_date.replace(hour=hour)
                hourly_data.append({
                    'datetime_iso': hour_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'temp': temp
                })
    
    # Create a DataFrame from the hourly data
    hourly_df = pd.DataFrame(hourly_data)
    
    # Save to CSV
    hourly_df.to_csv(output_file, index=False)
    print(f"Saved hourly temperature data to {output_file}")
    print(f"Generated {len(hourly_df)} hourly records from {len(daily_df)} daily records")
    
    return hourly_df

def add_temp_dev(df):
    
    print("Calculating temperature deviation...")
    
    
    # Add a new column for temperature deviation
    df['temp_dev'] = abs(df['temp'] - 22)  # Assuming 22 is the reference temperature
    
    # Round to 2 decimal places for consistency
    df['temp_dev'] = df['temp_dev'].round(2)
    
    print("Temperature deviation added.")

    df.to_csv('../data/results/temperatura_data_def.csv', index=False)
    
    return df

# Define the file paths
barna_file = '../data/source/temperatura/temp_barna.csv'
madrid_file = '../data/source/temperatura/temp_mad.csv'
sevilla_file = '../data/source/temperatura/temp_sev.csv'

# # Process each file
# barna_df = add_avg_temperature(barna_file)
# madrid_df = add_avg_temperature(madrid_file)
# sevilla_df = add_avg_temperature(sevilla_file)

# Display a sample of the results
# print("\nSample of Barcelona data with average temperature:")
# if barna_df is not None:
#     print(barna_df.head())

# print("\nSample of Madrid data with average temperature:")
# if madrid_df is not None:
#     print(madrid_df.head())

# print("\nSample of Sevilla data with average temperature:")
# if sevilla_df is not None:
#     print(sevilla_df.head())

# hourly_df = expand_daily_to_hourly('../data/results/temperatura_data.csv', '../data/results/temperatura_data_def.csv')

df = pd.read_csv('../data/results/temperatura_data.csv')

df = add_temp_dev(df)

print("\nSample of hourly temperature data:")
print(df.head(24))

print("\nProcess completed successfully!")