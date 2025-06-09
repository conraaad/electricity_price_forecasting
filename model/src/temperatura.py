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


# Define the file paths
barna_file = '../data/source/temperatura/temp_barna.csv'
madrid_file = '../data/source/temperatura/temp_mad.csv'
sevilla_file = '../data/source/temperatura/temp_sev.csv'

# # Process each file
# barna_df = add_avg_temperature(barna_file)
# madrid_df = add_avg_temperature(madrid_file)
# sevilla_df = add_avg_temperature(sevilla_file)

barna_df = pd.read_csv(barna_file)
madrid_df = pd.read_csv(madrid_file)
sevilla_df = pd.read_csv(sevilla_file)

temp_df = pd.DataFrame(barna_df['datetime_iso'])

# Display a sample of the results
print("\nSample of Barcelona data with average temperature:")
if barna_df is not None:
    print(barna_df.head())

print("\nSample of Madrid data with average temperature:")
if madrid_df is not None:
    print(madrid_df.head())

print("\nSample of Sevilla data with average temperature:")
if sevilla_df is not None:
    print(sevilla_df.head())

print("\nProcess completed successfully!")