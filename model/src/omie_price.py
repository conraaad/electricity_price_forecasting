
import glob
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

def descarregar_marginals(any):
    carpeta = f'../data/source/target_price/{any}'
    os.makedirs(carpeta, exist_ok=True)

    start_date = datetime(any, 1, 13)
    end_date = datetime(any, 12, 31)
    data = start_date

    while data <= end_date:
        data_str = data.strftime('%Y%m%d')
        url = f"https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{data_str}.1"
        desti = os.path.join(carpeta, f"marginalpdbc_{data_str}.1")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(desti, 'wb') as f:
                    f.write(response.content)
                print(f"✔ Descarregat: {data_str}")
            else:
                print(f"✘ No trobat: {data_str} (HTTP {response.status_code})")
        except Exception as e:
            print(f"⚠ Error amb {data_str}: {e}")

        data += timedelta(days=1)

def process_marginal_price_files(source_dir, output_file, year):
    """
    Process marginal price files for a specific year and save to CSV.
    
    Parameters:
    - source_dir: Directory containing year folders with marginal price files
    - output_file: Path to save the combined CSV file
    - year: The year to process (as integer or string)
    """
    year_str = str(year)
    year_path = os.path.join(source_dir, year_str)
    
    print(f"Processing marginal price files for year {year_str}")
    
    if not os.path.isdir(year_path):
        print(f"Error: Directory for year {year_str} does not exist at {year_path}")
        return None
    
    # List to store data from all files
    all_data = []
    
    # Get all files in the year folder
    files = glob.glob(os.path.join(year_path, "marginalpdbc_*.1"))
    
    if not files:
        print(f"No marginal price files found for year {year_str}")
        return None
    
    # Process each file
    for file_path in sorted(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header and footer lines
            data_lines = [line for line in lines if line.strip() and not line.startswith('MARGINALPDBC') and not line.startswith('*')]
            
            # Process each line
            for line in data_lines:
                # Split by semicolon and remove empty entries
                fields = [f.strip() for f in line.split(';') if f.strip()]
                
                if len(fields) >= 6:  # We need at least 6 fields
                    file_year = int(fields[0])
                    month = int(fields[1])
                    day = int(fields[2])
                    hour = int(fields[3]) - 1  # Convert from 1-24 to 0-23
                    price = float(fields[4])
                    
                    # Verify that the year in the file matches the expected year
                    if file_year != int(year):
                        print(f"Warning: Year mismatch in file {file_path}. Expected {year}, got {file_year}")
                    
                    # Create datetime string in ISO format
                    dt = datetime(file_year, month, day, hour)
                    datetime_iso = dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add to data list
                    all_data.append({
                        'datetime_iso': datetime_iso,
                        'target_price': price
                    })
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    if not all_data:
        print(f"No data was extracted from files for year {year_str}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by datetime
    df = df.sort_values('datetime_iso')
    
    # Create year-specific output file if none provided
    if not output_file:
        output_file = f'../data/results/marginal_price_data_{year_str}.csv'
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Processed {len(df)} records from {len(files)} files for year {year_str}")
    print(f"Data saved to {output_file}")
    
    # Display sample
    print("\nSample of processed data:")
    print(df.head())
    
    return df



    
source_dir = '../data/source/target_price'
year_to_process = 2024  # Change this to the year you want to process
output_file = f'../data/results/target_price/target_price_data_{year_to_process}.csv'

# Process the files
marginal_price_df = process_marginal_price_files(source_dir, output_file, year_to_process)

# Descarregar fitxers per 2023 i 2024
# descarregar_marginals(2023)
# descarregar_marginals(2024)
