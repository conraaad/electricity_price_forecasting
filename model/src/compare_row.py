import pandas as pd
import datetime

def find_missing_dates():
    # Load both datasets
    def_dataset = pd.read_csv('../data/results/def_dataset.csv')
    solar_data = pd.read_csv('../data/results/solar_data.csv')

    print(f"def_dataset has {len(def_dataset)} rows and {len(def_dataset.columns)} columns")
    print(f"solar_data has {len(solar_data)} rows and {len(solar_data.columns)} columns")
    
    # Convert datetime_iso to datetime objects for easier comparison
    def_dataset['datetime'] = pd.to_datetime(def_dataset['datetime_iso'])
    
    # Check if solar_data has datetime_iso column
    if 'datetime_iso' in solar_data.columns:
        solar_data['datetime'] = pd.to_datetime(solar_data['datetime_iso'])
    else:
        # Try to find a column that might contain datetime information
        datetime_columns = [col for col in solar_data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_columns:
            solar_data['datetime'] = pd.to_datetime(solar_data[datetime_columns[0]])
        else:
            print("Could not find a datetime column in solar_data. Available columns:", solar_data.columns.tolist())
            return
    
    # Sort both datasets by datetime
    def_dataset = def_dataset.sort_values('datetime')
    solar_data = solar_data.sort_values('datetime')
    
    # Get sets of datetimes from both datasets
    def_datetimes = set(def_dataset['datetime'])
    solar_datetimes = set(solar_data['datetime'])
    
    # Find dates in def_dataset but not in solar_data
    missing_in_solar = def_datetimes - solar_datetimes
    
    # Find dates in solar_data but not in def_dataset
    missing_in_def = solar_datetimes - def_datetimes
    
    print(f"Number of dates in def_dataset: {len(def_datetimes)}")
    print(f"Number of dates in solar_data: {len(solar_datetimes)}")
    
    print("\nDates missing in solar_data but present in def_dataset:")
    for dt in sorted(missing_in_solar):
        print(dt)
    
    print("\nDates missing in def_dataset but present in solar_data:")
    for dt in sorted(missing_in_def):
        print(dt)
    
    # Check if there are any gaps in the hourly sequence
    def_dates_list = sorted(def_datetimes)
    solar_dates_list = sorted(solar_datetimes)
    
    def check_sequence_gaps(dates_list, dataset_name):
        if not dates_list:
            return
        
        gaps = []
        for i in range(1, len(dates_list)):
            expected_delta = datetime.timedelta(hours=1)
            actual_delta = dates_list[i] - dates_list[i-1]
            if actual_delta != expected_delta:
                gaps.append((dates_list[i-1], dates_list[i], actual_delta))
        
        if gaps:
            print(f"\nGaps in {dataset_name} hourly sequence:")
            for prev_dt, curr_dt, delta in gaps:
                print(f"Gap between {prev_dt} and {curr_dt} ({delta})")
        else:
            print(f"\nNo gaps in {dataset_name} hourly sequence.")
    
    check_sequence_gaps(def_dates_list, "def_dataset")
    check_sequence_gaps(solar_dates_list, "solar_data")
    
    return missing_in_solar, missing_in_def


find_missing_dates()