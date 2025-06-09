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
    
    solar_data['datetime'] = pd.to_datetime(solar_data['datetime_iso'])
    
    
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




def find_duplicates_and_differences():
    # Load both datasets
    def_dataset = pd.read_csv('../data/results/def_dataset.csv')
    solar_data = pd.read_csv('../data/results/solar_data.csv')

    print(f"def_dataset has {len(def_dataset)} rows and {len(def_dataset.columns)} columns")
    print(f"solar_data has {len(solar_data)} rows and {len(solar_data.columns)} columns")
    
    # Convert datetime_iso to datetime objects
    def_dataset['datetime'] = pd.to_datetime(def_dataset['datetime_iso'])
    solar_data['datetime'] = pd.to_datetime(solar_data['datetime_iso'])
    
    # Check for duplicates in both datasets
    def_dataset_dupes = def_dataset['datetime'].duplicated().sum()
    solar_data_dupes = solar_data['datetime'].duplicated().sum()
    
    print(f"\nDuplicates in def_dataset: {def_dataset_dupes}")
    print(f"Duplicates in solar_data: {solar_data_dupes}")
    
    # If there are duplicates, show them
    if solar_data_dupes > 0:
        print("\nDuplicate datetimes in solar_data:")
        duplicates = solar_data[solar_data['datetime'].duplicated(keep=False)]
        print(duplicates.sort_values('datetime'))
    
    if def_dataset_dupes > 0:
        print("\nDuplicate datetimes in def_dataset:")
        duplicates = def_dataset[def_dataset['datetime'].duplicated(keep=False)]
        print(duplicates.sort_values('datetime'))
    
    # Also check the value counts for each datetime to see which ones appear multiple times
    print("\nDatetimes occurring more than once in solar_data:")
    duplicate_counts = solar_data['datetime'].value_counts()
    print(duplicate_counts[duplicate_counts > 1])
    
    # Check first and last rows of both datasets
    print("\nFirst rows of both datasets:")
    print("def_dataset first row:", def_dataset['datetime'].min())
    print("solar_data first row:", solar_data['datetime'].min())
    
    print("\nLast rows of both datasets:")
    print("def_dataset last row:", def_dataset['datetime'].max())
    print("solar_data last row:", solar_data['datetime'].max())
    
    # Compare row counts by date (without time) to see if a particular date has extra entries
    def_dataset['date'] = def_dataset['datetime'].dt.date
    solar_data['date'] = solar_data['datetime'].dt.date
    
    def_counts = def_dataset['date'].value_counts().sort_index()
    solar_counts = solar_data['date'].value_counts().sort_index()
    
    print("\nComparing row counts by date:")
    date_comparison = pd.DataFrame({
        'def_dataset': def_counts, 
        'solar_data': solar_counts
    }).fillna(0)
    
    date_comparison['difference'] = date_comparison['solar_data'] - date_comparison['def_dataset']
    date_differences = date_comparison[date_comparison['difference'] != 0]
    
    if not date_differences.empty:
        print(date_differences)
    else:
        print("No dates with different row counts found")


print("Starting date comparison...")

find_missing_dates()

find_duplicates_and_differences()

