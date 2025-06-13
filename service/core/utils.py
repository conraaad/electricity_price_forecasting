import random
from datetime import datetime, timedelta

def get_random_date_2023():
    """
    Returns a random date within 2023 in yyyy-mm-dd format
    """
    # Start and end dates for 2023
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Calculate the number of days between start and end
    days_between = (end_date - start_date).days
    
    # Generate a random number of days to add to start date
    random_days = random.randint(0, days_between)
    
    # Calculate the random date
    random_date = start_date + timedelta(days=random_days)
    
    # Return in yyyy-mm-dd format
    return random_date.strftime('%Y-%m-%d')
