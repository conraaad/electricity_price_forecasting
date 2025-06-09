import pandas as pd

def create_full_range_df(path):
    df = pd.read_csv(path)
    df['datetime_iso'] = pd.to_datetime(df['datetime_iso'])

    # Aquí es corregeix l'ús de min i max
    full_range = pd.date_range(start=df['datetime_iso'].min(), end=df['datetime_iso'].max(), freq='H')
    df_full = pd.DataFrame({'datetime_iso': full_range})

    df_merged = df_full.merge(df, on='datetime_iso', how='left')
    df_merged['gas_generation'] = df_merged['gas_generation'].interpolate(method='linear')

    df_merged.to_csv('../data/results/merged_gas_data.csv', index=False)

    print(f"Full range DataFrame created with {len(df_merged)} rows and {len(df_merged.columns)} columns")

create_full_range_df('../data/results/gas_data.csv')
