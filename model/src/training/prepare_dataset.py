
import pandas as pd
import datetime

csv_path_import = "../../data/def_data/training_dataset.csv"
csv_path_export = "../../data/def_data/training_dataset.csv"

df = pd.read_csv(csv_path_import)

# columns_to_drop = [
#   'datetime_iso',  # codificat ja
#   'month',         # redundant amb month_1...month_12
#   'day',           # substituït per sin/cos
#   'hour',          # substituït per sin/cos
#   'weekday',       # substituït per is_mond...is_sun
#   'type_of_day'    # substituït per one-hot type_day_*
#   'temp',          # substituït per temp_dev
# ]

df = df.drop(columns=['temp'])

df = df[[
  'year',
  'month_1',
  'month_2',
  'month_3',
  'month_4',
  'month_5',
  'month_6',
  'month_7',
  'month_8',
  'month_9',
  'month_10',
  'month_11',
  'month_12',
  'day_sin',
  'day_cos',
  'hour_sin',
  'hour_cos',
  'is_mond',
  'is_tues',
  'is_wed',
  'is_thurs',
  'is_fri',
  'is_sat',
  'is_sun',
  'type_day_workday',
  'type_day_sat',
  'type_day_sun',
  'type_day_holiday',
  'holiday_coef',
  'demand',
  'solar',
  'solar_share_demand',
  'wind',
  'wind_share_demand',
  'gas_generation',
  'gas_price',
  'residual_demand',
  'interchange_balance',
  'temp_dev',
  'price_es_24h',
  'target_price'
]]


df.to_csv(csv_path_export, index=False)

print(f"Dataset prepared with {len(df)} rows and {len(df.columns)} columns")

print(df.head())

print(f"Dataset saved to {csv_path_export}")