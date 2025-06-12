

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from metrics_utils import smape, analyze_worst_predictions

df = pd.read_csv("../../data/datasets/training_dataset_2024.csv")

df['gas_generation_share'] = round(df['gas_generation'] / df['demand'], 4)
df['target_price'] = abs(df['target_price'])  # Evitar preus negatius
df['renewable_ratio'] = round((df['solar'] + df['wind']) / df['demand'], 4)
df['is_sunday_or_holiday'] = ((df['is_sun'] == 1) | (df['type_day_holiday'] == 1)).astype(int)
df['high_renewable_ratio'] = (df['renewable_ratio'] > 0.8).astype(int)
df['low_demand'] = (df['demand'] < df['demand'].quantile(0.2)).astype(int)
# df['wind_to_solar_ratio'] = df['wind_share_demand'] / (df['solar_share_demand'] + 1e-5)
df['renewables_to_gas'] = (df['solar_share_demand'] + df['wind_share_demand']) / (df['gas_generation_share'] + 1e-5)
df['demand_per_gas'] = df['demand'] / (df['gas_generation_share'] + 1e-5)
# df['price_es_24h'] = np.log1p(df['price_es_24h'])
df["price_rolling_3h"] = df["target_price"].rolling(window=3).mean().shift(1)
df["gas_price_lag1"] = df["gas_price"].shift(1)

features_reduced = [
    # 'day_sin', 'day_cos',       # Codificació cíclica del dia
    'is_mond', 'is_tues','is_wed','is_thurs','is_fri','is_sat','is_sun',  # Indicadors de dia de la setmana 
    'is_sunday_or_holiday',
    # 'month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12',
    'hour_sin', 'hour_cos',     # Codificació cíclica de l’hora
    'type_day_workday','type_day_sat','type_day_sun','type_day_holiday',
    'holiday_coef',             # Percentatge de població en festiu
    'demand', 'low_demand',  # Indicador de baixa demanda
    'solar_share_demand', 'wind_share_demand',  # Valors bruts
    'gas_generation_share', 'gas_price',
    'residual_demand',
    'interchange_balance',
    'renewable_ratio',
    'high_renewable_ratio',
    'temp_dev',
    'price_es_24h',
    # 'wind_to_solar_ratio',
    'renewables_to_gas',
    'demand_per_gas',
    'price_rolling_3h',
    'gas_price_lag1'
]

all_features = [col for col in df.columns if col != 'target_price']
# X = df[all_features]

X = df[features_reduced]  # o les 40 si vols comparar
y = df['target_price']
y_log = np.log1p(y)

# Crear llistes per emmagatzemar les mètriques
mae_scores, rmse_scores, smape_scores = [], [], []

# Validació temporal: mantenim l'ordre cronològic
tscv = TimeSeriesSplit(n_splits=5)

for i, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Crear i entrenar el model XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    # Només analitzem els errors del darrer split (o escull un altre)
    # if i == tscv.get_n_splits() - 1:
    if i == 3:  
        analysis_df = analyze_worst_predictions(y_val, y_pred, X_val)
        print("Mostrant els 10 pitjors errors de predicció:")
        print(analysis_df)

    # Avaluar
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    rmse_scores.append(root_mean_squared_error(y_val, y_pred))
    smape_scores.append(smape(y_val, y_pred))


print("RESUM DE RENDIMENT DEL MODEL:\n")

print("MAE mitjà:", np.mean(mae_scores))
print("RMSE mitjà:", np.mean(rmse_scores))
print("SMAPE mitjà:", np.mean(smape_scores) * 100, "%")

plt.plot(mae_scores, label="MAE")
plt.plot(rmse_scores, label="RMSE")
plt.plot(smape_scores, label="SMAPE")
plt.legend()
plt.title("Rendiment del model per split")
plt.xlabel("Split temporal")
plt.ylabel("Error")
plt.show()