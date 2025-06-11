

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from metrics_utils import smape, analyze_worst_predictions

df = pd.read_csv("../../data/def_data/training_dataset_no_2021.csv")

df['gas_generation_share'] = round(df['gas_generation'] / df['demand'], 4)

features_reduced = [
    # 'day_sin', 'day_cos',                         # Codificació cíclica del dia
    'month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12',
    'hour_sin', 'hour_cos',                         # Codificació cíclica de l’hora
    'type_day_workday','type_day_sat','type_day_sun','type_day_holiday',
    'holiday_coef',                                 # Percentatge de població en festiu
    'demand', 'solar_share_demand', 'wind_share_demand',  # Valors bruts
    'gas_generation_share', 'gas_price',
    'residual_demand',
    'interchange_balance',
    'temp_dev',
    'price_es_24h'
]

all_features = [col for col in df.columns if col != 'target_price']
# X = df[all_features]

X = df[features_reduced]  # o les 40 si vols comparar
y = df['target_price']

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
    if i == tscv.get_n_splits() - 1:
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