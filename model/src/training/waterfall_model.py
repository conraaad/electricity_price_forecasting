
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import numpy as np
from metrics_utils import smape, analyze_worst_predictions


df = pd.read_csv("../../data/def_data/training_dataset.csv")

df['gas_generation_share'] = round(df['gas_generation'] / df['demand'], 4)
df['is_zero'] = (df['target_price'] == 0).astype(int)

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
X = df[features_reduced]  # O les teves 40 features

y_price = df['target_price']
y_zero = df['is_zero']


mae_scores, rmse_scores, smape_scores = [], [], []

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    # Separar en conjunt d'entrenament i validació
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_price = y_price.iloc[train_idx]
    y_val_price = y_price.iloc[val_idx]
    y_train_zero = y_zero.iloc[train_idx]
    y_val_zero = y_zero.iloc[val_idx]

    # Fase 1: entrenar classificador
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
    )
    clf.fit(X_train, y_train_zero)

    # Fase 2: entrenar regressor només amb valors no-zero
    X_train_reg = X_train[y_train_price > 0]
    y_train_reg = y_train_price[y_train_price > 0]

    reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    reg.fit(X_train_reg, y_train_reg)

    # Predicció fase 1 (classificació)
    is_zero_pred = clf.predict(X_val)

    # Predicció fase 2 (regressió per als que NO són zero)
    y_pred_combined = []
    for i, is_zero in enumerate(is_zero_pred):
        if is_zero == 1:
            y_pred_combined.append(0.0)
        else:
            x_instance = X_val.iloc[i:i+1]
            y_hat = reg.predict(x_instance)[0]
            y_pred_combined.append(y_hat)

    # Convertir a array per a mètriques
    y_pred_combined = np.array(y_pred_combined)

    # Mètriques finals
    mae_scores.append(mean_absolute_error(y_val_price, y_pred_combined))
    rmse_scores.append(root_mean_squared_error(y_val_price, y_pred_combined))
    smape_scores.append(smape(y_val_price.values, y_pred_combined))

print("RESUM DE RENDIMENT DEL MODEL:\n")

print("MAE mitjà:", np.mean(mae_scores))
print("RMSE mitjà:", np.mean(rmse_scores))
print("SMAPE mitjà:", np.mean(smape_scores) * 100, "%")  # en percentatge

print("\nErrors per split:")
plt.plot(mae_scores, label="MAE")
plt.plot(rmse_scores, label="RMSE")
plt.plot(smape_scores, label="MAPE")
plt.legend()
plt.title("Rendiment del model per split")
plt.xlabel("Split temporal")
plt.ylabel("Error")
plt.show()