import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_pred - y_true)
    return np.mean(diff / denominator) * 100

def analyze_worst_predictions(y_true, y_pred, X_val):
    # Make sure inputs are numpy arrays or pandas Series with same indices
    y_pred = pd.Series(y_pred, index=X_val.index)
    y_true = pd.Series(y_true, index=X_val.index)
    
    # Calculate errors
    errors = np.abs(y_true - y_pred)
    rel_error = np.where(y_true != 0, errors / np.abs(y_true), np.inf)
    
    # Create DataFrame with all data
    error_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'abs_error': errors,
        'rel_error': rel_error
    }, index=X_val.index)
    
    # Join with features
    result = pd.concat([error_df, X_val], axis=1)
    
    # Sort by relative error and get top 10
    worst = result.sort_values(by='rel_error', ascending=False).head(10)
    
    return worst

df = pd.read_csv("../../data/def_data/training_dataset.csv")

df['gas_genration_share'] = round(df['gas_generation'] / df['demand'], 4)

features_reduced = [
    # 'day_sin', 'day_cos',       # Codificació cíclica del dia
    'month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12',
    'hour_sin', 'hour_cos',     # Codificació cíclica de l’hora
    'type_day_workday','type_day_sat','type_day_sun','type_day_holiday','holiday_coef',             # Percentatge de població en festiu
    'demand', 'solar_share_demand', 'wind_share_demand',  # Valors bruts
    'gas_generation', 'gas_price',
    'residual_demand',
    'interchange_balance',
    'temp_dev',
    'price_es_24h'
]


# Definim les variables d'entrada (X) i la variable objectiu (y)
# feature_cols = [col for col in df.columns if col != 'target_price']
# X = df[feature_cols]

X = df[features_reduced]
y = df['target_price']

tscv = TimeSeriesSplit(n_splits=5)  # Es faran 5 divisions successives

mae_scores, rmse_scores, smape_scores = [], [], []

for i, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Creació i entrenament del model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicció
    y_pred = model.predict(X_val)

    # Calcul de les mètriques
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    rmse_scores.append(root_mean_squared_error(y_val, y_pred))
    smape_scores.append(smape(y_val, y_pred))

    # Només analitzem els errors del darrer split (o escull un altre)
    if i == tscv.get_n_splits() - 1:
        analysis_df = analyze_worst_predictions(y_val, y_pred, X_val)
        print("Mostrant els 10 pitjors errors de predicció:")
        print(analysis_df)

print("RESUM DE RENDIMENT DEL MODEL:\n")

print("MAE mitjà:", np.mean(mae_scores))
print("RMSE mitjà:", np.mean(rmse_scores))
print("SMAPE mitjà:", np.mean(smape_scores) * 100, "%")  # en percentatge

# print("\nErrors per split:")
# plt.plot(mae_scores, label="MAE")
# plt.plot(rmse_scores, label="RMSE")
# plt.plot(smape_scores, label="MAPE")
# plt.legend()
# plt.title("Rendiment del model per split")
# plt.xlabel("Split temporal")
# plt.ylabel("Error")
# plt.show()