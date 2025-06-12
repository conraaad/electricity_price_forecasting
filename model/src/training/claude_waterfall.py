import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from typing import Tuple, List

def safe_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Safe SMAPE calculation that handles zero values properly
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Add small epsilon to avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred) + epsilon) / 2
    
    # Calculate SMAPE
    smape_val = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape_val

def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error with protection against zero division
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Only calculate MAPE for non-zero true values
    non_zero_mask = np.abs(y_true) > epsilon
    if np.sum(non_zero_mask) == 0:
        return 0.0
    
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def analyze_zero_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Analyze how well the model predicts zero values
    """
    zero_mask = (y_true == 0)
    n_zeros = np.sum(zero_mask)
    
    if n_zeros == 0:
        return {"n_zeros": 0, "mean_pred_for_zeros": None, "max_pred_for_zeros": None}
    
    predictions_for_zeros = y_pred[zero_mask]
    
    return {
        "n_zeros": n_zeros,
        "percentage_zeros": n_zeros / len(y_true) * 100,
        "mean_pred_for_zeros": np.mean(predictions_for_zeros),
        "max_pred_for_zeros": np.max(predictions_for_zeros),
        "predictions_close_to_zero": np.sum(predictions_for_zeros < 1.0)
    }

class ElectricityPricePredictor:
    def __init__(self, model_type='xgboost', handle_zeros='direct'):
        """
        model_type: 'xgboost' or 'random_forest'
        handle_zeros: 'direct', 'log_transform', or 'waterfall'
        """
        self.model_type = model_type
        self.handle_zeros = handle_zeros
        self.model = None
        self.classifier = None  # For waterfall approach
        
    def _create_model(self):
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )
        else:  # random_forest
            return RandomForestRegressor(
                n_estimators=200,    
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.handle_zeros == 'direct':
            # Direct regression approach
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
            
        elif self.handle_zeros == 'log_transform':
            # Log transform approach
            self.model = self._create_model()
            y_log = np.log1p(y_train)  # log(1 + x) to handle zeros
            self.model.fit(X_train, y_log)
            
        elif self.handle_zeros == 'waterfall':
            # Two-stage approach: classification + regression
            # Stage 1: Classify zero vs non-zero
            y_is_zero = (y_train == 0).astype(int)
            
            if y_is_zero.sum() < 5:
                # Fallback to direct approach if too few zeros
                self.handle_zeros = 'direct'
                self.fit(X_train, y_train)
                return
            
            # Balance classes
            pos_weight = (len(y_is_zero) - y_is_zero.sum()) / y_is_zero.sum()
            
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                random_state=42,
                verbosity=0
            )
            self.classifier.fit(X_train, y_is_zero)
            
            # Stage 2: Regression for non-zero values
            non_zero_mask = y_train > 0
            X_train_reg = X_train[non_zero_mask]
            y_train_reg = y_train[non_zero_mask]
            
            self.model = self._create_model()
            self.model.fit(X_train_reg, y_train_reg)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.handle_zeros == 'direct':
            predictions = self.model.predict(X_test)
            # Ensure non-negative predictions
            return np.maximum(predictions, 0)
            
        elif self.handle_zeros == 'log_transform':
            y_log_pred = self.model.predict(X_test)
            predictions = np.expm1(y_log_pred)  # exp(x) - 1
            return np.maximum(predictions, 0)
            
        elif self.handle_zeros == 'waterfall':
            # First predict if price is zero
            is_zero_pred = self.classifier.predict(X_test)
            
            # Then predict actual prices for non-zero cases
            predictions = np.zeros(len(X_test))
            non_zero_mask = is_zero_pred == 0
            
            if np.any(non_zero_mask):
                X_non_zero = X_test[non_zero_mask]
                price_pred = self.model.predict(X_non_zero)
                predictions[non_zero_mask] = np.maximum(price_pred, 0)
            
            return predictions

def evaluate_models(df: pd.DataFrame, features: List[str]) -> dict:
    """
    Evaluate different model configurations
    """
    X = df[features]
    y = df['target_price']
    
    # Model configurations to test
    configs = [
        ('XGBoost Direct', 'xgboost', 'direct'),
        ('XGBoost Log Transform', 'xgboost', 'log_transform'),
        ('XGBoost Waterfall', 'xgboost', 'waterfall'),
        ('Random Forest Direct', 'random_forest', 'direct'),
        ('Random Forest Log Transform', 'random_forest', 'log_transform'),
    ]
    
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)
    
    for config_name, model_type, zero_handling in configs:
        print(f"\nEvaluating {config_name}...")
        
        mae_scores, rmse_scores, smape_scores, mape_scores = [], [], [], []
        zero_analyses = []
        
        for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            predictor = ElectricityPricePredictor(model_type, zero_handling)
            predictor.fit(X_train, y_train)
            
            # Make predictions
            y_pred = predictor.predict(X_val)
            
            # Calculate metrics
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            smape_scores.append(safe_smape(y_val.values, y_pred))
            mape_scores.append(mape(y_val.values, y_pred))
            
            # Analyze zero predictions
            zero_analyses.append(analyze_zero_predictions(y_val.values, y_pred))
        
        # Store results
        results[config_name] = {
            'MAE': np.mean(mae_scores),
            'RMSE': np.mean(rmse_scores),
            'SMAPE': np.mean(smape_scores),
            'MAPE': np.mean(mape_scores),
            'MAE_std': np.std(mae_scores),
            'RMSE_std': np.std(rmse_scores),
            'zero_analysis': zero_analyses[-1]  # Last fold analysis
        }
        
        print(f"MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
        print(f"RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
        print(f"SMAPE: {np.mean(smape_scores):.2f}%")
        print(f"MAPE: {np.mean(mape_scores):.2f}%")
    
    return results

# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../../data/def_data/training_dataset_2024.csv")
    
    # Feature engineering
    df['gas_generation_share'] = round(df['gas_generation'] / df['demand'], 4)
    df['target_price'] = abs(df['target_price'])  # Evitar preus negatius
    df['renewable_ratio'] = round((df['solar'] + df['wind']) / df['demand'], 4)


    features_reduced = [
        # 'day_sin', 'day_cos',       # Codificació cíclica del dia
        'is_mond', 'is_tues','is_wed','is_thurs','is_fri','is_sat','is_sun',  # Indicadors de dia de la setmana 
        # 'month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12',
        'hour_sin', 'hour_cos',     # Codificació cíclica de l’hora
        'type_day_workday','type_day_sat','type_day_sun','type_day_holiday','holiday_coef',             # Percentatge de població en festiu
        'demand', 'solar_share_demand', 'wind_share_demand',  # Valors bruts
        'gas_generation_share', 'gas_price',
        'residual_demand',
        'interchange_balance',
        'renewable_ratio',
        'temp_dev',
        'price_es_24h'
    ]
    
    print("Dataset shape:", df.shape)
    print("Zero prices:", (df['target_price'] == 0).sum(), f"({(df['target_price'] == 0).mean():.2%})")
    print("Price statistics:")
    print(df['target_price'].describe())
    
    # Evaluate all models
    results = evaluate_models(df, features_reduced)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results).T
    print(results_df[['MAE', 'RMSE', 'SMAPE', 'MAPE']].round(2))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['MAE', 'RMSE', 'SMAPE', 'MAPE']
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        values = [results[model][metric] for model in results.keys()]
        bars = ax.bar(range(len(results)), values)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(results.keys(), rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()