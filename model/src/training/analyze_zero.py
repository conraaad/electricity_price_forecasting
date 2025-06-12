import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

def analyze_zero_price_patterns(df: pd.DataFrame) -> None:
    """
    Comprehensive analysis of zero price patterns in the dataset
    """
    print("="*60)
    print("ZERO PRICE PATTERN ANALYSIS")
    print("="*60)
    
    # Basic statistics
    zero_mask = (df['target_price'] == 0)
    n_zeros = np.sum(zero_mask)
    total_samples = len(df)
    
    print(f"Total samples: {total_samples}")
    print(f"Zero prices: {n_zeros} ({n_zeros/total_samples*100:.2f}%)")
    print(f"Non-zero prices: {total_samples - n_zeros} ({(total_samples-n_zeros)/total_samples*100:.2f}%)")
    
    if n_zeros == 0:
        print("No zero prices found in dataset!")
        return
    
    # Price distribution
    print(f"\nPrice statistics:")
    print(df['target_price'].describe())
    
    # Temporal patterns of zero prices
    print(f"\nTemporal patterns of zero prices:")
    
    # By hour
    if 'hour_sin' in df.columns and 'hour_cos' in df.columns:
        # Reconstruct hour from sin/cos encoding
        hours = np.arctan2(df['hour_sin'], df['hour_cos']) * 12 / np.pi
        hours = ((hours + 24) % 24).astype(int)  # Convert to int here
        
        zero_by_hour = pd.DataFrame({
            'hour': hours,
            'is_zero': zero_mask
        }).groupby('hour')['is_zero'].agg(['count', 'sum', 'mean']).reset_index()
        zero_by_hour['zero_rate'] = zero_by_hour['mean'] * 100
        
        print("Zero rate by hour:")
        for _, row in zero_by_hour.iterrows():
            if row['sum'] > 0:
                # Convert hour to int for formatting
                hour_int = int(row['hour'])
                print(f"  Hour {hour_int:2d}: {row['sum']:3.0f}/{row['count']:3.0f} ({row['zero_rate']:5.1f}%)")

    # By month
    month_cols = [col for col in df.columns if col.startswith('month_')]
    if month_cols:
        print("Zero rate by month:")
        for i, col in enumerate(month_cols, 1):
            if df[col].sum() > 0:  # This month exists in data
                month_mask = df[col] == 1
                month_zeros = np.sum(zero_mask & month_mask)
                month_total = np.sum(month_mask)
                if month_total > 0:
                    print(f"  Month {i:2d}: {month_zeros:3d}/{month_total:4d} ({month_zeros/month_total*100:5.1f}%)")
    
    # By day type
    day_type_cols = ['type_day_workday', 'type_day_sat', 'type_day_sun', 'type_day_holiday']
    day_type_names = ['Workday', 'Saturday', 'Sunday', 'Holiday']
    
    print(f"\nZero rate by day type:")
    for col, name in zip(day_type_cols, day_type_names):
        if col in df.columns:
            day_mask = df[col] == 1
            day_zeros = np.sum(zero_mask & day_mask)
            day_total = np.sum(day_mask)
            if day_total > 0:
                print(f"  {name:8s}: {day_zeros:3d}/{day_total:4d} ({day_zeros/day_total*100:5.1f}%)")
    
    # Feature correlations with zero prices
    print(f"\nFeature correlations with zero prices:")
    
    # Key features to check
    key_features = ['demand', 'solar_share_demand', 'wind_share_demand', 
                   'gas_generation_share', 'gas_price', 'residual_demand', 
                   'interchange_balance', 'temp_dev', 'price_es_24h']
    
    correlations = []
    for feature in key_features:
        if feature in df.columns:
            # Calculate correlation with zero prices
            if feature == 'gas_generation_share':
                # Handle potential division by zero
                safe_feature = df[feature].fillna(0)
            else:
                safe_feature = df[feature]
                
            corr = np.corrcoef(zero_mask.astype(int), safe_feature)[0, 1]
            correlations.append((feature, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, corr in correlations:
        print(f"  {feature:20s}: {corr:6.3f}")
    
    # Conditions analysis
    print(f"\nConditions when prices are zero:")
    zero_data = df[zero_mask]
    non_zero_data = df[~zero_mask]
    
    for feature in key_features[:5]:  # Top 5 features
        if feature in df.columns:
            zero_mean = zero_data[feature].mean()
            nonzero_mean = non_zero_data[feature].mean()
            print(f"  {feature:20s}: Zero={zero_mean:8.2f}, Non-zero={nonzero_mean:8.2f}")

def test_zero_prediction_strategies(df: pd.DataFrame, features: list) -> None:
    """
    Test different strategies specifically for zero price prediction
    """
    print("\n" + "="*60)
    print("ZERO PREDICTION STRATEGY TESTING")
    print("="*60)
    
    X = df[features]
    y = df['target_price']
    
    # Create binary target for classification
    y_binary = (y == 0).astype(int)
    
    print(f"Class distribution: {np.sum(y_binary)} zeros, {np.sum(~y_binary)} non-zeros")
    
    # Test different classification thresholds
    tscv = TimeSeriesSplit(n_splits=3)
    
    strategies = [
        ("Direct Regression + Threshold 0.5", "threshold", 0.5),
        ("Direct Regression + Threshold 1.0", "threshold", 1.0),
        ("Direct Regression + Threshold 2.0", "threshold", 2.0),
        ("Binary Classification", "classification", None),
        ("Log Transform", "log", None),
    ]
    
    results = {}
    
    for strategy_name, method, threshold in strategies:
        print(f"\nTesting: {strategy_name}")
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if method == "threshold":
                # Direct regression with threshold
                model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_pred[y_pred < threshold] = 0
                y_pred = np.maximum(y_pred, 0)
                
            elif method == "classification":
                # Binary classification approach
                y_binary_train = (y_train == 0).astype(int)
                y_binary_val = (y_val == 0).astype(int)
                
                model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
                model.fit(X_train, y_binary_train)
                y_pred_binary = model.predict(X_val)
                
                # Simple strategy: predict 0 for zeros, mean price for non-zeros
                mean_nonzero_price = y_train[y_train > 0].mean()
                y_pred = np.where(y_pred_binary == 1, 0, mean_nonzero_price)
                
            elif method == "log":
                # Log transformation
                model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                y_log_train = np.log1p(y_train)
                model.fit(X_train, y_log_train)
                y_log_pred = model.predict(X_val)
                y_pred = np.expm1(y_log_pred)
                y_pred = np.maximum(y_pred, 0)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            # Zero-specific metrics
            zero_mask_val = (y_val == 0)
            if np.sum(zero_mask_val) > 0:
                zero_mae = mean_absolute_error(y_val[zero_mask_val], y_pred[zero_mask_val])
                zero_predictions_for_zeros = y_pred[zero_mask_val]
                correct_zero_predictions = np.sum(zero_predictions_for_zeros < 1.0)
                zero_accuracy = correct_zero_predictions / np.sum(zero_mask_val)
            else:
                zero_mae = 0
                zero_accuracy = 0
            
            fold_results.append({
                'mae': mae,
                'rmse': rmse,
                'zero_mae': zero_mae,
                'zero_accuracy': zero_accuracy
            })
        
        # Average results
        avg_results = {
            'mae': np.mean([r['mae'] for r in fold_results]),
            'rmse': np.mean([r['rmse'] for r in fold_results]), 
            'zero_mae': np.mean([r['zero_mae'] for r in fold_results]),
            'zero_accuracy': np.mean([r['zero_accuracy'] for r in fold_results])
        }
        
        results[strategy_name] = avg_results
        
        print(f"  MAE: {avg_results['mae']:.2f}")
        print(f"  RMSE: {avg_results['rmse']:.2f}")
        print(f"  Zero MAE: {avg_results['zero_mae']:.2f}")
        print(f"  Zero Accuracy: {avg_results['zero_accuracy']:.2%}")
    
    # Summary
    print(f"\n{'Strategy':<35} {'MAE':<8} {'RMSE':<8} {'Zero MAE':<8} {'Zero Acc':<8}")
    print("-" * 70)
    for strategy, result in results.items():
        print(f"{strategy:<35} {result['mae']:<8.2f} {result['rmse']:<8.2f} "
              f"{result['zero_mae']:<8.2f} {result['zero_accuracy']:<8.1%}")

def plot_price_distribution_analysis(df: pd.DataFrame) -> None:
    """
    Visualize price distribution and zero price patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Price distribution
    axes[0, 0].hist(df['target_price'], bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Price (€/MWh)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Zero Price')
    axes[0, 0].legend()
    
    # 2. Log-scale price distribution (excluding zeros)
    non_zero_prices = df['target_price'][df['target_price'] > 0]
    if len(non_zero_prices) > 0:
        axes[0, 1].hist(non_zero_prices, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Price (€/MWh)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Non-Zero Price Distribution')
        axes[0, 1].set_yscale('log')
    
    # 3. Zero prices over time (if datetime available)
    if 'datetime_iso' in df.columns:
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['datetime_iso'])
        df_copy['date'] = df_copy['datetime'].dt.date
        
        daily_zeros = df_copy.groupby('date').agg({
            'target_price': ['count', lambda x: (x == 0).sum()]
        }).reset_index()
        daily_zeros.columns = ['date', 'total', 'zeros']
        daily_zeros['zero_rate'] = daily_zeros['zeros'] / daily_zeros['total']
        
        axes[1, 0].plot(daily_zeros['date'], daily_zeros['zero_rate'] * 100)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Zero Price Rate (%)')
        axes[1, 0].set_title('Zero Price Rate Over Time')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Price vs key features
    if 'residual_demand' in df.columns:
        scatter = axes[1, 1].scatter(df['residual_demand'], df['target_price'], 
                                   c=(df['target_price'] == 0), alpha=0.5)
        axes[1, 1].set_xlabel('Residual Demand')
        axes[1, 1].set_ylabel('Price (€/MWh)')
        axes[1, 1].set_title('Price vs Residual Demand')
        plt.colorbar(scatter, ax=axes[1, 1], label='Is Zero Price')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("../../data/def_data/training_dataset_2024.csv")
    
    # Feature engineering
    df['gas_generation_share'] = df['gas_generation'] / df['demand']
    df['target_price'] = abs(df['target_price'])  # Avoid negative prices
    
    # Define features
    features_reduced = [
        'month_1','month_2','month_3','month_4','month_5','month_6',
        'month_7','month_8','month_9','month_10','month_11','month_12',
        'hour_sin', 'hour_cos',
        'type_day_workday','type_day_sat','type_day_sun','type_day_holiday',
        'holiday_coef',
        'demand', 'solar_share_demand', 'wind_share_demand',
        'gas_generation_share', 'gas_price',
        'residual_demand',
        'interchange_balance',
        'temp_dev',
        'price_es_24h'
    ]
    
    print("Dataset shape:", df.shape)
    
    # Run comprehensive analysis
    analyze_zero_price_patterns(df)
    test_zero_prediction_strategies(df, features_reduced)
    plot_price_distribution_analysis(df)