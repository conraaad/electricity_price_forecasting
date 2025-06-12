
import numpy as np
import pandas as pd


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
    worst = result.sort_values(by='abs_error', ascending=False).head(20)
    
    return worst