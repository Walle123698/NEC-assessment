

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from typing import List


def calculate_plant_selection_error_grouped(y_true: np.ndarray, 
                                            y_pred: np.ndarray,
                                            demand_ids: np.ndarray) -> np.ndarray:
    
    df = pd.DataFrame({
        'Demand_ID': demand_ids,
        'Actual_Cost': y_true,
        'Predicted_Cost': y_pred
    })
    
    errors = []
    
    for demand_id, group in df.groupby('Demand_ID'):
        # Oracle: minimum actual cost for this demand across all plants
        oracle_cost = group['Actual_Cost'].min()
        
        # Model selection: select plant with lowest predicted cost
        selected_idx = group['Predicted_Cost'].idxmin()
        selected_actual_cost = group.loc[selected_idx, 'Actual_Cost']
        
        # Error: how much more expensive is the selected plant vs oracle
        # Positive error means we overpaid (worse than oracle)
        # Zero error means we selected the optimal plant
        error = selected_actual_cost - oracle_cost
        errors.append(error)
    
    return np.array(errors)


def plant_selection_error_scorer(estimator, X, y, demand_ids=None):
   
    if demand_ids is None:
        raise ValueError("demand_ids must be provided to plant_selection_error_scorer")
    
    # Predict costs
    y_pred = estimator.predict(X)
    
    # Calculate selection errors
    errors = calculate_plant_selection_error_grouped(y, y_pred, demand_ids)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # Return negative RMSE (sklearn convention: higher is better)
    return -rmse


def create_custom_scorer():
    
    # We cannot use make_scorer directly with additional parameters in GridSearchCV
    # So we'll return the function itself and handle scoring manually
    return plant_selection_error_scorer


def calculate_rmse(errors: np.ndarray) -> float:
  
    return np.sqrt(np.mean(errors ** 2))


def calculate_error_statistics(errors: np.ndarray) -> dict:
    
    if len(errors) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'rmse': 0.0,
            'count': 0
        }
    
    return {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'rmse': float(calculate_rmse(errors)),
        'count': len(errors)
    }
