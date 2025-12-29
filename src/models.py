

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from typing import Dict, Any


class ModelFactory:
    """Factory class for creating ML models"""
    
    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any], random_seed: int):
        """
        Create a model instance based on type and parameters.
        
        Args:
            model_type: One of 'random_forest', 'gradient_boosting', or 'lasso' 
            params: Model hyperparameters
            random_seed: Random seed for reproducibility
            
        Returns:
            Configured sklearn model instance
            
        Raises:
            ValueError: If model_type is not recognized 
        """
        
        # Add random_state to params for reproducibility
        params_with_seed = params.copy()
        params_with_seed['random_state'] = random_seed
        
        if model_type == 'random_forest':
            return RandomForestRegressor(**params_with_seed)
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**params_with_seed)
        
        elif model_type == 'lasso':
            return Lasso(**params_with_seed)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_name(model_type: str) -> str:
        """Get human-readable model name"""
        
        names = {
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'lasso': 'Lasso Regression'
        }
        return names.get(model_type, model_type)
    
    @staticmethod
    def get_model_description(model_type: str) -> str:
        """Get model description for reporting"""
       
        descriptions = {
            'random_forest': 'Ensemble method using bagging with decision trees. Robust and interpretable.',
            'gradient_boosting': 'Ensemble method using boosting for sequential improvement. Often superior performance.',
            'lasso': 'Linear regression with L1 regularization. Fast and interpretable with feature selection.'
        }
        return descriptions.get(model_type, '')
