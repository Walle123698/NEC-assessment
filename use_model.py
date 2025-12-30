#!/usr/bin/env python3
"""
Script to use the trained final model to select the best plant for a given demand scenario.
The model takes demand characteristics and plant features, and recommends the optimal plant.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add pipeline src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Define paths
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "final_model.pkl"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.json"
PLANTS_PATH = Path(__file__).parent / "data" / "raw" / "plants.csv"


def load_model_artifacts():
    """Load the trained model, preprocessor, and feature names."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = json.load(f)
    
    # Load plants data without converting 'NA' to NaN
    plants_df = pd.read_csv(PLANTS_PATH, keep_default_na=False, na_values='')
    
    return model, preprocessor, feature_names, plants_df


def prepare_input_data(demand_features, plant_features):
    """
    Prepare input data for prediction.
    
    Args:
        demand_features (list): List of 12 demand feature values (DF1-DF12)
        plant_features (list): List of plant feature values (PF1-PF18)
    
    Returns:
        np.ndarray: Prepared feature array
    """
    # Combine all features in the correct order
    all_features = demand_features + plant_features
    return np.array(all_features).reshape(1, -1)


def get_plant_id_from_index(plant_index, plants_df):
    """Convert numerical plant index to plant ID."""
    plant_index = int(round(plant_index))
    plant_index = max(0, min(plant_index, len(plants_df) - 1))
    return plants_df.iloc[plant_index]['Plant ID']


def make_prediction(demand_features, plant_id=None):
    """
    Make a prediction using the trained model to select the best plant.
    
    Args:
        demand_features (list): List of 12 demand feature values (DF1-DF12)
        plant_id (str, optional): If provided, predicts cost for that specific plant.
                                 If None, finds the best plant across all available plants.
    
    Returns:
        dict: Prediction result with plant selection and details
    """
    # Load artifacts
    model, preprocessor, feature_names, plants_df = load_model_artifacts()
    
    if plant_id is not None:
        # Single plant prediction
        plant_row = plants_df[plants_df['Plant ID'] == plant_id].iloc[0]
        plant_features = [plant_row[col] for col in feature_names['plant_features']]
        X = prepare_input_data(demand_features, plant_features)
        
        try:
            X_processed = preprocessor.transform(X)
        except:
            X_processed = X
        
        plant_index = model.predict(X_processed)[0]
        predicted_plant = get_plant_id_from_index(plant_index, plants_df)
        
        return {
            'selected_plant': predicted_plant,
            'plant_type': plant_row['Plant Type'],
            'region': plant_row['Region'],
            'raw_prediction': plant_index,
            'feature_count': X.shape[1]
        }
    
    else:
        # Find best plant across all plants
        best_plant = None
        best_prediction = float('inf')
        
        for _, plant_row in plants_df.iterrows():
            plant_features = [plant_row[col] for col in feature_names['plant_features']]
            X = prepare_input_data(demand_features, plant_features)
            
            try:
                X_processed = preprocessor.transform(X)
            except:
                X_processed = X
            
            prediction = model.predict(X_processed)[0]
            
            if prediction < best_prediction:
                best_prediction = prediction
                best_plant = plant_row['Plant ID']
        
        best_plant_row = plants_df[plants_df['Plant ID'] == best_plant].iloc[0]
        
        return {
            'selected_plant': best_plant,
            'plant_type': best_plant_row['Plant Type'],
            'region': best_plant_row['Region'],
            'raw_prediction': best_prediction,
            'feature_count': 30
        }


def example_prediction():
    """Run an example prediction with dummy demand scenario."""
    print("=" * 70)
    print("NEC Smart Plant Selection - Example Demand Scenario")
    print("=" * 70)
    
    # Create a random demand scenario
    print("\n📊 Generating Random Demand Scenario...")
    demand_features = np.random.randn(12).tolist()  # 12 demand features (DF1-DF12)
    
    print("\n📈 Input Features:")
    print(f"  Demand Characteristics (DF1-DF12):")
    for i, val in enumerate(demand_features, 1):
        print(f"    DF{i}: {val:7.2f}")
    
    # Make prediction - find best plant across all plants
    print("\n🔍 Making prediction...")
    result = make_prediction(demand_features)
    
    print("\n✅ Plant Selection Result:")
    print(f"  Selected Plant: {result['selected_plant']}")
    print(f"  Plant Type: {result['plant_type']}")
    print(f"  Region: {result['region']}")
    print(f"  Features Used: {result['feature_count']}")
    
    return result


if __name__ == "__main__":
    # Run example prediction
    result = example_prediction()
    