"""
Code purely dedicated to generating predictions from the trained
hybrid model and exporting those predictions to the outputs folder.
"""

import pandas as pd
# from src.models import load_trained_hybrid_model
# from src.preprocessing import load_and_preprocess_test_data
from src.utils import save_predictions, save_metrics

def main():
    print("Loading model and test data...")
    # TODO: Load CNN-LSTM-XGBoost
    
    print("Generating predictions...")
    # TODO: Predict using model
    
    # Mock data for architecture testing
    mock_preds = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
        "actual_load": [100, 110, 120, 115, 130, 140, 135, 125, 110, 105],
        "predicted_load": [102, 112, 118, 117, 128, 142, 133, 128, 112, 106]
    })
    
    mock_metrics = {"MAE": 2.5, "RMSE": 3.1, "R2": 0.94}
    
    save_predictions(mock_preds)
    save_metrics(mock_metrics)

if __name__ == "__main__":
    main()
