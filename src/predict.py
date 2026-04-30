"""
Generates predictions from the trained hybrid model and exports
predictions/metrics to the outputs folder.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.train import build_model
from src.utils import (
    DataConfig,
    MODEL_DIR,
    apply_scaler,
    chronological_split,
    create_sequences,
    save_metrics,
    save_predictions,
)


def load_artifacts(model_dir: str = MODEL_DIR) -> Tuple[DataConfig, List[str], object, List[xgb.XGBRegressor]]:
    config_path = os.path.join(model_dir, "config.json")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    xgb_path = os.path.join(model_dir, "xgb_models.joblib")

    with open(config_path, "r") as f:
        payload = json.load(f)

    config = DataConfig(**payload["config"])
    feature_columns = payload["feature_columns"]
    scaler = joblib.load(scaler_path)
    xgb_models = joblib.load(xgb_path)
    return config, feature_columns, scaler, xgb_models


def build_prediction_dataframe(
    timestamps: pd.Series,
    target_columns: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    result = pd.DataFrame({"timestamp": timestamps.values})
    for idx, name in enumerate(target_columns):
        if y_true is not None:
            result[f"actual_{name}"] = y_true[:, idx]
        result[f"predicted_{name}"] = y_pred[:, idx]
    return result


def compute_metrics(target_columns: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(target_columns):
        actual = y_true[:, idx]
        predicted = y_pred[:, idx]
        error = predicted - actual
        abs_error = np.abs(error)
        squared_error = np.square(error)
        nonzero_mask = actual != 0
        mape = np.nan
        if np.any(nonzero_mask):
            mape = np.mean(abs_error[nonzero_mask] / np.abs(actual[nonzero_mask])) * 100

        smape_denominator = np.abs(actual) + np.abs(predicted)
        smape_mask = smape_denominator != 0
        smape = np.nan
        if np.any(smape_mask):
            smape = np.mean((2 * abs_error[smape_mask]) / smape_denominator[smape_mask]) * 100

        metrics[name] = {
            "MAE": float(mean_absolute_error(actual, predicted)),
            "MSE": float(mean_squared_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "MedianAE": float(np.median(abs_error)),
            "MaxAE": float(np.max(abs_error)),
            "MAPE_percent": float(mape),
            "sMAPE_percent": float(smape),
            "MBE": float(np.mean(error)),
            "Error_STD": float(np.std(error)),
            "Mean_Squared_Error": float(np.mean(squared_error)),
            "R2": float(r2_score(actual, predicted)),
        }
    return metrics


def main() -> None:
    print("Loading model artifacts...")
    config, feature_columns, scaler, xgb_models = load_artifacts()
    print(f"Targets: {config.target_columns}")
    print(f"Features: {feature_columns}")

    print("Loading dataset...")
    df = pd.read_csv("data/microgrid.csv")
    train_df, val_df, test_df = chronological_split(df, config)

    if config.time_column not in test_df.columns:
        raise ValueError(f"Missing time column: {config.time_column}")

    missing_features = [c for c in feature_columns if c not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    print("Scaling test data...")
    test_df = apply_scaler(test_df, feature_columns, scaler)

    print("Building test sequences...")
    x_test, y_test = create_sequences(
        test_df,
        feature_columns,
        config.target_columns,
        config.look_back,
        config.horizon,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, input_channels=len(feature_columns)).to(device)
    model_path = os.path.join(MODEL_DIR, "cnn_lstm.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Extracting embeddings...")
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_test).to(device)
        embeddings = model.extract_features(x_tensor).cpu().numpy()

    print("Running XGBoost inference...")
    preds = []
    for model_xgb in xgb_models:
        preds.append(model_xgb.predict(embeddings))
    y_pred = np.stack(preds, axis=1)

    start_idx = config.look_back + config.horizon - 1
    timestamps = test_df[config.time_column].iloc[start_idx : start_idx + len(y_pred)]

    y_true = y_test if y_test.size else None
    predictions_df = build_prediction_dataframe(timestamps, config.target_columns, y_true, y_pred)
    print("Saving predictions...")
    save_predictions(predictions_df)

    if y_true is not None:
        print("Computing metrics...")
        metrics = compute_metrics(config.target_columns, y_true, y_pred)
        save_metrics(metrics)

    print("Prediction complete.")


if __name__ == "__main__":
    main()
