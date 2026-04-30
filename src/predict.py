"""
Generates predictions from the trained hybrid model and exports
predictions/metrics to the outputs folder.
"""

import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        metrics[name] = {
            "MAE": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
            "RMSE": float(mean_squared_error(y_true[:, idx], y_pred[:, idx], squared=False)),
            "R2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
        }
    return metrics


def main() -> None:
    print("Loading model artifacts...")
    config, feature_columns, scaler, xgb_models = load_artifacts()

    df = pd.read_csv("data/microgrid.csv")
    train_df, val_df, test_df = chronological_split(df, config)

    if config.time_column not in test_df.columns:
        raise ValueError(f"Missing time column: {config.time_column}")

    missing_features = [c for c in feature_columns if c not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    test_df = apply_scaler(test_df, feature_columns, scaler)

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

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_test).to(device)
        embeddings = model.extract_features(x_tensor).cpu().numpy()

    preds = []
    for model_xgb in xgb_models:
        preds.append(model_xgb.predict(embeddings))
    y_pred = np.stack(preds, axis=1)

    start_idx = config.look_back + config.horizon - 1
    timestamps = test_df[config.time_column].iloc[start_idx : start_idx + len(y_pred)]

    y_true = y_test if y_test.size else None
    predictions_df = build_prediction_dataframe(timestamps, config.target_columns, y_true, y_pred)
    save_predictions(predictions_df)

    if y_true is not None:
        metrics = compute_metrics(config.target_columns, y_true, y_pred)
        save_metrics(metrics)

    print("Prediction complete.")


if __name__ == "__main__":
    main()
