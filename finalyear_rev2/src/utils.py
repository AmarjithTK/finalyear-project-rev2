"""
Utility functions for data handling, saving/loading models,
and writing structured outputs (JSON/CSV) to the outputs/ directory.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")


@dataclass
class DataConfig:
    time_column: str
    target_columns: List[str]
    feature_columns: List[str]
    optional_columns: Optional[List[str]] = None
    look_back: int = 24
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    shuffle_train: bool = True


DEFAULT_CONFIG = DataConfig(
    time_column="timestamp",
    target_columns=[
        "grid_load_demand",
        "solar_pv_output",
        "wind_power_output",
    ],
    feature_columns=[
        "solar_irradiance",
        "wind_speed",
        "temperature",
        "humidity",
        "atmospheric_pressure",
        "hour_of_day",
        "day_of_week",
    ],
    optional_columns=[
        "battery_state_of_charge",
        "battery_charging_rate",
        "battery_discharging_rate",
    ],
    look_back=24,
    horizon=1,
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=64,
    shuffle_train=True,
)


def validate_schema(df: pd.DataFrame, config: DataConfig) -> Tuple[List[str], List[str]]:
    missing_required = [c for c in config.feature_columns + config.target_columns if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    optional_columns = config.optional_columns or []
    present_optional = [c for c in optional_columns if c in df.columns]
    leakage = [c for c in config.target_columns if c in config.feature_columns]
    if leakage:
        raise ValueError(f"Target columns leaked into features: {leakage}")

    return present_optional, missing_required


def resolve_feature_columns(df: pd.DataFrame, config: DataConfig) -> List[str]:
    present_optional, _ = validate_schema(df, config)
    return config.feature_columns + present_optional


def chronological_split(df: pd.DataFrame, config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.time_column in df.columns:
        df = df.sort_values(config.time_column).reset_index(drop=True)

    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame, feature_columns: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])
    return scaler


def apply_scaler(df: pd.DataFrame, feature_columns: List[str], scaler: StandardScaler) -> pd.DataFrame:
    scaled = df.copy()
    scaled[feature_columns] = scaler.transform(df[feature_columns])
    return scaled


def create_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    look_back: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[target_columns].to_numpy(dtype=np.float32)

    x_list = []
    y_list = []
    max_start = len(df) - look_back - horizon + 1
    for i in range(max_start):
        x_window = features[i : i + look_back]
        y_index = i + look_back + horizon - 1
        y_value = targets[y_index]
        x_list.append(x_window)
        y_list.append(y_value)

    x = np.stack(x_list)
    y = np.stack(y_list)

    # Convert to (N, C, L) for Conv1d
    x = np.transpose(x, (0, 2, 1))
    return x, y


class TimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_dataloaders(
    df: pd.DataFrame,
    config: DataConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, List[str]]:
    feature_columns = resolve_feature_columns(df, config)
    train_df, val_df, test_df = chronological_split(df, config)

    scaler = fit_scaler(train_df, feature_columns)
    train_df = apply_scaler(train_df, feature_columns, scaler)
    val_df = apply_scaler(val_df, feature_columns, scaler)
    test_df = apply_scaler(test_df, feature_columns, scaler)

    x_train, y_train = create_sequences(
        train_df, feature_columns, config.target_columns, config.look_back, config.horizon
    )
    x_val, y_val = create_sequences(
        val_df, feature_columns, config.target_columns, config.look_back, config.horizon
    )
    x_test, y_test = create_sequences(
        test_df, feature_columns, config.target_columns, config.look_back, config.horizon
    )

    train_loader = DataLoader(
        TimeSeriesDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        drop_last=False,
    )
    val_loader = DataLoader(
        TimeSeriesDataset(x_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(x_test, y_test),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, scaler, feature_columns

def save_predictions(df: pd.DataFrame, filename="predictions.csv"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/{filename}", index=False)
    print(f"Predictions saved to {OUTPUT_DIR}/{filename}")

def save_metrics(metrics_dict: dict, filename="metrics.json"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/{filename}", 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {OUTPUT_DIR}/{filename}")

def save_scenarios(scenarios_dict: dict, filename="scenarios.json"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/{filename}", 'w') as f:
        json.dump(scenarios_dict, f, indent=4)
    print(f"Scenarios saved to {OUTPUT_DIR}/{filename}")


def save_hybrid_model(
    cnn_lstm: torch.nn.Module,
    xgb_models,
    scaler: StandardScaler,
    config: DataConfig,
    feature_columns: List[str],
    model_dir: str = MODEL_DIR,
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    torch.save(cnn_lstm.state_dict(), os.path.join(model_dir, "cnn_lstm.pth"))
    joblib.dump(xgb_models, os.path.join(model_dir, "xgb_models.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    config_payload = {
        "config": asdict(config),
        "feature_columns": feature_columns,
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_payload, f, indent=4)
    print(f"Models saved to {model_dir}")
