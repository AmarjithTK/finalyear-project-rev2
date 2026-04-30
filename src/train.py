"""
Defines and pre-trains the CNN-LSTM feature extractor used in the hybrid pipeline.
"""

import os
import sys
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from torch.optim import Adam

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import DataConfig, DEFAULT_CONFIG, build_dataloaders, save_hybrid_model


class CNNLSTMHybrid(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        pool_size: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        embedding_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv1d(input_channels, conv_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.embedding = nn.Linear(lstm_hidden, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embedding_dim, output_dim)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        embedding = self.embedding(last)
        embedding = self.dropout(self.relu(embedding))
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        return self.output(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_features(x)


def build_model(config: DataConfig, input_channels: int) -> CNNLSTMHybrid:
    output_dim = len(config.target_columns)
    return CNNLSTMHybrid(input_channels=input_channels, output_dim=output_dim)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(loader.dataset)


def maybe_stop_early(best_loss: float, current_loss: float, patience: int, counter: int) -> Tuple[float, int, bool]:
    if current_loss < best_loss:
        return current_loss, 0, False
    counter += 1
    if counter >= patience:
        return best_loss, counter, True
    return best_loss, counter, False


def extract_embeddings(
    model: CNNLSTMHybrid,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            emb = model.extract_features(x_batch).cpu().numpy()
            features.append(emb)
            targets.append(y_batch.numpy())

    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def main(config: Optional[DataConfig] = None) -> None:
    config = config or DEFAULT_CONFIG
    print(f"Config: {asdict(config)}")

    print("Loading dataset...")
    df = pd.read_csv("data/microgrid.csv")
    print(f"Rows loaded: {len(df)}")

    print("Building dataloaders...")
    train_loader, val_loader, _, scaler, feature_columns = build_dataloaders(df, config)

    print(f"Features: {feature_columns}")
    print(f"Targets: {config.target_columns}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(config, input_channels=len(feature_columns)).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 30
    patience = 5
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    print("Starting CNN-LSTM pretraining...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        best_val, patience_counter, should_stop = maybe_stop_early(
            best_val, val_loss, patience, patience_counter
        )
        if should_stop:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Extracting embeddings...")
    train_features, train_targets = extract_embeddings(model, train_loader, device)
    val_features, val_targets = extract_embeddings(model, val_loader, device)

    print("Training XGBoost models...")
    xgb_models = []
    for target_idx, target_name in enumerate(config.target_columns):
        model_xgb = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
        )
        model_xgb.fit(
            train_features,
            train_targets[:, target_idx],
            eval_set=[(val_features, val_targets[:, target_idx])],
            verbose=False,
            early_stopping_rounds=30,
        )
        xgb_models.append(model_xgb)
        print(f"XGBoost trained for target: {target_name}")

    print("Saving hybrid models...")
    save_hybrid_model(model, xgb_models, scaler, config, feature_columns)
    print("Training complete.")


if __name__ == "__main__":
    main()
