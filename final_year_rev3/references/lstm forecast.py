"""
Kerala IEEE 13-Bus — LSTM Voltage Forecaster
=============================================
Trains an LSTM on the 30-day load flow dataset to predict
per-bus voltages 1–24 hours ahead.

Usage:
    python lstm_forecast.py                     # train + evaluate
    python lstm_forecast.py --predict            # load saved model, forecast next 24h
"""

import numpy as np
import pandas as pd
import argparse
import os

# ──────────────────────────────────────────────────────────────
# 1.  COLUMN DEFINITIONS  (matched to loadflow_training.csv)
# ──────────────────────────────────────────────────────────────

VOLTAGE_BUSES = [
    "V_650_pu", "V_632_pu", "V_633_pu", "V_645_pu", "V_646_pu",
    "V_671HV_pu", "V_684HV_pu", "V_692HV_pu",
    "V_675_pu", "V_680_pu", "V_611_pu",
    "V_634_pu", "V_671_pu", "V_684_pu", "V_692_pu",
]

FEATURE_COLS = VOLTAGE_BUSES + [
    "Res_kW", "Com_kW", "Ind_kW", "Crit_kW", "Total_Load_kW",
    "Solar_kW", "Wind_kW", "Total_DER_kW",
    "Solar_Irr_pu", "Net_Import_kW", "Loss_kW",
    "L_Main_kW", "L_671_kW", "L_684_kW", "L_692_kW", "L_675_kW", "L_680_kW",
    "Hour",
]

WINDOW       = 24   # look-back: 24 hours
HORIZON      = 1    # 1-step-ahead forecast
BATCH_SIZE   = 32
EPOCHS       = 50
HIDDEN_UNITS = 128


# ──────────────────────────────────────────────────────────────
# 2.  LOAD DATA
# ──────────────────────────────────────────────────────────────

def load_dataset(csv_path="loadflow_training.csv"):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Only keep rows where load-flow converged
    if "Converged" in df.columns:
        before = len(df)
        df = df[df["Converged"] == 1].reset_index(drop=True)
        print(f"Kept {len(df)}/{before} converged rows")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    return df


# ──────────────────────────────────────────────────────────────
# 3.  SCALE + WINDOWING
# ──────────────────────────────────────────────────────────────

def scale_and_window(df, window=WINDOW, target_cols=None):
    """MinMax-scale features, build sliding (X, y) windows."""
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    if target_cols is None:
        target_cols = VOLTAGE_BUSES   # predict ALL bus voltages

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_raw = scaler_X.fit_transform(df[FEATURE_COLS].values)
    y_raw = scaler_y.fit_transform(df[target_cols].values)

    X_seq, y_seq = [], []
    for i in range(len(X_raw) - window - HORIZON + 1):
        X_seq.append(X_raw[i : i + window])
        y_seq.append(y_raw[i + window])   # target = next timestep

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )
    return X_tr, X_te, y_tr, y_te, scaler_X, scaler_y


# ──────────────────────────────────────────────────────────────
# 4.  MODEL
# ──────────────────────────────────────────────────────────────

def build_model(input_shape, output_dim):
    """Stacked LSTM → predicts voltages at all buses simultaneously."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input   # type: ignore

    inp = Input(shape=input_shape, name="sequence_input")
    x   = layers.LSTM(HIDDEN_UNITS, return_sequences=True, name="lstm1")(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(HIDDEN_UNITS // 2, return_sequences=False, name="lstm2")(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(output_dim, name="voltage_output")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ──────────────────────────────────────────────────────────────
# 5.  TRAIN
# ──────────────────────────────────────────────────────────────

def train(csv_path="loadflow_training.csv"):
    import tensorflow as tf
    import joblib

    df = load_dataset(csv_path)
    X_tr, X_te, y_tr, y_te, scaler_X, scaler_y = scale_and_window(df)

    print(f"\nX_train : {X_tr.shape}   y_train : {y_tr.shape}")
    print(f"X_test  : {X_te.shape}   y_test  : {y_te.shape}")
    print(f"Features: {len(FEATURE_COLS)}   Targets: {len(VOLTAGE_BUSES)} buses\n")

    model = build_model(
        input_shape=(X_tr.shape[1], X_tr.shape[2]),
        output_dim=len(VOLTAGE_BUSES),
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint("kerala_lstm_best.keras", save_best_only=True),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────
    y_pred_scaled = model.predict(X_te)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_te)

    results = pd.DataFrame(y_true, columns=[f"True_{b}" for b in VOLTAGE_BUSES])
    for i, bus in enumerate(VOLTAGE_BUSES):
        results[f"Pred_{bus}"] = y_pred[:, i]

    print("\n── Per-Bus Voltage MAE (p.u.) ──")
    for bus in VOLTAGE_BUSES:
        mae = np.mean(np.abs(results[f"True_{bus}"] - results[f"Pred_{bus}"]))
        print(f"  {bus:<15} MAE = {mae:.5f} pu")

    results.to_csv("lstm_predictions.csv", index=False)
    print("\n✓ Predictions saved → lstm_predictions.csv")

    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    np.save("feature_cols.npy", np.array(FEATURE_COLS))
    print("✓ Scalers saved → scaler_X.pkl / scaler_y.pkl")

    return model, history, results


# ──────────────────────────────────────────────────────────────
# 6.  FORECAST NEXT 24 HOURS (rolling)
# ──────────────────────────────────────────────────────────────

def forecast_24h(model=None, csv_path="loadflow_training.csv"):
    """Roll forecast across next 24 hours using last window as seed."""
    import joblib
    import tensorflow as tf

    if model is None:
        model = tf.keras.models.load_model("kerala_lstm_best.keras")
        print("Loaded model from kerala_lstm_best.keras")

    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    df    = load_dataset(csv_path)
    X_raw = scaler_X.transform(df[FEATURE_COLS].values)

    seed      = X_raw[-WINDOW:].copy()   # (24, n_features)
    forecasts = []

    for step in range(24):
        inp        = seed[np.newaxis, :, :]              # (1, 24, n_features)
        pred_sc    = model.predict(inp, verbose=0)       # (1, n_buses)
        pred_pu    = scaler_y.inverse_transform(pred_sc)[0]
        forecasts.append(pred_pu)

        # Shift window: drop oldest row, append updated row
        new_row = seed[-1].copy()
        for i, bus in enumerate(VOLTAGE_BUSES):
            feat_idx          = FEATURE_COLS.index(bus)
            new_row[feat_idx] = pred_sc[0, i]           # use scaled prediction
        seed = np.vstack([seed[1:], new_row])

    fc_df = pd.DataFrame(forecasts, columns=VOLTAGE_BUSES)
    fc_df.insert(0, "Forecast_Hour", range(1, 25))
    fc_df.to_csv("forecast_next24h.csv", index=False)

    print("\n── 24-Hour Voltage Forecast (p.u.) ──")
    print(fc_df.round(4).to_string(index=False))
    print("\n✓ Saved → forecast_next24h.csv")
    return fc_df


# ──────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict", action="store_true",
        help="Load saved model and forecast next 24h (skip training)",
    )
    parser.add_argument(
        "--csv", default="loadflow_training.csv",
        help="Path to the load-flow CSV (default: loadflow_training.csv)",
    )
    args = parser.parse_args()

    if args.predict:
        forecast_24h(csv_path=args.csv)
    else:
        model, history, results = train(csv_path=args.csv)
        print("\nRunning 24-hour forecast from trained model …")
        forecast_24h(model=model, csv_path=args.csv)