import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
import joblib

# --- 1. Configurations ---
DATA_FILE = "../datasets/unified_microgrid_24h_results.csv"
SEQ_LENGTH = 24  # Use 24 hours of history to predict the next hour
N_ESTIMATORS = 20
MAX_DEPTH = 2
LEARNING_RATE = 0.03
SUBSAMPLE = 0.65
COLSAMPLE_BYTREE = 0.65

# --- 2. Load Data ---
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# Input features (Weather)
features = ['Hour', 'Temperature_C', 'Humidity_pct', 'Wind_Speed_ms', 'Cloud_Cover_pct', 'Solar_Irradiance_Wm2']

# Target variables to predict (Generation and Loads)
targets = [
    'Solar_MW', 'Wind_MW',
    'Residential_Load_MW', 'Commercial_Load_MW', 
    'Industrial_Load_MW', 'Critical_Load_MW'
]

print(f"Dataset shape: {df.shape}")

# Combine features and targets for sequential processing
data = df[features + targets].values

# --- 3. Scaling ---
print("Scaling Data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --- 4. Sequence Generation (Time Series windowing) ---
def create_sequences(data, seq_length, num_targets):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        # XGBoost requires 2D input (samples, features), so we flatten the sequence out
        X.append(data[i:i + seq_length, :].flatten())  
        y.append(data[i + seq_length, -num_targets:])  
    return np.array(X), np.array(y)

def calculate_metrics(actuals, predictions):
    epsilon = 1e-8
    non_zero_actuals = np.abs(actuals) > epsilon
    smape_denominator = np.abs(actuals) + np.abs(predictions)
    non_zero_smape = smape_denominator > epsilon

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions, multioutput='uniform_average')
    explained_variance = explained_variance_score(actuals, predictions, multioutput='uniform_average')
    median_ae = median_absolute_error(actuals, predictions)
    max_ae = np.max(np.abs(actuals - predictions))
    mape = np.mean(np.abs((actuals[non_zero_actuals] - predictions[non_zero_actuals]) / actuals[non_zero_actuals])) * 100
    smape = np.mean(
        2 * np.abs(predictions[non_zero_smape] - actuals[non_zero_smape]) / smape_denominator[non_zero_smape]
    ) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_variance,
        'median_ae': median_ae,
        'max_ae': max_ae,
        'mape': mape,
        'smape': smape,
    }

def print_metrics(title, metrics):
    print(f"\n--- Summary of {title} Predictions ---")
    print(f"Mean Squared Error (MSE):             {metrics['mse']:.5f}")
    print(f"Root Mean Sq Error (RMSE):            {metrics['rmse']:.5f}")
    print(f"Mean Absolute Error (MAE):            {metrics['mae']:.5f}")
    print(f"R-squared (R2):                       {metrics['r2']:.5f}")
    print(f"Explained Variance Score:             {metrics['explained_variance']:.5f}")
    print(f"Median Absolute Error (MedAE):        {metrics['median_ae']:.5f}")
    print(f"Maximum Absolute Error (MaxAE):       {metrics['max_ae']:.5f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    print(f"Symmetric MAPE (SMAPE):               {metrics['smape']:.2f}%")

X, y = create_sequences(scaled_data, SEQ_LENGTH, len(targets))

# Train/Test Split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 5. XGBoost Model definition and Training ---
print("Training compact XGBoost baseline (reduced tree depth/count)...")
xgb_estimator = XGBRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    subsample=SUBSAMPLE,
    colsample_bytree=COLSAMPLE_BYTREE,
    reg_lambda=3.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)
model = MultiOutputRegressor(xgb_estimator)

model.fit(X_train, y_train)

# --- 6. Evaluation Metrics ---
print("\nEvaluating model on Test Set...")
predictions = model.predict(X_test)
actuals = y_test

# Calculate metrics (on 0-1 scaled data)
metrics = calculate_metrics(actuals, predictions)
print_metrics("XGBoost", metrics)

# --- 7. Save Model and Scaler ---
print("\nSaving model to 'xgboost_model.joblib'...")
joblib.dump(model, 'xgboost_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and Scaler saved successfully!")
