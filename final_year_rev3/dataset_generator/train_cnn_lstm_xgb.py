import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- 1. Configurations ---
DATA_FILE = "kerala_microgrid_hourly_dataset.csv"
SEQ_LENGTH = 24  # Use 24 hours of history
BATCH_SIZE = 32
CNN_FILTERS = 64
LSTM_HIDDEN_SIZE = 64
NUM_LSTM_LAYERS = 1
EPOCHS = 30
LEARNING_RATE = 0.001

# --- 2. Load Data ---
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Use the full dataset timeline
df_filtered = df.copy()

df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

# Input features: weather + time + previous load/generation
features = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'solar_irradiance']
# Target variables to predict
targets = [
    'residential_load_MW',
    'commercial_load_MW',
    'industrial_load_MW',
    'critical_load_MW',
    'solar_MW',
    'wind_MW',
]

print(f"Dataset shape after filtering: {df_filtered.shape}")
data = df_filtered[features + targets].values

# --- 3. Scaling ---
print("Scaling Data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --- 4. Sequence Generation ---
def create_sequences(data, seq_length, num_features, num_targets):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
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

X, y = create_sequences(scaled_data, SEQ_LENGTH, len(features), len(targets))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

# --- 5. Stage 1: CNN-LSTM Feature Extractor ---
class MicrogridCNNLSTM_Extractor(nn.Module):
    def __init__(self, input_size, cnn_filters, lstm_hidden_size, num_lstm_layers, output_size):
        super(MicrogridCNNLSTM_Extractor, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, output_size)
    
    def forward(self, x, extract_features=False):
        x = x.permute(0, 2, 1)
        c_out = self.relu(self.conv1d(x))
        c_out = c_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(c_out)
        features = lstm_out[:, -1, :] # The latent features (hidden state)
        
        if extract_features:
            return features
            
        out = self.fc(features)
        return out

model = MicrogridCNNLSTM_Extractor(input_size=len(features)+len(targets), 
                                   cnn_filters=CNN_FILTERS,
                                   lstm_hidden_size=LSTM_HIDDEN_SIZE, 
                                   num_lstm_layers=NUM_LSTM_LAYERS, 
                                   output_size=len(targets))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Stage 1: Training CNN-LSTM...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.6f}")

# --- 6. Extract Latent Features ---
print("Extracting Deep Learning Features for XGBoost...")
model.eval()
with torch.no_grad():
    X_train_features = model(X_train_t, extract_features=True).numpy()
    X_test_features = model(X_test_t, extract_features=True).numpy()

# --- 7. Stage 2: XGBoost Regression ---
print("Stage 2: Training XGBoost MultiOutput Regressor (Tuned)...")
# We use MultiOutputRegressor because base XGBRegressor predicts only 1 target
xgb_base = XGBRegressor(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=4, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    reg_alpha=0.1, 
    reg_lambda=1.0, 
    random_state=42
)
xgb_model = MultiOutputRegressor(xgb_base)

xgb_model.fit(X_train_features, y_train)

# --- 8. Evaluation Metrics ---
print("Evaluating Hybrid Model on Test Set...")
predictions = xgb_model.predict(X_test_features)
actuals = y_test

metrics = calculate_metrics(actuals, predictions)
print_metrics("CNN-LSTM-XGBoost", metrics)

# --- 9. Save Models ---
print("\nSaving Base PyTorch model and XGBoost MultiOutputRegressor...")
import joblib
import torch
torch.save(model.state_dict(), 'cnn_lstm_base_model.pth')
joblib.dump(xgb_model, 'xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Models and Scaler saved successfully!")
