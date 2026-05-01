import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib

# --- 1. Configurations ---
DATA_FILE = "kerala_microgrid_hourly_dataset.csv"
MODEL_FILE = "lstm_model.pth"
SCALER_FILE = "scaler.joblib"
# These should match train_lstm.py
SEQ_LENGTH = 24  
HIDDEN_SIZE = 64
NUM_LAYERS = 2

# --- 2. Define Model Class ---
class MicrogridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MicrogridLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 3. Prep Data & Load Models ---
print("Loading data and transforming...")
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)

features = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'solar_irradiance']
targets = ['residential_load_MW', 'commercial_load_MW', 'industrial_load_MW', 'solar_MW', 'wind_MW']
data = df[features + targets].values

# Load Scaler
try:
    scaler = joblib.load(SCALER_FILE)
    scaled_data = scaler.transform(data)
except FileNotFoundError:
    print(f"Error: Run the training scripts to generate {SCALER_FILE} and {MODEL_FILE}")
    exit()

def create_sequences(data, seq_length, num_targets):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :]) 
        y.append(data[i + seq_length, -num_targets:]) 
    return np.array(X), np.array(y)

X, _ = create_sequences(scaled_data, SEQ_LENGTH, len(targets))

# Isolate Test Set (last 20%)
train_size = int(len(X) * 0.8)
X_test = torch.FloatTensor(X[train_size:])

# Dates representing the test set (shift by train_size + SEQ_LENGTH)
test_dates = df['timestamp'].iloc[train_size + SEQ_LENGTH:].values

# Load Model
model = MicrogridLSTM(input_size=len(features)+len(targets), 
                      hidden_size=HIDDEN_SIZE, 
                      num_layers=NUM_LAYERS, 
                      output_size=len(targets))
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

# --- 4. Generate Predictions ---
print("Generating predictions for the test dataset...")
with torch.no_grad():
    preds_scaled = model(X_test).numpy()

# We only have the target predictions scaled, but the scaler was fit on [features + targets].
# To inverse_transform, we need a dummy array matching the original num of columns.
dummy_array_actuals = np.zeros((len(X_test), len(features) + len(targets)))
dummy_array_preds = np.zeros((len(X_test), len(features) + len(targets)))

# Fill the target columns (last 5) with our data
dummy_array_actuals[:, -len(targets):] = scaled_data[train_size + SEQ_LENGTH:, -len(targets):]
dummy_array_preds[:, -len(targets):] = preds_scaled

# Inverse transform
actuals_unscaled = scaler.inverse_transform(dummy_array_actuals)[:, -len(targets):]
preds_unscaled = scaler.inverse_transform(dummy_array_preds)[:, -len(targets):]

# --- 5. Plotting a 7-Day "Chaos" Window ---
print("Plotting the results...")
# Pick a random 7-day window (7 days * 24 hours = 168 hours) inside the test set
# Let's grab days somewhere in the middle of testing
START = 2000
END = START + (7 * 24)

# Create a Beautiful Matplotlib chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Graph 1: Industrial Load (Shows Random Spikes)
target_idx_ind = targets.index('industrial_load_MW')
ax1.plot(test_dates[START:END], actuals_unscaled[START:END, target_idx_ind], label='Actual Physics (Noisy + Spikes)', color='darkred', linewidth=1.5, alpha=0.8)
ax1.plot(test_dates[START:END], preds_unscaled[START:END, target_idx_ind], label='LSTM Predicted Baseline', color='blue', linewidth=2, linestyle='--')
ax1.set_title("Machine Learning Baseline vs Real-World Grid Anomalies (Industrial Spikes)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Power (MW)", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Graph 2: Solar Generation (Shows Cloud Drops)
target_idx_sol = targets.index('solar_MW')
ax2.plot(test_dates[START:END], actuals_unscaled[START:END, target_idx_sol], label='Actual Solar Output (Sudden Cloud Drops)', color='goldenrod', linewidth=1.5, alpha=0.8)
ax2.plot(test_dates[START:END], preds_unscaled[START:END, target_idx_sol], label='LSTM Predicted Solar Envelope', color='darkorange', linewidth=2, linestyle='--')
ax2.set_title("Solar Baseline vs Localized Cloud Fades", fontsize=14, fontweight='bold')
ax2.set_ylabel("Power (MW)", fontsize=12)
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Time (Hourly)", fontsize=12)

plt.tight_layout()
plt.savefig("chaos_predictions_plot.png", dpi=300)
print("Finished! The visualization has been saved as 'chaos_predictions_plot.png'.")

