import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 1. Configurations ---
DATA_FILE = "kerala_microgrid_hourly_dataset.csv"
SEQ_LENGTH = 24  # Use 24 hours of history to predict the next hour
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 20
LEARNING_RATE = 0.001

# --- 2. Load and Filter Data (2023-01-01 to 2024-01-01) ---
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter one year of data: Jan 1, 2023 to Jan 1, 2024
mask = (df['timestamp'] >= '2023-01-01') & (df['timestamp'] <= '2024-01-01')
df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    print("Warning: No data found for 2023-2024. Using the entire dataset instead.")
    df_filtered = df.copy()

# Features for predicting (X) and Targets (Y)
# Input features: weather + time + previous load/generation
features = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'solar_irradiance']
# Target variables to predict
targets = ['residential_load_MW', 'commercial_load_MW', 'industrial_load_MW', 'solar_MW', 'wind_MW']

# Combine features and targets for sequential processing
data = df_filtered[features + targets].values

# --- 3. Scaling ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --- 4. Sequence Generation ---
def create_sequences(data, seq_length):
    X = []
    y = []
    # targets are the last columns
    num_features = data.shape[1]
    num_targets = len(targets)
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, -num_targets:])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Train/Test Split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# --- 5. PyTorch LSTM Model ---
class MicrogridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MicrogridLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = MicrogridLSTM(input_size=len(features)+len(targets), 
                      hidden_size=HIDDEN_SIZE, 
                      num_layers=NUM_LAYERS, 
                      output_size=len(targets))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. Training Loop ---
print("Starting training...")
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
        
    train_loss /= len(train_loader)
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.6f}")

# --- 7. Evaluation (MSE, MAE) ---
print("Evaluating model...")
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = model(batch_X)
        predictions.append(preds.numpy())
        actuals.append(batch_y.numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)

# Calculate metrics (on scaled data)
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)

print("\n--- Model Performance (Scaled) ---")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

print("\nNotes:")
print("- We used PyTorch for the LSTM.")
print("- Handled Train/Test split.")
print("- Computed MSE and MAE on the output.")
