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

# Ensure chronological order
df_filtered = df_filtered.sort_values(by='timestamp').reset_index(drop=True)

# Input features: weather + time + previous load/generation
features = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'solar_irradiance']
# Target variables to predict (Load & Generation)
targets = ['residential_load_MW', 'commercial_load_MW', 'industrial_load_MW', 'solar_MW', 'wind_MW']

print(f"Dataset shape after filtering: {df_filtered.shape}")

# Combine features and targets for sequential processing
data = df_filtered[features + targets].values

# --- 3. Scaling ---
print("Scaling Data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --- 4. Sequence Generation (Time Series windowing) ---
def create_sequences(data, seq_length, num_features, num_targets):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :]) # All features and targets inside window
        y.append(data[i + seq_length, -num_targets:]) # Only predict targets
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LENGTH, len(features), len(targets))

# Train/Test Split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train) # Shape: (samples, sequence, features+targets)
y_train_t = torch.FloatTensor(y_train) # Shape: (samples, targets)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# --- 5. PyTorch CNN-LSTM Hybrid Model definition ---
class MicrogridCNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_filters, lstm_hidden_size, num_lstm_layers, output_size):
        super(MicrogridCNNLSTM, self).__init__()
        
        # 1D Convolutional Layer for Feature Extraction
        # input_size = number of channels in Conv1D
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # LSTM Layer for Sequential Modeling
        # input to LSTM will be the cnn_filters sized vector at each time step
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)
        
        # Fully Connected Output Layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)
    
    def forward(self, x):
        # Permute input from [batch, sequence, features] to [batch, features, sequence] for Conv1d
        x = x.permute(0, 2, 1)
        
        # Pass through CNN
        c_out = self.relu(self.conv1d(x))
        
        # Permute back to [batch, sequence, filters] for LSTM
        c_out = c_out.permute(0, 2, 1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(c_out)
        
        # Take the output of the last time step from LSTM sequence
        out = self.fc(lstm_out[:, -1, :])
        return out

# Model instantiation
model = MicrogridCNNLSTM(input_size=len(features)+len(targets), 
                         cnn_filters=CNN_FILTERS,
                         lstm_hidden_size=LSTM_HIDDEN_SIZE, 
                         num_lstm_layers=NUM_LSTM_LAYERS, 
                         output_size=len(targets))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. Training Loop ---
print("Starting training (epochs)...")
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
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.6f}")

# --- 7. Evaluation (MSE, MAE) ---
print("\nEvaluating model on Test Set...")
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

# Calculate metrics (on 0-1 scaled data)
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mse)

print("\n--- Summary of CNN-LSTM Predictions ---")
print(f"Mean Squared Error (MSE):  {mse:.5f}")
print(f"Root Mean Sq Error (RMSE): {rmse:.5f}")
print(f"Mean Absolute Error (MAE): {mae:.5f}")

