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

# --- 1. Configurations ---
DATA_FILE = "unified_microgrid_24h_results.csv"
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
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Ensure chronological order
df = df.sort_values(by='Timestamp').reset_index(drop=True)

# Input features (Weather and Time)
features = ['Hour', 'Temperature_C', 'Humidity_pct', 'Wind_Speed_ms', 'Cloud_Cover_pct', 'Solar_Irradiance_Wm2']

# Target variables to predict (Full Grid state, Load, Gen, Voltages, Loadings, Losses)
targets = [
    'Solar_MW', 'Wind_MW', 'Total_Load_MW',
    'Grid_Import_MW', 'Grid_Import_MVAR',
    'V_Sub_650', 'V_Split_632', 'V_Solar_633', 'V_Wind_675', 
    'V_Res_634', 'V_Ind_671', 'V_Com_684', 'V_Crit_692', 
    'V_Min_pu', 'V_Max_pu',
    'L_Main_650_632_pct', 'L_IndWind_632_671_pct', 'L_Solar_632_633_pct', 'Max_Line_Loading_pct',
    'Total_Loss_MW', 'Total_Loss_MVAR'
]

print(f"Dataset shape: {df.shape}")

# Combine features and targets for sequential processing
data = df[features + targets].values

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

# --- 7. Evaluation Metrics ---
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
metrics = calculate_metrics(actuals, predictions)
print_metrics("CNN-LSTM", metrics)

# --- 8. Save Model ---
print("\nSaving model to 'cnn_lstm_model.pth'...")
torch.save(model.state_dict(), 'cnn_lstm_model.pth')
import joblib
joblib.dump(scaler, 'scaler.joblib')
print("Model and Scaler saved successfully!")
