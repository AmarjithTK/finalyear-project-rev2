# CNN-LSTM-XGBoost Hybrid Architecture Plan

## 1. The Core Concept
The standard neural network is great at recognizing complex patterns, while XGBoost is the state-of-the-art for making final robust predictions. This hybrid model leverages the strengths of all three:
- **CNN (1D-Convolution):** Scans the time-series sliding windows to detect sudden local changes (e.g., quick drops in solar power, sudden load spikes).
- **LSTM:** Takes the sequence evaluated by the CNN and remembers the long-term temporal dependencies (e.g., weather patterns over the week, battery cyclic behavior).
- **XGBoost:** Replaces the standard fully-connected output layer of the neural network to act as the final regressor, reducing overfitting and handling nonlinear boundaries more effectively.

## 2. The Data Flow
Instead of training all models simultaneously, we use a **Pre-train -> Extract -> Regress** pipeline.

### Step A: Pre-train the Deep Learning Network
We will build a standard CNN-LSTM model using PyTorch (`nn.Module`):
```
Input (batch_size, channels=features, sequence_length=time_steps) -> nn.Conv1d -> nn.MaxPool1d -> nn.LSTM -> nn.Linear (Embedding Layer) -> Output nn.Linear(1)
```
We train this model normally using the actual target (e.g., Microgrid Load) with PyTorch optimizers and loss functions so that the CNN and LSTM layers learn how to extract meaningful representations of the data. Note: `Conv1d` expects `(N, C, L)` while LSTM with `batch_first=True` expects `(N, L, C)`, so we must `permute` after the CNN/Pool block.

### Step B: Feature Extraction (The "Embedding")
Once the model is trained, we chop off the final `Output Dense(1)` layer. 
We then pass our train and test data through this truncated network. The output is no longer a prediction, but a numeric **feature embedding** representing the deep sequential patterns of the time-series.

### Step C: XGBoost Training
We feed these extracted **embeddings** (along with any other static features like Day of Week) natively into an `XGBRegressor`. The XGBoost model fits on these deep features to make the final prediction.

### Step D: Critical Implementation Notes (Fixes)
- **Time-based split first:** split train/val/test on chronological order before scaling or sequence generation to avoid leakage.
- **Scaler on train only:** fit scalers using only the training split; save scaler parameters and feature order for consistent inference.
- **Sequence/label alignment:** for a horizon $h$, ensure each input window maps to the target at $t+h$ and keep aligned timestamps.
- **Tensor layout:** after `Conv1d`, permute to `(N, L, C)` for LSTM with `batch_first=True`.
- **Feature extraction safety:** use `model.eval()` and `torch.no_grad()` when extracting embeddings.
- **Multi-target handling:** if predicting multiple targets, set output dim accordingly and train one XGBoost per target or a multi-output wrapper.
- **Metrics on original scale:** inverse-transform predictions and targets before MAE/RMSE/R2 if you scaled data.

## 3. How It Maps to Your Codebase

### `src/utils.py`
- **`create_sequences(df, look_back, horizon)`:** Creates sequences after a time-based split and maps them to PyTorch `DataLoader` objects.
- **`fit_scaler(train_df)` and `apply_scaler(df, scaler)`:** Fit on train only, apply to val/test, and persist scaler state.
- **`save_hybrid_model(cnn_lstm, xgboost_model, scaler, config)`:** Saves the PyTorch `state_dict` (`.pth`), XGBoost model, scaler, and sequence config for consistent inference.

### `src/train.py`
1. Load data and initialize PyTorch `DataLoader` instances.
2. Define the CNN-LSTM class by inheriting from `torch.nn.Module`.
3. Train the CNN-LSTM on the training set using a standard PyTorch training loop (`loss.backward()`, `optimizer.step()`).
4. Build the feature extractor: Modify the `forward` method or add an `extract_features` method that skips the final `nn.Linear` output layer.
5. Extract `train_features = cnn_lstm.extract_features(X_train_tensor).detach().cpu().numpy()` using `model.eval()` and `torch.no_grad()`.
6. Initialize and fit `xgb = XGBRegressor()` using train embeddings only (use val embeddings for early stopping if needed).
7. Fit `xgb` on `train_features` and `y_train`.
8. Save both models and scaler/config.

### `src/predict.py`
1. Load both the CNN-LSTM extractor and the XGBoost model.
2. Structure new Test data into 3D sequences using the saved scaler/config.
3. Pass through extractor (eval/no_grad) -> get test features.
4. Pass test features through XGBoost -> get final predictions.
5. Inverse-transform predictions (if scaled) before metrics.
6. Save `predictions.csv`.

---

## 4. Step-by-Step Task Breakdown

*Note on Modularization: We will start by building inside the 4 core files (`utils.py`, `train.py`, `predict.py`, `scenarios.py`). If a specific domain (like the PyTorch neural network logic) becomes too massive (e.g., spanning hundreds of lines), we will extract it into a dedicated, importable file (e.g., `src/model_defs.py`) using Object-Oriented paradigms. We will explicitly avoid creating dozens of tiny files; modules will only be created when they significantly clean up the execution scripts.*

**Task 1: Build the Data Pipeline (`src/utils.py`)**
- Read `microgrid.csv`, split chronologically into train/val/test, and handle missing values.
- Fit scaler on train only and persist it for inference.
- Write the function to generate `(batch_size, channels, sequence_length)` sliding windows with a defined horizon.
- Create a custom PyTorch `Dataset` class and initialize the `DataLoader`.

**Task 2: Define the PyTorch Network Architecture**
- Define the `CNNLSTMHybrid` class inheriting from `nn.Module`.
- Implement `forward(self, x)` for the pre-training regression.
- Implement `extract_features(self, x)` to grab the embedding representations right before the final `nn.Linear` output layer.
- Ensure correct tensor permutation between CNN and LSTM (`(N, C, L)` -> `(N, L, C)`).
- *(If this class gets too large, this is where we will branch it into `src/model_defs.py`)*

**Task 3: Pre-train the Deep Learning Feature Extractor (`src/train.py`)**
- Initialize the `CNNLSTMHybrid` model, the optimizer (e.g., Adam), and the loss function (MSE).
- Write the PyTorch training loop (epochs, forward pass, calculate loss, `loss.backward()`, `optimizer.step()`).
- Include basic early-stopping or validation logic without using the test split.

**Task 4: Extract Embeddings & Train XGBoost (`src/train.py`)**
- Pass the training data through the trained model using `extract_features(...)`.
- Convert the output tensors back into NumPy arrays (`.detach().cpu().numpy()`).
- Initialize the `XGBRegressor` and fit it exclusively on these extracted embeddings.
- Use validation embeddings for early stopping and avoid any test data usage.
- Save the PyTorch `.pth` state and the XGBoost model to `outputs/` or `models/`.

**Task 5: Implement the Prediction Pipeline (`src/predict.py`)**
- Load the saved `state_dict` and XGBoost model.
- Load and prepare the Test dataset using the `utils.py` pipeline.
- Extract embeddings from the test sequences using the PyTorch model.
- Pass the test embeddings to XGBoost to generate final predictions.
- Calculate and save evaluation metrics (MAE, RMSE, R²) on the original (inverse-transformed) scale to `outputs/metrics.json`.
- Save actual vs. predicted load to `outputs/predictions.csv`.

**Task 6: Setup Scenario Simulation (`src/scenarios.py`)**
- Read `outputs/predictions.csv`.
- Apply domain logic to evaluate stress/risk scenarios (e.g., sudden loss of predicted solar generation).
- Output findings cleanly to `outputs/scenarios.json`.

**Task 7: Build the Jupyter Dashboard (`main.ipynb`)**
- Ensure the notebook cleanly executes the `.py` scripts (using `!python src/train.py`, etc.).
- Load the JSONs and CSVs from `outputs/`.
- Plot final visualizations and print the scenario tables cleanly for your final report.

---

## 5. Flexibility Plan (Variable Columns / Different Datasets)

**Goal:** Make the pipeline resilient to different CSV schemas (solar-only, wind-only, missing battery, bus-wise loads, etc.) while keeping a minimal file structure.

**A. Configuration-Driven Features**
- Maintain a single config dictionary in `utils.py` (or a compact `config.py` if it grows) that defines:
	- `target_columns` (what to predict)
	- `feature_columns` (what to use as inputs)
	- `optional_columns` (use if present, ignore if missing)
	- `time_column` (timestamp column name)
	- `horizon`, `look_back`, `frequency`

**B. Automatic Schema Validation**
- Add a `validate_schema(df, config)` function that:
	- Confirms all required columns exist.
	- Logs which optional columns are missing.
	- Warns if target columns accidentally appear in features (leakage guard).

**C. Flexible Dataset Loading**
- The data loader should accept a `config` object and:
	- Dynamically assemble `feature_columns` based on the CSV.
	- Drop columns that are not in the config.
	- Support datasets without battery or solar by using only the available features.

**D. Model Shape Adaptation**
- CNN input channels should be derived from `len(feature_columns)`.
- Output dimension should be derived from `len(target_columns)`.
- If predicting multiple targets, train one XGBoost model per target (clean and stable) unless multi-output is needed later.

**E. Minimal Modular Files**
- Keep everything in `utils.py` and `train.py` unless it becomes too large.
- If the configuration grows, add **one** file `src/config.py` and import it everywhere.
- If the model class grows, add **one** file `src/model_defs.py` and keep all network classes there.

**F. Example Config Shapes**
- **Solar-only dataset:** targets = `solar_pv_output`, features = `solar_irradiance`, `temperature`, `humidity`, `hour_of_day`, `day_of_week`.
- **Wind-only dataset:** targets = `wind_power_output`, features = `wind_speed`, `atmospheric_pressure`, `temperature`, `hour_of_day`, `day_of_week`.
- **No battery columns:** ignore battery features entirely; no change in pipeline required.
- **Bus-wise load dataset:** define `target_columns` as `load_bus_1`, `load_bus_2`, ... and train one XGBoost per bus.
