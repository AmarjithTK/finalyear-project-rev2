# Action Plan: CNN-LSTM-XGBoost Hybrid Model

## Overview
This plan outlines the steps to build a hybrid model combining Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), and XGBoost for time-series forecasting or classification. The CNN extracts local spatial/sequential features, the LSTM captures long-term temporal dependencies, and XGBoost serves as the final robust predictor using the deep learning features.

## Phase 1: Data Preprocessing (`src/preprocessing.py`)
1. **Load Data:** Read `data/microgrid.csv`.
2. **Exploratory Data Analysis (EDA):** Identify trends, seasonality, and handle missing values or outliers.
3. **Scaling:** Normalize or standardize features (e.g., using `MinMaxScaler` or `StandardScaler`) to aid deep learning convergence.
4. **Sequence Generation:** Create sliding windows (look-back periods) to transform the dataset into a 3D structure `[samples, time_steps, features]` required by CNN/LSTM layers.
5. **Data Split:** Partition data into Train, Validation, and Test sets (ensure chronological splitting to avoid data leakage).

## Phase 2: CNN-LSTM Feature Extractor (`src/models.py`)
1. **Architecture Definition:**
   - **Input Layer:** Shape `(time_steps, features)`.
   - **CNN Component:** 1D Convolutional layers (`Conv1D`) followed by MaxPooling (`MaxPooling1D`) to extract robust local features from the time series.
   - **LSTM Component:** One or more LSTM layers to capture temporal dynamics from the CNN-extracted features.
   - **Dense Layer:** A dense mechanism primarily for pre-training the neural network.
2. **Feature Extraction Output:** Configure the model so that after pre-training, we can extract the activations of the final hidden layer (the dense embedding vector) to feed into the XGBoost model.

## Phase 3: Training the Hybrid System (`src/train.py`)
1. **Pre-train the Neural Network:**
   - Compile the CNN-LSTM with a temporary output dense layer matching your target (e.g., linear for regression).
   - Train on the training set with early stopping using the validation set.
2. **Extract Deep Features:** 
   - Truncate the output layer of the trained CNN-LSTM model.
   - Pass the Train, Validation, and Test sets through the truncated network to generate high-level feature embeddings.
3. **Train XGBoost:**
   - Initialize an XGBoost model (`XGBRegressor` or `XGBClassifier`).
   - Fit the XGBoost target on the new Train embeddings.
   - Tune hyperparameters using Validation embeddings.

## Phase 4: Evaluation & Experimentation (`notebooks/experiment.ipynb`)
1. **Evaluate:** Generate predictions on the Test set embeddings via XGBoost.
2. **Metrics:** Calculate appropriate metrics (MAE, RMSE, MAPE for regression; F1-score, ROC-AUC for classification).
3. **Baseline Comparison:** Compare the hybrid approach against standalone LSTM, standalone XGBoost, and naive baselines.
4. **Hyperparameter Tuning:** Use techniques like GridSearch or Optuna to optimize CNN kernel sizes, LSTM units, and XGBoost tree parameters.
