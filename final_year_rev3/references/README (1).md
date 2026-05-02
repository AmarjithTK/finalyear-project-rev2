# Kerala IEEE 13-Bus — DSS + LSTM Pipeline

## Project Structure
```
KeralaIEEE13.dss          ← OpenDSS network model (all buses, DER)
loadflow_generator.py     ← Run DSS load flow for all 24h × all buses
lstm_forecast.py          ← Train LSTM + forecast next 24 hours
```

---

## Step 1 — Generate Load Flow Data

```bash
python loadflow_generator.py
```

**Outputs:**
| File | Rows | Description |
|---|---|---|
| `loadflow_baseline.csv` | 24 | Single clean day, no noise |
| `loadflow_training.csv` | 720 | 30 days with noise (LSTM training data) |
| `lstm_X.npy` | (672, 24, 27) | Sliding-window sequences |
| `lstm_y.npy` | (672,) | Target voltage at Bus 634 |

**Columns generated (per hour):**

| Category | Columns |
|---|---|
| Bus voltages (p.u.) | V_650, V_632, V_633, V_634, V_671, V_675, V_680, V_684, V_692 |
| LV bus voltages | V_634LV, V_671LV, V_684LV, V_692LV |
| Sector loads (kW) | Res_kW, Com_kW, Ind_kW, Crit_kW, Total_Load_kW |
| DER output | Solar_kW, Wind_kW, Total_DER_kW, Solar_Irr_pu |
| Grid | Grid_kW, Net_Import_kW |
| Losses | Loss_kW, Loss_kVAR |
| Line flows | L1_P_kW, L1_Q_kVAR, L1_Loading_pct, L3_P_kW, L3_Q_kVAR |

---

## Step 2 — Train LSTM

```bash
python lstm_forecast.py
```

**Outputs:**
- `kerala_lstm_best.keras` — best model weights
- `lstm_predictions.csv` — predicted vs actual voltages for test set
- `scaler_X.pkl`, `scaler_y.pkl` — MinMaxScaler objects
- `forecast_next24h.csv` — rolling 24-hour voltage forecast

**Architecture:**
```
Input (24 timesteps × 27 features)
  → LSTM(128)  + Dropout(0.2)
  → LSTM(64)   + Dropout(0.2)
  → Dense(64, relu)
  → Dense(13)           ← predicts all bus voltages simultaneously
```

---

## Step 3 — Forecast Only (after training)

```bash
python lstm_forecast.py --predict
```

---

## Load Profiles (Kerala-specific)

| Sector | Bus | Peak | Pattern |
|---|---|---|---|
| Residential | 634 | 1500 kW | Evening peak 19-22h |
| Commercial | 684 | 800 kW | Working hours 09-18h |
| Industrial | 671 | 1100 kW | Daytime stable |
| Critical | 692 | 400 kW | Flat 24/7 (hospital/telecom) |

**DER:** Solar PV 1.2 MW (Bus 633) + Wind 0.8 MW (Bus 675)

---

## Requirements

```
opendssdirect.py
pandas
numpy
scikit-learn
tensorflow >= 2.12
joblib
```

Install:
```bash
pip install opendssdirect[extras] pandas numpy scikit-learn tensorflow joblib
```
