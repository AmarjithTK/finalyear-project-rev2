# Dataset Generation Plan

This plan outlines the methodology for creating a highly realistic, engineering-grade synthetic dataset for a 4 MW microgrid, leveraging daily weather data as the backbone.

## 1. System Specifications

*   **Total Peak Load:** 4.0 MW
*   **Solar Capacity:** 1.2 MW
*   **Wind Capacity:** 0.8 MW
*   **Total DER Capacity:** 2.0 MW (50% penetration)

## 2. Weather Data Baseline (from Kaggle Dataset)

We will use the daily weather data from the Kaggle dataset to drive the generation and load equations.
Since the dataset provides daily values, each row will be **expanded into 24 hourly rows** (timestamp expansion) adding hour-of-varying profiles and small randomness to ensure realism.

**Base weather variables needed:**
*   `temperature`
*   `humidity`
*   `wind_speed`
*   `cloud_cover`
*   `solar_irradiance`

## 3. Power Generation Modeling

### Solar Generation (1.2 MW)
Follows a daytime bell-curve peaking at noon, influenced by irradiance and cloud cover.
*   **Hourly Shape:** `solar_shape = max(0, sin(pi * (hour - 6) / 12))` (Zero before 6 AM and after 6 PM)
*   **Cloud Factor:** `cloud_factor = 1 - 0.006 * cloud_cover`
*   **Hourly Generation:** `solar_MW = solar_capacity * solar_shape * (irradiance/1000) * cloud_factor`

### Wind Generation (0.8 MW)
Does not follow a clean time curve. Uses daily wind speed with hourly random variation/noise.
*   **Hourly Wind Speed:** `v = daily_wind_speed + hourly_random_noise`
*   **Power Curve:**
    *   $< 3$ m/s: `0 MW` (Cut-in)
    *   $3$ to $12$ m/s: `wind_capacity * ((v - 3) / (12 - 3))^3` (Cubic increase)
    *   $12$ to $25$ m/s: `wind_capacity` (Rated power)
    *   $> 25$ m/s: `0 MW` (Cut-out)

## 4. Load Modeling (4 MW Peak)

The 4 MW peak load is split into four distinct sectors to simulate a realistic Indian/Kerala feeder. Each sector has its own hourly activity profile.

1.  **Residential (40% - 1.6 MW Peak)**
    *   Profile: Medium morning/night, low day, **strong evening peak**.
2.  **Industrial (30% - 1.2 MW Peak)**
    *   Profile: High morning, **steady high day**, slight drop evening, low/off night.
3.  **Commercial (20% - 0.8 MW Peak)**
    *   Profile: Rising morning, **high day peak**, medium evening, near zero night.
4.  **Critical (10% - 0.4 MW Peak)**
    *   Profile: **Constant 24h** flat line (e.g., hospitals, telecom).

**Weather Influence on Load:**
Hot and humid weather increases cooling loads. This factor is applied to the aggregate or residential/commercial loads.
*   `weather_factor = 1 + 0.01 * max(temperature - 28, 0) + 0.002 * max(humidity - 70, 0)`

## 5. Microgrid Topology & OpenDSS Map (IEEE 13-Bus)

The generated hourly data will be distributed across the standard IEEE 13-node test feeder. Unused buses act as essential electrical network junctions.

*   **Slack Bus:** `650` (Utility Grid connection)
*   **Loads:**
    *   Residential: `Bus 634`
    *   Industrial: `Bus 671`
    *   Commercial: `Bus 684`
    *   Critical: `Bus 692`
*   **DERs:**
    *   Solar: `Bus 675`
    *   Wind: `Bus 680`

## 6. Project Pipeline & LSTM Objective

Note: Simple aggregations like `total_load_MW`, `grid_import_MW` and `grid_export_MW` can easily be calculated later during data preprocessing, so they don't strictly need to be hardcoded into the base dataset. 

### The Correct ML Pipeline (Prediction vs. Validation)
The generated dataset is designed specifically to train a time-series forecasting model (like an LSTM). The LSTM's job is **only to predict future load and generation values**, not to detect anomalies.

If we include OpenDSS anomaly labels in the training dataset, it confuses the objective of the LSTM since physics-based anomaly detection operates on an entirely different premise than time-series prediction. Here is the actual sequence:

1.  **LSTM Forecasting:** Train the LSTM on the historical weather and load/generation data.
2.  **Future Prediction:** Use the LSTM to forecast future `solar_MW`, `wind_MW`, and the `load_MW` for each sector.
3.  **OpenDSS Validation:** Feed the **predicted** future values into OpenDSS to solve for power flow. 
4.  **Anomaly Detection:** OpenDSS will flag physical network violations in the forecasted future state:
    *   **Undervoltage:** If any predicted bus voltage < 0.95 pu
    *   **Overload:** If any predicted line loading > 100% capacity
    *   **Normal:** No violations

## 7. Final Dataset Table Structure (For LSTM Training)

The finalized hourly dataset should only contain the fundamental values required for time-series forecasting, leaving the anomaly detection to the post-prediction OpenDSS phase. It should look like this:

| timestamp | temperature (°C) | humidity (%) | wind_speed (m/s) | cloud_cover (%) | solar_irradiance (W/m²) | residential_load_MW | commercial_load_MW | industrial_load_MW | critical_load_MW | solar_MW | wind_MW |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2019-01-01 00:00 | 25 | 80 | 4.2 | 10 | 0 | 0.8 | 0 | 0.3 | 0.4 | 0 | 0.1 |
| 2019-01-01 12:00 | 32 | 60 | 6.5 | 20 | 850 | 0.6 | 0.8 | 1.2 | 0.4 | 0.9 | 0.3 |
| 2019-01-01 19:00 | 28 | 75 | 3.1 | 50 | 0 | 1.5 | 0.6 | 1.0 | 0.4 | 0 | 0 |

This table provides the fundamental **weather backbone** and the derived **hourly profiles** for loads and DER generation. OpenDSS will strictly be used *after* the ML prediction phase to test the stability of the forecasted microgrid behavior.
