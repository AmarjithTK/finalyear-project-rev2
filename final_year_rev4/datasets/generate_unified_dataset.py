"""
========================================================================================
Kerala Microgrid Project: Weather & Load/DER Dataset Generator
========================================================================================

Generates a realistic hourly dataset with weather variables, solar PV generation,
wind generation, and dynamic load profiles (Residential, Commercial, Industrial, Critical).
No OpenDSS power flow — pure weather-to-load/DER mapping.
========================================================================================
"""

import pandas as pd
import numpy as np
import math
import os

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════

# Input/Output paths
WEATHER_INPUT = "solar_power_dataset.csv"
OUTPUT_CSV = "unified_microgrid_24h_results.csv"

# DER Capacities (MW)
SOLAR_CAPACITY = 1.2
WIND_CAPACITY = 0.8

# Load Peak Capacities (MW)
PEAK_RES = 1.6   # 40% of 4 MW
PEAK_IND = 1.2   # 30% of 4 MW
PEAK_COM = 0.8   # 20% of 4 MW
PEAK_CRIT = 0.4  # 10% of 4 MW

# Hours array
HOURS = list(range(24))

# ──────────────────────────────────────────────────────────────
# 2. LOAD PROFILES (Hourly normalized 0-1)
# ──────────────────────────────────────────────────────────────

# Residential: Medium morning, low day, high evening
RES_PROFILE = [0.4, 0.3, 0.3, 0.3, 0.4, 0.5, 0.7, 0.8, 0.7, 0.6, 0.5, 0.5,
               0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

# Commercial: Near zero night, rising morning, high day peak, medium evening
COM_PROFILE = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]

# Industrial: Low night, high morning, steady high day, slight drop evening
IND_PROFILE = [0.2, 0.2, 0.2, 0.2, 0.3, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.2]

# Critical: Constant 24/7
CRIT_PROFILE = [1.0] * 24


# ──────────────────────────────────────────────────────────────
# 3. ADVANCED WEATHER GENERATION
# ──────────────────────────────────────────────────────────────

def generate_hourly_weather_and_der(daily_row, hour):
    """
    Expands a single daily weather row into an hourly weather snapshot
    with realistic physics for solar/wind generation and load variation.
    """
    # 1. Base Weather Variables
    temp = daily_row['temperature'] + np.random.uniform(-2, 2)
    humidity = daily_row['humidity'] + np.random.uniform(-5, 5)
    humidity = min(100, max(0, humidity))
    
    # Kerala coastal winds: sea breeze boost during day
    sea_breeze_boost = 1.6 + 1.4 * max(0, math.sin(math.pi * (hour - 6) / 14))
    
    # Extrapolate ground-level weather data wind spread to hub-height (multiplier ~3.5x for robust generation)
    hub_height_factor = 3.5
    wind_speed = (daily_row['wind_speed'] * hub_height_factor) + sea_breeze_boost + np.random.uniform(-0.5, 1.5)
    wind_speed = max(0, wind_speed)
    
    cloud_cover = daily_row['cloud_cover'] + np.random.uniform(-5, 5)
    cloud_cover = min(100, max(0, cloud_cover))
    
    irradiance = daily_row['solar_irradiance'] + np.random.uniform(-50, 50)
    irradiance = max(0, irradiance)
    
    # 2. SOLAR GENERATION
    if 6 <= hour <= 18:
        solar_shape = max(0, math.sin(math.pi * (hour - 6) / 12))
    else:
        solar_shape = 0.0
        irradiance = 0.0
        
    cloud_factor = max(0, 1 - 0.006 * cloud_cover)
    solar_MW = SOLAR_CAPACITY * solar_shape * (irradiance / 1000) * cloud_factor
    
    # 3. WIND GENERATION (IEC Class III low-wind turbine profile)
    if wind_speed < 3:
        wind_MW = 0.0
    elif wind_speed < 9:  # Lowered rated speed to 9 m/s to ensure it reaches 0.8MW frequently
        wind_MW = WIND_CAPACITY * ((wind_speed - 3) / (9 - 3)) ** 3
    elif wind_speed <= 25:
        wind_MW = WIND_CAPACITY
    else:
        wind_MW = 0.0
    
    # 4. ANOMALIES
    if 7 <= hour <= 17 and np.random.random() < 0.02:
        solar_MW *= np.random.uniform(0.20, 0.60)
    if np.random.random() < 0.01:
        wind_MW *= np.random.uniform(0.30, 0.70)
    
    # 5. WEATHER IMPACT ON LOAD
    weather_factor = 1 + 0.01 * max(temp - 28, 0) + 0.002 * max(humidity - 70, 0)
    
    # 6. BASE LOAD
    res_MW = PEAK_RES * RES_PROFILE[hour] * weather_factor * np.random.normal(1.0, 0.04)
    com_MW = PEAK_COM * COM_PROFILE[hour] * weather_factor * np.random.normal(1.0, 0.02)
    ind_MW = PEAK_IND * IND_PROFILE[hour] * weather_factor * np.random.normal(1.0, 0.02)
    crit_MW = PEAK_CRIT * CRIT_PROFILE[hour] * np.random.normal(1.0, 0.01)
    
    # 7. LOAD SPIKES
    if np.random.random() < 0.01:
        ind_MW *= np.random.uniform(1.20, 1.40)
    if 17 <= hour <= 22 and np.random.random() < 0.015:
        res_MW *= np.random.uniform(1.25, 1.50)
    if 10 <= hour <= 16 and np.random.random() < 0.005:
        com_MW *= np.random.uniform(1.15, 1.30)
    
    return {
        'temperature': round(temp, 2),
        'humidity': round(humidity, 2),
        'wind_speed': round(wind_speed, 2),
        'cloud_cover': round(cloud_cover, 2),
        'solar_irradiance': round(irradiance, 2),
        'solar_MW': round(solar_MW, 6),
        'wind_MW': round(wind_MW, 6),
        'residential_load_MW': round(res_MW, 6),
        'commercial_load_MW': round(com_MW, 6),
        'industrial_load_MW': round(ind_MW, 6),
        'critical_load_MW': round(crit_MW, 6),
    }


# ──────────────────────────────────────────────────────────────
# 4. UNIFIED SIMULATION (Weather + Loads + DER only)
# ──────────────────────────────────────────────────────────────

def run_unified_simulation(num_days=30, seed=42):
    if seed is not None:
        np.random.seed(seed)
    
    if not os.path.exists(WEATHER_INPUT):
        raise FileNotFoundError(f"Weather input file not found: {WEATHER_INPUT}")
    
    df_daily = pd.read_csv(WEATHER_INPUT)
    n_days_available = len(df_daily)
    
    records = []
    
    print(f"\nRunning {num_days} days × 24 hours simulation...")
    
    for day in range(num_days):
        daily_idx = day % n_days_available
        daily_row = df_daily.iloc[daily_idx].to_dict()
        
        for hour in HOURS:
            weather = generate_hourly_weather_and_der(daily_row, hour)
            
            current_date = daily_row.get('date', f"Day_{day+1}")
            
            rec = {
                "Timestamp": f"{current_date} {hour:02d}:00:00",
                "Hour": hour,
                # Weather
                "Temperature_C": weather['temperature'],
                "Humidity_pct": weather['humidity'],
                "Wind_Speed_ms": weather['wind_speed'],
                "Cloud_Cover_pct": weather['cloud_cover'],
                "Solar_Irradiance_Wm2": weather['solar_irradiance'],
                # DER Generation
                "Solar_MW": weather['solar_MW'],
                "Wind_MW": weather['wind_MW'],
                # Loads
                "Residential_Load_MW": weather['residential_load_MW'],
                "Commercial_Load_MW": weather['commercial_load_MW'],
                "Industrial_Load_MW": weather['industrial_load_MW'],
                "Critical_Load_MW": weather['critical_load_MW'],
            }
            records.append(rec)
            
            if hour % 6 == 0:
                total_load_MW = round(
                    weather['residential_load_MW'] +
                    weather['commercial_load_MW'] +
                    weather['industrial_load_MW'] +
                    weather['critical_load_MW'], 4
                )
                total_der_MW = round(weather['solar_MW'] + weather['wind_MW'], 4)
                print(f"  Day {day+1:2d} H{hour:02d} | Load={total_load_MW:6.3f}MW  DER={total_der_MW:5.3f}MW")
                
    return pd.DataFrame(records)


if __name__ == "__main__":
    try:
        df_res = run_unified_simulation(num_days=730)
        df_res.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved weather + load/DER dataset to {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error: {e}")
