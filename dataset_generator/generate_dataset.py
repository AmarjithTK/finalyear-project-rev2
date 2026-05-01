import pandas as pd
import numpy as np
import math
import os

# --- Configurations ---
INPUT_DATASET = "./solar_power_dataset.csv"
OUTPUT_DATASET = "kerala_microgrid_hourly_dataset.csv"

# --- Capacity Assumptions ---
PEAK_RES = 1.6  # 40% of 4 MW
PEAK_IND = 1.2  # 30% of 4 MW
PEAK_COM = 0.8  # 20% of 4 MW
PEAK_CRIT = 0.4 # 10% of 4 MW

SOLAR_CAPACITY = 1.2
WIND_CAPACITY = 0.8

# --- Hourly Load Profiles (Normalized 0 to 1) ---
# Residential: Medium morning, low day, high evening
res_profile = [0.4, 0.3, 0.3, 0.3, 0.4, 0.5, 0.7, 0.8, 0.7, 0.6, 0.5, 0.5,
               0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

# Commercial: Near zero night, rising morning, high day peak, medium evening
com_profile = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]

# Industrial: Low night, high morning, steady high day, slight drop evening
ind_profile = [0.2, 0.2, 0.2, 0.2, 0.3, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0,
               1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.2]

# Critical: Constant 24/7 (hospitals, telecom)
crit_profile = [1.0] * 24

def generate_hourly_data(row, hour):
    """
    Takes a single daily weather row and expands it dynamically into an hour.
    Applies constraints, formulas, profiles, and noise.
    """
    # 1. Base Weather Variables (with slight hourly noise for realism)
    temp = row['temperature'] + np.random.uniform(-2, 2)
    humidity = row['humidity'] + np.random.uniform(-5, 5)
    humidity = min(100, max(0, humidity))
    
    wind_speed = row['wind_speed'] + np.random.uniform(-1, 1)
    wind_speed = max(0, wind_speed)
    
    cloud_cover = row['cloud_cover'] + np.random.uniform(-5, 5)
    cloud_cover = min(100, max(0, cloud_cover))
    
    irradiance = row['solar_irradiance'] + np.random.uniform(-50, 50)
    irradiance = max(0, irradiance)

    # 2. Solar Generation Calculation
    if 6 <= hour <= 18:
        # Bell curve starting at 6, peaking at 12, ending at 18
        solar_shape = max(0, math.sin(math.pi * (hour - 6) / 12))
    else:
        solar_shape = 0.0
        irradiance = 0.0 # Force irradiance to 0 at night
        
    cloud_factor = max(0, 1 - 0.006 * cloud_cover)
    solar_MW = SOLAR_CAPACITY * solar_shape * (irradiance / 1000) * cloud_factor

    # 3. Wind Generation Calculation
    if wind_speed < 3:
        wind_MW = 0.0
    elif wind_speed < 12:
        wind_MW = WIND_CAPACITY * ((wind_speed - 3) / (12 - 3)) ** 3
    elif wind_speed <= 25:
        wind_MW = WIND_CAPACITY
    else:
        wind_MW = 0.0

    # 4. Weather Impact on Load
    # Hot and humid weather increases cooling/AC loads
    weather_factor = 1 + 0.01 * max(temp - 28, 0) + 0.002 * max(humidity - 70, 0)

    # 5. Load Generation
    res_MW = PEAK_RES * res_profile[hour] * weather_factor
    com_MW = PEAK_COM * com_profile[hour] * weather_factor
    ind_MW = PEAK_IND * ind_profile[hour] * weather_factor
    # Critical loads typically don't scale strongly with weather, keeping it flat
    crit_MW = PEAK_CRIT * crit_profile[hour]

    # Format timestamp
    timestamp = f"{row['date']} {hour:02d}:00"

    return {
        'timestamp': timestamp,
        'temperature': round(temp, 2),
        'humidity': round(humidity, 2),
        'wind_speed': round(wind_speed, 2),
        'cloud_cover': round(cloud_cover, 2),
        'solar_irradiance': round(irradiance, 2),
        'residential_load_MW': round(res_MW, 3),
        'commercial_load_MW': round(com_MW, 3),
        'industrial_load_MW': round(ind_MW, 3),
        'critical_load_MW': round(crit_MW, 3),
        'solar_MW': round(solar_MW, 3),
        'wind_MW': round(wind_MW, 3)
    }

def main():
    print(f"Loading input daily dataset from '{INPUT_DATASET}'...")
    if not os.path.exists(INPUT_DATASET):
        print(f"Error: Could not find {INPUT_DATASET}")
        return

    df_daily = pd.read_csv(INPUT_DATASET)
    
    print("Expanding daily dataset to hourly rows...")
    hourly_rows = []
    for _, row in df_daily.iterrows():
        for hour in range(24):
            hourly_rows.append(generate_hourly_data(row, hour))
            
    df_hourly = pd.DataFrame(hourly_rows)
    
    print(f"Saving generated dataset to '{OUTPUT_DATASET}'...")
    df_hourly.to_csv(OUTPUT_DATASET, index=False)
    
    print(f"Success! Generated {len(df_hourly)} hourly rows.")
    print("Data Preview:")
    print(df_hourly.head())

if __name__ == "__main__":
    main()
