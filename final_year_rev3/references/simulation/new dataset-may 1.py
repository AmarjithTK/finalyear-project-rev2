# ==========================================================
# MICROGRID HOURLY DATASET GENERATOR
# Modified IEEE 13-Bus Smart Microgrid
# Creates 1-year hourly dataset with:
# Weather + Loads + Solar + Wind
# ==========================================================

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# ==========================================================
# CONFIGURATION
# ==========================================================

OUTPUT_FILE = "kerala_microgrid_hourly_dataset.csv"

DAYS = 365                 # Number of days
START_DATE = "2025-01-01"

# Peak Loads (MW)
PEAK_RES = 1.6
PEAK_COM = 0.8
PEAK_IND = 1.2
PEAK_CRIT = 0.4

# Renewable Capacities (MW)
SOLAR_CAPACITY = 1.2
WIND_CAPACITY = 0.8

# ==========================================================
# HOURLY LOAD SHAPES (0 to 1)
# ==========================================================

res_profile = [
0.40,0.35,0.30,0.30,0.35,0.50,0.70,0.80,0.75,0.65,0.55,0.50,
0.50,0.55,0.60,0.70,0.80,0.90,1.00,0.95,0.85,0.75,0.60,0.50
]

com_profile = [
0.10,0.10,0.10,0.10,0.10,0.20,0.40,0.70,0.90,1.00,1.00,1.00,
1.00,1.00,1.00,0.90,0.80,0.70,0.50,0.30,0.20,0.10,0.10,0.10
]

ind_profile = [
0.20,0.20,0.20,0.20,0.30,0.60,0.80,0.90,1.00,1.00,1.00,1.00,
1.00,1.00,1.00,1.00,1.00,0.90,0.80,0.60,0.40,0.30,0.20,0.20
]

crit_profile = [1.0] * 24

# ==========================================================
# WEATHER FUNCTIONS
# ==========================================================

def seasonal_temp(day):
    """Yearly seasonal temperature swing"""
    return 28 + 5 * math.sin(2 * math.pi * day / 365)

def daily_temp(base, hour):
    """Daily temperature variation"""
    return base + 4 * math.sin((hour - 6) * math.pi / 12)

# ==========================================================
# GENERATOR LOOP
# ==========================================================

rows = []
start = datetime.strptime(START_DATE, "%Y-%m-%d")

for day in range(DAYS):

    date = start + timedelta(days=day)

    for hour in range(24):

        timestamp = date + timedelta(hours=hour)

        # -------------------------------
        # WEATHER
        # -------------------------------
        base_temp = seasonal_temp(day)
        temp = daily_temp(base_temp, hour) + np.random.uniform(-1,1)

        humidity = 70 + 15*np.sin(2*np.pi*hour/24) + np.random.uniform(-5,5)
        humidity = np.clip(humidity, 40, 95)

        wind_speed = 5 + 3*np.sin(2*np.pi*(hour+3)/24) + np.random.uniform(-1,1)
        wind_speed = max(0, wind_speed)

        cloud_cover = np.random.uniform(10, 80)

        # -------------------------------
        # SOLAR IRRADIANCE
        # -------------------------------
        if 6 <= hour <= 18:
            solar_shape = math.sin(math.pi * (hour - 6) / 12)
            irradiance = 1000 * solar_shape * (1 - cloud_cover/120)
            irradiance = max(0, irradiance)
        else:
            solar_shape = 0
            irradiance = 0

        # -------------------------------
        # SOLAR GENERATION
        # -------------------------------
        solar_MW = SOLAR_CAPACITY * (irradiance / 1000)

        # -------------------------------
        # WIND GENERATION
        # -------------------------------
        if wind_speed < 3:
            wind_MW = 0
        elif wind_speed < 12:
            wind_MW = WIND_CAPACITY * ((wind_speed - 3)/9)**3
        elif wind_speed <= 25:
            wind_MW = WIND_CAPACITY
        else:
            wind_MW = 0

        # -------------------------------
        # WEATHER LOAD FACTOR
        # -------------------------------
        weather_factor = 1 + 0.01*max(temp-28,0) + 0.002*max(humidity-70,0)

        # -------------------------------
        # LOADS
        # -------------------------------
        res = PEAK_RES * res_profile[hour] * weather_factor
        com = PEAK_COM * com_profile[hour] * weather_factor
        ind = PEAK_IND * ind_profile[hour] * weather_factor
        crit = PEAK_CRIT * crit_profile[hour]

        total_load = res + com + ind + crit
        renewable = solar_MW + wind_MW
        grid_import = max(0, total_load - renewable)

        # -------------------------------
        # SAVE ROW
        # -------------------------------
        rows.append({
            "timestamp": timestamp,
            "temperature_C": round(temp,2),
            "humidity_%": round(humidity,2),
            "wind_speed_mps": round(wind_speed,2),
            "cloud_cover_%": round(cloud_cover,2),
            "solar_irradiance_Wm2": round(irradiance,2),

            "residential_load_MW": round(res,3),
            "commercial_load_MW": round(com,3),
            "industrial_load_MW": round(ind,3),
            "critical_load_MW": round(crit,3),

            "solar_MW": round(solar_MW,3),
            "wind_MW": round(wind_MW,3),

            "total_load_MW": round(total_load,3),
            "renewable_MW": round(renewable,3),
            "grid_import_MW": round(grid_import,3)
        })

# ==========================================================
# EXPORT CSV
# ==========================================================

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("Dataset created successfully!")
print("Rows:", len(df))
print("Saved as:", OUTPUT_FILE)
print(df.head())