# ==============================================================
# IEEE 13 BUS POWERED MICROGRID DATASET GENERATOR (CORRECTED)
# OpenDSS + IEEE13Nodeckt.dss
# Output: ieee13_microgrid_dataset2.csv
# ==============================================================
# REQUIREMENTS:
# pip install opendssdirect.py pandas numpy
#
# PLACE FILES IN SAME FOLDER:
#   IEEE13Nodeckt.dss
#   IEEELineCodes.dss
#   this_script.py
# ==============================================================

import opendssdirect as dss
import pandas as pd
import numpy as np
import os
import random
from math import sin, pi
from datetime import datetime, timedelta

# --------------------------------------------------------------
# PATH
# --------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(BASE, "IEEE13Nodeckt.dss")

# --------------------------------------------------------------
# CHECK FILE
# --------------------------------------------------------------
if not os.path.exists(MASTER):
    raise FileNotFoundError(f"Cannot find file: {MASTER}")

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
START = datetime(2025, 1, 1, 0, 0)
END   = datetime(2025, 12, 31, 23, 0)

STEP_HOURS = 1

SOLAR_CAP   = 15.0   # MW
WIND_CAP    = 10.0   # MW
THERMAL_CAP = 12.0   # MW

PEAK_LOAD = 35.0     # MW

# --------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------
def season_factor(month):
    if month in [4, 5, 6]:
        return 1.08
    elif month in [7, 8, 9]:
        return 1.03
    elif month in [10, 11]:
        return 0.97
    return 1.00


def residential_profile(h):
    return 0.50 + 0.20*np.exp(-((h-7)/2)**2) + 0.38*np.exp(-((h-20)/3)**2)


def commercial_profile(h):
    return 0.08 + 0.95*np.exp(-((h-14)/4)**2)


def industrial_profile(h):
    return 0.70 + 0.18*np.exp(-((h-11)/5)**2)


def critical_profile(h):
    return 0.92


def solar_pu(hour, month):
    sunrise = 6
    sunset = 18

    if month in [11, 12, 1]:
        sunset = 17.5
    elif month in [4, 5]:
        sunset = 18.5

    if hour < sunrise or hour > sunset:
        return 0.0

    x = (hour - sunrise) / (sunset - sunrise)
    base = sin(pi * x)

    # Cloud effect
    if month in [6, 7, 8]:
        cloud = np.random.uniform(0.35, 0.75)
    else:
        cloud = np.random.uniform(0.75, 1.0)

    return max(0.0, base * cloud)


def wind_pu(month):
    if month in [6, 7, 8]:
        mu = 0.65
    elif month in [3, 4, 5]:
        mu = 0.50
    else:
        mu = 0.38

    x = np.random.normal(mu, 0.15)
    return min(max(x, 0.0), 1.0)


# --------------------------------------------------------------
# SAFE COMPILE
# --------------------------------------------------------------
def compile_model():
    dss.Basic.ClearAll()
    dss.Text.Command(f'compile "{MASTER}"')


# --------------------------------------------------------------
# DATA STORAGE
# --------------------------------------------------------------
rows = []

# --------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------
ts = START

while ts <= END:

    try:
        hour = ts.hour
        month = ts.month
        dow = ts.weekday()

        # ------------------------------------------------------
        # 1. LOAD MODEL FRESH EACH STEP
        # ------------------------------------------------------
        compile_model()

        # ------------------------------------------------------
        # 2. LOAD MODEL
        # ------------------------------------------------------
        sf = season_factor(month)
        weekend = 0.95 if dow >= 5 else 1.0

        res = PEAK_LOAD * 0.45 * residential_profile(hour)
        com = PEAK_LOAD * 0.25 * commercial_profile(hour) * weekend
        ind = PEAK_LOAD * 0.20 * industrial_profile(hour)
        cri = PEAK_LOAD * 0.10 * critical_profile(hour)

        total_load = (res + com + ind + cri) * sf
        total_load *= np.random.uniform(0.97, 1.03)

        load_mult = total_load / PEAK_LOAD
        dss.Text.Command(f"set loadmult={load_mult:.4f}")

        # ------------------------------------------------------
        # 3. MICROGRID GENERATORS
        # ------------------------------------------------------
        solar = SOLAR_CAP * solar_pu(hour, month)
        wind = WIND_CAP * wind_pu(month)

        renewable = solar + wind
        deficit = max(total_load - renewable, 0.0)

        thermal = min(deficit, THERMAL_CAP)

        # Use unique names each loop to avoid clashes
        dss.Text.Command(
            f"new generator.SOLAR1 bus1=680 phases=3 kv=4.16 kw={solar*1000:.2f} pf=1"
        )

        dss.Text.Command(
            f"new generator.WIND1 bus1=675 phases=3 kv=4.16 kw={wind*1000:.2f} pf=1"
        )

        dss.Text.Command(
            f"new generator.THERMAL1 bus1=634 phases=3 kv=4.16 kw={thermal*1000:.2f} pf=0.95"
        )

        # ------------------------------------------------------
        # 4. RANDOM OUTAGE
        # ------------------------------------------------------
        outage_flag = 0
        grid_mode = "grid_connected"

        if random.random() < 0.002:
            # disable source
            dss.Text.Command("edit vsource.source enabled=no")
            outage_flag = 1
            grid_mode = "islanded"

        # ------------------------------------------------------
        # 5. SOLVE
        # ------------------------------------------------------
        dss.Text.Command("solve")

        # ------------------------------------------------------
        # 6. READ RESULTS
        # ------------------------------------------------------
        total_kw, total_kvar = dss.Circuit.TotalPower()

        # OpenDSS returns watts, vars
        losses_w, losses_var = dss.Circuit.Losses()
        feeder_losses_kw = losses_w / 1000.0

        buses = dss.Circuit.AllBusNames()

        volts = []

        for b in buses:
            dss.Circuit.SetActiveBus(b)
            pu = dss.Bus.puVmagAngle()

            # puVmagAngle returns [V1pu, ang1, V2pu, ang2...]
            if len(pu) >= 1:
                phase_vals = pu[0::2]
                volts.extend(phase_vals)

        if len(volts) == 0:
            avg_v = 1.0
            min_v = 1.0
            max_v = 1.0
        else:
            avg_v = float(np.mean(volts))
            min_v = float(np.min(volts))
            max_v = float(np.max(volts))

        # TotalPower returns negative for load served
        total_system_mw = abs(total_kw) / 1000.0

        grid_import = max(total_system_mw - renewable - thermal, 0.0)
        grid_export = max((renewable + thermal) - total_system_mw, 0.0)

        frequency = 50.02 - 0.08*(grid_import/10.0) + np.random.uniform(-0.02, 0.02)
        frequency = min(max(frequency, 49.0), 50.5)

        pf = abs(total_kw) / (np.sqrt(total_kw**2 + total_kvar**2) + 1e-9)

        overload_flag = 1 if total_load > 34 else 0
        undervoltage_flag = 1 if min_v < 0.93 else 0

        anomaly = "normal"

        if outage_flag:
            anomaly = "grid_outage"
        elif overload_flag:
            anomaly = "peak_overload"
        elif undervoltage_flag:
            anomaly = "voltage_sag"
        elif solar < 2 and 10 <= hour <= 15:
            anomaly = "cloudy_low_solar"
        elif wind < 1:
            anomaly = "wind_lull"

        # ------------------------------------------------------
        # 7. SAVE
        # ------------------------------------------------------
        rows.append({
            "timestamp": ts,
            "hour": hour,
            "month": month,
            "day_of_week": dow,

            "residential_load_MW": round(res, 3),
            "commercial_load_MW": round(com, 3),
            "industrial_load_MW": round(ind, 3),
            "critical_load_MW": round(cri, 3),
            "total_load_MW": round(total_load, 3),

            "solar_MW": round(solar, 3),
            "wind_MW": round(wind, 3),
            "thermal_MW": round(thermal, 3),
            "grid_import_MW": round(grid_import, 3),
            "grid_export_MW": round(grid_export, 3),

            "avg_voltage_pu": round(avg_v, 4),
            "min_voltage_pu": round(min_v, 4),
            "max_voltage_pu": round(max_v, 4),

            "frequency_Hz": round(frequency, 3),
            "feeder_losses_kW": round(feeder_losses_kw, 3),
            "power_factor": round(pf, 4),

            "grid_mode": grid_mode,
            "outage_flag": outage_flag,
            "undervoltage_flag": undervoltage_flag,
            "overload_flag": overload_flag,
            "anomaly_label": anomaly
        })

    except Exception as e:
        print("Error at:", ts, "->", e)

    ts += timedelta(hours=STEP_HOURS)

# --------------------------------------------------------------
# SAVE CSV
# --------------------------------------------------------------
df = pd.DataFrame(rows)

OUTFILE = os.path.join(BASE, "new microgrid dataset.csv")
df.to_csv(OUTFILE, index=False)

print("================================================")
print("Dataset created successfully!")
print("Rows:", len(df))
print("Saved:", OUTFILE)
print("================================================")