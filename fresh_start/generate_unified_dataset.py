"""
========================================================================================
Kerala Microgrid Project: Unified Weather & OpenDSS Power Flow Simulator
========================================================================================

PROJECT AIM:
To generate a comprehensive, highly realistic machine learning dataset for a localized 
microgrid. This script bridges advanced physical weather modeling (temperature, solar 
irradiance, wind speed, anomalies) with accurate electrical power flow simulations using 
OpenDSS. The output dataset provides the foundation for advanced grid analytics such as 
load forecasting, voltage stability analysis, anomaly detection, and islanding feasibility.

MICROGRID TOPOLOGY (Modified IEEE 13-Bus System parameterized for Kerala):
- Main Substation: 33/11 kV, 5 MVA Transformer connecting SourceBus to backbone Bus 650.
- Distribution Grid: 11 kV primary backbone line varying out to LV (0.433 kV / 415 V) loads.

DISTRIBUTED ENERGY RESOURCES (DERs):
- Solar PV: 1.2 MW Generator connected at Bus 633 (11 kV), at the end of a main branch.
- Wind Turbine: 0.8 MW Generator connected at Bus 675 (11 kV).

LOAD DISTRIBUTION (Dynamic, Weather-Dependent, and Profile-Driven):
- Residential Center: Connected locally at Bus 634 via 1.5 MVA DT_Res. Peak ~1.6 MW.
- Industrial Branch: Connected locally at Bus 671 via 1.5 MVA DT_Ind. Peak ~1.2 MW.
- Commercial Branch: Connected locally at Bus 684 via 1.0 MVA DT_Com. Peak ~0.8 MW.
- Critical Load: Connected locally at Bus 692 via 0.5 MVA DT_Crit. Peak ~0.4 MW (Constant 24/7).

SIMULATION WORKFLOW:
1. Loads historic daily weather baselines (e.g., from solar_power_dataset.csv).
2. Computes hourly variations using physics models (wind sea-breeze boosts, solar/cloud curves).
3. Interfaces with the OpenDSS COM, injecting exact active loads and DER generation.
5. Integrates with downstream pipelines: Generated telemetries are forecasted by CNN-LSTM
   hybrid models, which are then fed into LLMs (e.g., Gemini, OpenRouter) against strict 
   grid physics rules to generate automated actionable reports on undervoltages, 
   overvoltages, bottleneck loading, and islanding strategies.
========================================================================================
"""

import pandas as pd
import numpy as np
import math
import os
import opendssdirect as dss

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════

# Input/Output paths
WEATHER_INPUT = "solar_dataset_2022_2023.csv"
DSS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opendss", "KeralaIEEE13.dss")
OUTPUT_CSV = "unified_microgrid_24h_results.csv"

# DER Capacities (MW)
SOLAR_CAPACITY = 1.2
WIND_CAPACITY = 0.8

# Load Peak Capacities (MW)
PEAK_RES = 1.6   # 40% of 4 MW
PEAK_IND = 1.2   # 30% of 4 MW
PEAK_COM = 0.8   # 20% of 4 MW
PEAK_CRIT = 0.4  # 10% of 4 MW

# OpenDSS Peak values (kW) - Ensure OpenDSS is loaded in kW matching these base capacities
# e.g., if total Res peak is 1.6 MW = 1600 kW

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
# 3. ADVANCED WEATHER GENERATION (from generate_dataset.py)
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
    wind_speed = daily_row['wind_speed'] + sea_breeze_boost + np.random.uniform(-0.8, 1.4)
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
    
    # 3. WIND GENERATION
    if wind_speed < 3:
        wind_MW = 0.0
    elif wind_speed < 12:
        wind_MW = WIND_CAPACITY * ((wind_speed - 3) / (12 - 3)) ** 3
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
# 4. OpenDSS HELPERS
# ──────────────────────────────────────────────────────────────

def init_dss():
    dss.Text.Command("Clear")
    dss.Text.Command(f"Compile [{DSS_FILE}]")
    err = dss.Error.Description()
    if err:
        raise RuntimeError(f"DSS compile error: {err}")

def set_load(load_name, kw, kvar_ratio=0.30):
    kvar = round(kw * kvar_ratio, 2)
    dss.Text.Command(f"Edit Load.{load_name} kw={kw:.2f} kvar={kvar:.2f}")

def get_all_voltages():
    names = dss.Circuit.AllBusNames()
    mag_pu = dss.Circuit.AllBusMagPu()
    bus_phase_vals = {}
    idx = 0
    for bus in names:
        dss.Circuit.SetActiveBus(bus)
        phases = dss.Bus.NumNodes()
        vals = mag_pu[idx: idx + phases]
        bus_phase_vals[bus.lower()] = vals
        idx += phases
    
    result = {b: round(float(np.mean(v)), 5) if len(v) else 0.0
              for b, v in bus_phase_vals.items()}
    return result

def get_line_flow(line_name):
    try:
        dss.Circuit.SetActiveElement(f"Line.{line_name}")
        powers = dss.CktElement.Powers()
        p_kw = abs(powers[0])
        q_kvar = abs(powers[1])
        norm_amps = dss.CktElement.NormalAmps()
        curr = dss.CktElement.CurrentsMagAng()
        i_mag = curr[0] if curr else 0
        loading = round(100.0 * i_mag / norm_amps, 2) if norm_amps else 0.0
        return round(p_kw, 2), round(q_kvar, 2), loading
    except Exception:
        return 0.0, 0.0, 0.0


# ──────────────────────────────────────────────────────────────
# 5. UNIFIED SIMULATION
# ──────────────────────────────────────────────────────────────

def run_unified_simulation(num_days=30, seed=42):
    if seed is not None:
        np.random.seed(seed)
    
    if not os.path.exists(WEATHER_INPUT):
        raise FileNotFoundError(f"Weather input file not found: {WEATHER_INPUT}")
    
    df_daily = pd.read_csv(WEATHER_INPUT)
    n_days_available = len(df_daily)
    
    init_dss()
    records = []
    
    print(f"\nRunning {num_days} days × 24 hours simulation...")
    
    for day in range(num_days):
        daily_idx = day % n_days_available
        daily_row = df_daily.iloc[daily_idx].to_dict()
        
        for hour in HOURS:
            weather = generate_hourly_weather_and_der(daily_row, hour)
            
            # MW -> kW for OpenDSS loads
            res_kw = weather['residential_load_MW'] * 1000
            com_kw = weather['commercial_load_MW'] * 1000
            ind_kw = weather['industrial_load_MW'] * 1000
            crit_kw = weather['critical_load_MW'] * 1000
            
            # 3 phase distribution
            for phase in ["A", "B", "C"]:
                set_load(f"Res_{phase}", res_kw / 3)
                set_load(f"Com_{phase}", com_kw / 3)
                set_load(f"Ind_{phase}", ind_kw / 3)
                set_load(f"Crit_{phase}", crit_kw / 3)
            
            # DER Updates
            solar_kw = weather['solar_MW'] * 1000
            wind_kw = weather['wind_MW'] * 1000
            
            solar_pu = solar_kw / 1200  # assuming base 1200kVA
            dss.Text.Command(f"Edit PVSystem.SolarPV irradiance={solar_pu:.4f} kva={max(solar_kw,1):.2f} pmpp={max(solar_kw,1):.2f}")
            dss.Text.Command(f"Edit Generator.WindGen kw={max(wind_kw,0.1):.2f}")
            
            # Solve
            dss.Solution.Solve()
            
            voltages = get_all_voltages()
            
            # Line flows and loadings
            p_L650_632, q_L650_632, ld_L650_632 = get_line_flow("L650_632") # Main Feeder
            p_L632_671, q_L632_671, ld_L632_671 = get_line_flow("L632_671") # Industrial/Wind Split
            p_L632_633, q_L632_633, ld_L632_633 = get_line_flow("L632_633") # Solar PV Branch
            
            # Max line loading
            max_line_loading = max(ld_L650_632, ld_L632_671, ld_L632_633)
            
            # Total System Losses
            losses = dss.Circuit.Losses()
            loss_kw = round(losses[0] / 1000, 3)
            loss_kvar = round(losses[1] / 1000, 3)
            
            # Substation / Grid Import (TotalPower returns negative for import)
            total_pwr = dss.Circuit.TotalPower()
            grid_import_kw = round(-total_pwr[0], 2)
            grid_import_kvar = round(-total_pwr[1], 2)
            
            # Voltage Stability
            valid_volts = [v for v in voltages.values() if v > 0.0]
            v_min = min(valid_volts) if valid_volts else 0.0
            v_max = max(valid_volts) if valid_volts else 0.0
            
            # Form real timestamp explicitly using the actual date from dataset
            current_date = daily_row.get('date', f"Day_{day+1}")
            
            rec = {
                "Timestamp": f"{current_date} {hour:02d}:00:00",
                "Date": current_date,
                "Hour": hour,
                # Weather & Generation
                "Temperature_C": weather['temperature'],
                "Humidity_pct": weather['humidity'],
                "Wind_Speed_ms": weather['wind_speed'],
                "Cloud_Cover_pct": weather['cloud_cover'],
                "Solar_Irradiance_Wm2": weather['solar_irradiance'],
                "Solar_MW": weather['solar_MW'],
                "Wind_MW": weather['wind_MW'],
                "Total_Load_MW": round((res_kw + com_kw + ind_kw + crit_kw) / 1000, 4),
                "Grid_Import_MW": round(grid_import_kw / 1000, 4),
                "Grid_Import_MVAR": round(grid_import_kvar / 1000, 4),
                
                # Critical Bus Voltages (PU)
                "V_Sub_650": voltages.get("650", 0.0),
                "V_Split_632": voltages.get("632", 0.0),
                "V_Solar_633": voltages.get("633", 0.0),
                "V_Wind_675": voltages.get("675", 0.0),
                "V_Res_634": voltages.get("634", 0.0),
                "V_Ind_671": voltages.get("671", 0.0),
                "V_Com_684": voltages.get("684", 0.0),
                "V_Crit_692": voltages.get("692", 0.0),
                "V_Min_pu": v_min,
                "V_Max_pu": v_max,
                
                # Critical Branch Loadings (%)
                "L_Main_650_632_pct": ld_L650_632,
                "L_IndWind_632_671_pct": ld_L632_671,
                "L_Solar_632_633_pct": ld_L632_633,
                "Max_Line_Loading_pct": max_line_loading,
                
                # System Losses
                "Total_Loss_MW": round(loss_kw / 1000, 4),
                "Total_Loss_MVAR": round(loss_kvar / 1000, 4)
            }
            records.append(rec)
            
            if hour % 6 == 0:
                print(f"  Day {day+1:2d} H{hour:02d} | Load={rec['Total_Load_MW']:6.3f}MW  DER={rec['Solar_MW']+rec['Wind_MW']:5.3f}MW  V632={rec['V_Split_632']:.4f}pu  Loss={rec['Total_Loss_MW']:5.4f}MW")
                
    return pd.DataFrame(records)

if __name__ == "__main__":
    try:
        # Generate 2 Years (730 Days) worth of hourly data (approx 17520 records)
        df_res = run_unified_simulation(num_days=730)
        df_res.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved tightly-coupled weather & OpenDSS dataset to {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error: {e}")
