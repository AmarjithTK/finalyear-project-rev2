"""
Kerala IEEE 13-Bus - 24-Hour Load Flow Generator
=================================================
Generates complete bus-wise power flow data for all buses across 24 hours
with realistic Kerala demand patterns. Output feeds directly into LSTM training.

Buses monitored: 650, 632, 633, 634, 671, 675, 680, 684, 692
LV buses:        634LV, 671LV, 684LV, 692LV

Load Sectors (from design slide):
  - Residential (Bus 634):  Peak 1500 kW  → Evening peak pattern
  - Commercial  (Bus 684):  Peak  800 kW  → Working hours pattern
  - Industrial  (Bus 671):  Peak 1100 kW  → Daytime stability pattern
  - Critical    (Bus 692):  Peak  400 kW  → Always-on flat pattern

DER:
  - Solar PV = 1.2 MW  (Bus 633, daytime only)
  - Wind     = 0.8 MW  (Bus 675, variable)
  - Total DER = 2.0 MW
"""

import opendssdirect as dss
import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────────────────────
# 1.  REALISTIC 24-HOUR MULTIPLIER PROFILES (Kerala context)
# ──────────────────────────────────────────────────────────────

HOURS = list(range(24))

# Residential: low morning, peak 19-22h (evening / AC / cooking)
RES_PROFILE = [
    0.40, 0.35, 0.30, 0.28, 0.28, 0.32,   # 00-05
    0.45, 0.60, 0.65, 0.55, 0.50, 0.48,   # 06-11
    0.50, 0.52, 0.50, 0.52, 0.58, 0.70,   # 12-17
    0.85, 1.00, 0.98, 0.90, 0.75, 0.55,   # 18-23
]

# Commercial: zero at night, ramp 09-18h
COM_PROFILE = [
    0.10, 0.10, 0.10, 0.10, 0.10, 0.12,   # 00-05
    0.20, 0.40, 0.65, 0.85, 0.95, 1.00,   # 06-11
    0.98, 1.00, 0.98, 0.95, 0.90, 0.80,   # 12-17
    0.60, 0.40, 0.25, 0.15, 0.12, 0.10,   # 18-23
]

# Industrial: steady daytime, slight overnight base
IND_PROFILE = [
    0.50, 0.50, 0.50, 0.50, 0.52, 0.55,   # 00-05
    0.65, 0.80, 0.92, 1.00, 1.00, 1.00,   # 06-11
    0.98, 1.00, 1.00, 0.98, 0.95, 0.90,   # 12-17
    0.80, 0.70, 0.65, 0.60, 0.55, 0.52,   # 18-23
]

# Critical: flat (hospital/telecom/water)
CRIT_PROFILE = [1.0] * 24

# Solar irradiance (p.u.) – Kerala sun curve
SOLAR_PROFILE = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.02,   # 00-05
    0.10, 0.30, 0.55, 0.75, 0.90, 0.98,   # 06-11
    1.00, 0.98, 0.92, 0.80, 0.60, 0.35,   # 12-17
    0.10, 0.02, 0.00, 0.00, 0.00, 0.00,   # 18-23
]

# Wind generation (p.u.) – variable, stronger at night/morning
WIND_PROFILE = [
    0.70, 0.75, 0.80, 0.85, 0.80, 0.75,   # 00-05
    0.60, 0.50, 0.45, 0.40, 0.40, 0.45,   # 06-11
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75,   # 12-17
    0.80, 0.85, 0.90, 0.80, 0.75, 0.72,   # 18-23
]

# Peak values (kW)
PEAK = {"Res": 1500, "Com": 800, "Ind": 1100, "Crit": 400}

# All 11kV buses to monitor
HV_BUSES = ["650", "632", "633", "634", "671", "675", "680", "684", "692"]
# LV buses (0.433 kV)
LV_BUSES  = ["634lv", "671lv", "684lv", "692lv"]
ALL_BUSES = HV_BUSES + LV_BUSES

# ──────────────────────────────────────────────────────────────
# 2.  DSS INITIALISE
# ──────────────────────────────────────────────────────────────

DSS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KeralaIEEE13.dss")

def init_dss():
    dss.Text.Command("Clear")
    dss.Text.Command(f"Compile [{DSS_FILE}]")
    err = dss.Error.Description()
    if err:
        raise RuntimeError(f"DSS compile error: {err}")
    print("✓ DSS compiled successfully")

# ──────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────

def set_load(load_name, kw, kvar_ratio=0.30):
    """Update a single load element."""
    kvar = round(kw * kvar_ratio, 2)
    dss.Text.Command(f"Edit Load.{load_name} kw={kw:.2f} kvar={kvar:.2f}")

def get_bus_voltage(bus_dict, bus):
    """Return average per-unit voltage for a bus (mean of phases)."""
    bus_lower = bus.lower()
    # opendssdirect returns lowercase bus names
    vals = []
    for key, v in bus_dict.items():
        if key.startswith(bus_lower):
            if isinstance(v, (int, float)):
                vals.append(v)
    return round(float(np.mean(vals)), 5) if vals else 0.0

def get_all_voltages():
    """Return dict {busname: avg_pu_voltage} for all buses."""
    names  = dss.Circuit.AllBusNames()     # list of strings
    # AllBusMagPu returns flat list of phase voltages
    mag_pu = dss.Circuit.AllBusMagPu()    # list of floats

    # Build per-bus averages
    bus_phase_count = {}
    bus_phase_vals  = {}
    idx = 0
    for bus in names:
        num_phases = dss.Bus.NumNodes()   # need to set active bus first
        dss.Circuit.SetActiveBus(bus)
        phases = dss.Bus.NumNodes()
        vals = mag_pu[idx: idx + phases]
        bus_phase_vals[bus.lower()] = vals
        idx += phases

    result = {b: round(float(np.mean(v)), 5) if len(v) else 0.0
              for b, v in bus_phase_vals.items()}
    return result

def get_line_flow(line_name):
    """Return (P_kW, Q_kVAR, loading_pct) for a line element."""
    try:
        dss.Circuit.SetActiveElement(f"Line.{line_name}")
        powers = dss.CktElement.Powers()   # [P1,Q1,P2,Q2, ...] kW/kVAR
        p_kw  = abs(powers[0])
        q_kvar = abs(powers[1])
        norm_amps = dss.CktElement.NormalAmps()
        curr = dss.CktElement.CurrentsMagAng()
        i_mag = curr[0] if curr else 0
        loading = round(100.0 * i_mag / norm_amps, 2) if norm_amps else 0.0
        return round(p_kw, 2), round(q_kvar, 2), loading
    except Exception:
        return 0.0, 0.0, 0.0

# ──────────────────────────────────────────────────────────────
# 4.  MAIN 24-HOUR SIMULATION LOOP
# ──────────────────────────────────────────────────────────────

def run_24h_simulation(num_days=1, add_noise=True):
    """
    Run load flow for every hour of the day (optionally multiple days).
    Returns a DataFrame with bus voltages, sector loads, losses, DER output.
    """
    init_dss()
    records = []

    for day in range(num_days):
        for h in HOURS:
            # ── Sector load multipliers ──
            res_mult  = RES_PROFILE[h]
            com_mult  = COM_PROFILE[h]
            ind_mult  = IND_PROFILE[h]
            crit_mult = CRIT_PROFILE[h]

            if add_noise:
                noise = lambda: np.random.uniform(0.97, 1.03)
                res_mult  *= noise()
                com_mult  *= noise()
                ind_mult  *= noise()

            res_kw  = round(PEAK["Res"]  * res_mult,  2)
            com_kw  = round(PEAK["Com"]  * com_mult,  2)
            ind_kw  = round(PEAK["Ind"]  * ind_mult,  2)
            crit_kw = round(PEAK["Crit"] * crit_mult, 2)

            # ── Update DSS loads ──
            # Residential (634LV) — 3 phases
            per_phase_res = res_kw / 3
            set_load("Res_A", per_phase_res)
            set_load("Res_B", per_phase_res)
            set_load("Res_C", per_phase_res)

            # Commercial (684LV)
            per_phase_com = com_kw / 3
            set_load("Com_A", per_phase_com)
            set_load("Com_B", per_phase_com)
            set_load("Com_C", per_phase_com)

            # Industrial (671LV)
            per_phase_ind = ind_kw / 3
            set_load("Ind_A", per_phase_ind)
            set_load("Ind_B", per_phase_ind)
            set_load("Ind_C", per_phase_ind)

            # Critical (692LV)
            per_phase_crit = crit_kw / 3
            set_load("Crit_A", per_phase_crit)
            set_load("Crit_B", per_phase_crit)
            set_load("Crit_C", per_phase_crit)

            # ── Update Solar PV ──
            solar_pu = SOLAR_PROFILE[h]
            solar_kw = round(1200 * solar_pu, 2)
            dss.Text.Command(f"Edit PVSystem.SolarPV irradiance={solar_pu:.4f} kva={max(solar_kw,1):.2f} pmpp={max(solar_kw,1):.2f}")

            # ── Update Wind Generator ──
            wind_pu = WIND_PROFILE[h]
            if add_noise:
                wind_pu *= np.random.uniform(0.95, 1.05)
            wind_kw = round(800 * wind_pu, 2)
            dss.Text.Command(f"Edit Generator.WindGen kw={wind_kw:.2f}")

            # ── Solve power flow ──
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                print(f"  ⚠ Hour {h}: power flow did not converge")

            # ── Collect bus voltages ──
            bus_voltages = get_all_voltages()

            def vpu(bus):
                return bus_voltages.get(bus.lower(), 0.0)

            # ── Collect line flows (key lines) ──
            p_l1, q_l1, ld_l1 = get_line_flow("L650_632")
            p_l3, q_l3, ld_l3 = get_line_flow("L632_671")

            # ── System losses ──
            losses = dss.Circuit.Losses()  # [W_real, VAR_reactive]
            loss_kw   = round(losses[0] / 1000, 3)
            loss_kvar = round(losses[1] / 1000, 3)

            # ── Total generation & load ──
            total_pwr = dss.Circuit.TotalPower()   # [P, Q] in kW, kVAR (source injection)
            grid_kw   = round(abs(total_pwr[0]), 2)

            total_load_kw = res_kw + com_kw + ind_kw + crit_kw
            der_kw        = solar_kw + wind_kw

            row = {
                # ── Time ──
                "Day":  day + 1,
                "Hour": h,

                # ── Bus voltages (per unit) ──
                "V_650":   vpu("650"),
                "V_632":   vpu("632"),
                "V_633":   vpu("633"),
                "V_634":   vpu("634"),
                "V_671":   vpu("671"),
                "V_675":   vpu("675"),
                "V_680":   vpu("680"),
                "V_684":   vpu("684"),
                "V_692":   vpu("692"),
                "V_634LV": vpu("634lv"),
                "V_671LV": vpu("671lv"),
                "V_684LV": vpu("684lv"),
                "V_692LV": vpu("692lv"),

                # ── Sector loads (kW) ──
                "Res_kW":  res_kw,
                "Com_kW":  com_kw,
                "Ind_kW":  ind_kw,
                "Crit_kW": crit_kw,
                "Total_Load_kW": round(total_load_kw, 2),

                # ── Load multipliers ──
                "Res_Mult":  round(res_mult, 4),
                "Com_Mult":  round(com_mult, 4),
                "Ind_Mult":  round(ind_mult, 4),

                # ── DER output (kW) ──
                "Solar_kW":    solar_kw,
                "Wind_kW":     round(wind_kw, 2),
                "Total_DER_kW": round(solar_kw + wind_kw, 2),
                "Solar_Irr_pu": round(solar_pu, 4),

                # ── Grid / source ──
                "Grid_kW":  grid_kw,
                "Net_Import_kW": round(total_load_kw - der_kw, 2),

                # ── Losses ──
                "Loss_kW":   loss_kw,
                "Loss_kVAR": loss_kvar,

                # ── Key line flows ──
                "L1_P_kW":  p_l1,  "L1_Q_kVAR": q_l1,  "L1_Loading_pct": ld_l1,
                "L3_P_kW":  p_l3,  "L3_Q_kVAR": q_l3,  "L3_Loading_pct": ld_l3,
            }
            records.append(row)

            if h % 6 == 0:
                print(f"  Day {day+1} Hour {h:02d} | Load={total_load_kw:.0f}kW  "
                      f"DER={der_kw:.0f}kW  Loss={loss_kw:.1f}kW  "
                      f"V632={vpu('632'):.4f}pu  V634={vpu('634'):.4f}pu")

    df = pd.DataFrame(records)
    return df

# ──────────────────────────────────────────────────────────────
# 5.  GENERATE SYNTHETIC WEATHER VARIATION (multi-day)
# ──────────────────────────────────────────────────────────────

def generate_weather_csv(num_days=30, path="weather.csv"):
    """
    Generate realistic Kerala weather CSV for multi-day simulation.
    Temp range: 24-34°C, Monsoon season has higher rain probability.
    """
    rows = []
    for day in range(num_days):
        is_monsoon = (day % 365) // 30 in [5, 6, 7, 8, 9]  # Jun-Oct
        for h in HOURS:
            temp  = 28 + 3*np.sin(np.pi*(h-6)/12) + np.random.normal(0, 1)
            rain  = int(np.random.random() < (0.6 if is_monsoon else 0.2))
            solar = max(0, SOLAR_PROFILE[h] * (0.5 if rain else 1.0) * np.random.uniform(0.9, 1.1))
            wind  = max(0, WIND_PROFILE[h]  * (1.2 if is_monsoon else 1.0) * np.random.uniform(0.85, 1.15))
            rows.append({"Day": day+1, "Hour": h, "Temp": round(temp,2),
                         "Rain": rain, "Solar": round(solar,4), "Wind": round(wind,4)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"✓ Weather CSV saved: {path}  ({len(df)} rows)")
    return df

# ──────────────────────────────────────────────────────────────
# 6.  LSTM-READY FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def prepare_lstm_features(df, target_col="V_634", window=24):
    """
    Prepare sliding-window sequences for LSTM.
    Returns X (samples, timesteps, features), y (samples,)
    """
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    feature_cols = [
        "V_650","V_632","V_633","V_634","V_671","V_675","V_680","V_684","V_692",
        "V_634LV","V_671LV","V_684LV","V_692LV",
        "Res_kW","Com_kW","Ind_kW","Crit_kW","Total_Load_kW",
        "Solar_kW","Wind_kW","Total_DER_kW",
        "Grid_kW","Net_Import_kW","Loss_kW",
        "L1_P_kW","L3_P_kW",
        "Hour"
    ]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    X, y = [], []
    target_idx = feature_cols.index(target_col)

    for i in range(len(scaled) - window):
        X.append(scaled[i: i + window])
        y.append(scaled[i + window][target_idx])

    return np.array(X), np.array(y), scaler, feature_cols

# ──────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("Kerala IEEE 13-Bus — 24-Hour Load Flow Generator")
    print("=" * 60)

    # Step A: single clean day (no noise) — baseline
    print("\n[1/3] Running baseline 24-hour simulation …")
    df_baseline = run_24h_simulation(num_days=1, add_noise=False)
    df_baseline.to_csv("loadflow_baseline.csv", index=False)
    print(f"✓ Baseline saved → loadflow_baseline.csv  ({len(df_baseline)} rows)")

    # Step B: 30-day training dataset with noise
    print("\n[2/3] Running 30-day training dataset (with noise) …")
    df_train = run_24h_simulation(num_days=30, add_noise=True)
    df_train.to_csv("loadflow_training.csv", index=False)
    print(f"✓ Training data saved → loadflow_training.csv  ({len(df_train)} rows)")

    # Step C: LSTM features
    print("\n[3/3] Preparing LSTM sequences …")
    X, y, scaler, feat_cols = prepare_lstm_features(df_train, target_col="V_634", window=24)
    np.save("lstm_X.npy", X)
    np.save("lstm_y.npy", y)
    print(f"✓ LSTM arrays saved → lstm_X.npy {X.shape}, lstm_y.npy {y.shape}")

    # Summary
    print("\n" + "=" * 60)
    print("OUTPUT FILES:")
    print("  loadflow_baseline.csv  — 24 rows, 1 clean day")
    print("  loadflow_training.csv  — 720 rows, 30 days with noise")
    print("  lstm_X.npy             — (samples, 24h window, features)")
    print("  lstm_y.npy             — target voltage sequence")
    print("=" * 60)

    print("\n── BASELINE VOLTAGE SUMMARY (per bus) ──")
    v_cols = [c for c in df_baseline.columns if c.startswith("V_")]
    print(df_baseline[v_cols].describe().round(4).to_string())

    print("\n── HOURLY LOAD & DER (baseline) ──")
    print(df_baseline[["Hour","Total_Load_kW","Total_DER_kW",
                        "Net_Import_kW","Loss_kW"]].to_string(index=False))