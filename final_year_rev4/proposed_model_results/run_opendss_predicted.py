import os
import pandas as pd
import numpy as np
import opendssdirect as dss

# --- Configuration ---
# Define paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTED_CSV = os.path.join(BASE_DIR, "predicted_results.csv")
DSS_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "opendss", "KeralaIEEE13.dss"))
OUTPUT_CSV = os.path.join(BASE_DIR, "predicted_loadflow_results.csv")

REQUIRED_LOADS = [
    "Res_A", "Res_B", "Res_C",
    "Com_A", "Com_B", "Com_C",
    "Ind_A", "Ind_B", "Ind_C",
    "Crit_A", "Crit_B", "Crit_C",
]
REQUIRED_LINES = ["L650_632", "L632_671", "L632_633"]
REQUIRED_PVS = ["SolarPV"]
REQUIRED_GENERATORS = ["WindGen"]
SOLAR_RATING_KW = 1200.0
WIND_KVAR_RATIO = 100.0 / 800.0

# ──────────────────────────────────────────────────────────────
# OpenDSS Helpers
# ──────────────────────────────────────────────────────────────

def init_dss():
    dss.Text.Command("Clear")
    dss.Text.Command(f"Compile [{DSS_FILE}]")
    err = dss.Error.Description()
    if err:
        raise RuntimeError(f"DSS compile error: {err}")
    validate_circuit()

def _missing(required_names, actual_names):
    actual = {name.lower() for name in actual_names}
    return [name for name in required_names if name.lower() not in actual]

def validate_circuit():
    missing = {
        "loads": _missing(REQUIRED_LOADS, dss.Loads.AllNames()),
        "lines": _missing(REQUIRED_LINES, dss.Lines.AllNames()),
        "PV systems": _missing(REQUIRED_PVS, dss.PVsystems.AllNames()),
        "generators": _missing(REQUIRED_GENERATORS, dss.Generators.AllNames()),
    }
    missing = {group: names for group, names in missing.items() if names}
    if missing:
        detail = "; ".join(f"{group}: {', '.join(names)}" for group, names in missing.items())
        raise RuntimeError(f"DSS file does not match the expected predicted-loadflow circuit. Missing {detail}")

def run_dss_command(command):
    dss.Text.Command(command)
    err = dss.Error.Description()
    if err:
        raise RuntimeError(f"DSS command failed: {command}\n{err}")

def set_load(load_name, kw, kvar_ratio=0.30):
    kvar = round(kw * kvar_ratio, 2)
    run_dss_command(f"Edit Load.{load_name} kw={kw:.2f} kvar={kvar:.2f}")

def get_all_voltages():
    names = dss.Circuit.AllBusNames()
    result = {}
    for bus in names:
        dss.Circuit.SetActiveBus(bus)
        vals = dss.Bus.puVmagAngle()[::2]
        if not vals:
            continue
        result[bus.lower()] = round(float(np.mean(vals)), 5)
    return result

def get_line_flow(line_name):
    if not dss.Circuit.SetActiveElement(f"Line.{line_name}"):
        raise RuntimeError(f"Line.{line_name} was not found in the active DSS circuit")

    powers = dss.CktElement.Powers()
    currents = dss.CktElement.CurrentsMagAng()[::2]
    norm_amps = dss.CktElement.NormalAmps()
    if not norm_amps:
        raise RuntimeError(f"Line.{line_name} has no NormalAmps value, so loading cannot be calculated")

    p_kw = sum(abs(powers[i]) for i in range(0, min(len(powers), 6), 2))
    q_kvar = sum(abs(powers[i]) for i in range(1, min(len(powers), 6), 2))
    max_current = max(currents[:3]) if currents else 0.0
    loading = round(100.0 * max_current / norm_amps, 2)
    return round(p_kw, 2), round(q_kvar, 2), loading

# ──────────────────────────────────────────────────────────────
# Main Loadflow Runner
# ──────────────────────────────────────────────────────────────

def run_predicted_loadflow():
    if not os.path.exists(PREDICTED_CSV):
        raise FileNotFoundError(f"Predicted CSV not found: {PREDICTED_CSV}")
    
    df = pd.read_csv(PREDICTED_CSV)
    print(f"Loaded {len(df)} simulated hours from {PREDICTED_CSV}")
    
    init_dss()
    results = []
    
    for i, row in df.iterrows():
        # 1. Read MW values from predictions and convert to kW
        res_kw = row['Residential_Load_MW'] * 1000
        com_kw = row['Commercial_Load_MW'] * 1000
        ind_kw = row['Industrial_Load_MW'] * 1000
        crit_kw = row['Critical_Load_MW'] * 1000
        
        solar_kw_raw = row['Solar_MW'] * 1000
        wind_kw_raw = row['Wind_MW'] * 1000
        solar_kw = max(0.0, solar_kw_raw)
        wind_kw = max(0.0, wind_kw_raw)
        
        # 2. Update OpenDSS Loads (Distributed across 3 phases evenly)
        for phase in ["A", "B", "C"]:
            set_load(f"Res_{phase}", res_kw / 3)
            set_load(f"Com_{phase}", com_kw / 3)
            set_load(f"Ind_{phase}", ind_kw / 3)
            set_load(f"Crit_{phase}", crit_kw / 3)
            
        # 3. Update DER Generation
        solar_pu = min(1.0, solar_kw / SOLAR_RATING_KW)
        run_dss_command(
            f"Edit PVSystem.SolarPV irradiance={solar_pu:.4f} pmpp={SOLAR_RATING_KW:.2f} kva={SOLAR_RATING_KW:.2f}"
        )
        run_dss_command(f"Edit Generator.WindGen kw={wind_kw:.2f} kvar={wind_kw * WIND_KVAR_RATIO:.2f}")
        
        # 4. Run Power Flow Solution
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError(f"OpenDSS solution did not converge at {row['Timestamp']}")
        
        # 5. Extract Voltages
        voltages = get_all_voltages()
        
        # 6. Extract Line Flows & Loading
        p_L650_632, q_L650_632, ld_L650_632 = get_line_flow("L650_632") # Main Feeder
        p_L632_671, q_L632_671, ld_L632_671 = get_line_flow("L632_671") # Industrial/Wind Split
        p_L632_633, q_L632_633, ld_L632_633 = get_line_flow("L632_633") # Solar PV Branch
        max_line_loading = max(ld_L650_632, ld_L632_671, ld_L632_633)
        
        # 7. Extract System Losses & Grid Import/Export
        losses_w, losses_var = dss.Circuit.Losses()
        total_pwr = dss.Circuit.TotalPower()
        grid_import_mw = -total_pwr[0] / 1000  # TotalPower returns negative for import
        
        valid_volts = [v for v in voltages.values() if v > 0.0]
        v_min = min(valid_volts) if valid_volts else 0.0
        v_max = max(valid_volts) if valid_volts else 0.0
        
        # Append to results
        rec = {
            "Timestamp": row['Timestamp'],
            "Total_Load_Predicted_MW": res_kw/1000 + com_kw/1000 + ind_kw/1000 + crit_kw/1000,
            "Total_DER_Predicted_MW": solar_kw/1000 + wind_kw/1000,
            "Grid_Import_MW": round(grid_import_mw, 4),
            "System_Loss_MW": round(losses_w / 1_000_000, 4),
            "V_Sub_650_pu": voltages.get("650", 0.0),
            "V_Split_632_pu": voltages.get("632", 0.0),
            "V_Res_634_pu": voltages.get("634", 0.0),
            "V_Ind_671_pu": voltages.get("671", 0.0),
            "V_Min_pu": v_min,
            "V_Max_pu": v_max,
            "L_Main_650_632_pct": ld_L650_632,
            "L_Ind_632_671_pct": ld_L632_671,
            "L_Solar_632_633_pct": ld_L632_633,
            "Max_Line_Loading_pct": max_line_loading
        }
        results.append(rec)
        
        if (i+1) % 6 == 0 or i == 0:
            print(f"Hour {i:02d} | Grid Import: {grid_import_mw:6.3f} MW | V_Min: {v_min:.4f} pu | Max Loading: {max_line_loading:.1f}%")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Power flow generated and saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    try:
        run_predicted_loadflow()
    except Exception as e:
        print(f"❌ Error running loadflow: {e}")
