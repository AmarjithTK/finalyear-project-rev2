import dss
import os
import pandas as pd
import numpy as np

def run_daily_simulation():
    # 1. Initialize OpenDSS
    dss_engine = dss.DSS
    dss_text = dss_engine.Text
    dss_circuit = dss_engine.ActiveCircuit

    # Compile the Master file
    master_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Master_Kerala11kV.dss")
    dss_text.Command = f"Compile [{master_path}]"

    # Define 24-hour load multiplier arrays (Predefined load shapes)
    hours = range(24)
    # Kerala Evening Peak Profile
    res_shape  = [0.3, 0.25, 0.25, 0.25, 0.3, 0.4, 0.6, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.35, 0.4, 0.5, 0.7, 0.9, 1.0, 1.0, 0.9, 0.7, 0.5, 0.4]
    # Commercial Daytime Peak
    com_shape  = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
    # Relatively Flat Profiles
    ind_shape  = [0.7, 0.7, 0.7, 0.7, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.7, 0.7]
    crit_shape = [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8]
    # Solar daytime generation curve
    pv_shape   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0, 1.0, 0.9, 0.7, 0.5, 0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Wind Profile
    wind_shape = [0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7, 0.6]

    # Base Load Ratings in kW (must match the .dss definitions)
    base_kw = {
        "Res_Load": 80,
        "Com_Load": 200,
        "Ind_Load": 400,
        "Crit_Load": 120
    }
    base_gen = {
        "PV_675": 200, # kW
        "Wind_680": 250
    }

    results = []

    # 2. Run simulation loop for 24 hours manually
    print(f"\n{'Hour':>4} | {'Total(kW)':>9} | {'Grid(kW)':>9} | {'Min_V(pu)':>9} | {'Max_Load(%)':>11} | {'UV_Flag'} | {'OL_Flag'}")
    print("-" * 75)
    for h in hours:
        # ---- UPDATE LOADS ----
        dss_circuit.Loads.Name = "Res_Load"
        dss_circuit.Loads.kW = base_kw["Res_Load"] * res_shape[h]

        dss_circuit.Loads.Name = "Com_Load"
        dss_circuit.Loads.kW = base_kw["Com_Load"] * com_shape[h]

        dss_circuit.Loads.Name = "Ind_Load"
        dss_circuit.Loads.kW = base_kw["Ind_Load"] * ind_shape[h]

        dss_circuit.Loads.Name = "Crit_Load"
        dss_circuit.Loads.kW = base_kw["Crit_Load"] * crit_shape[h]

        # ---- UPDATE GENERATORS ----
        # To avoid error if generator isn't found, checking first
        dss_circuit.Generators.Name = "PV_675"
        dss_circuit.Generators.kW = base_gen["PV_675"] * pv_shape[h]

        dss_circuit.Generators.Name = "Wind_680"
        dss_circuit.Generators.kW = base_gen["Wind_680"] * wind_shape[h]

        # ---- SOLVE SNAPSHOT ----
        dss_text.Command = "Solve Mode=Snap"

        if not dss_circuit.Solution.Converged:
            print(f"Warning: Did not converge at hour {h}")

        # ---- EXTRACT METRICS ----
        # 1. Total Power and Losses
        grid_import_kw = dss_circuit.TotalPower[0] * -1 # Active power drawn from source (kW)
        losses_kw = dss_circuit.Losses[0] / 1000 # Total active losses (kW)
        
        # 2. Key Bus Voltages (pu)
        def get_voltage(bus_name):
            dss_circuit.SetActiveBus(bus_name)
            return dss_circuit.ActiveBus.puVoltages[0] if len(dss_circuit.ActiveBus.puVoltages) > 0 else 0.0

        v_source = get_voltage("SourceBus")
        v_res    = get_voltage("634_LT")
        v_com    = get_voltage("684_LT")
        v_ind    = get_voltage("671_LT")
        v_crit   = get_voltage("692_LT")
        v_pv     = get_voltage("675")
        v_wind   = get_voltage("680")

        # 3. Component Loadings (%)
        def get_loading(element_class, element_name):
            dss_circuit.SetActiveElement(f"{element_class}.{element_name}")
            amps = dss_circuit.ActiveCktElement.CurrentsMagAng[0::2] # Get magnitudes
            nphases = dss_circuit.ActiveCktElement.NumPhases
            
            # Avoid numPy truth array ambiguity and only check Terminal 1 (first nphases)
            # This fixes the gigantic % loadings because terminal 2 (LT) has huge currents
            if amps is not None and np.size(amps) >= nphases:
                max_amps = float(np.max(amps[:nphases]))
            else:
                max_amps = 0.0
                
            rated = dss_circuit.ActiveCktElement.NormalAmps
            
            # If standard NormalAmps isn't reliable for Transformers, use KVA logic
            if element_class.lower() == "transformer":
                dss_circuit.Transformers.Name = element_name
                kva = dss_circuit.Transformers.kVA
                kv = dss_circuit.Transformers.kV
                if kva > 0 and kv > 0:
                    rated = kva / (np.sqrt(3) * kv)

            return (max_amps / rated) * 100.0 if rated > 0 else 0.0

        tr_res_load  = get_loading("Transformer", "TR_Res634")
        tr_com_load  = get_loading("Transformer", "TR_Com684")
        tr_ind_load  = get_loading("Transformer", "TR_Ind671")
        tr_crit_load = get_loading("Transformer", "TR_Crit692")

        line_main = get_loading("Line", "L_Source_650")
        line_671_680 = get_loading("Line", "L_671_680") # Wind connection
        line_671_684 = get_loading("Line", "L_671_684") # Com connection
        line_671_692 = get_loading("Line", "L_671_692") # Crit connection

        total_load_kw = (base_kw["Res_Load"] * res_shape[h] + 
                         base_kw["Com_Load"] * com_shape[h] +
                         base_kw["Ind_Load"] * ind_shape[h] +
                         base_kw["Crit_Load"] * crit_shape[h])
                         
        pv_kw = base_gen["PV_675"] * pv_shape[h]
        wind_kw = base_gen["Wind_680"] * wind_shape[h]
        
        all_voltages = dss_circuit.AllBusVmagPu
        min_v = float(np.min(all_voltages)) if all_voltages is not None and np.size(all_voltages) > 0 else 0.0
        
        max_load = max([tr_res_load, tr_com_load, tr_ind_load, tr_crit_load, line_main, line_671_680, line_671_684, line_671_692])
        uv_flag = 1 if min_v < 0.95 else 0
        ol_flag = 1 if max_load > 100.0 else 0

        # Store to records combining old columns and the new LLM optimal ones
        results.append({
            "timestamp": f"2026-05-02 {h:02d}:00",
            "total_load_kW": round(total_load_kw, 2),
            "solar_kW": round(pv_kw, 2),
            "wind_kW": round(wind_kw, 2),
            "grid_import_kW": round(grid_import_kw, 2),
            "losses_kW": round(losses_kw, 2),
            "min_voltage_pu": round(min_v, 4),
            "v_res_pu": round(v_res, 4),
            "v_ind_pu": round(v_ind, 4),
            "v_com_pu": round(v_com, 4),
            "v_crit_pu": round(v_crit, 4),
            "max_loading_pct": round(max_load, 2),
            "tr_res_loading_pct": round(tr_res_load, 2),
            "tr_ind_loading_pct": round(tr_ind_load, 2),
            "tr_com_loading_pct": round(tr_com_load, 2),
            "tr_crit_loading_pct": round(tr_crit_load, 2),
            "line_main_loading_pct": round(line_main, 2),
            "undervoltage_flag": uv_flag,
            "overload_flag": ol_flag
        })
        
        # Print meaningful hourly output to console
        print(f"{h:4d} | {total_load_kw:9.1f} | {grid_import_kw:9.1f} | {min_v:9.4f} | {max_load:10.1f}% | {uv_flag:7d} | {ol_flag:7d}")

    # 3. Export Data
    df = pd.DataFrame(results)
    
    # Save to a CSV similar to what you'd use for PyTorch
    out_file = "simulated_24h_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved full results to {out_file}")

if __name__ == "__main__":
    run_daily_simulation()