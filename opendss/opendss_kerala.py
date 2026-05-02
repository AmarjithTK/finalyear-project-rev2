import dss
import os
import pandas as pd
import numpy as np

# Total rows in the CSV: 48216 (hourly data)
# To run the last 10 days: 48216 - 240 = 47976
START_ROW = 47976
END_ROW = 48216

def run_dataset_simulation():
    # 1. Initialize OpenDSS
    dss_engine = dss.DSS
    dss_text = dss_engine.Text
    dss_circuit = dss_engine.ActiveCircuit

    # Compile the Master file
    master_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Master_Kerala11kV.dss")
    dss_text.Command = f"Compile [{master_path}]"

    # Read the dataset
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_generator", "kerala_microgrid_hourly_dataset.csv")
    df_data = pd.read_csv(csv_path)

    # Select the range
    df_subset = df_data.iloc[START_ROW:END_ROW].copy()
    
    results = []

    print(f"\nRunning simulation for rows {START_ROW} to {END_ROW} (Last {END_ROW - START_ROW} hours)")
    print(f"\n{'Hour':>4} | {'Total(MW)':>9} | {'Grid(MW)':>9} | {'Min_V(pu)':>9} | {'Max_Load(%)':>11} | {'UV_Flag'} | {'OL_Flag'}")
    print("-" * 75)

    for i, row in df_subset.iterrows():
        # Read values in MW
        res_mw = row['residential_load_MW']
        com_mw = row['commercial_load_MW']
        ind_mw = row['industrial_load_MW']
        crit_mw = row['critical_load_MW']
        pv_mw = row['solar_MW']
        wind_mw = row['wind_MW']
        timestamp = row['timestamp']

        # ---- UPDATE LOADS (OpenDSS uses kW internally) ----
        dss_circuit.Loads.Name = "Res_Load"
        dss_circuit.Loads.kW = res_mw * 1000.0

        dss_circuit.Loads.Name = "Com_Load"
        dss_circuit.Loads.kW = com_mw * 1000.0

        dss_circuit.Loads.Name = "Ind_Load"
        dss_circuit.Loads.kW = ind_mw * 1000.0

        dss_circuit.Loads.Name = "Crit_Load"
        dss_circuit.Loads.kW = crit_mw * 1000.0

        # ---- UPDATE GENERATORS (OpenDSS uses kW internally) ----
        dss_circuit.Generators.Name = "PV_675"
        dss_circuit.Generators.kW = pv_mw * 1000.0

        dss_circuit.Generators.Name = "Wind_680"
        dss_circuit.Generators.kW = wind_mw * 1000.0

        # ---- SOLVE SNAPSHOT ----
        dss_text.Command = "Solve Mode=Snap"

        if not dss_circuit.Solution.Converged:
            print(f"Warning: Did not converge at timestamp {timestamp}")

        # ---- EXTRACT METRICS ----
        # 1. Total Power and Losses (convert to MW)
        grid_import_mw = (dss_circuit.TotalPower[0] * -1) / 1000.0 
        losses_mw = dss_circuit.Losses[0] / 1000000.0 
        
        # 2. Key Bus Voltages (pu)
        def get_voltage(bus_name):
            dss_circuit.SetActiveBus(bus_name)
            return dss_circuit.ActiveBus.puVoltages[0] if len(dss_circuit.ActiveBus.puVoltages) > 0 else 0.0

        v_source = get_voltage("SourceBus")
        v_res    = get_voltage("634_LT")
        v_com    = get_voltage("684_LT")
        v_ind    = get_voltage("671_LT")
        v_crit   = get_voltage("692_LT")

        # 3. Component Loadings (%)
        def get_loading(element_class, element_name):
            dss_circuit.SetActiveElement(f"{element_class}.{element_name}")
            amps = dss_circuit.ActiveCktElement.CurrentsMagAng[0::2]
            nphases = dss_circuit.ActiveCktElement.NumPhases
            
            if amps is not None and np.size(amps) >= nphases:
                max_amps = float(np.max(amps[:nphases]))
            else:
                max_amps = 0.0
                
            rated = dss_circuit.ActiveCktElement.NormalAmps
            
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

        total_load_mw = res_mw + com_mw + ind_mw + crit_mw
        
        all_voltages = dss_circuit.AllBusVmagPu
        min_v = float(np.min(all_voltages)) if all_voltages is not None and np.size(all_voltages) > 0 else 0.0
        
        max_load = max([tr_res_load, tr_com_load, tr_ind_load, tr_crit_load, line_main])
        uv_flag = 1 if min_v < 0.95 else 0
        ol_flag = 1 if max_load > 100.0 else 0

        # Store to records combining old columns and the new LLM optimal ones
        results.append({
            "timestamp": timestamp,
            "total_load_MW": round(total_load_mw, 4),
            "solar_MW": round(pv_mw, 4),
            "wind_MW": round(wind_mw, 4),
            "grid_import_MW": round(grid_import_mw, 4),
            "losses_MW": round(losses_mw, 4),
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
        # Limit to printing once per day to not overload terminal or just a few
        if int(i) % 24 == 0 or i > END_ROW-5:
            print(f"{i:4d} | {total_load_mw:9.3f} | {grid_import_mw:9.3f} | {min_v:9.4f} | {max_load:10.1f}% | {uv_flag:7d} | {ol_flag:7d}")

    # 3. Export Data
    df_out = pd.DataFrame(results)
    
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opendss_kerala_output.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved full results to {out_file}")

if __name__ == "__main__":
    run_dataset_simulation()