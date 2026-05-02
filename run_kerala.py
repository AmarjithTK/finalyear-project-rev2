import dss

# Initialize OpenDSS
dss_engine = dss.DSS
dss_text = dss_engine.Text
dss_circuit = dss_engine.ActiveCircuit

# Compile the Master file
dss_text.Command = "Compile [Master_Kerala11kV.dss]"

# Solve Snapshot
dss_text.Command = "Solve"

# Check for convergence
if dss_circuit.Solution.Converged:
    print("Circuit Solved Successfully!")
    print(f"Total Active Power: {dss_circuit.TotalPower[0] * -1:.2f} kW")
    print(f"Total Losses: {dss_circuit.Losses[0] / 1000:.2f} kW")
else:
    print("Circuit did NOT converge!")
