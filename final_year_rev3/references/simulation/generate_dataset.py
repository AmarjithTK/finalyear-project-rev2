import opendssdirect as dss
import pandas as pd
import numpy as np
from pathlib import Path

# Set base path
base_dir = Path(__file__).parent.absolute()
file_path = base_dir / "IEEE13Nodeckt.dss"

# Verify file exists
if not file_path.exists():
    raise FileNotFoundError(f"DSS file not found at: {file_path}")

print(f"Loading DSS file from: {file_path}")

# Compile IEEE feeder
dss.Text.Command(f'compile "{file_path}"')

rows = []

# Get all buses once
buses = dss.Circuit.AllBusNames()
print(f"Total buses found: {len(buses)}")

# Simulation parameters
num_days = 30
num_hours = 24
np.random.seed(42)  # For reproducibility

print("Starting simulation...")

for day in range(1, num_days + 1):
    for hour in range(num_hours):
        
        # Generate random load multiplier
        load_mult = np.random.uniform(0.7, 1.3)
        
        # Set DSS commands
        dss.Text.Command(f"set loadmult={load_mult}")
        dss.Text.Command(f"set hour={hour}")
        dss.Text.Command("solve")
        
        # Get total circuit power
        total_power = dss.Circuit.TotalPower()
        total_kw = abs(total_power[0])
        total_kvar = abs(total_power[1])
        
        # Iterate through each bus
        for bus in buses:
            try:
                dss.Circuit.SetActiveBus(bus)
                
                # Get voltage magnitude in per unit
                v_mag_angle = dss.Bus.puVmagAngle()
                voltage_pu = v_mag_angle[0] if len(v_mag_angle) > 0 else 1.0
                
                # Get bus voltage angle (in degrees)
                voltage_angle = v_mag_angle[1] if len(v_mag_angle) > 1 else 0.0
                
                # Get number of nodes
                num_nodes = dss.Bus.NumNodes()
                
                # Get bus power
                try:
                    bus_powers = dss.Bus.Powers()
                    bus_kw = abs(bus_powers[0]) if len(bus_powers) > 0 else 0.0
                    bus_kvar = abs(bus_powers[1]) if len(bus_powers) > 1 else 0.0
                except:
                    bus_kw = 0.0
                    bus_kvar = 0.0
                
                rows.append({
                    "day": day,
                    "hour": hour,
                    "bus": bus,
                    "load_mult": round(load_mult, 4),
                    "voltage_pu": round(voltage_pu, 6),
                    "voltage_angle_deg": round(voltage_angle, 2),
                    "bus_kw": round(bus_kw, 2),
                    "bus_kvar": round(bus_kvar, 2),
                    "total_kw": round(total_kw, 2),
                    "total_kvar": round(total_kvar, 2),
                    "num_nodes": num_nodes
                })
            except Exception as e:
                print(f"Error processing bus {bus}: {e}")
                continue
    
    if day % 5 == 0:
        print(f"Completed {day}/{num_days} days...")

# Create DataFrame
df = pd.DataFrame(rows)

# Create data directory if it doesn't exist
data_dir = base_dir.parent / "data"
data_dir.mkdir(exist_ok=True)

# Save to CSV
output_file = data_dir / "ieee13_dataset.csv"
df.to_csv(output_file, index=False)

print(f"\n✓ Dataset created successfully!")
print(f"✓ Total records: {len(df)}")
print(f"✓ Saved to: {output_file}")
print(f"\nDataset Statistics:")
print(df.describe())