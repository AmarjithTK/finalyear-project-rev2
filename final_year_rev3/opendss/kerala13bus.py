from opendssdirect import dss

# 1. Clear anything already loaded
dss.Basic.ClearAll()

# 2. Compile the DSS circuit
dss(f'Compile "kerala13bus.dss"')

# 3. Solve power flow
dss.Solution.Solve()

# 4. Check convergence
print("Converged:", dss.Solution.Converged())

# 5. Circuit-level results
print("Total circuit power (kW, kvar):", dss.Circuit.TotalPower())
print("Total losses (W, var):", dss.Circuit.Losses())

# 6. Bus names
bus_names = dss.Circuit.AllBusNames()
print("Number of buses:", len(bus_names))
print("Buses:", bus_names)

# 7. Per-bus voltage magnitudes in pu
for bus in bus_names:
    dss.Circuit.SetActiveBus(bus)
    vmag_pu = dss.Bus.puVmagAngle()   # [V1_pu, angle1, V2_pu, angle2, ...]
    print(f"\nBus: {bus}")
    print("pu voltage magnitudes and angles:", vmag_pu)

# 8. Line results
line_names = dss.Lines.AllNames()
for line in line_names:
    dss.Lines.Name(line)
    currents = dss.CktElement.CurrentsMagAng()
    powers = dss.CktElement.Powers()
    print(f"\nLine: {line}")
    print("Currents (mag, angle):", currents)
    print("Powers (kW, kvar by conductor/terminal):", powers)