# run_simulation.py

import opendssdirect as dss
import pandas as pd
import matplotlib.pyplot as plt

# Load DSS model
dss.Text.Command("Clear")
dss.Text.Command("Compile dss file.dss")

# Solve once
dss.Text.Command("Solve")

# Get all bus names
print("Buses:", dss.Circuit.AllBusNames())

# Get voltage magnitudes pu
voltages = dss.Circuit.AllBusMagPu()
print("Voltages:", voltages)

# Total losses (watts, vars)
print("Losses:", dss.Circuit.Losses())