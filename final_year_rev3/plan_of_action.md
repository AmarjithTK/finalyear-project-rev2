# Plan of Action: Creating a Kerala 11kV Feeder Model in OpenDSS

This plan outlines the structured steps to build a customized, localized 11kV distribution feeder model representing typical Kerala State Electricity Board (KSEB) grid parameters, integrated with your machine learning dataset generation.

## Phase 1: Define Network Architecture & Data Collection
1. **Source Substation Requirements:** 110kV/11kV or 33kV/11kV source substation definition.
2. **Conductor Specs:** Gather line parameters (Resistance, Reactance, Ampacity) for standard Indian/KSEB AAC/ACSR conductors typically used in 11kV lines (e.g., *Rabbit*, *Raccoon*, *Dog*).
3. **Distribution Transformers (DTRs):** Typical 11kV / 433V (or 400V) step-down transformers (Ratings: 100kVA, 160kVA, 250kVA).
4. **Topology:** Design a radial or weakly meshed feeder backbone representing urban/semi-urban Kerala distribution.

## Phase 2: Create Modular OpenDSS Files
Instead of one massive file, split the model into logical components for better debugging and scaling.

* `Master_Kerala11kV.dss`: The main entry point. Compiles the circuit, redirects to other files, and defines simulation parameters.
* `Kerala_LineCodes.dss`: A clean library holding geometry and impedance metrics of KSEB conductors.
* `Kerala_Lines.dss`: Definitions of the feeder segments connecting nodes/buses with specific lengths.
* `Kerala_Transformers.dss`: Definitions of the DTRs supplying the loads.
* `Kerala_Loads.dss`: Bus assignments allocating kW/kVAr for specific sectors (Residential, Commercial, Industrial, Agricultural).
* `Kerala_LoadShapes.dss`: Defining the typical high evening-peak load curve (6 PM - 10 PM) unique to Kerala.

## Phase 3: Integrate Distributed Energy Resources (DERs)
1. **Solar PV Integration:** 
   - Define `PVSystem` elements to represent Kerala's rooftop solar initiatives (e.g., *Soura* project).
   - Attach a solar generation curve (normalized irradiance profile) using `Loadshape`.
2. **Battery Energy Storage (BESS):**
   - Define `Storage` elements attached to critical buses.
   - Set up standard charging/discharging states compatible with evening peaks.

## Phase 4: Time-Series Automation & Python Interfacing
Since you have machine learning models (LSTM/CNN-LSTM in `dataset_generator`), you need dynamic data.
1. Use `dss-python` (already available in `.venv`) to interface with `Master_Kerala11kV.dss`.
2. Run **QSTS (Quasi-Static Time Series)** power flow simulations.
3. Automatically export hourly metrics:
   - Node voltage profiles (to check for under/over voltages).
   - Line power flows and system losses.
   - Transformer loading percentages.
4. Format the outputs directly into `kerala_microgrid_hourly_dataset.csv` formats capable of training your existing pyTorch files (`train_lstm.py`, etc.).

## Next Immediate Steps
1. **Draft `Master_Kerala11kV.dss`** to define the `Vsource`.
2. **Draft `Kerala_LineCodes.dss`** converting Rabbit/Dog conductor specs into OpenDSS matrices.
3. Modify your existing `kerala13bus.py` to compile the new `Master_Kerala11kV.dss` rather than the old IEEE files.
