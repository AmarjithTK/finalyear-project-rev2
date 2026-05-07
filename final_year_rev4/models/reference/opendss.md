# OpenDSS: Power Flow, Kerala Feeder, and Viva Notes

OpenDSS (Open Distribution System Simulator) is an electric distribution system simulation tool used for power flow, loss analysis, voltage profile studies, and DER (distributed energy resources) integration. It is widely used for unbalanced, multi-phase distribution networks and time-series simulations.

## What OpenDSS Is

- **Distribution system simulator**: Built for feeders, transformers, loads, and DERs at distribution voltage levels.
- **Power flow engine**: Solves bus voltages, branch currents, power flows, and losses under given loading and generation.
- **Time-series capable**: Supports daily and yearly load shapes and PV/wind profiles.
- **Scriptable**: Uses .dss files for models and Python interfaces (opendssdirect or COM) for automation.

## Uses and Benefits

- **Voltage regulation studies**: Check undervoltage and overvoltage across the feeder.
- **Thermal loading**: Detect line and transformer overloads.
- **Loss analysis**: Quantify feeder losses under different loading and DER scenarios.
- **DER integration**: Model PV and wind, evaluate voltage rise and reverse power flow.
- **Planning**: Evaluate feeder upgrades, capacitor placement, and load growth.

## Power Flow Basics

Power flow solves the steady-state operating point of a network. Given topology, impedances, and power injections, it computes:

- **Bus voltages** (magnitude and angle)
- **Branch currents** and power flows
- **System losses**

Mathematically, it solves nonlinear equations of the form:
$$S_i = V_i I_i^* = V_i \left(\sum_{j=1}^{n} Y_{ij} V_j\right)^*$$
where $S_i = P_i + jQ_i$ and $Y_{ij}$ are elements of the network admittance matrix.

## Gauss-Seidel Power Flow (Concept)

Gauss-Seidel is an iterative method to update bus voltages one at a time using the latest available estimates. A common update form is:
$$V_i^{(k+1)} = \frac{1}{Y_{ii}} \left( \frac{P_i - jQ_i}{(V_i^{(k)})^*} - \sum_{j \ne i} Y_{ij} V_j^{(k+1,k)} \right)$$
It is simple to implement but can be slow for large or ill-conditioned systems. OpenDSS also supports more advanced solvers for speed and robustness.

## Safe Voltage Limits and Loading

- **Typical safe voltage range**: 0.95 to 1.05 pu for normal operation.
- **Project monitoring**: We flag undervoltage if $V_{min} < 0.95$ pu in scripts.
- **Load model limits**: Some Kerala loads use $Vminpu=0.85$ and $Vmaxpu=1.15$ for modeling, but planning limits are tighter.
- **Loading**: Percent loading is computed as:
$$\text{Loading (percent)} = 100 \times \frac{I_{max}}{I_{rated}}$$
For transformers, $I_{rated}$ is derived from kVA and kV ratings.

## Kerala Feeder Specs (From .dss Files)

### A) KeralaIEEE13.dss (Predicted Loadflow Model)

- **Base**: 33 kV source, 33/11 kV transformer rated 5 MVA, 50 Hz.
- **Voltage bases**: 33 kV, 11 kV, 0.433 kV.
- **Linecodes**:
  - KLine_HV: r1=0.32, x1=0.38, r0=0.75, x0=1.10 (ohm per km)
  - KLine_MV: r1=0.50, x1=0.45, r0=0.90, x0=1.20 (ohm per km)
- **Backbone lines**: 650-632-671-675-680 and 671-684-692-634 segments.
- **Transformers**: DT_Res (1500 kVA), DT_Com (1000 kVA), DT_Ind (1500 kVA), DT_Crit (500 kVA) with 11 kV / 0.433 kV.
- **Loads**: Residential, Commercial, Industrial, Critical (3 single-phase loads each, wye-connected).
- **DERs**: Solar PV 1.2 MW at bus 675; Wind 0.8 MW at bus 680.

### B) KeralaIEEE13Nodeckt.dss (Modified IEEE13 Topology)

- **Base**: 33 kV source, 33/11 kV transformer rated 10 MVA.
- **Linecodes**: K11 (3-phase OH line) and K11_1ph (1-phase spur).
- **Feeders**: 650-632-670-671-680 chain, plus single-phase spurs to 645, 646, 684, 611, 652.
- **Transformers**: 500 kVA at 633, 250 kVA at 675, 100 kVA at 646.
- **Loads**: LV residential, commercial, and rural at 0.24 kV; 11 kV single-phase loads at 645, 611, 652.
- **Capacitors**: 300 kvar at 675 and 50 kvar at 611.3.
- **PV**: 150 kVA at bus 633.

### C) Kerala11kV Modular Feeder (Master_Kerala11kV.dss)

- **Base**: 11 kV source, 50 Hz, short-circuit MVA defined at the source.
- **Redirects**: Kerala_LineCodes, Kerala_Lines, Kerala_Transformers, Kerala_LoadShapes, Kerala_Loads, Kerala_DERs.
- **Linecodes**: Rabbit, Raccoon, Dog, LT_ABC_50 with typical Indian conductor parameters.
- **Transformers**: TR_Res634 100 kVA, TR_Ind671 500 kVA, TR_Com684 250 kVA, TR_Crit692 160 kVA.
- **Loads**: Res 80 kW, Ind 400 kW, Com 200 kW, Crit 120 kW with daily load shapes.
- **DERs**: PV_675 200 kW and Wind_680 250 kW with daily profiles.

## How Our System Uses OpenDSS

- **Predicted loadflow**: `run_opendss_predicted.py` compiles KeralaIEEE13.dss, updates Res/Com/Ind/Crit loads, updates PV and wind, runs the power flow, and logs voltages, line loading, losses, and grid import for each predicted hour.
- **Historical simulation**: `opendss_kerala.py` compiles Master_Kerala11kV.dss, injects measured dataset values, solves the circuit, and writes voltage and loading metrics to CSV.
- **Daily profile simulation**: `simulator.py` uses predefined 24-hour shapes and runs a full-day power flow trace.
- **Quick check**: `kerala13bus.py` compiles a DSS file, solves, and prints bus voltages and line flows.

## Viva Questions

1. **What is OpenDSS and why is it used in distribution systems?**
2. **What does a power flow study compute?**
3. **Why is power flow essential before deploying DERs?**
4. **What is the difference between transmission and distribution power flow models?**
5. **What are the typical outputs of OpenDSS after a solve?**
6. **What is the role of the admittance matrix $Y_{bus}$ in power flow?**
7. **What is Gauss-Seidel power flow and why is it iterative?**
8. **Write the Gauss-Seidel voltage update equation for a PQ bus.**
9. **What are the advantages and disadvantages of Gauss-Seidel?**
10. **What is the meaning of per-unit (pu) voltage?**
11. **What is considered a safe operating voltage range in distribution systems?**
12. **How do we flag undervoltage in our scripts?**
13. **What is line or transformer loading, and how is it computed?**
14. **What does 120 percent loading indicate?**
15. **What is reverse power flow and when does it occur?**
16. **What is the difference between kW and kVA in OpenDSS results?**
17. **Why does OpenDSS need linecodes?**
18. **What is the difference between a linecode and a line element?**
19. **What is a load shape and why is it important?**
20. **What is the purpose of capacitors in the feeder?**
21. **What is the main function of a distribution transformer in OpenDSS models?**
22. **Why are loads often connected at 0.433 kV in Kerala models?**
23. **What is the difference between wye and delta connections?**
24. **Why do we model three single-phase loads for a three-phase LV bus?**
25. **What is meant by a bus in OpenDSS?**
26. **What happens if we try to plan a feeder without power flow analysis?**
27. **How does OpenDSS handle unbalanced loads?**
28. **What is the purpose of `CalcVoltageBases`?**
29. **Why is base frequency set to 50 Hz in Kerala models?**
30. **Which buses host the major DERs in KeralaIEEE13.dss?**
31. **What is the difference between KeralaIEEE13.dss and KeralaIEEE13Nodeckt.dss?**
32. **Which file defines the Kerala11kV feeder topology?**
33. **What does `Solve Mode=Snap` do?**
34. **Why do we monitor both voltage and loading in each run?**
35. **What are system losses in OpenDSS and why do they matter?**
36. **What is grid import in our context?**
37. **How is maximum line loading computed in the predicted-loadflow script?**
38. **What does `NormalAmps` represent and why is it used?**
39. **Why do we divide loads across phases in KeralaIEEE13.dss?**
40. **What is the difference between PVSystem and Generator elements?**
41. **Why is OpenDSS good for distribution studies compared to a generic solver?**
42. **What is the main advantage of using OpenDSS with Python?**
43. **What is voltage regulation, and how do capacitors help?**
44. **What is the impact of high R/X ratio in distribution lines?**
45. **What is the purpose of short-circuit MVA settings in a circuit?**
46. **How does a load model use `Vminpu` and `Vmaxpu`?**
47. **What is a feeder backbone and why is it critical?**
48. **What is the difference between losses in W and losses in MW in our reports?**
49. **How does OpenDSS report bus voltages and angles?**
50. **What is the practical meaning of convergence in a power flow solve?**

## Answers

**1. What is OpenDSS and why is it used in distribution systems?**
OpenDSS is a distribution system simulator that solves power flow for feeders, transformers, loads, and DERs, especially for unbalanced and time-varying systems.

**2. What does a power flow study compute?**
It computes steady-state bus voltages, line currents, power flows, and system losses for given loading and generation.

**3. Why is power flow essential before deploying DERs?**
It shows whether voltage rise, reverse power flow, or thermal overloads will occur when DERs are connected.

**4. What is the difference between transmission and distribution power flow models?**
Distribution models are often radial, unbalanced, and have higher R/X ratios, while transmission models are typically meshed and more balanced.

**5. What are the typical outputs of OpenDSS after a solve?**
Bus voltages, branch currents, element power flows, losses, and convergence status.

**6. What is the role of the admittance matrix $Y_{bus}$ in power flow?**
It relates bus voltages to injected currents and is the core of the power flow equations.

**7. What is Gauss-Seidel power flow and why is it iterative?**
It updates bus voltages one by one using previous estimates, repeating until changes are small.

**8. Write the Gauss-Seidel voltage update equation for a PQ bus.**
$$V_i^{(k+1)} = \frac{1}{Y_{ii}} \left( \frac{P_i - jQ_i}{(V_i^{(k)})^*} - \sum_{j \ne i} Y_{ij} V_j^{(k+1,k)} \right)$$

**9. What are the advantages and disadvantages of Gauss-Seidel?**
It is simple and low memory, but can be slow to converge for large or ill-conditioned systems.

**10. What is the meaning of per-unit (pu) voltage?**
It is voltage normalized by a base value, making comparisons across voltage levels easier.

**11. What is considered a safe operating voltage range in distribution systems?**
Typically 0.95 to 1.05 pu, although device limits may allow wider bands.

**12. How do we flag undervoltage in our scripts?**
We set an undervoltage flag if the minimum bus voltage is below 0.95 pu.

**13. What is line or transformer loading, and how is it computed?**
It is the percent of rated current used, computed as $100 \times I_{max} / I_{rated}$.

**14. What does 120 percent loading indicate?**
The element is overloaded beyond its thermal rating and may overheat or fail.

**15. What is reverse power flow and when does it occur?**
Power flows from downstream to upstream when local generation exceeds local load.

**16. What is the difference between kW and kVA in OpenDSS results?**
kW is real power, kVA includes real and reactive power magnitude.

**17. Why does OpenDSS need linecodes?**
Linecodes define electrical parameters (resistance, reactance, capacitance) used by line elements.

**18. What is the difference between a linecode and a line element?**
A linecode defines conductor parameters; a line element applies them to a specific bus-to-bus segment.

**19. What is a load shape and why is it important?**
It is a time series multiplier that models how load varies across the day.

**20. What is the purpose of capacitors in the feeder?**
They supply reactive power locally to improve voltage and reduce losses.

**21. What is the main function of a distribution transformer in OpenDSS models?**
It steps voltage down (11 kV to 0.433 kV) for LV loads and defines impedance between levels.

**22. Why are loads often connected at 0.433 kV in Kerala models?**
That is the standard 415/240 V three-phase LV distribution level in India.

**23. What is the difference between wye and delta connections?**
Wye has a neutral point and supports phase-to-neutral loads; delta does not and has different fault behavior.

**24. Why do we model three single-phase loads for a three-phase LV bus?**
It represents phase-wise allocation and allows unbalance modeling.

**25. What is meant by a bus in OpenDSS?**
A bus is a node where elements connect and voltages are calculated.

**26. What happens if we try to plan a feeder without power flow analysis?**
We would not detect voltage violations, overloads, losses, or reverse power flow risks.

**27. How does OpenDSS handle unbalanced loads?**
It models each phase separately, solving a full unbalanced three-phase power flow.

**28. What is the purpose of `CalcVoltageBases`?**
It computes per-unit bases for each voltage level so voltage results are normalized.

**29. Why is base frequency set to 50 Hz in Kerala models?**
Because the Indian grid operates at 50 Hz.

**30. Which buses host the major DERs in KeralaIEEE13.dss?**
Solar PV is at bus 675 and wind generation is at bus 680.

**31. What is the difference between KeralaIEEE13.dss and KeralaIEEE13Nodeckt.dss?**
KeralaIEEE13 is a simplified predicted-loadflow model, while KeralaIEEE13Nodeckt retains a modified IEEE13 topology with additional spurs, capacitors, and LV loads.

**32. Which file defines the Kerala11kV feeder topology?**
Kerala_Lines.dss (redirected by Master_Kerala11kV.dss).

**33. What does `Solve Mode=Snap` do?**
It solves a single steady-state snapshot of the circuit at that time step.

**34. Why do we monitor both voltage and loading in each run?**
Voltage ensures power quality and safety, while loading ensures thermal limits are not exceeded.

**35. What are system losses in OpenDSS and why do they matter?**
They are real and reactive power lost in lines and transformers, affecting efficiency and cost.

**36. What is grid import in our context?**
The net power drawn from the upstream grid after accounting for local DER generation.

**37. How is maximum line loading computed in the predicted-loadflow script?**
It computes loading for key lines and takes the maximum percent loading among them.

**38. What does `NormalAmps` represent and why is it used?**
It is the rated current of a line element, used to compute percent loading.

**39. Why do we divide loads across phases in KeralaIEEE13.dss?**
To model a three-phase LV system realistically and allow phase-wise load allocation.

**40. What is the difference between PVSystem and Generator elements?**
PVSystem models inverter-based PV with irradiance and power limits; Generator is a generic source with specified kW and kvar.

**41. Why is OpenDSS good for distribution studies compared to a generic solver?**
It is optimized for unbalanced, radial feeders with detailed equipment models and time-series support.

**42. What is the main advantage of using OpenDSS with Python?**
Automation: batch runs, data integration, and result extraction for ML pipelines.

**43. What is voltage regulation, and how do capacitors help?**
Voltage regulation is maintaining voltages within limits; capacitors supply reactive power to boost voltage.

**44. What is the impact of high R/X ratio in distribution lines?**
Voltage drops are more sensitive to real power flow, making voltage control harder.

**45. What is the purpose of short-circuit MVA settings in a circuit?**
They define source strength and impact fault currents and voltage profiles.

**46. How does a load model use `Vminpu` and `Vmaxpu`?**
It defines voltage ranges where the load remains active or begins to drop off.

**47. What is a feeder backbone and why is it critical?**
It is the main trunk line carrying bulk power; its loading and voltage determine feeder performance.

**48. What is the difference between losses in W and losses in MW in our reports?**
OpenDSS returns losses in W; scripts convert to MW for reporting.

**49. How does OpenDSS report bus voltages and angles?**
It provides per-unit magnitudes and phase angles for each bus phase.

**50. What is the practical meaning of convergence in a power flow solve?**
The iterative solver has found a stable solution that satisfies network equations within tolerance.
