import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv

# --- 1. Configurations ---
# Loads environment variables from a .env file located in the same directory or project root
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "20"))
model = None
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

# Grid Thresholds for Physics Rules
V_MIN_THRESHOLD = 0.95
V_MAX_THRESHOLD = 1.05
LINE_LOADING_WARNING = 80.0
LINE_LOADING_CRITICAL = 95.0

def load_predictions(predictions_csv):
    """Loads the forecasted telemetry predictions CSV."""
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"{predictions_csv} not found.")
    df = pd.read_csv(predictions_csv)
    return df

def value_from(row, *columns, default=0.0):
    for column in columns:
        if column in row and not pd.isna(row[column]):
            return float(row[column])
    return default

def series_from(df, *columns, default=0.0):
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)

def analyze_grid_state(df_hour):
    """
    Analyzes a single hour of forecasted grid data against strict physics rules
    to generate a context block for the LLM.
    """
    issues = []
    
    # Check Voltages
    v_min = value_from(df_hour, "V_Min_pu")
    v_max = value_from(df_hour, "V_Max_pu")
    line_loading = value_from(df_hour, "Max_Line_Loading_pct")

    if v_min < V_MIN_THRESHOLD:
        issues.append(f"UNDERVOLTAGE: System minimum voltage dropped to {v_min:.3f} p.u. (Threshold: {V_MIN_THRESHOLD}).")
    if v_max > V_MAX_THRESHOLD:
        issues.append(f"OVERVOLTAGE: System maximum voltage reached {v_max:.3f} p.u. (Threshold: {V_MAX_THRESHOLD}).")
        
    # Check line loadings
    if line_loading > LINE_LOADING_CRITICAL:
        issues.append(f"CRITICAL BOTTLENECK: Line loading at {line_loading:.1f}%. Risk of thermal overload/failure.")
    elif line_loading > LINE_LOADING_WARNING:
        issues.append(f"WARNING: High line loading detected at {line_loading:.1f}%.")
        
    return issues

def generate_local_report(df_forecast):
    load = series_from(df_forecast, "Total_Load_Predicted_MW", "Total_Load_MW")
    der = series_from(df_forecast, "Total_DER_Predicted_MW")
    grid_import = series_from(df_forecast, "Grid_Import_MW")
    losses = series_from(df_forecast, "System_Loss_MW", "Total_Loss_MW")
    v_min = series_from(df_forecast, "V_Min_pu")
    v_max = series_from(df_forecast, "V_Max_pu")
    loading = series_from(df_forecast, "Max_Line_Loading_pct")

    critical_hours = []
    warning_hours = []
    for _, row in df_forecast.iterrows():
        issues = analyze_grid_state(row)
        if issues:
            timestamp = row.get("Timestamp", f"Hour {row.name}")
            target = critical_hours if any("UNDERVOLTAGE" in issue or "OVERVOLTAGE" in issue or "CRITICAL" in issue for issue in issues) else warning_hours
            target.append(f"- {timestamp}: " + " | ".join(issues))

    stability = "Stable forecast"
    if critical_hours:
        stability = "Needs attention"
    elif warning_hours:
        stability = "Watch list"

    peak_load_time = df_forecast.loc[load.idxmax(), "Timestamp"] if "Timestamp" in df_forecast.columns and len(load) else "N/A"
    peak_der_time = df_forecast.loc[der.idxmax(), "Timestamp"] if "Timestamp" in df_forecast.columns and len(der) else "N/A"

    report = f"""# AI Microgrid Summary

## Executive View
The {len(df_forecast)}-hour forecast is classified as **{stability}**. Average demand is **{load.mean():.2f} MW** with a peak of **{load.max():.2f} MW** at **{peak_load_time}**. DER output averages **{der.mean():.2f} MW** and peaks at **{der.max():.2f} MW** at **{peak_der_time}**.

## Operating Metrics
- Average grid import: **{grid_import.mean():.4f} MW**
- Maximum grid import: **{grid_import.max():.4f} MW**
- Cumulative system losses: **{losses.sum():.4f} MW**
- Voltage range: **{v_min.min():.5f} p.u.** to **{v_max.max():.5f} p.u.**
- Maximum line loading: **{loading.max():.2f}%**

## Rule Checks
"""
    if not critical_hours and not warning_hours:
        report += "No voltage or line-loading rule violations were detected under the configured thresholds.\n"
    else:
        report += "\n".join(critical_hours + warning_hours) + "\n"

    report += """
## Recommended Actions
- Verify the zero-valued bus and line-loading fields in the OpenDSS extraction path before using this result for final operational claims.
- Compare peak load hours with DER availability to decide whether grid import, storage discharge, or demand response should be scheduled.
- Re-run the load-flow after fixing any missing meter/line mappings so the AI summary can distinguish true stability from incomplete telemetry.
"""
    return report

def generate_llm_report(df_forecast):
    """
    Feeds the forecasted anomalies and general grid state into Gemini
    to generate an actionable grid operation report.
    """
    print("Analyzing predictions and generating prompt...")
    
    # We'll aggregate the worst issues over the forecasted period
    total_hours = len(df_forecast)
    critical_hours = []
    
    for _, row in df_forecast.iterrows():
        issues = analyze_grid_state(row)
        if issues:
            timestamp = row.get('Timestamp', f"Hour {row.get('Hour', 'Unknown')}")
            critical_hours.append(f"[{timestamp}] Issues: " + " | ".join(issues))
            
    # Extract overall metrics for context. Supports both the older ML forecast
    # columns and the newer OpenDSS predicted load-flow result columns.
    load = series_from(df_forecast, "Total_Load_Predicted_MW", "Total_Load_MW")
    der = series_from(df_forecast, "Total_DER_Predicted_MW")
    solar = series_from(df_forecast, "Solar_MW")
    wind = series_from(df_forecast, "Wind_MW")
    grid_import = series_from(df_forecast, "Grid_Import_MW")
    losses = series_from(df_forecast, "System_Loss_MW", "Total_Loss_MW")
    v_min = series_from(df_forecast, "V_Min_pu")
    v_max = series_from(df_forecast, "V_Max_pu")
    loading = series_from(df_forecast, "Max_Line_Loading_pct")
    
    prompt = f"""
You are an expert microgrid operator and automated grid controller. 
I am providing you with a forecasted power flow summary generated by a CNN-LSTM model for a modified IEEE 13-Bus system in Kerala over a {total_hours}-hour period.

**Grid Analytics Summary:**
- Average Load: {load.mean():.2f} MW (Peak: {load.max():.2f} MW)
- Average DER Generation: {der.mean():.2f} MW (Peak: {der.max():.2f} MW)
- Average Solar Gen: {solar.mean():.2f} MW
- Average Wind Gen: {wind.mean():.2f} MW
- Average Grid Import: {grid_import.mean():.4f} MW
- Cumulative Predicted System Losses: {losses.sum():.4f} MW
- Voltage Range: {v_min.min():.5f} p.u. to {v_max.max():.5f} p.u.
- Maximum Line Loading: {loading.max():.2f}%

**Critical Detected Rule Violations:**
"""
    if not critical_hours:
        prompt += "No physics rule violations detected. Grid operates within nominal bounds.\n"
    else:
        prompt += "\n".join(critical_hours) + "\n"

    prompt += """
**Your Task:**
1. Provide a concise executive summary of the forecasted grid stability.
2. If violations (undervoltage, overvoltage, overloading) are listed, identify the likely root causes based on the generation vs load balance.
3. Recommend actionable strategies (e.g., active/reactive power curtailment from DERs, load shedding, islanding feasibility) to mitigate these forecasted issues before they happen in real-time.
4. If telemetry appears incomplete, explicitly flag it and explain what should be verified before making operational claims.
"""
    
    if model is None:
        return generate_local_report(df_forecast)

    print("\nQuerying Gemini API...")
    try:
        response = model.generate_content(prompt, request_options={"timeout": GEMINI_TIMEOUT_SECONDS})
        print("\n" + "="*60)
        print("🤖 AI MICROGRID ANALYST REPORT")
        print("="*60)
        print(response.text)
        print("="*60)
        
        # Save report
        with open("gemini_prediction_report.md", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("\nReport saved to 'gemini_prediction_report.md'.")
        return response.text
        
    except Exception as e:
        error_msg = f"Failed to generate report from Gemini: {e}"
        print(error_msg)
        return f"**Error**: {error_msg}"

if __name__ == "__main__":
    # Ensure you have a predictions CSV file generated by your test/eval script
    predictions_file = "forecast_predictions.csv" 
    
    if not os.path.exists(predictions_file):
        print(f"Please ensure your ML script saves a '{predictions_file}' before running this summarizer.")
        print("Creating a dummy file for demonstration...")
        # Create a dummy for demonstration if real one doesn't exist
        dummy_data = pd.DataFrame({
            'Hour': [0, 1, 2],
            'Timestamp': ['2023-01-01 12:00:00', '2023-01-01 13:00:00', '2023-01-01 14:00:00'],
            'Total_Load_MW': [3.5, 3.8, 3.9],
            'Solar_MW': [0.8, 0.9, 0.4],
            'Wind_MW': [0.2, 0.1, 0.0],
            'V_Min_pu': [0.96, 0.94, 0.91], # Trigger undervoltage
            'V_Max_pu': [1.01, 1.01, 1.01],
            'Max_Line_Loading_pct': [75.0, 85.0, 98.0], # Trigger overload
            'Total_Loss_MW': [0.05, 0.08, 0.12]
        })
        dummy_data.to_csv(predictions_file, index=False)

    
    df_pred = load_predictions(predictions_file)
    generate_llm_report(df_pred)
