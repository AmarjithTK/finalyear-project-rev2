from flask import Flask, render_template, jsonify
import os
import pandas as pd
from gemini_summarizer import load_predictions, generate_llm_report

app = Flask(__name__)

# Ensure the template directory exists
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOADFLOW_RESULTS_CSV = os.path.abspath(
    os.path.join(BASE_DIR, "..", "proposed_model_results", "predicted_loadflow_results.csv")
)

DISPLAY_COLUMNS = [
    "Timestamp",
    "Total_Load_Predicted_MW",
    "Total_DER_Predicted_MW",
    "Grid_Import_MW",
    "System_Loss_MW",
    "V_Min_pu",
    "V_Max_pu",
    "Max_Line_Loading_pct",
    "V_Sub_650_pu",
    "V_Split_632_pu",
    "V_Res_634_pu",
    "V_Ind_671_pu",
    "L_Main_650_632_pct",
    "L_Ind_632_671_pct",
    "L_Solar_632_633_pct",
]

V_MIN_THRESHOLD = 0.95
V_MAX_THRESHOLD = 1.05
LINE_LOADING_WARNING = 80.0
LINE_LOADING_CRITICAL = 95.0


def _read_loadflow_results():
    if not os.path.exists(LOADFLOW_RESULTS_CSV):
        raise FileNotFoundError(f"Load-flow results not found: {LOADFLOW_RESULTS_CSV}")
    return pd.read_csv(LOADFLOW_RESULTS_CSV)


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value, precision=4):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return round(float(value), precision)
    return value


def _status_for_row(row):
    v_min = _safe_float(row.get("V_Min_pu"))
    v_max = _safe_float(row.get("V_Max_pu"))
    loading = _safe_float(row.get("Max_Line_Loading_pct"))

    if v_min < V_MIN_THRESHOLD or v_max > V_MAX_THRESHOLD or loading >= LINE_LOADING_CRITICAL:
        return "Critical"
    if loading >= LINE_LOADING_WARNING:
        return "Warning"
    return "Normal"


def _build_summary(df):
    total_rows = len(df)
    if total_rows == 0:
        return {
            "hours": 0,
            "avg_load_mw": 0,
            "peak_load_mw": 0,
            "avg_der_mw": 0,
            "peak_der_mw": 0,
            "avg_grid_import_mw": 0,
            "max_grid_import_mw": 0,
            "total_loss_mw": 0,
            "min_voltage_pu": 0,
            "max_voltage_pu": 0,
            "max_line_loading_pct": 0,
            "normal_hours": 0,
            "warning_hours": 0,
            "critical_hours": 0,
            "health": "No data",
            "insight": "Run the predicted load-flow script to populate the dashboard.",
        }

    statuses = df.apply(_status_for_row, axis=1)
    min_voltage = _safe_float(df["V_Min_pu"].min())
    max_voltage = _safe_float(df["V_Max_pu"].max())
    max_loading = _safe_float(df["Max_Line_Loading_pct"].max())
    peak_load_idx = df["Total_Load_Predicted_MW"].idxmax()
    peak_der_idx = df["Total_DER_Predicted_MW"].idxmax()
    critical_count = int((statuses == "Critical").sum())
    warning_count = int((statuses == "Warning").sum())

    if critical_count:
        health = "Needs attention"
    elif warning_count:
        health = "Watch list"
    else:
        health = "Stable forecast"

    insight = (
        f"Peak load occurs at {df.loc[peak_load_idx, 'Timestamp']} while DER output peaks at "
        f"{df.loc[peak_der_idx, 'Timestamp']}. Voltage and line-loading checks currently classify "
        f"{total_rows - warning_count - critical_count} of {total_rows} hours as normal."
    )

    return {
        "hours": int(total_rows),
        "avg_load_mw": round(_safe_float(df["Total_Load_Predicted_MW"].mean()), 3),
        "peak_load_mw": round(_safe_float(df["Total_Load_Predicted_MW"].max()), 3),
        "avg_der_mw": round(_safe_float(df["Total_DER_Predicted_MW"].mean()), 3),
        "peak_der_mw": round(_safe_float(df["Total_DER_Predicted_MW"].max()), 3),
        "avg_grid_import_mw": round(_safe_float(df["Grid_Import_MW"].mean()), 4),
        "max_grid_import_mw": round(_safe_float(df["Grid_Import_MW"].max()), 4),
        "total_loss_mw": round(_safe_float(df["System_Loss_MW"].sum()), 4),
        "min_voltage_pu": round(min_voltage, 5),
        "max_voltage_pu": round(max_voltage, 5),
        "max_line_loading_pct": round(max_loading, 2),
        "normal_hours": int((statuses == "Normal").sum()),
        "warning_hours": warning_count,
        "critical_hours": critical_count,
        "health": health,
        "insight": insight,
    }


def _table_rows(df):
    rows = []
    for _, row in df.iterrows():
        row_data = {column: _fmt(row.get(column)) for column in DISPLAY_COLUMNS if column in df.columns}
        row_data["Status"] = _status_for_row(row)
        row_data["Net_Load_MW"] = round(
            _safe_float(row.get("Total_Load_Predicted_MW")) - _safe_float(row.get("Total_DER_Predicted_MW")),
            4,
        )
        rows.append(row_data)
    return rows

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/loadflow_results')
def loadflow_results():
    try:
        df = _read_loadflow_results()
        return jsonify({
            "status": "success",
            "source": os.path.relpath(LOADFLOW_RESULTS_CSV, os.path.dirname(BASE_DIR)),
            "columns": ["Status", "Net_Load_MW"] + [c for c in DISPLAY_COLUMNS if c in df.columns],
            "summary": _build_summary(df),
            "rows": _table_rows(df),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        df_pred = load_predictions(LOADFLOW_RESULTS_CSV)
        report_markdown = generate_llm_report(df_pred)
        return jsonify({"status": "success", "report": report_markdown})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
