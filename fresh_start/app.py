from flask import Flask, render_template, jsonify
import os
import pandas as pd
from gemini_summarizer import load_predictions, generate_llm_report

app = Flask(__name__)

# Ensure the template directory exists
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    predictions_file = "forecast_predictions.csv" 
    
    # Use dummy data if predictions aren't available matching the original script logic
    if not os.path.exists(predictions_file):
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

    try:
        df_pred = load_predictions(predictions_file)
        report_markdown = generate_llm_report(df_pred)
        return jsonify({"status": "success", "report": report_markdown})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
