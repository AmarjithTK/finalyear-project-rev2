"""
Utility functions for data handling, saving/loading models,
and writing structured outputs (JSON/CSV) to the outputs/ directory.
"""

import json
import pandas as pd
import os

OUTPUT_DIR = "outputs"

def save_predictions(df: pd.DataFrame, filename="predictions.csv"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/{filename}", index=False)
    print(f"Predictions saved to {OUTPUT_DIR}/{filename}")

def save_metrics(metrics_dict: dict, filename="metrics.json"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/{filename}", 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {OUTPUT_DIR}/{filename}")

def save_scenarios(scenarios_dict: dict, filename="scenarios.json"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/{filename}", 'w') as f:
        json.dump(scenarios_dict, f, indent=4)
    print(f"Scenarios saved to {OUTPUT_DIR}/{filename}")
