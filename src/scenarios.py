"""
Derives microgrid risk scenarios from model predictions.
"""

from typing import Dict, List

import pandas as pd

from src.utils import save_scenarios


def _detect_target_columns(df: pd.DataFrame) -> List[str]:
    return [col.replace("predicted_", "") for col in df.columns if col.startswith("predicted_")]


def _find_sudden_drops(series: pd.Series, threshold: float) -> pd.Series:
    pct_change = series.pct_change()
    return pct_change <= -threshold


def _find_sudden_surges(series: pd.Series, threshold: float) -> pd.Series:
    pct_change = series.pct_change()
    return pct_change >= threshold


def _summarize_events(mask: pd.Series, timestamps: pd.Series, max_samples: int = 5) -> Dict:
    indices = mask[mask].index
    sample_times = timestamps.loc[indices].astype(str).head(max_samples).tolist()
    return {
        "count": int(mask.sum()),
        "sample_timestamps": sample_times,
    }


def run_scenarios() -> None:
    print("Running risk scenarios on generated predictions...")

    try:
        preds = pd.read_csv("outputs/predictions.csv")
    except FileNotFoundError:
        print("Missing outputs/predictions.csv. Run predict.py first.")
        return

    if "timestamp" not in preds.columns:
        raise ValueError("predictions.csv must include a timestamp column")

    preds["timestamp"] = pd.to_datetime(preds["timestamp"])
    target_columns = _detect_target_columns(preds)

    scenarios_summary: Dict[str, Dict] = {}

    for target in target_columns:
        pred_col = f"predicted_{target}"
        if pred_col not in preds.columns:
            continue

        series = preds[pred_col].astype(float)

        if any(key in target.lower() for key in ["solar", "wind", "renewable", "energy"]):
            drop_mask = _find_sudden_drops(series, threshold=0.5)
            scenarios_summary[f"Scenario_Sudden_Drop_{target}"] = {
                "description": "50%+ one-step drop in predicted output.",
                "impact_assessment": "High" if drop_mask.any() else "Low",
                "battery_intervention_required": bool(drop_mask.any()),
                **_summarize_events(drop_mask, preds["timestamp"]),
            }

        if "load" in target.lower():
            surge_mask = _find_sudden_surges(series, threshold=0.2)
            scenarios_summary[f"Scenario_Peak_Surge_{target}"] = {
                "description": "20%+ one-step surge in predicted load.",
                "impact_assessment": "Medium" if surge_mask.any() else "Low",
                "battery_intervention_required": bool(surge_mask.any()),
                **_summarize_events(surge_mask, preds["timestamp"]),
            }

    if not scenarios_summary:
        scenarios_summary["Scenario_None"] = {
            "description": "No scenario rules matched available prediction columns.",
            "impact_assessment": "N/A",
            "battery_intervention_required": False,
        }

    save_scenarios(scenarios_summary)


if __name__ == "__main__":
    run_scenarios()
