## Run In Google Colab

This project is designed to run in Colab with a clear, repeatable sequence.

The forecasting pipeline uses a CNN-LSTM-XGBoost hybrid model for day-ahead
microgrid forecasting. The model predicts exactly three targets:

- `grid_load_demand`
- `solar_pv_output`
- `wind_power_output`

`total_predicted_energy` is derived after inference as:

```text
predicted_solar_pv_output + predicted_wind_power_output
```

These outputs are prepared for later mapping into a modified IEEE 13-bus
microgrid model for hourly load-flow analysis.

### Recommended Sequence
1. Open `main.ipynb` in Colab.
2. Run Cell 1. It auto-detects Colab, clones the repo into `/content/finalyear-project-rev2`, and sets `basedir`.
3. (Optional) Install dependencies with:
	- `!pip install -r {basedir}requirements.txt`
4. Run the training pipeline cell to execute:
	- `python -m src.train`
	- `python -m src.predict`
	- `python -m src.scenarios`
5. Run the outputs cell to load `predictions.csv`, `metrics.json`, and `scenarios.json`.
6. Run the visualization and scenario reporting cells.

### Notes
- The execution flow is intentionally script-first: `.py` files do the heavy work, `main.ipynb` only displays results.
- If you change datasets or columns, update the config in `src/utils.py` before re-running.
