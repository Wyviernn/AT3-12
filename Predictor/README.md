# Multi-Subject Academic Predictor

Usage:

Install requirements:

```bash
pip install -r Predictor/requirements.txt
```

Run the predictor (uses `Data/data.csv` and predicts all missing values):

```bash
python3 Predictor/run_final.py
```

Outputs:
- `Predictor/PREDICTIONS.txt` — formatted predictions for all students with missing values
- Model evaluation printed to console (RMSE and cross-validation R² per column)
