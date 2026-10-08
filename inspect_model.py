import joblib
import pandas as pd
import numpy as np

model_path = "models/ncaa_model.pkl"
model = joblib.load(model_path)

# Mock input data based on what's used in app.py
# FEATURE_COLS:
# "neutral_site",
# "h_roll_pts_scored", "h_roll_ypp", "h_roll_ppm", "h_roll_turnovers", "h_sos",
# "a_roll_pts_scored", "a_roll_ypp", "a_roll_ppm", "a_roll_turnovers", "a_sos"

# Just use dummy data to test
input_data = pd.DataFrame([{
    "neutral_site": 0,
    "h_roll_pts_scored": 30.0,
    "h_roll_ypp": 6.0,
    "h_roll_ppm": 0.5,
    "h_roll_turnovers": 1.0,
    "h_sos": 20.0,
    "a_roll_pts_scored": 20.0,
    "a_roll_ypp": 5.0,
    "a_roll_ppm": 0.4,
    "a_roll_turnovers": 2.0,
    "a_sos": 25.0,
}])

if isinstance(model, dict):
    winner_model = model["winner"]
else:
    winner_model = model

print(f"Model type: {type(winner_model)}")
if hasattr(winner_model, "predict_proba"):
    probabilities = winner_model.predict_proba(input_data)[0]
    print(f"Probabilities: {probabilities}")
