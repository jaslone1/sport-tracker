import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent 
OUT_DIR = BASE_DIR / "ml_model" / "pytorch_training_winner"
# Points to the file created by your Feature Engineering script
FEATURES_DATA_PATH = BASE_DIR / "data" / "ml_ready_features.csv" 

MODEL_PATH = OUT_DIR / "winner_model.pth"
SCALER_PATH = OUT_DIR / "scaler.joblib"
FEATURE_COLUMNS_PATH = OUT_DIR / "feature_columns.json"

DEVICE = torch.device("cpu")

# --- MODEL DEFINITION ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_assets():
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        cols = json.load(f)
    scaler = joblib.load(SCALER_PATH)
    model = MLP(input_dim=len(cols))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load the latest stats for every team
    df = pd.read_csv(FEATURES_DATA_PATH)
    return model, scaler, cols, df

model, scaler, feature_cols, all_stats_df = load_assets()

# --- PREDICTION LOGIC ---
def get_prediction(home_team, away_team):
    # 1. Get the most recent rolling stats for both teams
    # We look at the very last entry in our features file for these teams
    home_stats = all_stats_df[all_stats_df['home_team'] == home_team].iloc[-1:]
    away_stats = all_stats_df[all_stats_df['away_team'] == away_team].iloc[-1:]
    
    if home_stats.empty or away_stats.empty:
        return None, "Team stats not found"

    # 2. Construct the feature row (Home Prev Stats vs Away Prev Stats)
    # We extract the 'home_prev_...' columns for the home team 
    # and the 'away_prev_...' columns for the away team.
    input_data = {}
    for col in feature_cols:
        if col.startswith("home_prev_"):
            input_data[col] = home_stats[col].values[0]
        elif col.startswith("away_prev_"):
            # We use the 'home_prev' data from the away_stats because that's 
            # how that team performed in their last game.
            generic_col = col.replace("away_prev_", "home_prev_")
            input_data[col] = away_stats[generic_col].values[0]
            
    input_df = pd.DataFrame([input_data])[feature_cols]
    
    # 3. Scale and Predict
    X_scaled = scaler.transform(input_df.values)
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32)).item()
        prob = 1.0 / (1.0 + np.exp(-logits))
    
    return prob, "Success"

# --- UI ---
st.title("🏈 CFB Advanced Stat Predictor")

teams = sorted(all_stats_df['home_team'].unique())
col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("Home Team", teams, index=0)
with col2:
    a_team = st.selectbox("Away Team", teams, index=1)

if st.button("Predict Outcome"):
    prob, status = get_prediction(h_team, a_team)
    if prob is not None:
        st.metric(f"{h_team} Win Probability", f"{prob:.1%}")
        st.progress(prob)
        if prob > 0.5:
            st.success(f"Prediction: {h_team} is favored to win.")
        else:
            st.warning(f"Prediction: {a_team} is favored to win.")
    else:
        st.error(f"Could not find recent stats for those teams.")
