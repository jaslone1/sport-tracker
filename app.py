import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent 
# Updated to match the model saved by train_model.py
MODEL_PATH = BASE_DIR / "models" / "ncaa_model.pkl"
FEATURES_DATA_PATH = BASE_DIR / "data" / "ml_ready_features.csv" 

@st.cache_resource
def load_assets():
    # Load the Random Forest model
    model = joblib.load(MODEL_PATH)
    
    # Load the features file to get team names and latest stats
    df = pd.read_csv(FEATURES_DATA_PATH)
    
    # Standardize team names for the dropdown
    df['home_team'] = df['home_team'].astype(str).str.strip()
    df['away_team'] = df['away_team'].astype(str).str.strip()
    
    return model, df

model, all_stats_df = load_assets()

# --- PREDICTION LOGIC ---
def get_prediction(home_team, away_team):
    # 1. Get stats
    home_latest = all_stats_df[all_stats_df['home_team'] == home_team].iloc[-1:]
    away_latest = all_stats_df[all_stats_df['away_team'] == away_team].iloc[-1:]
    
    if home_latest.empty or away_latest.empty:
        return None, "Team stats not found"

    # 2. Extract stats
    h_pts = float(home_latest['h_avg_pts'].values[0])
    a_pts = float(away_latest['a_avg_pts'].values[0])
    
    # 3. Create a DataFrame with the SAME column names used in training
    # This prevents Scikit-Learn 'feature name' warnings/errors
    input_df = pd.DataFrame([[h_pts, a_pts]], columns=['h_avg_pts', 'a_avg_pts'])
    
    # 4. Predict
    try:
        # predict_proba returns [[prob_0, prob_1]]
        prob = model.predict_proba(input_df)[0][1]
        return prob, "Success"
    except Exception as e:
        return None, f"Prediction Error: {str(e)}"

# --- UI ---
st.set_page_config(page_title="CFB Predictor", page_icon="🏈")
st.title("🏈 CFB Rolling Stat Predictor")
st.markdown("This model predicts winners based on **rolling scoring averages**.")

# Get unique list of all teams
teams = sorted(list(set(all_stats_df['home_team'].unique()) | set(all_stats_df['away_team'].unique())))

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("🏠 Home Team", teams)
with col2:
    a_team = st.selectbox("✈️ Away Team", teams)

if st.button("Run Prediction", use_container_width=True):
    with st.spinner("Analyzing team matchups..."):
        prob, status = get_prediction(h_team, a_team)
        
        if prob is not None:
            st.divider()
            st.subheader(f"Win Probability: {prob:.1%}")
            st.progress(prob)
            
            if prob > 0.5:
                st.success(f"**{h_team}** is projected to win at home.")
            else:
                st.warning(f"**{a_team}** is projected to pull the upset.")
                
            # Show the stats being used
            with st.expander("See Matchup Stats"):
                home_latest = all_stats_df[all_stats_df['home_team'] == h_team].iloc[-1:]
                away_latest = all_stats_df[all_stats_df['away_team'] == a_team].iloc[-1:]
                st.write(f"{h_team} Rolling Avg Points: **{home_latest['h_avg_pts'].values[0]:.1f}**")
                st.write(f"{a_team} Rolling Avg Points: **{away_latest['a_avg_pts'].values[0]:.1f}**")
        else:
            st.error(f"Error: {status}")
