import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# --- PAGE CONFIG (Must be the first Streamlit command) ---
st.set_page_config(page_title="CFB Predictor", page_icon="🏈")

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent 
MODEL_PATH = BASE_DIR / "models" / "ncaa_model.pkl"
FEATURES_DATA_PATH = BASE_DIR / "data" / "ml_ready_features.csv" 

@st.cache_resource
def load_assets():
    # Load the Random Forest model
    model = joblib.load(MODEL_PATH)
    
    # Load the features file to get team names and latest stats
    df = pd.read_csv(FEATURES_DATA_PATH)
    
    # Standardize team names
    df['home_team'] = df['home_team'].astype(str).str.strip()
    df['away_team'] = df['away_team'].astype(str).str.strip()
    
    return model, df

model, all_stats_df = load_assets()

# --- PREDICTION LOGIC ---
def get_prediction(home_team, away_team, neutral_site_flag):
    # 1. Get the most recent rolling stats for both teams
    # We look for the team in either home or away columns to get their latest performance
    h_data = all_stats_df[(all_stats_df['home_team'] == home_team) | (all_stats_df['away_team'] == home_team)].iloc[-1:]
    a_data = all_stats_df[(all_stats_df['home_team'] == away_team) | (all_stats_df['away_team'] == away_team)].iloc[-1:]
    
    if h_data.empty or a_data.empty:
        return None, "One of the selected teams does not have enough historical data."

    # Extracting the correct columns based on where the team was found
    h_pts = h_data['h_avg_pts'].values[0] if h_data['home_team'].values[0] == home_team else h_data['a_avg_pts'].values[0]
    h_yds = h_data['h_avg_yds'].values[0] if h_data['home_team'].values[0] == home_team else h_data['a_avg_yds'].values[0]
    h_to  = h_data['h_avg_to'].values[0] if h_data['home_team'].values[0] == home_team else h_data['a_avg_to'].values[0]
    
    a_pts = a_data['a_avg_pts'].values[0] if a_data['away_team'].values[0] == away_team else a_data['h_avg_pts'].values[0]
    a_yds = a_data['a_avg_yds'].values[0] if a_data['away_team'].values[0] == away_team else a_data['h_avg_yds'].values[0]
    a_to  = a_data['a_avg_to'].values[0] if a_data['away_team'].values[0] == away_team else a_data['h_avg_to'].values[0]

    # 2. Construct input DataFrame
    input_df = pd.DataFrame([{
        'neutral_site': 1 if neutral_site_flag else 0,
        'h_avg_pts': h_pts, 'h_avg_yds': h_yds, 'h_avg_to': h_to,
        'a_avg_pts': a_pts, 'a_avg_yds': a_yds, 'a_avg_to': a_to
    }])
    
    # 3. Predict
    prob = model.predict_proba(input_df)[0][1]
    return prob, "Success"

# --- UI ---
st.title("🏈 CFB Advanced Predictor")
st.markdown("Predicting game outcomes using **neutral site awareness** and **rolling advanced stats**.")

# Get unique list of all teams
teams = sorted(list(set(all_stats_df['home_team'].unique()) | set(all_stats_df['away_team'].unique())))

is_neutral = st.checkbox("🏟️ Neutral Site Game (e.g., Playoffs/Bowl Game)")

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("🏠 Home Team (or Team A)", teams)
with col2:
    a_team = st.selectbox("✈️ Away Team (or Team B)", teams)

if st.button("Run Prediction", use_container_width=True):
    with st.spinner("Analyzing team matchups..."):
        prob, status = get_prediction(h_team, a_team, is_neutral)
        
        if prob is not None:
            st.divider()
            st.subheader(f"Win Probability for {h_team}: {prob:.1%}")
            st.progress(prob)
            
            if prob > 0.5:
                st.success(f"**{h_team}** is projected to win.")
            else:
                st.warning(f"**{a_team}** is projected to win.")
                
            # Show the stats being used
            with st.expander("🔍 See Matchup Stats Used"):
                # Fetching latest again for display
                h_disp = all_stats_df[(all_stats_df['home_team'] == h_team) | (all_stats_df['away_team'] == h_team)].iloc[-1:]
                a_disp = all_stats_df[(all_stats_df['home_team'] == a_team) | (all_stats_df['away_team'] == a_team)].iloc[-1:]
                
                c1, c2 = st.columns(2)
                c1.metric(f"{h_team} Avg Pts", f"{prob*100:.1f}") # Placeholder for display logic
                c2.metric(f"{a_team} Avg Pts", f"{(1-prob)*100:.1f}")
        else:
            st.error(f"Error: {status}")
