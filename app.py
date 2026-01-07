import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="CFB Predictor", page_icon="🏈", layout="wide")

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent 
MODEL_PATH = BASE_DIR / "models" / "ncaa_model.pkl"
# Use the lookup file created in Feature Engineering
LOOKUP_DATA_PATH = BASE_DIR / "data" / "team_lookup.csv" 

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    lookup_df = pd.read_csv(LOOKUP_DATA_PATH)
    lookup_df['team'] = lookup_df['team'].astype(str).str.strip()
    return model, lookup_df

model, lookup_df = load_assets()

# --- UI ---
st.title("🏈 CFB Advanced Matchup Predictor")
st.markdown("Utilizing **Rolling 3-Game Averages** for Yards, Turnovers, and Scoring.")

teams = sorted(lookup_df['team'].unique())

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("🏠 Select Home Team (or Team A)", teams, index=teams.index("Georgia") if "Georgia" in teams else 0)
with col2:
    a_team = st.selectbox("✈️ Select Away Team (or Team B)", teams, index=teams.index("Alabama") if "Alabama" in teams else 0)

is_neutral = st.checkbox("🏟️ Neutral Site Game (e.g., Playoffs/Bowl Game)")

# --- PREDICTION LOGIC ---
if st.button("Analyze Matchup", use_container_width=True):
    # 1. Pull Latest Stats from Lookup
    h_stats = lookup_df[lookup_df['team'] == h_team].iloc[0]
    a_stats = lookup_df[lookup_df['team'] == a_team].iloc[0]
    
    # 2. Construct input for the model
    # CRITICAL: These must EXACTLY match the 'features' list from train_model.py
    input_df = pd.DataFrame([{
        'neutral_site': 1 if is_neutral else 0,
        'h_roll_pts_scored': h_stats['roll_pts_scored'],
        'h_roll_ypp': h_stats['roll_ypp'],
        'h_roll_ppm': h_stats['roll_ppm'],
        'h_roll_turnovers': h_stats['roll_turnovers'],
        'h_sos': h_stats['opp_def_strength'],
        'a_roll_pts_scored': a_stats['roll_pts_scored'],
        'a_roll_ypp': a_stats['roll_ypp'],
        'a_roll_ppm': a_stats['roll_ppm'],
        'a_roll_turnovers': a_stats['roll_turnovers'],
        'a_sos': a_stats['opp_def_strength']
    }])
    
    # 3. Predict
    # Use predict_proba to get the percentage chance of a Home Win (column 1)
    prob = model.predict_proba(input_df)[0][1]
    
    # --- DISPLAY RESULTS ---
    st.divider()
    
    if prob > 0.5:
        st.success(f"### 🏆 Projected Winner: **{h_team}**")
        win_prob = prob
    else:
        st.warning(f"### 🏆 Projected Winner: **{a_team}**")
        win_prob = 1 - prob

    st.write(f"Win Probability: **{win_prob:.1% Rose}**")
    st.progress(win_prob)

    # --- ADVANCED STATS COMPARISON ---
    st.subheader("📊 Advanced Matchup Metrics")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.write("**Yards Per Play**")
        st.metric(h_team, f"{h_stats['roll_ypp']:.2f}")
        st.metric(a_team, f"{a_stats['roll_ypp']:.2f}")

    with c2:
        st.write("**Points Per Min**")
        st.metric(h_team, f"{h_stats['roll_ppm']:.2f}")
        st.metric(a_team, f"{a_stats['roll_ppm']:.2f}")

    with c3:
        st.write("**Turnovers**")
        st.metric(h_team, f"{h_stats['roll_turnovers']:.1f}")
        st.metric(a_team, f"{a_stats['roll_turnovers']:.1f}")
        
    with c4:
        st.write("**SOS (Opp Pts Allowed)**")
        st.metric(h_team, f"{h_stats['opp_def_strength']:.1f}")
        st.metric(a_team, f"{a_stats['opp_def_strength']:.1f}")
