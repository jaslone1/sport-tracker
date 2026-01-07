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
    # Note: These names MUST match the features list in train_model.py
    input_df = pd.DataFrame([{
        'neutral_site': 1 if is_neutral else 0,
        'h_roll_pts': h_stats['roll_pts_scored'],
        'h_roll_yds': h_stats['roll_yards'],
        'h_roll_to': h_stats['roll_turnovers'],
        'h_roll_opp_pts': h_stats['roll_pts_allowed'],
        'a_roll_pts': a_stats['roll_pts_scored'],
        'a_roll_yds': a_stats['roll_yards'],
        'a_roll_to': a_stats['roll_turnovers'],
        'a_roll_opp_pts': a_stats['roll_pts_allowed']
    }])
    
    # 3. Predict
    prob = model.predict_proba(input_df)[0][1]
    
    # --- DISPLAY RESULTS ---
    st.divider()
    
    # Display Winner Header
    if prob > 0.5:
        st.success(f"### 🏆 Projected Winner: **{h_team}**")
        win_prob = prob
    else:
        st.warning(f"### 🏆 Projected Winner: **{a_team}**")
        win_prob = 1 - prob

    st.write(f"Confidence Level: **{win_prob:.1%}**")
    st.progress(win_prob)

    # --- ADVANCED STATS COMPARISON ---
    st.subheader("📊 Matchup Breakdown (Last 3 Games)")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write("**Avg Points**")
        st.metric(h_team, f"{h_stats['roll_pts_scored']:.1f}")
        st.metric(a_team, f"{a_stats['roll_pts_scored']:.1f}")

    with c2:
        st.write("**Avg Yards**")
        st.metric(h_team, f"{h_stats['roll_yards']:.0f}")
        st.metric(a_team, f"{a_stats['roll_yards']:.0f}")

    with c3:
        st.write("**Avg Turnovers**")
        st.metric(h_team, f"{h_stats['roll_turnovers']:.1f}")
        st.metric(a_team, f"{a_stats['roll_turnovers']:.1f}")
