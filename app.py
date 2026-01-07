import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="CFB Predictor", page_icon="🏈", layout="wide")

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent 
MODEL_PATH = BASE_DIR / "models" / "ncaa_model.pkl"
LOOKUP_DATA_PATH = BASE_DIR / "data" / "team_lookup.csv" 

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    lookup_df = pd.read_csv(LOOKUP_DATA_PATH)
    lookup_df['team'] = lookup_df['team'].astype(str).str.strip()
    return model, lookup_df

model, lookup_df = load_assets()

# --- UI ---
st.title("🏈 CFB Matchup Predictor & Simulator")
st.markdown("Predicting winners and simulating scores based on advanced efficiency metrics.")

teams = sorted(lookup_df['team'].unique())

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("🏠 Select Home Team", teams, index=teams.index("Indiana") if "Indiana" in teams else 0)
with col2:
    a_team = st.selectbox("✈️ Select Away Team", teams, index=teams.index("Alabama") if "Alabama" in teams else 0)

is_neutral = st.checkbox("🏟️ Neutral Site Game")

# --- PREDICTION & SIMULATION LOGIC ---
if st.button("Analyze & Simulate Matchup", use_container_width=True):
    h_stats = lookup_df[lookup_df['team'] == h_team].iloc[0]
    a_stats = lookup_df[lookup_df['team'] == a_team].iloc[0]
    
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
    
    prob = model.predict_proba(input_df)[0][1]
    
    # --- SCORE SIMULATION ALGORITHM ---
    # We use PPM (Points per Minute) x 30 minutes of possession
    # Then we adjust by the opponent's SOS (Defensive strength)
    avg_pts_allowed = lookup_df['roll_pts_allowed'].mean()
    
    h_sim_base = h_stats['roll_ppm'] * 30
    a_sim_base = a_stats['roll_ppm'] * 30
    
    # Adjust score based on SOS (If opponent allows fewer pts than avg, reduce score)
    h_adj = (a_stats['roll_pts_allowed'] / avg_pts_allowed) if avg_pts_allowed > 0 else 1
    a_adj = (h_stats['roll_pts_allowed'] / avg_pts_allowed) if avg_pts_allowed > 0 else 1
    
    h_final = round(h_sim_base * h_adj + (0 if is_neutral else 3)) # Home field bump
    a_final = round(a_sim_base * a_adj)
    
    # --- RESULTS DISPLAY ---
    st.divider()
    
    # Scoreboard View
    st.subheader("🏟️ Predicted Final Score")
    sb1, sb2, sb3 = st.columns([2, 1, 2])
    sb1.metric(h_team, h_final)
    sb2.markdown("<h2 style='text-align: center;'>vs</h2>", unsafe_allow_html=True)
    sb3.metric(a_team, a_final)

    if prob > 0.5:
        st.success(f"### 🏆 Projected Winner: **{h_team}**")
        win_prob = prob
        winner_name = h_team
    else:
        st.warning(f"### 🏆 Projected Winner: **{a_team}**")
        win_prob = 1 - prob
        winner_name = a_team

    st.write(f"Win Probability: **{win_prob:.1%}**")
    st.progress(win_prob)

    # --- KEYS TO THE GAME ---
    st.subheader(f"🔑 Keys to the Game for {winner_name}")
    keys = []
    if h_stats['roll_ypp'] > a_stats['roll_ypp'] and winner_name == h_team:
        keys.append(f"**Explosive Advantage:** {h_team} is gaining {h_stats['roll_ypp'] - a_stats['roll_ypp']:.1f} more yards per play.")
    elif a_stats['roll_ypp'] > h_stats['roll_ypp'] and winner_name == a_team:
        keys.append(f"**Explosive Advantage:** {a_team} is outpacing the defense in efficiency.")

    if h_stats['roll_turnovers'] < a_stats['roll_turnovers'] and winner_name == h_team:
        keys.append(f"**Clean Football:** {h_team} protects the ball better than {a_team}.")
    elif a_stats['roll_turnovers'] < h_stats['roll_turnovers'] and winner_name == a_team:
        keys.append(f"**Ball Security:** {a_team} wins the turnover battle on paper.")

    if h_stats['opp_def_strength'] > a_stats['opp_def_strength'] and winner_name == h_team:
        keys.append(f"**Battle Tested:** {h_team} has performed against tougher competition.")
    elif a_stats['opp_def_strength'] > h_stats['opp_def_strength'] and winner_name == a_team:
        keys.append(f"**Schedule Strength:** {a_team} is coming off a more difficult slate.")

    for key in keys[:3]:
        st.write(f"✅ {key}")

    # --- ADVANCED STATS COMPARISON ---
    st.subheader("📊 Matchup Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("YPP (Efficiency)", f"{h_stats['roll_ypp']:.1f}", f"{h_stats['roll_ypp'] - a_stats['roll_ypp']:.1f}")
    with c2:
        st.metric("Points Per Min", f"{h_stats['roll_ppm']:.2f}", f"{h_stats['roll_ppm'] - a_stats['roll_ppm']:.2f}")
    with c3:
        st.metric("Turnovers", f"{h_stats['roll_turnovers']:.1f}", f"{h_stats['roll_turnovers'] - a_stats['roll_turnovers']:.1f}", delta_color="inverse")
    with c4:
        st.metric("SOS Rating", f"{h_stats['opp_def_strength']:.0f}", f"{h_stats['opp_def_strength'] - a_stats['opp_def_strength']:.0f}")
