import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="CFB Playoff Predictor", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("models/ncaa_model.pkl")
    lookup_df = pd.read_csv("data/team_lookup.csv")
    return model, lookup_df

model, lookup_df = load_assets()

# --- SIMULATION ENGINE ---
def simulate_matchup(h_name, a_name, neutral):
    h_stats = lookup_df[lookup_df['team'] == h_name].iloc[0]
    a_stats = lookup_df[lookup_df['team'] == a_name].iloc[0]
    
    # ML Prediction
    input_df = pd.DataFrame([{
        'neutral_site': 1 if neutral else 0,
        'h_roll_pts_scored': h_stats['roll_pts_scored'], 'h_roll_ypp': h_stats['roll_ypp'],
        'h_roll_ppm': h_stats['roll_ppm'], 'h_roll_turnovers': h_stats['roll_turnovers'], 'h_sos': h_stats['opp_def_strength'],
        'a_roll_pts_scored': a_stats['roll_pts_scored'], 'a_roll_ypp': a_stats['roll_ypp'],
        'a_roll_ppm': a_stats['roll_ppm'], 'a_roll_turnovers': a_stats['roll_turnovers'], 'a_sos': a_stats['opp_def_strength']
    }])
    
    prob = model.predict_proba(input_df)[0][1]
    
    # Score Simulation
    avg_pts_allowed = lookup_df['roll_pts_allowed'].mean()
    h_score = round((h_stats['roll_ppm'] * 30) * (a_stats['roll_pts_allowed'] / avg_pts_allowed) + (0 if neutral else 3))
    a_score = round((a_stats['roll_ppm'] * 30) * (h_stats['roll_pts_allowed'] / avg_pts_allowed))
    
    winner = h_name if prob > 0.5 else a_name
    return winner, prob, h_score, a_score

# --- UI HEADER ---
st.title("CFB Mini-Playoff Simulator")
st.markdown("Predicting the Semi-Finals and the Neutral-Site Championship.")

# --- SEMI-FINALS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Semi-Final 1")
    w1, p1, s1h, s1a = simulate_matchup("Ole Miss", "Miami", False)
    st.write(f"**Ole Miss {s1h} - Miami {s1a}**")
    st.info(f"Projected Winner: **{w1}** ({p1 if w1=='Ole Miss' else 1-p1:.1%} confidence)")

with col2:
    st.subheader("Semi-Final 2")
    w2, p2, s2h, s2a = simulate_matchup("Indiana", "Oregon", False)
    st.write(f"**Indiana {s2h} - Oregon {s2a}**")
    st.info(f"Projected Winner: **{w2}** ({p2 if w2=='Indiana' else 1-p2:.1%} confidence)")

# --- CHAMPIONSHIP (THE WINNERS PLAY HERE) ---
st.divider()
st.header("Championship Projection")
st.write(f"**Matchup: {w1} vs {w2} (Neutral Site)**")

cw, cp, csh, csa = simulate_matchup(w1, w2, True)

# Display Championship Scoreboard
sc1, sc2, sc3 = st.columns([2,1,2])
sc1.metric(w1, csh)
sc2.markdown("<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
sc3.metric(w2, csa)

st.success(f"### 🎊 AI Predicts **{cw}** to win the Championship!")
st.write(f"Confidence in {cw}: **{cp if cw==w1 else 1-cp:.1%}**")



# --- MANUAL CALCULATOR ---
with st.expander("Custom Game & Stats Deep Dive"):
    teams = sorted(lookup_df['team'].unique())
    sel_h = st.selectbox("Home", teams, index=teams.index("Georgia") if "Georgia" in teams else 0)
    sel_a = st.selectbox("Away", teams, index=teams.index("Ohio State") if "Ohio State" in teams else 0)
    
    if st.button("Run Custom Analysis"):
        res_w, res_p, res_sh, res_sa = simulate_matchup(sel_h, sel_a, False)
        st.write(f"**{sel_h} {res_sh} - {sel_a} {res_sa}**")
        
        # Pull stats for deep dive
        h_s = lookup_df[lookup_df['team'] == sel_h].iloc[0]
        a_s = lookup_df[lookup_df['team'] == sel_a].iloc[0]
        
        # Add "Explosiveness vs Efficiency" chart idea
        st.write("---")
        st.write("**Explosiveness Check:**")
        st.write(f"{sel_h} Yards Per Play: {h_s['roll_ypp']:.2f}")
        st.write(f"{sel_a} Yards Per Play: {a_s['roll_ypp']:.2f}")
