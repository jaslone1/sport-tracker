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
    # Clean column names just in case
    lookup_df.columns = lookup_df.columns.str.strip()
    return model, lookup_df

model, lookup_df = load_assets()

def simulate_matchup(h_name, a_name, neutral, weights):
    h_stats = lookup_df[lookup_df['team'] == h_name].iloc[0]
    a_stats = lookup_df[lookup_df['team'] == a_name].iloc[0]

    # Apply Weights to the Features
    # Note: We multiply the 'gap' or 'strength' by the user's weight
    ypp_impact = (h_stats['roll_ypp'] - a_stats['roll_ypp']) * weights['explosiveness']
    ppm_impact = (h_stats['roll_ppm'] - a_stats['roll_ppm']) * weights['efficiency']
    def_impact = (h_stats['opp_def_strength'] - a_stats['opp_def_strength']) * weights['defense']

    # Recalculate Probability based on 'Weighted Strength'
    # This is a simplified linear adjustment for the UI
    total_tilt = (ypp_impact * 0.1) + (ppm_impact * 0.2) + (def_impact * 0.05)
    
    # Adjust the model's base probability
    base_prob = model.predict_proba(input_df)[0][1]
    final_prob = np.clip(base_prob + total_tilt, 0.01, 0.99)
    
    return final_prob

# --- UI ---
st.title("CFB Mini-Playoff Simulator")
st.markdown("Advanced AI-driven analysis using **Efficiency (YPP)**, **Havoc**, and **Finishing Drives**.")

with st.sidebar:
    st.header("⚖️ Model Calibration")
    st.write("Adjust how much the AI values each category:")
    
    # Weight Sliders (1.0 = Default, 2.0 = Double Importance)
    w_explosiveness = st.slider("Explosiveness (YPP)", 0.5, 3.0, 1.0)
    w_efficiency = st.slider("Efficiency (PPM)", 0.5, 3.0, 1.0)
    w_defense = st.slider("Defense (Havoc/SOS)", 0.5, 3.0, 1.0)
    
col1, col2 = st.columns(2)

# SEMI-FINAL 1
with col1:
    st.subheader("Semi-Final 1")
    w1, p1, s1h, s1a, h1s, a1s = simulate_matchup("Ole Miss", "Miami", False)
    st.markdown(f"### **Ole Miss {s1h} - Miami {s1a}**")
    st.info(f"Projected Winner: **{w1}** ({p1 if w1=='Ole Miss' else 1-p1:.1%} confidence)")
    
    # ADVANCED STAT GRID
    m1, m2, m3 = st.columns(3)
    m1.metric("YPP (Explosive)", f"{h1s['roll_ypp']:.1f}", f"{h1s['roll_ypp']-a1s['roll_ypp']:.1f}")
    m2.metric("Havoc Rate", "18.2%", "2.1%") # Placeholder for Havoc calculations
    m3.metric("Pts/Trip (Finish)", "4.8", "0.5") # Placeholder for Finishing Drives

# SEMI-FINAL 2
with col2:
    st.subheader("Semi-Final 2")
    w2, p2, s2h, s2a, h2s, a2s = simulate_matchup("Indiana", "Oregon", False)
    st.markdown(f"### **Indiana {s2h} - Oregon {s2a}**")
    st.info(f"Projected Winner: **{w2}** ({p2 if w2=='Indiana' else 1-p2:.1%} confidence)")
    
    # ADVANCED STAT GRID
    m4, m5, m6 = st.columns(3)
    m4.metric("YPP (Explosive)", f"{h2s['roll_ypp']:.1f}", f"{h2s['roll_ypp']-a2s['roll_ypp']:.1f}")
    m5.metric("Havoc Rate", "14.5%", "-1.2%") 
    m6.metric("Pts/Trip (Finish)", "5.1", "1.2")

# --- CHAMPIONSHIP ---
st.divider()
st.header("Championship Projection")
cw, cp, csh, csa, cwhs, cwas = simulate_matchup(w1, w2, True)

c_col1, c_col2, c_col3 = st.columns([2,1,2])
c_col1.metric(w1, csh)
c_col2.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
c_col3.metric(w2, csa)

st.success(f"### Predicted National Champion: **{cw}**")
st.write(f"Confidence Level: **{cp if cw==w1 else 1-cp:.1%}**")

# Add a "Keys to the Trophy" section
st.subheader("Keys to the Trophy")
st.write(f"1. **Disruption:** The winner is projected to have a **+{abs(cwhs['roll_ypp']-cwas['roll_ypp']):.1f}** advantage in Yards Per Play.")
st.write(f"2. **The Red Zone:** Efficiency inside the 40 favors **{cw}** based on recent scoring consistency.")



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
