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
    lookup_df.columns = lookup_df.columns.str.strip()
    return model, lookup_df

model, lookup_df = load_assets()

# --- REVISED SIMULATION ENGINE ---
def simulate_matchup(h_name, a_name, neutral, weights):
    h_stats = lookup_df[lookup_df['team'] == h_name].iloc[0].copy()
    a_stats = lookup_df[lookup_df['team'] == a_name].iloc[0].copy()

    # 1. Prepare ML Input
    input_df = pd.DataFrame([{
        'neutral_site': 1 if neutral else 0,
        'h_roll_pts_scored': h_stats['roll_pts_scored'], 'h_roll_ypp': h_stats['roll_ypp'],
        'h_roll_ppm': h_stats['roll_ppm'], 'h_roll_turnovers': h_stats['roll_turnovers'], 'h_sos': h_stats['opp_def_strength'],
        'a_roll_pts_scored': a_stats['roll_pts_scored'], 'a_roll_ypp': a_stats['roll_ypp'],
        'a_roll_ppm': a_stats['roll_ppm'], 'a_roll_turnovers': a_stats['roll_turnovers'], 'a_sos': a_stats['opp_def_strength']
    }])

    # 2. Base ML Probability
    base_prob = model.predict_proba(input_df)[0][1]

    # 3. Apply User Weight Tilt
    ypp_gap = (h_stats['roll_ypp'] - a_stats['roll_ypp']) * (weights['explosiveness'] - 1.0)
    ppm_gap = (h_stats['roll_ppm'] - a_stats['roll_ppm']) * (weights['efficiency'] - 1.0)
    def_gap = (h_stats['opp_def_strength'] - a_stats['opp_def_strength']) * (weights['defense'] - 1.0)
    
    # Tilt the probability based on gaps and weights
    total_tilt = (ypp_gap * 0.05) + (ppm_gap * 0.1) + (def_gap * 0.02)
    prob = np.clip(base_prob + total_tilt, 0.01, 0.99)

    # 4. Generate Scores (Synchronized with Probability)
    projected_total = (h_stats['roll_ppm'] + a_stats['roll_ppm']) * 30
    margin = (prob - 0.5) * 25  # 25 is a scaling factor for spread
    h_score = round((projected_total + margin) / 2 + (0 if neutral else 3))
    a_score = round((projected_total - margin) / 2)
    
    winner = h_name if prob > 0.5 else a_name
    return winner, prob, h_score, a_score, h_stats, a_stats

# --- UI ---
st.title("🏆 CFB Mini-Playoff Simulator")

with st.sidebar:
    st.header("⚖️ Model Calibration")
    st.write("Fine-tune the AI's logic:")
    
    # Store slider values in a dictionary
    user_weights = {
        "explosiveness": st.slider("Explosiveness (YPP)", 0.5, 3.0, 1.0),
        "efficiency": st.slider("Efficiency (PPM)", 0.5, 3.0, 1.0),
        "defense": st.slider("Defense (Havoc/SOS)", 0.5, 3.0, 1.0)
    }

col1, col2 = st.columns(2)

# SEMI-FINAL 1
with col1:
    st.subheader("Semi-Final 1")
    # PASS THE WEIGHTS HERE
    w1, p1, s1h, s1a, h1s, a1s = simulate_matchup("Ole Miss", "Miami", False, user_weights)
    st.markdown(f"### **Ole Miss {s1h} - Miami {s1a}**")
    st.info(f"Projected Winner: **{w1}** ({p1 if w1=='Ole Miss' else 1-p1:.1%} confidence)")
    
    m1, m2 = st.columns(2)
    m1.metric("YPP (Explosive)", f"{h1s['roll_ypp']:.1f}")
    m2.metric("PPM (Efficiency)", f"{h1s['roll_ppm']:.1f}")

# SEMI-FINAL 2
with col2:
    st.subheader("Semi-Final 2")
    # PASS THE WEIGHTS HERE
    w2, p2, s2h, s2a, h2s, a2s = simulate_matchup("Indiana", "Oregon", False, user_weights)
    st.markdown(f"### **Indiana {s2h} - Oregon {s2a}**")
    st.info(f"Projected Winner: **{w2}** ({p2 if w2=='Indiana' else 1-p2:.1%} confidence)")

    m3, m4 = st.columns(2)
    m3.metric("YPP (Explosive)", f"{h2s['roll_ypp']:.1f}")
    m4.metric("PPM (Efficiency)", f"{h2s['roll_ppm']:.1f}")

# --- CHAMPIONSHIP ---
st.divider()
st.header("🏁 Championship Projection")
# PASS THE WEIGHTS HERE
cw, cp, csh, csa, cwhs, cwas = simulate_matchup(w1, w2, True, user_weights)

c_col1, c_col2, c_col3 = st.columns([2,1,2])
c_col1.metric(w1, csh)
c_col2.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
c_col3.metric(w2, csa)

st.success(f"### Predicted Champion: **{cw}**")

# --- FULL STAT COMPARISON TABLE ---
st.divider()
st.subheader("📊 The Tale of the Tape")
st.write("A deep dive into every available metric for the finalists.")

# Create a full stats comparison dataframe
comp_data = {
    "Metric": [
        "Pts Scored (Roll)", "Pts Allowed (Roll)", 
        "Yards Per Play", "Points Per Minute", 
        "Turnover Average", "Strength of Schedule"
    ],
    w1: [
        f"{cwhs['roll_pts_scored']:.1f}", f"{cwhs['roll_pts_allowed']:.1f}",
        f"{cwhs['roll_ypp']:.2f}", f"{cwhs['roll_ppm']:.2f}",
        f"{cwhs['roll_turnovers']:.1f}", f"{cwhs['opp_def_strength']:.1f}"
    ],
    w2: [
        f"{cwas['roll_pts_scored']:.1f}", f"{cwas['roll_pts_allowed']:.1f}",
        f"{cwas['roll_ypp']:.2f}", f"{cwas['roll_ppm']:.2f}",
        f"{cwas['roll_turnovers']:.1f}", f"{cwas['opp_def_strength']:.1f}"
    ]
}
st.table(pd.DataFrame(comp_data))

# --- AI LOGIC VISUALIZER ---
st.subheader("🧠 What the AI is Watching")
st.write("This chart shows the 'Fixed' weights the model learned from historical data.")

if hasattr(model, 'feature_importances_'):
    # Match importance scores to your input columns
    feat_names = [
        'Neutral Site', 'H: Pts', 'H: YPP', 'H: PPM', 'H: TOs', 'H: SOS',
        'A: Pts', 'A: YPP', 'A: PPM', 'A: TOs', 'A: SOS'
    ]
    importances = model.feature_importances_
    
    # Create a simple bar chart
    importance_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))
else:
    st.info("Model type does not support raw feature importance extraction.")
