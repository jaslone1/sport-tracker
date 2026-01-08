import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="CFB Playoff Predictor", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("models/ncaa_model.pkl")
    lookup_df = pd.read_csv("data/team_lookup.csv")
    lookup_df.columns = lookup_df.columns.str.strip()
    return model, lookup_df

model, lookup_df = load_assets()

# --- UTILITY: GET TALE OF THE TAPE ---
def get_tape_df(h_name, a_name):
    h_s = lookup_df[lookup_df['team'] == h_name].iloc[0]
    a_s = lookup_df[lookup_df['team'] == a_name].iloc[0]
    return pd.DataFrame({
        "Metric": ["Points/Game", "Yards Per Play", "Points/Minute", "Turnovers/Game", "SOS Defense"],
        h_name: [f"{h_s['roll_pts_scored']:.1f}", f"{h_s['roll_ypp']:.2f}", f"{h_s['roll_ppm']:.2f}", f"{h_s['roll_turnovers']:.1f}", f"{h_s['opp_def_strength']:.1f}"],
        a_name: [f"{a_s['roll_pts_scored']:.1f}", f"{a_s['roll_ypp']:.2f}", f"{a_s['roll_ppm']:.2f}", f"{a_s['roll_turnovers']:.1f}", f"{a_s['opp_def_strength']:.1f}"]
    })

# --- SIMULATION ENGINE ---
def simulate_matchup(h_name, a_name, neutral, weights=None):
    h_stats = lookup_df[lookup_df['team'] == h_name].iloc[0]
    a_stats = lookup_df[lookup_df['team'] == a_name].iloc[0]
    
    input_df = pd.DataFrame([{
        'neutral_site': 1 if neutral else 0,
        'h_roll_pts_scored': h_stats['roll_pts_scored'], 'h_roll_ypp': h_stats['roll_ypp'],
        'h_roll_ppm': h_stats['roll_ppm'], 'h_roll_turnovers': h_stats['roll_turnovers'], 'h_sos': h_stats['opp_def_strength'],
        'a_roll_pts_scored': a_stats['roll_pts_scored'], 'a_roll_ypp': a_stats['roll_ypp'],
        'a_roll_ppm': a_stats['roll_ppm'], 'a_roll_turnovers': a_stats['roll_turnovers'], 'a_sos': a_stats['opp_def_strength']
    }])
    
    prob = model.predict_proba(input_df)[0][1]
    
    # Apply user-defined weight tilt if provided
    if weights:
        ypp_gap = (h_stats['roll_ypp'] - a_stats['roll_ypp']) * (weights['explosiveness'] - 1.0)
        ppm_gap = (h_stats['roll_ppm'] - a_stats['roll_ppm']) * (weights['efficiency'] - 1.0)
        total_tilt = (ypp_gap * 0.05) + (ppm_gap * 0.1)
        prob = np.clip(prob + total_tilt, 0.01, 0.99)

    h_score = round(((h_stats['roll_ppm'] * 30) + (prob-0.5)*20) + (0 if neutral else 3))
    a_score = round(((a_stats['roll_ppm'] * 30) - (prob-0.5)*20))
    winner = h_name if prob > 0.5 else a_name
    return winner, prob, h_score, a_score

# --- SECTION 1: THE OFFICIAL PLAYOFF ---
st.title("🏆 The CFB Official Mini-Playoff")
st.write("Predictions based on raw historical data and standard AI weights.")

scol1, scol2 = st.columns(2)
with scol1:
    st.subheader("Semi-Final 1")
    w1, p1, s1h, s1a = simulate_matchup("Ole Miss", "Miami", False)
    st.info(f"**{w1}** win {s1h}-{s1a} ({p1 if w1=='Ole Miss' else 1-p1:.1%})")
    st.table(get_tape_df("Ole Miss", "Miami"))

with scol2:
    st.subheader("Semi-Final 2")
    w2, p2, s2h, s2a = simulate_matchup("Indiana", "Oregon", False)
    st.info(f"**{w2}** win {s2h}-{s2a} ({p2 if w2=='Indiana' else 1-p2:.1%})")
    st.table(get_tape_df("Indiana", "Oregon"))

# --- SECTION 2: WHAT-IF LABORATORY ---
st.divider()
st.header("🧪 The 'What-If' Laboratory")
st.write("Adjust the sliders below to see how changes in team focus affect the results.")

with st.sidebar:
    st.header("⚖️ Dynamic Weights")
    lab_weights = {
        "explosiveness": st.slider("Explosiveness (YPP) Weight", 0.5, 3.0, 1.0),
        "efficiency": st.slider("Efficiency (PPM) Weight", 0.5, 3.0, 1.0),
        "defense": st.slider("Defense (SOS) Weight", 0.5, 3.0, 1.0)
    }

l_col1, l_col2 = st.columns(2)
# Re-run same matchups but with Lab Weights
with l_col1:
    lw1, lp1, ls1h, ls1a = simulate_matchup("Ole Miss", "Miami", False, lab_weights)
    st.metric("Modified Prob (Miami)", f"{lp1 if lw1=='Miami' else 1-lp1:.1%}", f"{(lp1-p1)*100:.1f}%")

# --- SECTION 3: AI BRAIN ---
st.divider()
st.subheader("🧠 Model Feature Importance")
if hasattr(model, 'feature_importances_'):
    feat_imp = pd.DataFrame({'Feature': ['Neutral', 'H_Pts', 'H_YPP', 'H_PPM', 'H_TO', 'H_SOS', 'A_Pts', 'A_YPP', 'A_PPM', 'A_TO', 'A_SOS'], 
                             'Weight': model.feature_importances_}).sort_values('Weight', ascending=False)
    st.bar_chart(feat_imp.set_index('Feature'))
