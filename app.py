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

def get_tape_df(h_name, a_name):
    h_s = lookup_df[lookup_df['team'] == h_name].iloc[0]
    a_s = lookup_df[lookup_df['team'] == a_name].iloc[0]
    return pd.DataFrame({
        "Metric": ["Pts/Game", "Yards/Play", "Pts/Minute", "Turnovers", "SOS"],
        h_name: [f"{h_s['roll_pts_scored']:.1f}", f"{h_s['roll_ypp']:.2f}", f"{h_s['roll_ppm']:.2f}", f"{h_s['roll_turnovers']:.1f}", f"{h_s['opp_def_strength']:.1f}"],
        a_name: [f"{a_s['roll_pts_scored']:.1f}", f"{a_s['roll_ypp']:.2f}", f"{a_s['roll_ppm']:.2f}", f"{a_s['roll_turnovers']:.1f}", f"{a_s['opp_def_strength']:.1f}"]
    })

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
    
    if weights:
        ypp_gap = (h_stats['roll_ypp'] - a_stats['roll_ypp']) * (weights['explosiveness'] - 1.0)
        ppm_gap = (h_stats['roll_ppm'] - a_stats['roll_ppm']) * (weights['efficiency'] - 1.0)
        prob = np.clip(prob + (ypp_gap * 0.05 + ppm_gap * 0.1), 0.01, 0.99)

    h_score = round(((h_stats['roll_ppm'] * 30) + (prob-0.5)*20) + (0 if neutral else 3))
    a_score = round(((a_stats['roll_ppm'] * 30) - (prob-0.5)*20))
    return (h_name if prob > 0.5 else a_name), prob, h_score, a_score

# --- SECTION 1: OFFICIAL BRACKET ---
st.title("🏆 Official CFB Mini-Playoff")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Semi-Final 1")
    w1, p1, s1h, s1a = simulate_matchup("Ole Miss", "Miami", False)
    st.success(f"**{w1}** win {s1h}-{s1a} ({p1 if w1=='Ole Miss' else 1-p1:.1%} confidence)")
    st.table(get_tape_df("Ole Miss", "Miami"))

with col2:
    st.subheader("Semi-Final 2")
    w2, p2, s2h, s2a = simulate_matchup("Indiana", "Oregon", False)
    st.success(f"**{w2}** win {s2h}-{s2a} ({p2 if w2=='Indiana' else 1-p2:.1%} confidence)")
    st.table(get_tape_df("Indiana", "Oregon"))

st.divider()
st.header("🏁 National Championship")
cw, cp, csh, csa = simulate_matchup(w1, w2, True)
c1, c2, c3 = st.columns([2,1,2])
c1.metric(w1, csh)
c2.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
c3.metric(w2, csa)
st.success(f"### Predicted Champion: {cw} ({cp if cw==w1 else 1-cp:.1%})")
st.table(get_tape_df(w1, w2))

# --- SECTION 2: THE LABORATORY ---
st.divider()
st.header("🧪 The 'What-If' Laboratory")
st.write("Adjust weights to see how the AI re-evaluates the Championship matchup:")

# Sliders in the main body
l1, l2, l3 = st.columns(3)
lab_weights = {
    "explosiveness": l1.slider("YPP Weight", 0.5, 3.0, 1.0),
    "efficiency": l2.slider("PPM Weight", 0.5, 3.0, 1.0),
    "defense": l3.slider("SOS Weight", 0.5, 3.0, 1.0)
}

lw, lp, lsh, lsa = simulate_matchup(w1, w2, True, lab_weights)
st.write(f"**Laboratory Result:** {w1} {lsh} - {w2} {lsa}")
st.info(f"With these weights, **{lw}** wins with **{lp if lw==w1 else 1-lp:.1%}** confidence.")

# --- SECTION 3: AI BRAIN ---
st.divider()
st.subheader("🧠 Model Feature Importance")
if hasattr(model, 'feature_importances_'):
    feat_names = ['Neutral', 'H_Pts', 'H_YPP', 'H_PPM', 'H_TO', 'H_SOS', 'A_Pts', 'A_YPP', 'A_PPM', 'A_TO', 'A_SOS']
    feat_imp = pd.DataFrame({'Feature': feat_names, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=False)
    st.bar_chart(feat_imp.set_index('Feature'))
