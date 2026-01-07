import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="CFB 2026 Marquee Predictor", page_icon="🏈", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("models/ncaa_model.pkl")
    lookup_df = pd.read_csv("data/team_lookup.csv")
    return model, lookup_df

model, lookup_df = load_assets()

# --- HEADER ---
st.title("🏈 2026 Marquee Matchup Center")
st.markdown("Detailed AI analysis for this week's biggest games.")

# --- MARQUEE MATCHUPS ---
marquee_games = [
    {"label": "🔥 Game of the Week: Ole Miss vs Miami", "home": "Ole Miss", "away": "Miami", "neutral": False},
    {"label": "🌲 B1G Showdown: Indiana vs Oregon", "home": "Indiana", "away": "Oregon", "neutral": False}
]

selected_marquee = st.radio("Select a Featured Matchup to Analyze:", [g['label'] for g in marquee_games], horizontal=True)
current_game = next(g for g in marquee_games if g['label'] == selected_marquee)

# --- ANALYSIS FUNCTION ---
def run_analysis(h_name, a_name, neutral):
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
    winner = h_name if prob > 0.5 else a_name
    conf = prob if prob > 0.5 else 1 - prob
    
    # Display Result
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Projected Winner", winner, f"{conf:.1%} Confidence")
        # Visual Win Probability Gauge
        st.progress(conf if winner == h_name else 1-conf)
        
    with col_b:
        st.subheader("💡 Strategic Insight")
        if h_stats['roll_ypp'] > 7.0 or a_stats['roll_ypp'] > 7.0:
            st.info("🚨 **High Explosiveness Alert:** One or both teams are averaging over 7 yards per play. Expect a shootout.")
        if abs(h_stats['opp_def_strength'] - a_stats['opp_def_strength']) > 10:
            st.warning("⚖️ **Schedule Imbalance:** One team has faced significantly tougher defenses lately. The 'raw' stats may be deceptive.")

    # Detailed Stats Table
    comparison_data = {
        "Metric": ["Yards Per Play", "Points Per Minute", "Turnovers (Avg)", "SOS Rating"],
        h_name: [f"{h_stats['roll_ypp']:.2f}", f"{h_stats['roll_ppm']:.2f}", f"{h_stats['roll_turnovers']:.1f}", f"{h_stats['opp_def_strength']:.1f}"],
        a_name: [f"{a_stats['roll_ypp']:.2f}", f"{a_stats['roll_ppm']:.2f}", f"{a_stats['roll_turnovers']:.1f}", f"{a_stats['opp_def_strength']:.1f}"]
    }
    st.table(pd.DataFrame(comparison_data))

# Run the Marquee Analysis
st.divider()
run_analysis(current_game['home'], current_game['away'], current_game['neutral'])

# --- MANUAL CALCULATOR SECTION ---
st.divider()
with st.expander("🛠️ Custom Matchup Calculator"):
    st.write("Predict any other FBS matchup below.")
    teams = sorted(lookup_df['team'].unique())
    c1, c2 = st.columns(2)
    with c1:
        custom_h = st.selectbox("Home Team", teams, key="custom_h")
    with c2:
        custom_a = st.selectbox("Away Team", teams, key="custom_a")
    
    if st.button("Calculate Custom Odds"):
        run_analysis(custom_h, custom_a, False)
