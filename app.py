import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="FBS Weekly Predictions", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load("models/ncaa_model.pkl")
    lookup_df = pd.read_csv("data/team_lookup.csv")
    lookup_df.columns = lookup_df.columns.str.strip()
    return model, lookup_df


model, lookup_df = load_assets()


def simulate_matchup(h_name, a_name, neutral, weights=None):
    h_stats = lookup_df[lookup_df['team'] == h_name]
    a_stats = lookup_df[lookup_df['team'] == a_name]

    if h_stats.empty or a_stats.empty:
        return None, None, None, None

    h_stats = h_stats.iloc[0]
    a_stats = a_stats.iloc[0]

    # Check for NaN values in critical rolling stats to avoid conversion errors
    if pd.isna(h_stats.get('roll_ppm')) or pd.isna(a_stats.get('roll_ppm')):
        return None, None, None, None

    input_df = pd.DataFrame([{
        'neutral_site': 1 if neutral else 0,
        'h_roll_pts_scored': h_stats.get('roll_pts_scored', 0),
        'h_roll_ypp': h_stats.get('roll_ypp', 0),
        'h_roll_ppm': h_stats['roll_ppm'],
        'h_roll_turnovers': h_stats.get('roll_turnovers', 0),
        'h_sos': h_stats.get('opp_def_strength', 0),
        'a_roll_pts_scored': a_stats.get('roll_pts_scored', 0),
        'a_roll_ypp': a_stats.get('roll_ypp', 0),
        'a_roll_ppm': a_stats['roll_ppm'],
        'a_roll_turnovers': a_stats.get('roll_turnovers', 0),
        'a_sos': a_stats.get('opp_def_strength', 0)
    }])

    # Fill any remaining NaNs in the input DataFrame with 0
    input_df = input_df.fillna(0)

    prob = model.predict_proba(input_df)[0][1]
    prob = float(np.clip(prob, 0.0, 1.0))
    print(f"RAW PROBABILITY FOR {h_name} vs {a_name}: {prob}")

    if weights:
        ypp_gap = (h_stats.get('roll_ypp', 0) - a_stats.get('roll_ypp', 0)) * (weights['explosiveness'] - 1.0)
        ppm_gap = (h_stats['roll_ppm'] - a_stats['roll_ppm']) * (weights['efficiency'] - 1.0)
        prob = np.clip(prob + (ypp_gap * 0.05 + ppm_gap * 0.1), 0.01, 0.99)

    h_score = round(((h_stats['roll_ppm'] * 30) + (prob - 0.5) * 20) + (0 if neutral else 3))
    a_score = round(((a_stats['roll_ppm'] * 30) - (prob - 0.5) * 20))
    return (h_name if prob > 0.5 else a_name), prob, h_score, a_score


def get_tape_df(h_name, a_name):
    h_s = lookup_df[lookup_df['team'] == h_name]
    a_s = lookup_df[lookup_df['team'] == a_name]
    if h_s.empty or a_s.empty:
        return pd.DataFrame()

    h_s = h_s.iloc[0]
    a_s = a_s.iloc[0]

    return pd.DataFrame({
        "Metric": ["Pts/Game", "Yards/Play", "Pts/Minute", "Turnovers", "SOS"],
        h_name: [
            f"{h_s['roll_pts_scored']:.1f}",
            f"{h_s['roll_ypp']:.2f}",
            f"{h_s['roll_ppm']:.2f}",
            f"{h_s['roll_turnovers']:.1f}",
            f"{h_s['opp_def_strength']:.1f}"
        ],
        a_name: [
            f"{a_s['roll_pts_scored']:.1f}",
            f"{a_s['roll_ypp']:.2f}",
            f"{a_s['roll_ppm']:.2f}",
            f"{a_s['roll_turnovers']:.1f}",
            f"{a_s['opp_def_strength']:.1f}"
        ]
    })


st.title("🏈 FBS Weekly Predictions")

try:
    scheduled_df = pd.read_csv("data/scheduled_games.csv")
    if not scheduled_df.empty:
        st.info(f"📅 **{len(scheduled_df)} scheduled FBS games** | Week {scheduled_df['week'].iloc[0]}")
    else:
        st.warning("⚠️ No scheduled games found. Run `python data_fetch.py` to fetch this week's FBS schedule.")
except FileNotFoundError:
    st.warning("⚠️ No scheduled games found. Run `python data_fetch.py` to fetch this week's FBS schedule.")
    scheduled_df = pd.DataFrame()

if not scheduled_df.empty:
    predictions = []
    for _, game in scheduled_df.iterrows():
        h_team = game['home_team']
        a_team = game['away_team']
        neutral = bool(game['neutral_site'])
        winner, prob, h_score, a_score = simulate_matchup(h_team, a_team, neutral)

        if winner is not None:
            predictions.append({
                'Week': game['week'],
                'Home': h_team,
                'Away': a_team,
                'Prediction': winner,
                'Confidence': f"{max(prob, 1-prob):.1%}",
                'Projected Score': f"{h_score}-{a_score}",
                'Neutral': "Yes" if neutral else "No"
            })

    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df['confidence_numeric'] = pred_df['Confidence'].str.rstrip('%').astype(float)
        pred_df = pred_df.sort_values('confidence_numeric', ascending=False).drop(columns=['confidence_numeric'])

        st.subheader("All Predictions")
        st.dataframe(pred_df, use_container_width=True)

        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"fbs_predictions_week{pred_df['Week'].iloc[0]}.csv",
            mime="text/csv"
        )

        st.divider()
        st.header("🔍 Detailed Matchup Analysis")
        selected_game = st.selectbox(
            "Select a game to analyze:",
            [f"{row['Home']} @ {row['Away']}" for _, row in pred_df.iterrows()]
        )

        if selected_game:
            h_team, a_team = selected_game.split(" @ ")
            winner, prob, h_score, a_score = simulate_matchup(h_team, a_team, False)
            if winner:
                c1, c2, c3 = st.columns([2, 1, 2])
                c1.metric(h_team, h_score, delta="Home")
                c2.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
                c3.metric(a_team, a_score, delta="Away")
                st.success(f"### Predicted Winner: **{winner}** ({max(prob, 1-prob):.1%} confidence)")
                st.table(get_tape_df(h_team, a_team))
    else:
        st.error("❌ Could not generate predictions. Check that team names from the schedule match training data.")

st.divider()
st.markdown("**How it works:** The model is trained on rolling 3-game averages from the last 3 seasons plus the current season so far, then used to predict all scheduled FBS games this week.")
