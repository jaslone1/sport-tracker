import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="NCAA Football Predictor",
    page_icon="🏈",
    layout="centered",
)

# Explicit feature list matching train_model.py exactly
FEATURE_COLS = [
    "neutral_site",
    "h_roll_pts_scored",
    "h_roll_ypp",
    "h_roll_ppm",
    "h_roll_turnovers",
    "h_sos",
    "a_roll_pts_scored",
    "a_roll_ypp",
    "a_roll_ppm",
    "a_roll_turnovers",
    "a_sos",
]


@st.cache_resource
def load_model():
    model_path = "models/ncaa_model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_lookup_data():
    lookup_path = "data/team_lookup.csv"
    if not os.path.exists(lookup_path):
        return None
    return pd.read_csv(lookup_path)


def get_tape_df(h_name, a_name, lookup_df):
    h_s = lookup_df[lookup_df["team"] == h_name]
    a_s = lookup_df[lookup_df["team"] == a_name]

    if h_s.empty or a_s.empty:
        return pd.DataFrame()

    h_s = h_s.iloc[0]
    a_s = a_s.iloc[0]

    # Safe lookup for SOS/Opponent Defense Strength
    h_sos_val = h_s.get("opp_def_strength", h_s.get("roll_pts_allowed", 0.0))
    a_sos_val = a_s.get("opp_def_strength", a_s.get("roll_pts_allowed", 0.0))

    return pd.DataFrame(
        {
            "Metric": [
                "Pts / Game (Rolling)",
                "Yards / Play",
                "Pts / Minute",
                "Turnovers / Game",
                "Opp. Def Strength (SOS)",
            ],
            f"🏠 {h_name}": [
                f"{h_s.get('roll_pts_scored', 0.0):.1f}",
                f"{h_s.get('roll_ypp', 0.0):.2f}",
                f"{h_s.get('roll_ppm', 0.0):.2f}",
                f"{h_s.get('roll_turnovers', 0.0):.1f}",
                f"{h_sos_val:.1f}",
            ],
            f"✈️ {a_name}": [
                f"{a_s.get('roll_pts_scored', 0.0):.1f}",
                f"{a_s.get('roll_ypp', 0.0):.2f}",
                f"{a_s.get('roll_ppm', 0.0):.2f}",
                f"{a_s.get('roll_turnovers', 0.0):.1f}",
                f"{a_sos_val:.1f}",
            ],
        }
    )


def predict_matchup(h_team, a_team, is_neutral, model, lookup_df):
    h_s = lookup_df[lookup_df["team"] == h_team]
    a_s = lookup_df[lookup_df["team"] == a_team]

    if h_s.empty or a_s.empty:
        return None

    h_s = h_s.iloc[0]
    a_s = a_s.iloc[0]

    h_sos_val = h_s.get("opp_def_strength", h_s.get("roll_pts_allowed", 0.0))
    a_sos_val = a_s.get("opp_def_strength", a_s.get("roll_pts_allowed", 0.0))

    input_data = pd.DataFrame(
        [
            {
                "neutral_site": 1 if is_neutral else 0,
                "h_roll_pts_scored": h_s.get("roll_pts_scored", 0.0),
                "h_roll_ypp": h_s.get("roll_ypp", 0.0),
                "h_roll_ppm": h_s.get("roll_ppm", 0.0),
                "h_roll_turnovers": h_s.get("roll_turnovers", 0.0),
                "h_sos": h_sos_val,
                "a_roll_pts_scored": a_s.get("roll_pts_scored", 0.0),
                "a_roll_ypp": a_s.get("roll_ypp", 0.0),
                "a_roll_ppm": a_s.get("roll_ppm", 0.0),
                "a_roll_turnovers": a_s.get("roll_turnovers", 0.0),
                "a_sos": a_sos_val,
            }
        ]
    )[FEATURE_COLS]

    # Use predict_proba to obtain soft probabilities instead of hard classification labels
    probabilities = model.predict_proba(input_data)[0]
    away_win_prob = float(probabilities[0])
    home_win_prob = float(probabilities[1])

    return {
        "home_team": h_team,
        "away_team": a_team,
        "home_prob": home_win_prob,
        "away_prob": away_win_prob,
        "winner": h_team if home_win_prob >= away_win_prob else a_team,
        "confidence": max(home_win_prob, away_win_prob),
    }


# 2. Main UI Layout
st.title("🏈 NCAA Football Matchup Predictor")
st.write("Predict game outcomes and win probabilities based on recent rolling team statistics.")

model = load_model()
lookup_df = load_lookup_data()

if model is None:
    st.error(
        "❌ Model file `models/ncaa_model.pkl` not found. Please run `python train_model.py` first."
    )
    st.stop()

if lookup_df is None or lookup_df.empty:
    st.error(
        "❌ Team lookup data `data/team_lookup.csv` not found. Please run `python feature_engineering.py` first."
    )
    st.stop()

teams = sorted(lookup_df["team"].dropna().unique())

st.markdown("### Matchup Selection")
col1, col2 = st.columns(2)

with col1:
    h_team = st.selectbox("Select Home Team", teams, index=0)

with col2:
    default_away_idx = 1 if len(teams) > 1 else 0
    a_team = st.selectbox("Select Away Team", teams, index=default_away_idx)

is_neutral = st.checkbox("Neutral Site Game", value=False)

if h_team == a_team:
    st.warning("⚠️ Please select two different teams for a matchup.")
else:
    if st.button("🚀 Predict Matchup", use_container_width=True):
        res = predict_matchup(h_team, a_team, is_neutral, model, lookup_df)

        if res:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)

            c1.metric(
                label=f"🏠 {res['home_team']}",
                value=f"{res['home_prob']:.1%}",
                delta="Home" if not is_neutral else "Neutral",
            )
            c2.markdown(
                "<h3 style='text-align: center; margin-top: 15px;'>VS</h3>",
                unsafe_allow_html=True,
            )
            c3.metric(
                label=f"✈️ {res['away_team']}",
                value=f"{res['away_prob']:.1%}",
                delta="Away",
            )

            st.success(
                f"### Predicted Winner: **{res['winner']}** ({res['confidence']:.1%} confidence)"
            )

            st.markdown("### 📊 Tale of the Tape")
            tape_df = get_tape_df(h_team, a_team, lookup_df)
            if not tape_df.empty:
                st.table(tape_df)
        else:
            st.error("❌ Could not generate prediction. Check that team data exists.")