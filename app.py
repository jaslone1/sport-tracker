import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="NCAA Football Predictor",
    page_icon="🏈",
    layout="wide",
)

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


@st.cache_data
def load_latest_schedule():
    stats_path = "data/detailed_stats.csv"
    if not os.path.exists(stats_path):
        return None, None, None

    df = pd.read_csv(stats_path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0

    latest_year = df["year"].max()
    latest_week = df[df["year"] == latest_year]["week"].max()

    week_games = df[(df["year"] == latest_year) & (df["week"] == latest_week)]
    schedule = week_games[["h_team", "a_team", "neutral_site"]].drop_duplicates()

    return schedule, latest_year, latest_week


def get_tape_df(h_name, a_name, lookup_df):
    h_s = lookup_df[lookup_df["team"] == h_name]
    a_s = lookup_df[lookup_df["team"] == a_name]

    if h_s.empty or a_s.empty:
        return pd.DataFrame()

    h_s = h_s.iloc[0]
    a_s = a_s.iloc[0]

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


def render_prediction_display(res, is_neutral):
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
    tape_df = get_tape_df(res["home_team"], res["away_team"], lookup_df)
    if not tape_df.empty:
        st.table(tape_df)


# --- UI Setup ---
st.title("🏈 NCAA Football Matchup Predictor")

model = load_model()
lookup_df = load_lookup_data()
schedule, year, week = load_latest_schedule()

if model is None or lookup_df is None:
    st.error("❌ Missing required model or data files. Run `feature_engineering.py` and `train_model.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["📅 Weekend Schedule", "⚔️ Custom Matchup Builder"])

# TAB 1: WEEKEND GAMES SCHEDULE
with tab1:
    if schedule is None or schedule.empty:
        st.warning("No schedule data found in `data/detailed_stats.csv`.")
    else:
        st.subheader(f"Games for {year} - Week {week}")

        # Precompute predictions for all scheduled games
        game_list = []
        for _, row in schedule.iterrows():
            h, a, n = row["h_team"], row["a_team"], bool(row["neutral_site"])
            pred = predict_matchup(h, a, n, model, lookup_df)
            if pred:
                label = f"{a} @ {h}" if not n else f"{a} vs {h} (Neutral)"
                game_list.append({
                    "Matchup": label,
                    "Away Team": a,
                    "Home Team": h,
                    "Neutral": n,
                    "Predicted Winner": pred["winner"],
                    "Confidence": f"{pred['confidence']:.1%}",
                    "Home Win %": f"{pred['home_prob']:.1%}",
                    "Away Win %": f"{pred['away_prob']:.1%}",
                })

        schedule_df = pd.DataFrame(game_list)

        selected_game_label = st.selectbox(
            "👉 Select a game from this weekend to inspect stats:",
            options=schedule_df["Matchup"].tolist(),
            index=0,
        )

        st.markdown("---")

        selected_row = schedule_df[schedule_df["Matchup"] == selected_game_label].iloc[0]
        res = predict_matchup(
            selected_row["Home Team"],
            selected_row["Away Team"],
            selected_row["Neutral"],
            model,
            lookup_df,
        )

        if res:
            render_prediction_display(res, selected_row["Neutral"])

        st.markdown("### 📋 Full Weekend Slate Overview")
        st.dataframe(
            schedule_df[["Matchup", "Predicted Winner", "Confidence", "Home Win %", "Away Win %"]],
            use_container_width=True,
            hide_index=True,
        )

# TAB 2: MANUAL CUSTOM MATCHUP
with tab2:
    st.subheader("Custom Matchup")
    teams = sorted(lookup_df["team"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        h_team = st.selectbox("Select Home Team", teams, index=0, key="custom_h")
    with col2:
        default_away = 1 if len(teams) > 1 else 0
        a_team = st.selectbox("Select Away Team", teams, index=default_away, key="custom_a")

    is_neutral = st.checkbox("Neutral Site Game", value=False, key="custom_n")

    if h_team == a_team:
        st.warning("⚠️ Please select two different teams.")
    else:
        if st.button("🚀 Predict Custom Matchup", use_container_width=True):
            res = predict_matchup(h_team, a_team, is_neutral, model, lookup_df)
            if res:
                st.markdown("---")
                render_prediction_display(res, is_neutral)