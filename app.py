import os
import joblib
import numpy as np
import pandas as pd
import numpy as np

st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-team {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-prob {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-score {
        font-size: 1.1rem;
        color: #cbd5e1;
    }
    .vs-divider {
        font-size: 1.8rem;
        font-weight: 900;
        color: #94a3b8;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    /* Banner Display */
    .winner-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-left: 6px solid #22c55e;
        border-radius: 8px;
        padding: 16px 24px;
        margin-top: 16px;
        margin-bottom: 24px;
    }
    .winner-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
    }
    .winner-subtitle {
        font-size: 1.1rem;
        color: #4ade80;
        margin: 4px 0 0 0;
    }
    </style>
""",
    unsafe_allow_html=True,
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


# -----------------------------------------------------------------------------
# 2. Data & Model Loaders
# -----------------------------------------------------------------------------
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
    df = pd.read_csv(lookup_path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def load_full_stats():
    stats_path = "data/detailed_stats.csv"
    if not os.path.exists(stats_path):
        return None
    df = pd.read_csv(stats_path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0
    return df


@st.cache_data
def load_scheduled_games():
    scheduled_path = "data/scheduled_games.csv"
    if not os.path.exists(scheduled_path):
        return pd.DataFrame()
    df = pd.read_csv(scheduled_path)
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={"season": "year", "home_team": "h_team", "away_team": "a_team"}
    )
    return df


# -----------------------------------------------------------------------------
# 3. Helper Logic & Prediction Engine
# -----------------------------------------------------------------------------
def get_team_conference(team_name, lookup_df):
    if lookup_df is None or lookup_df.empty:
        return "Unknown"
    match = lookup_df[lookup_df["team"] == team_name]
    if not match.empty and "conference" in match.columns:
        conf = match.iloc[0]["conference"]
        if pd.notna(conf):
            return str(conf)
    return "Unknown"


def get_tape_df(h_name, a_name, lookup_df):
    h_s = lookup_df[lookup_df["team"] == h_name]
    a_s = lookup_df[lookup_df["team"] == a_name]

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

    h_sos_val = h_s.get("opp_def_strength", h_s.get("roll_pts_allowed", 0.0))
    a_sos_val = a_s.get("opp_def_strength", a_s.get("roll_pts_allowed", 0.0))

    input_data = pd.DataFrame(
        [
            {
                "neutral_site": 1 if is_neutral else 0,
                "h_roll_pts_scored": float(h_s.get("roll_pts_scored", 0.0)),
                "h_roll_ypp": float(h_s.get("roll_ypp", 0.0)),
                "h_roll_ppm": float(h_s.get("roll_ppm", 0.0)),
                "h_roll_turnovers": float(h_s.get("roll_turnovers", 0.0)),
                "h_sos": float(h_sos_val),
                "a_roll_pts_scored": float(a_s.get("roll_pts_scored", 0.0)),
                "a_roll_ypp": float(a_s.get("roll_ypp", 0.0)),
                "a_roll_ppm": float(a_s.get("roll_ppm", 0.0)),
                "a_roll_turnovers": float(a_s.get("roll_turnovers", 0.0)),
                "a_sos": float(a_sos_val),
            }
        ]
    )[FEATURE_COLS]

    # Handle dictionary model output vs single classifier
    if isinstance(model, dict):
        winner_model = model["winner"]
        h_score_model = model.get("h_score")
        a_score_model = model.get("a_score")
    else:
        winner_model = model
        h_score_model = None
        a_score_model = None

    if hasattr(winner_model, "predict_proba"):
        probabilities = winner_model.predict_proba(input_data)[0]
        away_win_prob = float(probabilities[0])
        home_win_prob = float(probabilities[1])
    else:
        st.error("❌ Could not generate predictions. Check that team names from the schedule match training data.")

st.divider()
st.markdown("**How it works:** The model is trained on rolling 3-game averages from the last 3 seasons plus the current season so far, then used to predict all scheduled FBS games this week.")

        st.markdown("---")
        st.markdown("### 📋 Full Week Matchup Overview")

        # Interactive Data Table with Selection Support
        event = st.dataframe(
            schedule_df[
                [
                    "Matchup",
                    "Predicted Winner",
                    "Win Prob",
                    "Proj. Score",
                    "Home Win %",
                    "Away Win %",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="games_table",
        )

        # Sync table click with view details if row selected
        if event and hasattr(event, "selection") and event.selection:
            selected_rows = event.selection.get("rows", [])
            if selected_rows:
                clicked_game = schedule_df.iloc[selected_rows[0]]["Matchup"]
                if clicked_game != selected_game_label:
                    st.info(f"Selected: **{clicked_game}**")

# -----------------------------------------------------------------------------
# TAB 2: MANUAL CUSTOM MATCHUP BUILDER
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Custom Matchup Simulator")
    st.write(
        "Simulate any custom matchup using current rolling team stats and home field advantage."
    )

    teams = sorted(lookup_df["team"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        h_team = st.selectbox("Home Team", teams, index=0, key="custom_h")
    with col2:
        default_away = 1 if len(teams) > 1 else 0
        a_team = st.selectbox(
            "Away Team", teams, index=default_away, key="custom_a"
        )

    is_neutral = st.checkbox("Neutral Site Game", value=False, key="custom_n")

    if h_team == a_team:
        st.warning("⚠️ Please select two different teams.")
    else:
        if st.button("🚀 Predict Custom Matchup", use_container_width=True):
            res = predict_matchup(h_team, a_team, is_neutral, model, lookup_df)
            if res:
                st.markdown("---")
                render_prediction_display(res, is_neutral, lookup_df)