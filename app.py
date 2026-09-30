import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page & Custom CSS Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NCAA Football Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    h_sos_val = h_s.get("opp_def_strength", h_s.get("roll_pts_allowed", 0.0))
    a_sos_val = a_s.get("opp_def_strength", a_s.get("roll_pts_allowed", 0.0))

    return pd.DataFrame(
        {
            "Metric": [
                "Pts / Game (Rolling)",
                "Yards / Game (Rolling)",
                "Yards / Play",
                "Pts / Minute",
                "Turnovers / Game",
                "Penalty Yards / Game (Rolling)",
                "Opp. Def Strength (SOS)",
            ],
            f"🏠 {h_name}": [
                f"{float(h_s.get('roll_pts_scored', 0.0)):.1f}",
                f"{float(h_s.get('roll_yards', 0.0)):.1f}",
                f"{float(h_s.get('roll_ypp', 0.0)):.2f}",
                f"{float(h_s.get('roll_ppm', 0.0)):.2f}",
                f"{float(h_s.get('roll_turnovers', 0.0)):.1f}",
                f"{float(h_s.get('roll_pen_yds', 0.0)):.1f}",
                f"{float(h_sos_val):.1f}",
            ],
            f"✈️ {a_name}": [
                f"{float(a_s.get('roll_pts_scored', 0.0)):.1f}",
                f"{float(a_s.get('roll_yards', 0.0)):.1f}",
                f"{float(a_s.get('roll_ypp', 0.0)):.2f}",
                f"{float(a_s.get('roll_ppm', 0.0)):.2f}",
                f"{float(a_s.get('roll_turnovers', 0.0)):.1f}",
                f"{float(a_s.get('roll_pen_yds', 0.0)):.1f}",
                f"{float(a_sos_val):.1f}",
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
        raw_pred = float(winner_model.predict(input_data)[0])
        home_win_prob = raw_pred
        away_win_prob = 1.0 - raw_pred

    home_win_prob = float(np.clip(home_win_prob, 0.0, 1.0))
    away_win_prob = float(np.clip(away_win_prob, 0.0, 1.0))

    h_score_pred = (
        float(h_score_model.predict(input_data)[0]) if h_score_model else None
    )
    a_score_pred = (
        float(a_score_model.predict(input_data)[0]) if a_score_model else None
    )

    return {
        "home_team": h_team,
        "away_team": a_team,
        "home_prob": home_win_prob,
        "away_prob": away_win_prob,
        "h_score": h_score_pred,
        "a_score": a_score_pred,
        "winner": h_team if home_win_prob >= away_win_prob else a_team,
        "confidence": max(home_win_prob, away_win_prob),
    }


def render_prediction_display(res, is_neutral, lookup_df):
    st.markdown("### 🎯 Matchup Prediction Details")

    c1, c2, c3 = st.columns([4, 1, 4])

    with c1:
        score_str = (
            f"<div class='metric-score'>Proj. Score: <b>{res['h_score']:.1f}</b></div>"
            if res["h_score"] is not None
            else ""
        )
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-team'>🏠 {res['home_team']}</div>
                <div class='metric-prob'>{res['home_prob']:.1%}</div>
                {score_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("<div class='vs-divider'>VS</div>", unsafe_allow_html=True)

    with c3:
        score_str = (
            f"<div class='metric-score'>Proj. Score: <b>{res['a_score']:.1f}</b></div>"
            if res["a_score"] is not None
            else ""
        )
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-team'>✈️ {res['away_team']}</div>
                <div class='metric-prob'>{res['away_prob']:.1%}</div>
                {score_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

    score_banner = ""
    if res["h_score"] is not None and res["a_score"] is not None:
        score_banner = f" | Projected Score: **{res['home_team']} {res['h_score']:.1f} - {res['a_score']:.1f} {res['away_team']}**"

    st.markdown(
        f"""
        <div class='winner-banner'>
            <p class='winner-title'>Predicted Winner: {res['winner']}</p>
            <p class='winner-subtitle'>Confidence: {res['confidence']:.1%}{score_banner}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 Tale of the Tape")
    tape_df = get_tape_df(res["home_team"], res["away_team"], lookup_df)
    if not tape_df.empty:
        st.table(tape_df)


# -----------------------------------------------------------------------------
# 4. Main App Layout
# -----------------------------------------------------------------------------
st.title("🏈 NCAA Football Game Predictor")

model = load_model()
lookup_df = load_lookup_data()
stats_df = load_full_stats()
scheduled_df = load_scheduled_games()

if model is None or lookup_df is None:
    st.error(
        "❌ Missing required model or lookup data. Ensure `models/ncaa_model.pkl` and `data/team_lookup.csv` exist."
    )
    st.stop()

# Build Combined Game Dataset
combined_df = pd.DataFrame()
if stats_df is not None and not stats_df.empty:
    combined_df = pd.concat([combined_df, stats_df], ignore_index=True, sort=False)
if scheduled_df is not None and not scheduled_df.empty:
    combined_df = pd.concat([combined_df, scheduled_df], ignore_index=True, sort=False)

if combined_df.empty:
    st.warning("No schedule or detailed stats data available.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filters")

# 1. Season Filter
available_years = sorted(combined_df["year"].dropna().unique(), reverse=True)
selected_year = st.sidebar.selectbox("Season / Year", available_years, index=0)

year_df = combined_df[combined_df["year"] == selected_year]

# 2. Week Filter
available_weeks = sorted(year_df["week"].dropna().unique(), reverse=True)
selected_week = st.sidebar.selectbox("Week", available_weeks, index=0)

week_df = year_df[year_df["week"] == selected_week]

# 3. Conference Filter
available_conferences = []
if "conference" in lookup_df.columns:
    available_conferences = sorted(
        lookup_df["conference"].dropna().astype(str).unique()
    )

selected_conference = "All Conferences"
if available_conferences:
    conf_options = ["All Conferences"] + available_conferences
    selected_conference = st.sidebar.selectbox("Conference Filter", conf_options, index=0)

# 4. Search Filter
search_query = st.sidebar.text_input("🔎 Search Team", "")

# Apply Filters to Games List
schedule = week_df[["h_team", "a_team", "neutral_site"]].drop_duplicates().dropna()

if selected_conference != "All Conferences":
    schedule["h_conf"] = schedule["h_team"].apply(
        lambda t: get_team_conference(t, lookup_df)
    )
    schedule["a_conf"] = schedule["a_team"].apply(
        lambda t: get_team_conference(t, lookup_df)
    )
    schedule = schedule[
        (schedule["h_conf"] == selected_conference)
        | (schedule["a_conf"] == selected_conference)
    ]

if search_query.strip():
    q = search_query.strip().lower()
    schedule = schedule[
        schedule["h_team"].str.lower().str.contains(q)
        | schedule["a_team"].str.lower().str.contains(q)
    ]

# Navigation Tabs
tab1, tab2 = st.tabs(["📅 Weekly Schedule & Predictions", "⚔️ Custom Matchup Builder"])

# -----------------------------------------------------------------------------
# TAB 1: SCHEDULE & PREDICTIONS EXPLORER
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"Schedule for {selected_year} — Week {selected_week}")

    if schedule.empty:
        st.info("No games match the current filter criteria.")
    else:
        # Pre-compute predictions for all filtered games
        game_list = []
        for _, row in schedule.iterrows():
            h, a, n = row["h_team"], row["a_team"], bool(row["neutral_site"])
            pred = predict_matchup(h, a, n, model, lookup_df)
            if pred:
                label = f"{a} @ {h}" if not n else f"{a} vs {h} (Neutral)"
                proj_score = (
                    f"{pred['h_score']:.0f}-{pred['a_score']:.0f}"
                    if pred["h_score"] is not None
                    else "N/A"
                )
                game_list.append(
                    {
                        "Matchup": label,
                        "Away Team": a,
                        "Home Team": h,
                        "Neutral": n,
                        "Predicted Winner": pred["winner"],
                        "Win Prob": f"{pred['confidence']:.1%}",
                        "Proj. Score": proj_score,
                        "Home Win %": f"{pred['home_prob']:.1%}",
                        "Away Win %": f"{pred['away_prob']:.1%}",
                    }
                )

        schedule_df = pd.DataFrame(game_list)

        # Interactive Game Selection
        st.markdown("#### 👆 Click or Select a Game to Inspect Details")

        selected_game_label = st.selectbox(
            "Select Game Matchup:",
            options=schedule_df["Matchup"].tolist(),
            index=0,
            key="game_selector",
        )

        st.markdown("---")

        # Render Detailed Prediction Card for Selected Game
        selected_row = schedule_df[
            schedule_df["Matchup"] == selected_game_label
        ].iloc[0]
        res = predict_matchup(
            selected_row["Home Team"],
            selected_row["Away Team"],
            selected_row["Neutral"],
            model,
            lookup_df,
        )

        if res:
            render_prediction_display(res, selected_row["Neutral"], lookup_df)

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