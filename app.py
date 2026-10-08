import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NCAA Football Predictor",
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
    .metric-score-large {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-prob {
        font-size: 1.1rem;
        color: #cbd5e1;
        margin-top: 4px;
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
            f"{h_name}": [
                f"{float(h_s.get('roll_pts_scored', 0.0)):.1f}",
                f"{float(h_s.get('roll_yards', 0.0)):.1f}",
                f"{float(h_s.get('roll_ypp', 0.0)):.2f}",
                f"{float(h_s.get('roll_ppm', 0.0)):.2f}",
                f"{float(h_s.get('roll_turnovers', 0.0)):.1f}",
                f"{float(h_s.get('roll_pen_yds', 0.0)):.1f}",
                f"{float(h_sos_val):.1f}",
            ],
            f"{a_name}": [
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
    if model is None or lookup_df is None:
        return None

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
    )

    # Compute differential features if present in feature dictionary
    input_data["diff_pts_scored"] = input_data["h_roll_pts_scored"] - input_data["a_roll_pts_scored"]
    input_data["diff_ypp"] = input_data["h_roll_ypp"] - input_data["a_roll_ypp"]
    input_data["diff_ppm"] = input_data["h_roll_ppm"] - input_data["a_roll_ppm"]
    input_data["diff_turnovers"] = input_data["h_roll_turnovers"] - input_data["a_roll_turnovers"]
    input_data["diff_sos"] = input_data["h_sos"] - input_data["a_sos"]

    # Slice features matching the exact training set schema
    if isinstance(model, dict) and "features" in model:
        features = model["features"]
        input_data = input_data[features]

    result = {}

    # --- NEW COHERENT FORMAT (Spread + Total + Calibrator) ---
    if isinstance(model, dict) and "spread_model" in model and "total_model" in model:
        pred_spread = float(model["spread_model"].predict(input_data)[0])
        pred_total = float(model["total_model"].predict(input_data)[0])

        home_score = max(0.0, (pred_total + pred_spread) / 2.0)
        away_score = max(0.0, (pred_total - pred_spread) / 2.0)

        # Win probability calibrated directly from predicted margin
        if "calibrator" in model and model["calibrator"] is not None:
            home_prob = float(model["calibrator"].predict_proba([[pred_spread]])[0][1])
        else:
            # Fallback logistic transformation (k ~ 0.16 for CFB point spread)
            home_prob = 1.0 / (1.0 + np.exp(-0.16 * pred_spread))

        result["home_score"] = home_score
        result["away_score"] = away_score
        result["spread"] = pred_spread
        result["total"] = pred_total
        result["home_win_prob"] = home_prob
        result["away_win_prob"] = 1.0 - home_prob

    # --- FALLBACK TO OLD FORMAT (Separate Classifier & Score Regressors) ---
    elif isinstance(model, dict) and "winner" in model:
        winner_model = model["winner"]
        h_score_model = model.get("h_score")
        a_score_model = model.get("a_score")

        if hasattr(winner_model, "predict_proba"):
            probs = winner_model.predict_proba(input_data)[0]
            result["away_win_prob"] = float(probs[0])
            result["home_win_prob"] = float(probs[1])

        if h_score_model and a_score_model:
            result["home_score"] = float(h_score_model.predict(input_data)[0])
            result["away_score"] = float(a_score_model.predict(input_data)[0])

    return result


def render_prediction_display(res, h_team, a_team, lookup_df):
    if not res:
        st.warning("Prediction data unavailable for this matchup.")
        return

    home_win = res["home_win_prob"] >= res["away_win_prob"]
    favored_team = h_team if home_win else a_team
    favored_prob = res["home_win_prob"] if home_win else res["away_win_prob"]

    home_score = res.get("home_score", 0.0)
    away_score = res.get("away_score", 0.0)
    spread = abs(res.get("spread", home_score - away_score))

    # Banner displaying projected outcome
    st.markdown(
        f"""
        <div class="winner-banner">
            <div class="winner-title"> Projected Winner: {favored_team} ({favored_prob:.1%} Win Prob)</div>
            <div class="winner-subtitle">Projected Scoreline: {h_team} {home_score:.1f} — {a_team} {away_score:.1f} (Margin: {spread:.1f} pts)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([4, 1, 4])
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-team"> {h_team}</div>
                <div class="metric-score-large">{home_score:.1f}</div>
                <div class="metric-prob">Probability: <b>{res['home_win_prob']:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-team">{a_team}</div>
                <div class="metric-score-large">{away_score:.1f}</div>
                <div class="metric-prob">Probability: <b>{res['away_win_prob']:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Key Advantage Summary (Explaining Outcome)
    st.markdown("### Key Matchup Drivers")
    h_s = lookup_df[lookup_df["team"] == h_team].iloc[0] if not lookup_df[
        lookup_df["team"] == h_team].empty else None
    a_s = lookup_df[lookup_df["team"] == a_team].iloc[0] if not lookup_df[
        lookup_df["team"] == a_team].empty else None

    if h_s is not None and a_s is not None:
        ypp_diff = float(h_s.get("roll_ypp", 0.0)) - float(a_s.get("roll_ypp", 0.0))
        to_diff = float(h_s.get("roll_turnovers", 0.0)) - float(a_s.get("roll_turnovers", 0.0))
        sos_diff = float(h_s.get("opp_def_strength", 0.0)) - float(a_s.get("opp_def_strength", 0.0))

        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            edge = f"{h_team} (+{abs(ypp_diff):.2f} YPP)" if ypp_diff > 0 else f"{a_team} (+{abs(ypp_diff):.2f} YPP)"
            st.info(f"**Yards / Play Edge:**\n\n{edge}")
        with e_col2:
            # Lower turnovers is better
            edge = f"{h_team} ({abs(to_diff):.1f} fewer TO/gm)" if to_diff < 0 else f"{a_team} ({abs(to_diff):.1f} fewer TO/gm)"
            st.info(f"**Turnover Advantage:**\n\n{edge}")
        with e_col3:
            edge = f"{h_team} (+{abs(sos_diff):.1f} SOS)" if sos_diff > 0 else f"{a_team} (+{abs(sos_diff):.1f} SOS)"
            st.info(f"**Schedule Strength Edge:**\n\n{edge}")

    st.markdown("### Matchup Tale of the Tape")
    tape_df = get_tape_df(h_team, a_team, lookup_df)
    if not tape_df.empty:
        st.table(tape_df)


@st.cache_data
def get_all_predictions(df, lookup_df):
    model = load_model()
    predictions = []
    for _, row in df.iterrows():
        h_team = row["h_team"]
        a_team = row["a_team"]
        res = predict_matchup(h_team, a_team, False, model, lookup_df)

        if not res:
            predictions.append(
                {
                    "Predicted Winner": "TBD",
                    "Win Prob": "TBD",
                    "Proj. Score": "TBD",
                }
            )
            continue

        winner = h_team if res["home_win_prob"] >= res["away_win_prob"] else a_team
        win_prob = max(res["home_win_prob"], res["away_win_prob"])
        proj_score = f"{h_team} {res.get('home_score', 0):.1f} - {a_team} {res.get('away_score', 0):.1f}"

        predictions.append(
            {
                "Predicted Winner": winner,
                "Win Prob": f"{win_prob:.1%}",
                "Proj. Score": proj_score,
            }
        )
    return pd.DataFrame(predictions, index=df.index)


# -----------------------------------------------------------------------------
# 4. App Execution & Layout
# -----------------------------------------------------------------------------
model = load_model()
lookup_df = load_lookup_data()
schedule_df = load_scheduled_games()

# Sidebar Filters
st.sidebar.markdown("### Filters")

if not schedule_df.empty:
    # Week Selection
    all_weeks = sorted(schedule_df["week"].dropna().unique().tolist())
    default_week = 6 if 6 in all_weeks else all_weeks[0]
    selected_week = st.sidebar.selectbox("Week", all_weeks, index=all_weeks.index(default_week))
    filtered_df = schedule_df[schedule_df["week"] == selected_week]

    # Division Selection
    all_divisions = sorted(
        list(
            set(filtered_df["home_classification"].dropna().unique())
            | set(filtered_df["away_classification"].dropna().unique())
        )
    )
    default_div = "fbs" if "fbs" in all_divisions else (all_divisions[0] if all_divisions else None)
    div_options = ["All"]
    if isinstance(all_divisions, list):
        div_options.extend(all_divisions)

    div_index = 0
    if default_div in div_options:
        div_index = div_options.index(default_div)
    selected_division = st.sidebar.selectbox("Division", div_options, index=div_index)

    if selected_division != "All":
        filtered_df = filtered_df[
            (filtered_df["home_classification"] == selected_division)
            | (filtered_df["away_classification"] == selected_division)
            ]

    # Conference Selection
    all_conferences = sorted(
        list(
            set(filtered_df["home_conference"].dropna().unique())
            | set(filtered_df["away_conference"].dropna().unique())
        )
    )
    selected_conference = st.sidebar.selectbox("Conference", ["All"] + all_conferences)

    if selected_conference != "All":
        filtered_df = filtered_df[
            (filtered_df["home_conference"] == selected_conference)
            | (filtered_df["away_conference"] == selected_conference)
            ]

    schedule_df = filtered_df

# Model Feature Importance Inspection in Sidebar
if isinstance(model, dict) and "spread_model" in model and "features" in model:
    with st.sidebar.expander("Model Feature Importance"):
        importances = model["spread_model"].feature_importances_
        feature_names = model["features"]
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        st.dataframe(
            fi_df.style.format({"Importance": "{:.1%}"}),
            use_container_width=True,
            hide_index=True
        )


st.title("NCAA Football Game Predictor")
st.markdown(
    "**How it works:** The engine predicts game spread and total points, then calculates "
    "a win probability calibrated directly to the predicted scoreline."
)

st.divider()
st.markdown("### Full Week Matchup Overview")

if not schedule_df.empty:
    schedule_df["Matchup"] = schedule_df["a_team"] + " @ " + schedule_df["h_team"]

    # Compute coherent predictions
    pred_df = get_all_predictions(schedule_df, lookup_df)
    display_df = pd.concat([schedule_df, pred_df], axis=1)

    event = st.dataframe(
        display_df[
            [
                "Matchup",
                "Predicted Winner",
                "Win Prob",
                "Proj. Score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="games_table",
    )

    # Sync table click with detailed prediction view
    if event and hasattr(event, "selection") and event.selection:
        selected_rows = event.selection.get("rows", [])
        if selected_rows:
            row = display_df.iloc[selected_rows[0]]
            h_team = row["h_team"]
            a_team = row["a_team"]

            st.divider()
            st.markdown(f"## Detailed Analysis: **{a_team} @ {h_team}**")

            raw_neutral = row.get("neutral_site", False)
            is_neutral = bool(raw_neutral) if pd.notna(raw_neutral) else False
            prediction = predict_matchup(h_team, a_team, is_neutral, model, lookup_df)
            if prediction:
                render_prediction_display(prediction, h_team, a_team, lookup_df)
else:
    st.info("No scheduled games found matching the selected filters.")