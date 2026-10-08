import os
import joblib
import pandas as pd
import streamlit as st

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
            f"{h_name}": [
                f"{float(h_s.get('roll_pts_scored', 0.0)):.1f}",
                f"{float(h_s.get('roll_yards', 0.0)):.1f}",
                f"{float(h_s.get('roll_ypp', 0.0)):.2f}",
                f"{float(h_s.get('roll_ppm', 0.0)):.2f}",
                f"{float(h_s.get('roll_turnovers', 0.0)):.1f}",
                f"{float(h_s.get('roll_pen_yds', 0.0)):.1f}",
                f"{float(h_sos_val):.1f}",
            ],
            f"✈{a_name}": [
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

    result = {}
    if hasattr(winner_model, "predict_proba"):
        probabilities = winner_model.predict_proba(input_data)[0]
        result["away_win_prob"] = float(probabilities[0])
        result["home_win_prob"] = float(probabilities[1])
    
    if h_score_model and a_score_model:
        result["home_score"] = float(h_score_model.predict(input_data)[0])
        result["away_score"] = float(a_score_model.predict(input_data)[0])
        
    return result


def render_prediction_display(res, h_team, a_team, lookup_df):
    st.subheader("Prediction Results")
    st.write(f"Home win probability: {res['home_win_prob']:.2%}")
    st.write(f"Away win probability: {res['away_win_prob']:.2%}")
    
    if "home_score" in res and "away_score" in res:
        st.write(f"### Projected Score: {h_team} {res['home_score']:.1f} - {a_team} {res['away_score']:.1f}")
    
    st.subheader("Matchup Stats Comparison")
    tape_df = get_tape_df(h_team, a_team, lookup_df)
    if not tape_df.empty:
        st.table(tape_df)

# Load Data
model = load_model()
lookup_df = load_lookup_data()
schedule_df = load_scheduled_games()

# Filters
st.sidebar.markdown("### Filters")
all_divisions = sorted(list(set(schedule_df["home_classification"].dropna().unique()) | set(schedule_df["away_classification"].dropna().unique())))
default_div = "fbs" if "fbs" in all_divisions else (all_divisions[0] if all_divisions else None)
selected_division = st.sidebar.selectbox("Division", ["All"] + all_divisions, index=(["All"] + all_divisions).index(default_div) if default_div in (["All"] + all_divisions) else 0)

filtered_df = schedule_df
if selected_division != "All":
    filtered_df = schedule_df[(schedule_df["home_classification"] == selected_division) | (schedule_df["away_classification"] == selected_division)]

all_conferences = sorted(list(set(filtered_df["home_conference"].dropna().unique()) | set(filtered_df["away_conference"].dropna().unique())))
selected_conference = st.sidebar.selectbox("Conference", ["All"] + all_conferences)

if selected_conference != "All":
    filtered_df = filtered_df[(filtered_df["home_conference"] == selected_conference) | (filtered_df["away_conference"] == selected_conference)]

schedule_df = filtered_df

st.divider()
st.markdown(f"**How it works:** The model is trained on rolling 3-game averages from the last 3 seasons plus the current season so far, then used to predict all scheduled {selected_division.upper()} games this week.")

st.markdown("---")
st.markdown("### Full Week Matchup Overview")

# Interactive Data Table with Selection Support
schedule_df["Matchup"] = schedule_df["a_team"] + " @ " + schedule_df["h_team"]
schedule_df["Predicted Winner"] = "TBD" # Placeholder
schedule_df["Win Prob"] = "TBD" # Placeholder
schedule_df["Proj. Score"] = "TBD" # Placeholder
schedule_df["Home Win %"] = "TBD" # Placeholder
schedule_df["Away Win %"] = "TBD" # Placeholder

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
        row = schedule_df.iloc[selected_rows[0]]
        h_team = row["h_team"]
        a_team = row["a_team"]
        st.info(f"Selected: **{a_team} @ {h_team}**")
        
        # Predict and render
        is_neutral = False # Assuming False for scheduled games in table
        prediction = predict_matchup(h_team, a_team, is_neutral, model, lookup_df)
        if prediction:
            render_prediction_display(prediction, h_team, a_team, lookup_df)