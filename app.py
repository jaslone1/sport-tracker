import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib 
import altair as alt # <--- NEW: Import Altair for better visualizations

# --- Configuration ---
st.set_page_config(
    page_title="NCAA FBS Football Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SEO META TAG INJECTION (Placed after config) ---
st.markdown("""
    <meta name="description" content="NCAA FBS football data analysis tool displaying team performance, Elo ratings, win rates, and conference comparisons. Filterable by conference.">
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# --- Data Loading and Filtering Functions (No Change to load_data, analysis functions) ---
# --------------------------------------------------------------------------

@st.cache_data
def load_data(path):
    """Load and filter the processed CSV data for FBS teams."""
    
    if not os.path.exists(path):
        st.error(f"Error: Processed file not found at {path}. Please ensure your data processing script was run successfully to create this file.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(path)
        
        # Ensure all necessary analytical columns are numeric
        numeric_cols = ['homePoints', 'awayPoints', 'homePregameElo', 'homePostgameElo', 
                        'awayPregameElo', 'awayPostgameElo', 'attendance', 'point_differential', 
                        'home_elo_change', 'home_win', 'away_elo_change', 'neutralSite']
        for col in numeric_cols:
             if col in df.columns:
                 # Coerce boolean 'neutralSite' to boolean/int for consistency
                 if col == 'neutralSite':
                     df[col] = df[col].astype(bool) 
                 else:
                     df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # NOTE: We need to calculate Elo difference here for unplayed games since 
        # the model needs it, but the training script calculates it for historical data.
        if 'homePregameElo' in df.columns and 'awayPregameElo' in df.columns:
            df['elo_diff'] = df['homePregameElo'] - df['awayPregameElo']
            # Also create the 'home_advantage' feature needed by the prediction models
            df['home_advantage'] = np.where(df['neutralSite'] == True, 0, 1)

        # Drop games with missing outcomes for historical analysis (though this is 'odds.csv' now)
        # We will keep all rows for prediction, but drop missing confs
        df.dropna(subset=['homeConference', 'awayConference'], inplace=True)
        
        # FBS FILTERING LOGIC: Exclude games involving clearly non-FBS teams
        non_fbs_keywords = ['FCS', 'II', 'III', 'D-2', 'D-3', 'NAIA']
        
        mask_home = ~df['homeConference'].astype(str).str.contains('|'.join(non_fbs_keywords), case=False, na=False)
        mask_away = ~df['awayConference'].astype(str).str.contains('|'.join(non_fbs_keywords), case=False, na=False)
        
        df = df[mask_home & mask_away]
        
        return df
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

@st.cache_data
def create_all_up_analysis(df):
    """
    Creates a single-row-per-team-per-game structure for all-up performance metrics
    by combining the home and away columns, and then aggregates results by team.
    """
    # ... (content remains the same as your original script for create_all_up_analysis)
    
    # 1. Create a DataFrame for Home Team Stats
    home_df = df.copy()
    home_df['Team'] = home_df['homeTeam']
    home_df['Conference'] = home_df['homeConference']
    home_df['Venue'] = 'Home'
    home_df['Points_Scored'] = home_df['homePoints']
    home_df['Points_Allowed'] = home_df['awayPoints']
    home_df['Elo_Change'] = home_df['home_elo_change']
    home_df['Game_Outcome'] = np.where(home_df.get('home_win', 0) == 1, 'Win', 'Loss')
    
    # 2. Create a DataFrame for Away Team Stats
    away_df = df.copy()
    away_df['Team'] = away_df['awayTeam']
    away_df['Conference'] = away_df['awayConference']
    away_df['Venue'] = 'Away'
    away_df['Points_Scored'] = away_df['awayPoints']
    away_df['Points_Allowed'] = away_df['homePoints']
    away_df['Elo_Change'] = away_df['away_elo_change']
    away_df['Game_Outcome'] = np.where(away_df.get('home_win', 0) == 0, 'Win', 'Loss') 

    # 3. Concatenate and select relevant columns
    all_games_df = pd.concat([
        home_df[['Team', 'Conference', 'Venue', 'Points_Scored', 'Points_Allowed', 'Elo_Change', 'Game_Outcome']].dropna(subset=['Points_Scored']),
        away_df[['Team', 'Conference', 'Venue', 'Points_Scored', 'Points_Allowed', 'Elo_Change', 'Game_Outcome']].dropna(subset=['Points_Scored'])
    ], ignore_index=True)

    # 4. Group by Team and aggregate metrics
    all_up_analysis = all_games_df.groupby('Team').agg(
        Total_Games=('Game_Outcome', 'count'),
        Wins=('Game_Outcome', lambda x: (x == 'Win').sum()),
        Losses=('Game_Outcome', lambda x: (x == 'Loss').sum()),
        Win_Rate=('Game_Outcome', lambda x: (x == 'Win').sum() / x.count()),
        Avg_Points_Scored=('Points_Scored', 'mean'),
        Avg_Points_Allowed=('Points_Allowed', 'mean'),
        Net_Elo_Change=('Elo_Change', 'sum')
    ).sort_values(by='Win_Rate', ascending=False)
    
    # 5. Final Formatting
    all_up_analysis['Conference'] = all_games_df.groupby('Team')['Conference'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
    all_up_analysis['Point_Diff_Per_Game'] = all_up_analysis['Avg_Points_Scored'] - all_up_analysis['Avg_Points_Allowed']
    all_up_analysis = all_up_analysis.reset_index()
    all_up_analysis = all_up_analysis[[
        'Team', 'Conference', 'Total_Games', 'Wins', 'Losses', 'Win_Rate', 
        'Avg_Points_Scored', 'Avg_Points_Allowed', 'Point_Diff_Per_Game', 'Net_Elo_Change'
    ]]

    return all_up_analysis

@st.cache_data
def create_game_log_analysis(df):
    """
    Creates a detailed game-by-game log for every team, including opponent and score.
    NOTE: Only works for games that have been played (homePoints/awayPoints are not null).
    """
    df_played = df.dropna(subset=['homePoints', 'awayPoints']).copy()
    if df_played.empty:
        return pd.DataFrame()
        
    # 1. Home Game Log
    home_log = df_played.copy()
    home_log['Team'] = home_log['homeTeam']
    home_log['Opponent'] = home_log['awayTeam']
    home_log['Venue'] = np.where(home_log['neutralSite'] == True, 'Neutral', 'Home')
    home_log['Score_Display'] = home_log['homePoints'].astype(int).astype(str) + " - " + home_log['awayPoints'].astype(int).astype(str)
    home_log['Outcome'] = np.where(home_log['home_win'] == 1, 'W', 'L')
    
    # 2. Away Game Log
    away_log = df_played.copy()
    away_log['Team'] = away_log['awayTeam']
    away_log['Opponent'] = away_log['homeTeam']
    away_log['Venue'] = np.where(away_log['neutralSite'] == True, 'Neutral', 'Away')
    away_log['Score_Display'] = away_log['awayPoints'].astype(int).astype(str) + " - " + away_log['homePoints'].astype(int).astype(str)
    away_log['Outcome'] = np.where(away_log['home_win'] == 0, 'W', 'L')

    # 3. Combine Logs
    game_log = pd.concat([home_log, away_log], ignore_index=True)
    
    # 4. Select and format final columns
    final_log = game_log[[
        'Team', 
        'Opponent', 
        'Venue', 
        'Outcome', 
        'Score_Display',
        'homePregameElo', 
        'home_elo_change'
    ]].sort_values(by=['Team', 'Venue'], ascending=[True, False])

    final_log.columns = [
        'Team', 'Opponent', 'Venue', 'Outcome', 'Score (Team - Opp)', 
        'Home Team Pregame Elo', 'Home Team Elo Change'
    ]
    
    return final_log

# --------------------------------------------------------------------------
# --- UPDATED: Machine Learning Prediction Function ---
# --------------------------------------------------------------------------

@st.cache_data
def get_upset_predictions(df_input):
    """
    Loads both Logistic Regression and XGBoost models and calculates the 
    upset probability for each unplayed game.
    """
    
    # Define model file paths
    LOGREG_MODEL_PATH = 'ml_model/logistic_regression_upset_model.pkl'
    XGBOOST_MODEL_PATH = 'ml_model/xgboost_classifier_upset_model.pkl' # <-- Updated Path!

    # Create the features needed by the models (must match training script)
    # NOTE: The training script used 'elo_diff' and 'neutralSite'. 
    # The models are generally trained on elo_diff and home_advantage (or neutralSite)
    
    df_features = df_input[['elo_diff', 'neutralSite']].copy()
    
    # --- Load and Predict with Logistic Regression ---
    try:
        logreg_model = joblib.load(LOGREG_MODEL_PATH)
        # Predict probability of class 1 (Upset)
        df_input['logreg_upset_proba'] = logreg_model.predict_proba(df_features)[:, 1]
    except FileNotFoundError:
        st.warning(f"LogReg model not found at {LOGREG_MODEL_PATH}. Skipping LogReg predictions.")
        df_input['logreg_upset_proba'] = 0.0

    # --- Load and Predict with XGBoost Classifier ---
    try:
        xgb_model = joblib.load(XGBOOST_MODEL_PATH)
        # Predict probability of class 1 (Upset)
        df_input['xgb_upset_proba'] = xgb_model.predict_proba(df_features)[:, 1]
    except FileNotFoundError:
        st.warning(f"XGBoost model not found at {XGBOOST_MODEL_PATH}. Skipping XGBoost predictions.")
        df_input['xgb_upset_proba'] = 0.0
    
    return df_input


# --------------------------------------------------------------------------
# --- Main Application Logic ---
# --------------------------------------------------------------------------

DATA_PATH = 'data/odds.csv'
df = load_data(DATA_PATH)
df = get_upset_predictions(df) # <-- Call prediction function

if df.empty:
    st.stop()
    
# Split data into played (historical) and unplayed (prediction)
df_played = df.dropna(subset=['homePoints', 'awayPoints']).copy()
df_unplayed = df[df['homePoints'].isna() | df['awayPoints'].isna()].copy()


# =========================================================================
# --- SIDEBAR: FILTERING ---
# =========================================================================

st.sidebar.title("Data Filters")
all_conferences = sorted(df['homeConference'].unique().tolist())

selected_conferences = st.sidebar.multiselect(
    "Select Home Conference(s):", 
    options=all_conferences,
    default=all_conferences
)

# Apply filter to the PLAYED data for historical analysis
if selected_conferences:
    df_filtered = df_played[df_played['homeConference'].isin(selected_conferences)].copy()
else:
    df_filtered = df_played[0:0] # Show empty DataFrame if nothing is selected

# Apply filter to the UNPLAYED data for prediction display
if selected_conferences:
    df_unplayed_filtered = df_unplayed[df_unplayed['homeConference'].isin(selected_conferences)].copy()
else:
    df_unplayed_filtered = df_unplayed[0:0]

# Generate analysis tables based on filtered PLAYED data
all_up_df = create_all_up_analysis(df_filtered)
game_log_df = create_game_log_analysis(df_filtered)

# =========================================================================
# --- DASHBOARD CONTENT ---
# =========================================================================

st.title("🏈 NCAA FBS Game Analysis & Upset Prediction")
st.markdown("Historical metrics for games involving **Football Bowl Subdivision (FBS)** teams, with **Home Conference** filtering enabled in the sidebar.")
st.caption(f"Showing **{len(df_filtered):,}** Played Games and **{len(df_unplayed_filtered):,}** Unplayed Games.")

col1, col2, col3 = st.columns(3)
col1.metric("Total FBS Played Games", f"{len(df_filtered):,}")
col2.metric("Average Home Point Differential (Played)", f"{df_filtered.get('point_differential', pd.Series()).mean():.2f} pts")
col3.metric("Home Win Percentage (Played)", f"{df_filtered.get('home_win', pd.Series()).mean() * 100:.2f}%")

st.markdown("---")

# -------------------------------------------------------------------------
# --- 1. AGGREGATED ALL-UP TEAM ANALYSIS ---
# ... (Historical sections 1 & 2 remain the same, using df_filtered)
# -------------------------------------------------------------------------
st.header("🌐 All-Up Team Performance Summary (Aggregated)")
st.markdown("Team performance across **all games**, sorted by **Overall Win Rate**.")
st.dataframe(all_up_df.round(2), use_container_width=True)

# -------------------------------------------------------------------------
# --- 2. DETAILED GAME LOG TABLE ---
# -------------------------------------------------------------------------
st.header("📝 Detailed Game Log: All Games")
st.markdown("A game-by-game breakdown for every team.")

team_list = sorted(game_log_df['Team'].unique().tolist())
selected_team = st.selectbox(
    "Select Team to View Game Log:", 
    team_list,
    index=None,
    placeholder="Select a Team..."
)

if selected_team:
    log_filtered = game_log_df[game_log_df['Team'] == selected_team]
    st.dataframe(log_filtered.reset_index(drop=True), use_container_width=True)
else:
    st.info("Select a team above to view its detailed game log.")

st.markdown("---")

# -------------------------------------------------------------------------
# --- 3. ML UPSET TRACKERS (TWO MODELS) --- <--- NEW COMPARISON SECTION
# -------------------------------------------------------------------------
st.header("💥 Machine Learning Upset Predictions (Unplayed Games)")
st.markdown("Predictive probabilities from two different models for games with **unknown results**.")

if not df_unplayed_filtered.empty:
    
    # Identify the underdog by Elo for display
    df_unplayed_filtered['Underdog'] = np.where(
        df_unplayed_filtered['elo_diff'] < 0, 
        df_unplayed_filtered['homeTeam'], 
        df_unplayed_filtered['awayTeam']
    )
    
    base_cols = ['homeTeam', 'awayTeam', 'homePregameElo', 'awayPregameElo', 'Underdog']
    
    col_logreg, col_xgb = st.columns(2)
    
    # --- Logistic Regression Column ---
    with col_logreg:
        st.subheader("📊 Model 1: Logistic Regression")
        st.markdown("**Strength: Interpretability & Calibration.** Simple linear model's perspective.")
        
        logreg_candidates = df_unplayed_filtered.sort_values(by='logreg_upset_proba', ascending=False).head(25)
        
        # Display table
        st.dataframe(logreg_candidates[base_cols + ['logreg_upset_proba']].round({'logreg_upset_proba': 3}), use_container_width=True)
        
        # Display chart
        st.caption("Distribution of Predicted Upset Probabilities")
        logreg_chart = alt.Chart(logreg_candidates).mark_bar().encode(
            x=alt.X('logreg_upset_proba', bin=alt.Bin(maxbins=10), title="Upset Probability (LogReg)"),
            y=alt.Y('count()', title="Number of Games"),
            tooltip=['logreg_upset_proba', 'count()']
        ).properties(height=200)
        st.altair_chart(logreg_chart, use_container_width=True)


    # --- XGBoost Column ---
    with col_xgb:
        st.subheader("🌳 Model 2: XGBoost Classifier")
        st.markdown("**Strength: Accuracy.** Advanced, non-linear model for complex patterns.")

        xgb_candidates = df_unplayed_filtered.sort_values(by='xgb_upset_proba', ascending=False).head(25)
        
        # Display table
        st.dataframe(xgb_candidates[base_cols + ['xgb_upset_proba']].round({'xgb_upset_proba': 3}), use_container_width=True)
        
        # Display chart
        st.caption("Distribution of Predicted Upset Probabilities")
        xgb_chart = alt.Chart(xgb_candidates).mark_bar(color='#008080').encode(
            x=alt.X('xgb_upset_proba', bin=alt.Bin(maxbins=10), title="Upset Probability (XGBoost)"),
            y=alt.Y('count()', title="Number of Games"),
            tooltip=['xgb_upset_proba', 'count()']
        ).properties(height=200)
        st.altair_chart(xgb_chart, use_container_width=True)
        
else:
    st.info("No unplayed games found in the selected conferences to predict upsets.")

st.markdown("---") 

# -------------------------------------------------------------------------
# --- 4. GAME PERFORMANCE (ELO) --- (Section numbers adjusted)
# -------------------------------------------------------------------------
st.header("📈 Game Performance Relative to Expectation (Elo Change)")
# ... (Rest of historical ELO sections 4 & 5 remain the same, using df_filtered)
st.subheader("Top 50 Games with Highest Home Elo Gain (Overachievers)")
elo_gain_df = df_filtered.sort_values(by='home_elo_change', ascending=False).head(50) 
st.dataframe(elo_gain_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                           'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True), use_container_width=True)

st.subheader("Top 50 Games with Highest Home Elo Loss (Underachievers)")
elo_loss_df = df_filtered.sort_values(by='home_elo_change', ascending=True).head(50) 
st.dataframe(elo_loss_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                           'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True), use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------------------
# --- 5. CONFERENCE & DISTRIBUTION --- (Section numbers adjusted)
# -------------------------------------------------------------------------
col_conf, col_chart = st.columns([1, 2])

with col_conf:
    st.subheader("🏆 Home Conference Performance")

    conference_analysis = df_filtered.groupby('homeConference').agg(
        Total_Games=('home_win', 'count'),  
        Home_Win_Rate=('home_win', 'mean'),
        Avg_Home_Elo_Change=('home_elo_change', 'mean'),
        Avg_Point_Differential=('point_differential', 'mean')
    ).sort_values(by='Home_Win_Rate', ascending=False).reset_index()

    conference_analysis.columns = ['Conference', 'Total Games', 'Home Win Rate', 
                                   'Avg Home Elo Change', 'Avg Point Differential']

    st.dataframe(conference_analysis.round(2), use_container_width=True)

with col_chart:
    st.subheader("Point Differential Distribution")
    st.caption("Histogram showing margins of victory (Home Score - Away Score).")
    chart_data = df_filtered['point_differential'].value_counts().sort_index()
    chart_data = chart_data[chart_data.index.to_series().between(-50, 50, inclusive='both')]
    st.bar_chart(chart_data)


# --- Raw Data Display (Optional) ---
st.markdown("---")
if st.checkbox('Show Raw Data Table'):
    st.subheader('Raw Data (All Games)')
    st.dataframe(df, use_container_width=True)
