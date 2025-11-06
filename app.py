import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Configuration ---
st.set_page_config(
    page_title="NCAA FBS Football Data Analysis", # Updated title for better SEO
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SEO META TAG INJECTION (Placed after config) ---
st.markdown("""
    <meta name="description" content="NCAA FBS football data analysis tool displaying team performance, Elo ratings, win rates, and conference comparisons. Filterable by conference.">
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# --- Data Loading and Filtering Functions ---
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
        
        df.dropna(subset=['homePoints', 'awayPoints', 'homeConference', 'awayConference'], inplace=True)
        
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
    # 1. Create a DataFrame for Home Team Stats
    home_df = df.copy()
    home_df['Team'] = home_df['homeTeam']
    home_df['Conference'] = home_df['homeConference']
    home_df['Venue'] = 'Home'
    home_df['Points_Scored'] = home_df['homePoints']
    home_df['Points_Allowed'] = home_df['awayPoints']
    home_df['Elo_Change'] = home_df['home_elo_change']
    home_df['Game_Outcome'] = np.where(home_df['home_win'] == 1, 'Win', 'Loss')
    
    # 2. Create a DataFrame for Away Team Stats
    away_df = df.copy()
    away_df['Team'] = away_df['awayTeam']
    away_df['Conference'] = away_df['awayConference']
    away_df['Venue'] = 'Away'
    away_df['Points_Scored'] = away_df['awayPoints']
    away_df['Points_Allowed'] = away_df['homePoints']
    away_df['Elo_Change'] = away_df['away_elo_change']
    away_df['Game_Outcome'] = np.where(away_df['home_win'] == 0, 'Win', 'Loss') 

    # 3. Concatenate and select relevant columns
    all_games_df = pd.concat([
        home_df[['Team', 'Conference', 'Venue', 'Points_Scored', 'Points_Allowed', 'Elo_Change', 'Game_Outcome']], 
        away_df[['Team', 'Conference', 'Venue', 'Points_Scored', 'Points_Allowed', 'Elo_Change', 'Game_Outcome']]
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
    """
    # 1. Home Game Log
    home_log = df.copy()
    home_log['Team'] = home_log['homeTeam']
    home_log['Opponent'] = home_log['awayTeam']
    home_log['Venue'] = np.where(home_log['neutralSite'] == True, 'Neutral', 'Home')
    home_log['Score_Display'] = home_log['homePoints'].astype(int).astype(str) + " - " + home_log['awayPoints'].astype(int).astype(str)
    home_log['Outcome'] = np.where(home_log['home_win'] == 1, 'W', 'L')
    
    # 2. Away Game Log
    away_log = df.copy()
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
# --- Main Application Logic ---
# --------------------------------------------------------------------------

DATA_PATH = 'data/odds.csv'
df = load_data(DATA_PATH)

if df.empty:
    st.stop()

# =========================================================================
# --- SIDEBAR: FILTERING ---
# =========================================================================

st.sidebar.title("Data Filters")
all_conferences = sorted(df['homeConference'].unique().tolist())

# Multi-select filter for conferences
selected_conferences = st.sidebar.multiselect(
    "Select Home Conference(s):", 
    options=all_conferences,
    default=all_conferences
)

# Apply filter
if selected_conferences:
    # Use .isin() for multi-select filtering!
    df_filtered = df[df['homeConference'].isin(selected_conferences)]
else:
    df_filtered = df[0:0] # Show empty DataFrame if nothing is selected

# Generate analysis tables based on filtered data
all_up_df = create_all_up_analysis(df_filtered)
game_log_df = create_game_log_analysis(df_filtered)

# =========================================================================
# --- DASHBOARD CONTENT ---
# =========================================================================

st.title("🏈 NCAA FBS Game Analysis")
st.markdown("Metrics for games involving **Football Bowl Subdivision (FBS)** teams, with **Home Conference** filtering enabled in the sidebar.")
st.caption(f"Showing **{len(df_filtered):,}** Games in the Selected Conference(s).")

col1, col2, col3 = st.columns(3)
col1.metric("Total FBS Games", f"{len(df_filtered):,}")
col2.metric("Average Home Point Differential", f"{df_filtered['point_differential'].mean():.2f} pts")
col3.metric("Home Win Percentage", f"{df_filtered['home_win'].mean() * 100:.2f}%")

st.markdown("---")

# -------------------------------------------------------------------------
# --- 1. AGGREGATED ALL-UP TEAM ANALYSIS ---
# -------------------------------------------------------------------------
st.header("🌐 All-Up Team Performance Summary (Aggregated)")
st.markdown("Team performance across **all games**, sorted by **Overall Win Rate**.")
st.dataframe(all_up_df.round(2), use_container_width=True)

# -------------------------------------------------------------------------
# --- 2. DETAILED GAME LOG TABLE ---
# -------------------------------------------------------------------------
st.header("📝 Detailed Game Log: All Games")
st.markdown("A game-by-game breakdown for every team, showing opponents, venue, and final score from the **perspective of the listed Team**.")

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
# --- 3. GAME PERFORMANCE (ELO) ---
# -------------------------------------------------------------------------
st.header("📈 Game Performance Relative to Expectation (Elo Change)")

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
# --- 4. CONFERENCE & DISTRIBUTION ---
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
# The corrected block only displays the raw data table
if st.checkbox('Show Raw Data Table'):
    st.subheader('Raw Data')
    st.dataframe(df_filtered, use_container_width=True)
