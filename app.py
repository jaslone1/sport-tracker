import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Configuration ---
st.set_page_config(
    page_title="NCAA Game Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# --- Data Loading and Filtering Functions ---
# --------------------------------------------------------------------------

# Use st.cache_data for fast reloading after initial load
@st.cache_data
def load_data(path):
    """Load and filter the processed CSV data for FBS teams."""
    
    # 1. Check if the processed file exists
    if not os.path.exists(path):
        st.error(f"Error: Processed file not found at {path}. Please ensure your data processing script was run successfully to create this file.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(path)
        
        # 2. Ensure all necessary analytical columns are numeric
        numeric_cols = ['homePoints', 'awayPoints', 'homePregameElo', 'homePostgameElo', 
                        'awayPregameElo', 'awayPostgameElo', 'attendance', 'point_differential', 
                        'home_elo_change', 'home_win', 'away_elo_change']
        for col in numeric_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Drop rows with critical missing data
        df.dropna(subset=['homePoints', 'awayPoints', 'homeConference', 'awayConference'], inplace=True)
        
        # 4. FBS FILTERING LOGIC
        # Exclude games involving clearly non-FBS teams (e.g., FCS, Division II, Division III)
        # We use common keywords to exclude lower divisions.
        non_fbs_keywords = ['FCS', 'II', 'III', 'D-2', 'D-3', 'NAIA']
        
        # Create boolean mask: Keep rows where NEITHER home nor away conference contains a non-FBS keyword.
        mask_home = ~df['homeConference'].astype(str).str.contains('|'.join(non_fbs_keywords), case=False, na=False)
        mask_away = ~df['awayConference'].astype(str).str.contains('|'.join(non_fbs_keywords), case=False, na=False)
        
        # Keep games only where BOTH teams are likely FBS (or Independent)
        df = df[mask_home & mask_away]
        
        return df
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

@st.cache_data
def create_all_up_analysis(df):
    """
    Creates a single-row-per-team-per-game structure for all-up performance metrics
    by combining the home and away columns.
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
    # Game outcome is the inverse of the home_win column
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
    # Ensure Conference is correct (use the most frequent value or first value)
    all_up_analysis['Conference'] = all_games_df.groupby('Team')['Conference'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
    all_up_analysis['Point_Diff_Per_Game'] = all_up_analysis['Avg_Points_Scored'] - all_up_analysis['Avg_Points_Allowed']
    all_up_analysis = all_up_analysis.reset_index()
    all_up_analysis = all_up_analysis[[
        'Team', 'Conference', 'Total_Games', 'Wins', 'Losses', 'Win_Rate', 
        'Avg_Points_Scored', 'Avg_Points_Allowed', 'Point_Diff_Per_Game', 'Net_Elo_Change'
    ]]

    return all_up_analysis

# --------------------------------------------------------------------------
# --- Main Application Logic ---
# --------------------------------------------------------------------------

# Load the data (uses FBS filtering)
DATA_PATH = 'data/odds.csv'
df = load_data(DATA_PATH)

if df.empty:
    st.stop()


# =========================================================================
# --- SIDEBAR: FILTERING ---
# =========================================================================

st.sidebar.title("Data Filters")

# Get unique conferences, sort them, and add 'All Conferences' option
all_conferences = sorted(df['homeConference'].unique().tolist())
all_conferences.insert(0, 'All Conferences')

selected_conference = st.sidebar.selectbox(
    "Select Home Conference:", 
    all_conferences,
    help="Filters the data by the home team's conference."
)

# Apply filter to the main DataFrame
if selected_conference != 'All Conferences':
    df_filtered = df[df['homeConference'] == selected_conference]
else:
    df_filtered = df.copy()

# Recalculate All-Up Analysis only on the filtered set
all_up_df = create_all_up_analysis(df_filtered)


# =========================================================================
# --- MAIN DASHBOARD CONTENT ---
# =========================================================================

st.title("🏈 NCAA FBS Game Analysis")
st.markdown("Metrics for games involving **Football Bowl Subdivision (FBS)** teams, with **Home Conference** filtering enabled in the sidebar.")
st.caption(f"Showing **{len(df_filtered):,}** Games in the Selected Conference(s).")

col1, col2, col3 = st.columns(3)

# Display Key Metrics (using filtered data)
col1.metric("Total FBS Games", f"{len(df_filtered):,}")
col2.metric("Average Home Point Differential", f"{df_filtered['point_differential'].mean():.2f} pts")
col3.metric("Home Win Percentage", f"{df_filtered['home_win'].mean() * 100:.2f}%")

st.markdown("---")

# -------------------------------------------------------------------------
# --- 1. ALL-UP TEAM ANALYSIS (Requested) ---
# -------------------------------------------------------------------------
st.header("🌐 All-Up Team Performance Summary (Home & Away)")
st.markdown("Team performance across **all games**, sorted by **Overall Win Rate**.")
st.dataframe(all_up_df.round(2), use_container_width=True)


# -------------------------------------------------------------------------
# --- 2. GAME PERFORMANCE (ELO) ---
# -------------------------------------------------------------------------
st.header("📈 Game Performance Relative to Expectation (Elo Change)")

# Top Home Elo Gain
st.subheader("Top 50 Games with Highest Home Elo Gain (Overachievers)")
elo_gain_df = df_filtered.sort_values(by='home_elo_change', ascending=False).head(50) 
st.dataframe(elo_gain_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                          'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True), use_container_width=True)

# Bottom Home Elo Loss
st.subheader("Top 50 Games with Highest Home Elo Loss (Underachievers)")
elo_loss_df = df_filtered.sort_values(by='home_elo_change', ascending=True).head(50) 
st.dataframe(elo_loss_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                          'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True), use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------------------
# --- 3. CONFERENCE & DISTRIBUTION ---
# -------------------------------------------------------------------------
col_conf, col_chart = st.columns([1, 2])

with col_conf:
    st.subheader("🏆 Home Conference Performance")

    # Recalculate Conference Analysis (uses df_filtered)
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
    # Limiting the chart data to prevent issues with massive outliers
    chart_data = df_filtered['point_differential'].value_counts().sort_index()
    # Ensure chart only shows relevant range, e.g., -50 to +50
    chart_data = chart_data[chart_data.index.to_series().between(-50, 50, inclusive='both')]
    st.bar_chart(chart_data)


# --- Raw Data Display (Optional) ---
st.markdown("---")
if st.checkbox('Show Raw Data Table'):
    st.subheader('Raw Data')
    st.dataframe(df_filtered, use_container_width=True)
# --- Raw Data Display (Optional) ---
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.dataframe(df)
