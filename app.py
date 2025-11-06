import streamlit as st
import pandas as pd
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="NCAA Game Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(path):
    """Load the processed CSV data."""
    try:
        df = pd.read_csv(path)
        # Ensure all necessary analytical columns are numeric
        required_cols = ['point_differential', 'home_elo_change', 'home_win']
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {path}. Please run the data processing script first.")
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
    # Away Elo change is the negative of home_elo_change (since it's a zero-sum calculation)
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
    all_up_analysis['Conference'] = all_games_df.groupby('Team')['Conference'].first()
    all_up_analysis['Point_Diff_Per_Game'] = all_up_analysis['Avg_Points_Scored'] - all_up_analysis['Avg_Points_Allowed']
    all_up_analysis = all_up_analysis.reset_index()
    all_up_analysis = all_up_analysis[[
        'Team', 'Conference', 'Total_Games', 'Wins', 'Losses', 'Win_Rate', 
        'Avg_Points_Scored', 'Avg_Points_Allowed', 'Point_Diff_Per_Game', 'Net_Elo_Change'
    ]]

    return all_up_analysis


# Load the data
DATA_PATH = 'data/odds.csv'
df = load_data(DATA_PATH)

# Check if data loaded successfully
if df.empty:
    st.stop()

# --- Title and Summary Stats ---
st.title("🏈 NCAA Game Analysis: Key Metrics")
st.markdown("Exploring the most important stats including **Point Differential** and **Performance vs. Expectation (Elo Change)**.")

col1, col2, col3 = st.columns(3)

# Display Key Metrics
col1.metric("Total Games Analyzed", f"{len(df):,}")
col2.metric("Average Home Point Differential", f"{df['point_differential'].mean():.2f} pts")
col3.metric("Home Win Percentage", f"{df['home_win'].mean() * 100:.2f}%")

# =========================================================================
# 🆕 NEW: ALL-UP TEAM ANALYSIS (Regardless of Venue)
# =========================================================================

all_up_df = create_all_up_analysis(df)

st.header("🌐 All-Up Team Performance Summary (Home & Away)")
st.markdown("This table summarizes each team's performance across **all games**, sorted by **Overall Win Rate**.")
st.dataframe(all_up_df.round(2))

# =========================================================================
# REST OF THE ORIGINAL DASHBOARD
# =========================================================================

# --- Visualization: Point Differential ---
st.header("Point Differential Distribution (Home Team Perspective)")
st.bar_chart(df['point_differential'].value_counts().sort_index().head(51).tail(51))
st.caption("A histogram showing how often different margins of victory occur (Home Score - Away Score).")

# --- Elo Analysis: Top/Bottom Performers ---
st.header("📈 Game Performance Relative to Expectation (Elo Change)")

# Top Home Elo Gain
st.subheader("Top 50 Games with Highest Home Elo Gain (Overachievers)")
elo_gain_df = df.sort_values(by='home_elo_change', ascending=False).head(50) # Updated to Top 50
st.dataframe(elo_gain_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                          'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True))

# Bottom Home Elo Loss
st.subheader("Bottom 50 Games with Highest Home Elo Loss (Underachievers)")
elo_loss_df = df.sort_values(by='home_elo_change', ascending=True).head(50) # Updated to Top 50
st.dataframe(elo_loss_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                          'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True))

# --- Conference Analysis (Grouping) ---
st.header("🏆 Home Conference Performance Summary")

# Recalculate Conference Analysis (as done in the correction)
conference_analysis = df.groupby('homeConference').agg(
    Total_Games=('home_win', 'count'),  
    Home_Win_Rate=('home_win', 'mean'),
    Avg_Home_Elo_Change=('home_elo_change', 'mean'),
    Avg_Point_Differential=('point_differential', 'mean')
).sort_values(by='Home_Win_Rate', ascending=False).reset_index()

conference_analysis.columns = ['Conference', 'Total Games', 'Home Win Rate', 
                               'Avg Home Elo Change', 'Avg Point Differential']

st.dataframe(conference_analysis.round(2))

# --- Raw Data Display (Optional) ---
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.dataframe(df)
