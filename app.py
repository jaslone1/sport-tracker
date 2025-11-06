import streamlit as st
import pandas as pd
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Testing site for NCAA Game Data Analysis",
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
col2.metric("Average Home Point Differential", f"{df['point_differential'].mean():.2f} pts", 
            help="The average margin of victory for the home team.")
col3.metric("Home Win Percentage", f"{df['home_win'].mean() * 100:.2f}%")

# --- Visualization: Point Differential ---
st.header("Point Differential Distribution")
st.markdown("A histogram showing how often different margins of victory occur. Zero is the most common!")

# Create a histogram of the point differential
st.bar_chart(df['point_differential'].value_counts().sort_index().head(51).tail(51))
st.caption("Values range from -50 to +50 points.")

# --- Elo Analysis: Top/Bottom Performers ---
st.header("📈 Performance Relative to Expectation (Elo Change)")

# Top Home Elo Gain
st.subheader("Top 10 Games with Highest Home Elo Gain (Overachievers)")
elo_gain_df = df.sort_values(by='home_elo_change', ascending=False).head(10)
st.dataframe(elo_gain_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                          'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True))

# Bottom Home Elo Loss
st.subheader("Bottom 10 Games with Highest Home Elo Loss (Underachievers)")
elo_loss_df = df.sort_values(by='home_elo_change', ascending=True).head(10)
st.dataframe(elo_loss_df[['homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 
                          'homePregameElo', 'home_elo_change', 'point_differential']].reset_index(drop=True))

# --- Conference Analysis (Grouping) ---
st.header("🏆 Conference Performance Summary (Home Games)")

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
