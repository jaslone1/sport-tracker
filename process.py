import pandas as pd

# 1. Load the Data
# Assuming your CSV file is named 'ncaa_games.csv'
try:
    df = pd.read_csv('data/games.csv')
except FileNotFoundError:
    print("Error: 'games.csv' not found. Please check the file path.")
    exit()

# Set display options for better viewing of results
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Data Loading and Initial Cleaning ---")
print(f"Initial shape: {df.shape}")

# 2. Data Cleaning and Type Conversion
# Convert Points and Elo columns to numeric, coercing errors to NaN
numeric_cols = ['homePoints', 'awayPoints', 'homePregameElo', 'homePostgameElo', 
                'awayPregameElo', 'awayPostgameElo', 'attendance']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with critical missing values (e.g., game scores)
df.dropna(subset=['homePoints', 'awayPoints'], inplace=True)

print(f"Shape after dropping rows with missing scores: {df.shape}\n")

# 3. Feature Engineering: Calculating Most Important Stats

# A. Margin of Victory (MoV) / Point Differential
df['point_differential'] = df['homePoints'] - df['awayPoints']

# B. Game Result (0 for Away Win, 1 for Home Win)
df['home_win'] = (df['point_differential'] > 0).astype(int)

# C. Actual vs. Expected Margin (Based on Line Score)
# Assuming 'homeLineScores' and 'awayLineScores' are strings or objects, you'll need to 
# clean them if you want to use a betting spread. For simple MoV, we use the score.

# D. Elo Rating Change (A strong indicator of performance relative to expectation)
df['home_elo_change'] = df['homePostgameElo'] - df['homePregameElo']
df['away_elo_change'] = df['awayPostgameElo'] - df['awayPregameElo']

# E. Elo Difference (Pre-game rating difference)
df['pregame_elo_diff'] = df['homePregameElo'] - df['awayPregameElo']

print("--- Feature Engineering Complete: New Metrics Created ---\n")

# 4. Core Statistical Analysis

## 📊 Summary of Game Outcomes
print("## 📊 Summary of Game Outcomes")
# Mean Point Differential (MoV for home team)
mean_diff = df['point_differential'].mean()
print(f"Average Point Differential (Home Score - Away Score): {mean_diff:.2f} points")
print(f"Home Win Percentage: {df['home_win'].mean() * 100:.2f}%\n")


## 📈 Elo Analysis: Performance vs. Expectation
print("## 📈 Elo Analysis: Performance vs. Expectation")

# Teams that exceeded expectations (large positive Elo change)
top_home_elo_gain = df.sort_values(by='home_elo_change', ascending=False).head(5)
print("Top 5 Home Teams by Elo Gain (Exceeded Pre-game Expectation):")
print(top_home_elo_gain[['homeTeam', 'awayTeam', 'home_elo_change', 'point_differential']])
print("\n")

# Teams that underperformed expectations (large negative Elo change)
bottom_home_elo_loss = df.sort_values(by='home_elo_change', ascending=True).head(5)
print("Bottom 5 Home Teams by Elo Loss (Underperformed Expectation):")
print(bottom_home_elo_loss[['homeTeam', 'awayTeam', 'home_elo_change', 'point_differential']])
print("\n")

## 🏈 Analysis by Home Conference (Example of Group Analysis)
print("## 🏈 Analysis by Home Conference")
conference_analysis = df.groupby('homeConference').agg(
    total_games=('home_win', 'count'),
    home_win_rate=('home_win', 'mean'),
    avg_elo_gain=('home_elo_change', 'mean'),
    avg_point_diff=('point_differential', 'mean')
).sort_values(by='home_win_rate', ascending=False)

# Rename the mean columns for clarity
conference_analysis.columns = ['Total Games', 'Home Win Rate', 'Avg Home Elo Change', 'Avg Point Differential']

print("Conference Performance Summary (Home Games Only):")
print(conference_analysis.head(5).round(2))
print("\n")


## 🏟️ Impact of Attendance
print("## 🏟️ Impact of Attendance")
# Correlation between attendance and point differential
attendance_corr = df['attendance'].corr(df['point_differential'])
print(f"Correlation between Attendance and Point Differential: {attendance_corr:.4f}")
if attendance_corr > 0.1:
    print("-> Suggests higher attendance weakly correlates with a higher Home Point Differential (home team performs better).")
elif attendance_corr < -0.1:
    print("-> Suggests higher attendance weakly correlates with a lower Home Point Differential (home team performs worse).")
else:
    print("-> Suggests little to no linear correlation between attendance and home team's performance margin.")

print("\n--- End of Analysis ---")



