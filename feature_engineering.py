import pandas as pd
import numpy as np

def create_ml_features(input_file="data/detailed_stats.csv"):
    df = pd.read_csv(input_file)
    
    # 1. Convert numeric columns to float (important as API data often comes as strings)
    # Identify stat columns (anything starting with home_ or away_ except team names)
    stat_cols = [col for col in df.columns if any(x in col for x in ['home_', 'away_']) 
                 and 'team' not in col and 'conference' not in col]
    
    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Reshape data so each row is a "Team Performance" 
    # (To calculate averages, we need a list of every time a team played)
    home_df = df[['year', 'week', 'home_team'] + [c for c in df.columns if c.startswith('home_')]].copy()
    away_df = df[['year', 'week', 'away_team'] + [c for c in df.columns if c.startswith('away_')]].copy()
    
    # Rename columns to be generic
    home_df.columns = [c.replace('home_', '') for c in home_df.columns]
    away_df.columns = [c.replace('away_', '') for c in away_df.columns]
    
    # Combine them into one long list of team performances
    team_performances = pd.concat([home_df, away_df]).sort_values(['team', 'year', 'week'])

    # 3. Calculate Rolling Averages (The "Magic" for ML)
    # We use a window of 5 games. 
    # .shift(1) is CRITICAL: It ensures we use stats FROM PREVIOUS GAMES only.
    metrics = ['points', 'rushingYards', 'passingYards', 'turnovers', 'totalYards'] 
    
    for metric in metrics:
        if metric in team_performances.columns:
            team_performances[f'rolling_{metric}'] = team_performances.groupby('team')[metric]\
                .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))

    # 4. Merge these averages back into the original game-by-game format
    # This gives us a row for a game with "Home team's recent form" vs "Away team's recent form"
    averages = team_performances[['year', 'week', 'team'] + [f'rolling_{m}' for m in metrics]]
    
    final_df = df[['game_id', 'year', 'week', 'home_team', 'away_team', 'home_win']].copy()
    
    # Merge for home team
    final_df = final_df.merge(averages, left_on=['year', 'week', 'home_team'], right_on=['year', 'week', 'team'], how='left')
    final_df = final_df.rename(columns={f'rolling_{m}': f'home_prev_{m}' for m in metrics}).drop(columns=['team'])
    
    # Merge for away team
    final_df = final_df.merge(averages, left_on=['year', 'week', 'away_team'], right_on=['year', 'week', 'team'], how='left')
    final_df = final_df.rename(columns={f'rolling_{m}': f'away_prev_{m}' for m in metrics}).drop(columns=['team'])

    # Save the final ML-ready dataset
    final_df.dropna(inplace=True) # Remove early season games that don't have history yet
    final_df.to_csv("data/ml_ready_features.csv", index=False)
    print(f"✅ Created ML features for {len(final_df)} games.")

if __name__ == "__main__":
    create_ml_features()
