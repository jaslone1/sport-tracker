import pandas as pd
import numpy as np
import os

def create_ml_features():
    # 1. Load Data
    df = pd.read_csv("data/detailed_stats.csv", sep=None, engine='python', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    
    # 2. Advanced Numeric Cleanup (Handling new efficiency columns)
    cols_to_fix = [
        'home_points', 'away_points', 'h_yds', 'a_yds', 'h_to', 'a_to', 
        'h_pos_sec', 'a_pos_sec', 'h_pen_yds', 'a_pen_yds', 
        'h_rushingAttempts', 'h_completionAttempts', 'a_rushingAttempts', 'a_completionAttempts'
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Create Per-Game Efficiency Metrics
    # Calculate total plays to get Yards Per Play (YPP)
    df['h_plays'] = df['h_rushingAttempts'] + df['h_completionAttempts']
    df['a_plays'] = df['a_rushingAttempts'] + df['a_completionAttempts']
    
    # Avoid division by zero with .replace(0, 1)
    df['h_ypp'] = df['h_yds'] / df['h_plays'].replace(0, 1)
    df['a_ypp'] = df['a_yds'] / df['a_plays'].replace(0, 1)
    
    # Points per Minute of Possession (PPM)
    df['h_ppm'] = df['home_points'] / (df['h_pos_sec'] / 60).replace(0, 1)
    df['a_ppm'] = df['away_points'] / (df['a_pos_sec'] / 60).replace(0, 1)

    # 4. Flatten for Rolling Averages (Long Format)
    # We include our new efficiency stats here
    h_cols = ['year', 'week', 'h_team', 'h_yds', 'h_to', 'h_ypp', 'h_ppm', 'h_pen_yds', 'home_points', 'away_points']
    a_cols = ['year', 'week', 'a_team', 'a_yds', 'a_to', 'a_ypp', 'a_ppm', 'a_pen_yds', 'away_points', 'home_points']
    
    shared_cols = ['year', 'week', 'team', 'yards', 'turnovers', 'ypp', 'ppm', 'pen_yds', 'pts_scored', 'pts_allowed']
    
    home_df = df[h_cols].copy()
    away_df = df[a_cols].copy()
    home_df.columns = shared_cols
    away_df.columns = shared_cols
    
    perf_df = pd.concat([home_df, away_df]).sort_values(['team', 'year', 'week'])

    # 5. Calculate Rolling Averages (Last 3 games)
    stats_to_roll = ['yards', 'turnovers', 'ypp', 'ppm', 'pen_yds', 'pts_scored', 'pts_allowed']
    for stat in stats_to_roll:
        perf_df[f'roll_{stat}'] = perf_df.groupby(['team', 'year'])[stat].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )

    # 6. STRENGTH OF SCHEDULE (SOS) Adjustment
    # We calculate how "tough" the opponents were by looking at the avg points they allowed
    perf_df['opp_def_strength'] = perf_df.groupby(['team', 'year'])['pts_allowed'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )

    # 7. Save Team Lookup (For Streamlit)
    latest_stats = perf_df.groupby('team').tail(1)
    os.makedirs("data", exist_ok=True)
    latest_stats.to_csv("data/team_lookup.csv", index=False)

    # 8. Merge Rolling Stats back into Matchup Format
    rolling_cols = ['year', 'week', 'team'] + [f'roll_{s}' for s in stats_to_roll] + ['opp_def_strength']
    rolling_lookup = perf_df[rolling_cols]

    # Home Merge
    df = df.merge(rolling_lookup, left_on=['year', 'week', 'h_team'], right_on=['year', 'week', 'team'], how='left')
    df = df.rename(columns={f'roll_{s}': f'h_roll_{s}' for s in stats_to_roll}).rename(columns={'opp_def_strength': 'h_sos'}).drop(columns=['team'])

    # Away Merge
    df = df.merge(rolling_lookup, left_on=['year', 'week', 'a_team'], right_on=['year', 'week', 'team'], how='left')
    df = df.rename(columns={f'roll_{s}': f'a_roll_{s}' for s in stats_to_roll}).rename(columns={'opp_def_strength': 'a_sos'}).drop(columns=['team'])

    # 9. Save Final ML-Ready Data
    df_ml = df.dropna(subset=['h_roll_pts_scored', 'a_roll_pts_scored'])
    df_ml.to_csv("data/ml_ready_features.csv", index=False)
    
    print(f"🚀 Success! Created dataset with {len(df_ml)} games and advanced efficiency metrics.")

if __name__ == "__main__":
    create_ml_features()