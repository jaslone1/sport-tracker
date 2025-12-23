import requests
import pandas as pd
from datetime import datetime
import os
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
CFB_API_KEY = "3yZC6fPALRy4yRPtRMjghq/Mmrpe+R7FvMDYWae+7NqbMON8tH40idSddmQ+Yc/N"
CURRENT_YEAR = datetime.now().year
START_YEAR = CURRENT_YEAR - 9 
# Use the team stats endpoint for richer ML features
CFB_STATS_URL = "https://api.collegefootballdata.com/games/teams"

def flatten_stats(game_data):
    """
    Converts the nested API stats list into a flat dictionary 
    with home_ and away_ prefixes.
    """
    row = {
        "game_id": game_data.get("id"),
        "year": game_data.get("season"),
        "week": game_data.get("week"),
        "season_type": game_data.get("season_type")
    }

    for team_entry in game_data.get("teams", []):
        prefix = "home_" if team_entry.get("homeAway") == "home" else "away_"
        row[f"{prefix}team"] = team_entry.get("school")
        row[f"{prefix}conference"] = team_entry.get("conference")
        row[f"{prefix}points"] = team_entry.get("points")

        # Flatten the stats list into individual columns
        for s in team_entry.get("stats", []):
            category = s.get("category")
            stat_val = s.get("stat")
            # Convert string stats (like '30:00' or '5-10') if necessary, 
            # but usually, numeric categories are standard strings.
            row[f"{prefix}{category}"] = stat_val

    return row

def fetch_fbs_stats():
    all_game_rows = []
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}

    print(f"🏈 Fetching box scores from {START_YEAR} to {CURRENT_YEAR}...")

    for year in range(START_YEAR, CURRENT_YEAR + 1):
        for season_type in ["regular", "postseason"]:
            params = {"year": year, "seasonType": season_type}
            response = requests.get(CFB_STATS_URL, headers=headers, params=params)

            if response.status_code != 200:
                print(f"🚨 Error {year} {season_type}: {response.status_code}")
                continue

            games = response.json()
            
            # Filter for FBS vs FBS and flatten
            for g in games:
                # Check if both teams have a conference (standard way to identify FBS)
                if all(t.get("conference") for t in g.get("teams", [])):
                    all_game_rows.append(flatten_stats(g))

            print(f"   ✅ Processed {len(games)} games for {year} {season_type}")
            time.sleep(0.5)

    # --- Save Results ---
    if not all_game_rows:
        print("❌ No data found.")
        return

    df = pd.DataFrame(all_game_rows)
    
    # Calculate Winner (useful for ML Target)
    df["home_win"] = df["home_points"].astype(float) > df["away_points"].astype(float)
    
    os.makedirs("data", exist_ok=True)
    output_path = "data/detailed_stats.csv"
    df.to_csv(output_path, index=False)
    print(f"\n🚀 Success! Saved {len(df)} games with detailed stats to {output_path}")

if __name__ == "__main__":
    fetch_fbs_stats()
