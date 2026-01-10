import requests
import pandas as pd
import time
import os

CFB_API_KEY = "3yZC6fPALRy4yRPtRMjghq/Mmrpe+R7FvMDYWae+7NqbMON8tH40idSddmQ+Yc/N"
HEADERS = {"Authorization": f"Bearer {CFB_API_KEY}", "Accept": "application/json"}
YEARS = [2022, 2023, 2024, 2025] # Added 2025

def fetch_and_merge():
    all_game_records = []

    for year in YEARS:
        print(f"📅 Fetching scores for {year}...")
        g_url = f"https://api.collegefootballdata.com/games?year={year}&seasonType=both"
        g_data = requests.get(g_url, headers=HEADERS).json()
        
        # Create a dictionary for quick score lookups
        game_context = {
            g['id']: {
                'neutral': 1 if g.get('neutral_site') else 0,
                'h_pts': g.get('home_points'),
                'a_pts': g.get('away_points')
            } for g in g_data if isinstance(g, dict)
        }

        for week in range(1, 16):
            print(f"📊 Processing {year} Week {week}...")
            t_url = f"https://api.collegefootballdata.com/games/teams?year={year}&week={week}&seasonType=regular"
            t_resp = requests.get(t_url, headers=HEADERS).json()

            if not isinstance(t_resp, list):
                continue

            for game in t_resp:
                gid = game.get('id')
                ctx = game_context.get(gid, {}) # Get context or empty dict

                row = {
                    "game_id": gid, 
                    "year": year, 
                    "week": week,
                    "neutral_site": ctx.get('neutral', 0)
                }

                # Pull points from context, but use team data as backup
                h_pts, a_pts = ctx.get('h_pts'), ctx.get('a_pts')

                for team in game.get("teams", []):
                    is_home = team.get("homeAway") == "home"
                    prefix = "h_" if is_home else "a_"
                    row[f"{prefix}team"] = team.get("team")
                    
                    # If context failed, grab the points from the team data itself
                    if is_home and h_pts is None: 
                        h_pts = team.get("points")
                    if not is_home and a_pts is None: 
                        a_pts = team.get("points")

                    for s in team.get("stats", []):
                        cat, val = s.get("category"), s.get("stat")
                        if cat == 'totalYards': 
                            row[f"{prefix}yds"] = float(val)
                        if cat == 'turnovers': 
                            row[f"{prefix}to"] = float(val)
                        if cat == 'possessionTime': 
                            minutes, seconds = map(int, val.split(':'))
                            row[f"{prefix}pos_sec"] = minutes * 60 + seconds
                        if cat == 'totalPenaltiesYards':
                            row[f"{prefix}pen_yds"] = float(val.split('-')[1])
                        if cat == 'rushingAttempts' or cat == 'completionAttempts':
                            row[f"{prefix}{cat}"] = float(val.split('-')[-1] if '-' in val else val)

                # Assign points and calculate win
                row["home_points"] = h_pts if h_pts is not None else 0
                row["away_points"] = a_pts if a_pts is not None else 0
                row["home_win"] = 1 if row["home_points"] > row["away_points"] else 0
                
                all_game_records.append(row)
            
            time.sleep(0.6)

    # Final cleanup and save
    if not os.path.exists("data"):
        os.makedirs("data")
        
    df = pd.DataFrame(all_game_records).drop_duplicates(subset='game_id')
    df.to_csv("data/detailed_stats.csv", index=False)
    print(f"🚀 Success! {len(df)} games saved with full scores.")

if __name__ == "__main__":
    fetch_and_merge()