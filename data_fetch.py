import requests
import pandas as pd
import time
import os
from datetime import datetime

CFB_API_KEY = "3yZC6fPALRy4yRPtRMjghq/Mmrpe+R7FvMDYWae+7NqbMON8tH40idSddmQ+Yc/N"
HEADERS = {"Authorization": f"Bearer {CFB_API_KEY}"}
YEARS = [2022, 2023, 2024, 2025, 2026]


def fetch_and_merge():
    """Fetch historical game data for model training from last 3 seasons + current season so far."""
    all_game_records = []

    for year in YEARS:
        print(f"📅 Fetching scores for {year}...")
        g_url = f"https://api.collegefootballdata.com/games?year={year}&seasonType=both"
        g_data = requests.get(g_url, headers=HEADERS).json()

        game_context = {
            g['id']: {
                'neutral': 1 if g.get('neutral_site') else 0,
                'h_pts': g.get('home_points'),
                'a_pts': g.get('away_points')
            }
            for g in g_data if isinstance(g, dict)
        }

        for week in range(1, 16):
            print(f"📊 Processing {year} Week {week}...")
            t_url = f"https://api.collegefootballdata.com/games/teams?year={year}&week={week}&seasonType=regular"
            t_resp = requests.get(t_url, headers=HEADERS).json()

            if not isinstance(t_resp, list):
                continue

            for game in t_resp:
                gid = game.get('id')
                ctx = game_context.get(gid, {})

                row = {
                    "game_id": gid,
                    "year": year,
                    "week": week,
                    "neutral_site": ctx.get('neutral', 0)
                }

                h_pts, a_pts = ctx.get('h_pts'), ctx.get('a_pts')

                for team in game.get("teams", []):
                    is_home = team.get("homeAway") == "home"
                    prefix = "h_" if is_home else "a_"
                    row[f"{prefix}team"] = team.get("team")

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

                row["home_points"] = h_pts if h_pts is not None else 0
                row["away_points"] = a_pts if a_pts is not None else 0
                row["home_win"] = 1 if row["home_points"] > row["away_points"] else 0
                all_game_records.append(row)

            time.sleep(0.6)

    if not os.path.exists("data"):
        os.makedirs("data")

    df = pd.DataFrame(all_game_records).drop_duplicates(subset='game_id')
    df.to_csv("data/detailed_stats.csv", index=False)
    print(f"🚀 Success! {len(df)} historical games saved with full scores.")


def fetch_scheduled_fbs_games():
    """Fetch all scheduled FBS games that have not yet been played."""
    print("📅 Fetching scheduled FBS games...")
    current_year = datetime.now().year

    g_url = f"https://api.collegefootballdata.com/games?year={current_year}&seasonType=regular"
    g_data = requests.get(g_url, headers=HEADERS).json()

    if not isinstance(g_data, list):
        print("❌ Failed to fetch scheduled games")
        return pd.DataFrame()

    scheduled_games = []
    for game in g_data:
        if (game.get('homePoints') is None or game.get('awayPoints') is None):
            scheduled_games.append({
                'game_id': game.get('id'),
                'season': game.get('season'),
                'week': game.get('week'),
                'home_team': game.get('homeTeam'),
                'away_team': game.get('awayTeam'),
                'neutral_site': 1 if game.get('neutralSite') else 0,
                'start_date': game.get('startDate')
            })

    df_scheduled = pd.DataFrame(scheduled_games)
    if not df_scheduled.empty:
        os.makedirs("data", exist_ok=True)
        df_scheduled.to_csv("data/scheduled_games.csv", index=False)
        print(f"🚀 Success! {len(df_scheduled)} scheduled FBS games saved.")
    else:
        print("⚠️ No scheduled games found")

    return df_scheduled


if __name__ == "__main__":
    fetch_and_merge()
    time.sleep(1)
    fetch_scheduled_fbs_games()
