import requests
import pandas as pd
from datetime import datetime
import os
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
CFB_API_KEY = "3yZC6fPALRy4yRPtRMjghq/Mmrpe+R7FvMDYWae+7NqbMON8tH40idSddmQ+Yc/N"  # Replace with your valid key
CURRENT_YEAR = datetime.now().year
START_YEAR = CURRENT_YEAR - 9  # last 10 seasons
CFB_GAMES_URL = "https://api.collegefootballdata.com/games"


# -----------------------------
# Fetch FBS Games by Season
# -----------------------------
def fetch_fbs_games_for_seasons():
    """
    Fetches all FBS vs FBS games (regular + postseason)
    from the last 10 seasons and saves them to data/games.csv.
    """
    all_games_data = []
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}

    print(f"🏈 Starting data fetch from {START_YEAR} to {CURRENT_YEAR}...")

    for year in range(START_YEAR, CURRENT_YEAR + 1):
        # Regular season
        params_reg = {"year": year, "seasonType": "regular"}
        r_reg = requests.get(CFB_GAMES_URL, headers=headers, params=params_reg)

        # Postseason
        params_post = {"year": year, "seasonType": "postseason"}
        r_post = requests.get(CFB_GAMES_URL, headers=headers, params=params_post)

        # Handle bad responses
        if r_reg.status_code != 200 or r_post.status_code != 200:
            print(f"🚨 Error fetching {year}: {r_reg.status_code}/{r_post.status_code}")
            print("Response sample:", r_reg.text[:300])
            continue

        # Combine both sets of games
        games = r_reg.json() + r_post.json()

        if not games:
            print(f"⚠️ No games returned for {year}.")
            continue

        print(f"   📦 Retrieved {len(games):,} total games for {year}")

        # --- Filter to FBS vs FBS matchups ---
        fbs_games = [
            g for g in games
            if g.get("homeConference") and g.get("awayConference")
        ]

        print(f"   ✅ {len(fbs_games):,} FBS games for {year}")
        all_games_data.extend(fbs_games)

        time.sleep(0.5)  # prevent rate limiting

    # --- Save Results ---
    if not all_games_data:
        print("❌ No FBS games were retrieved. Please check API key or endpoint.")
        return

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(all_games_data)

    # Normalize column names
    if "homeTeam" in df.columns:
        df = df.rename(columns={"homeTeam": "home_team", "awayTeam": "away_team"})

    # Add winner column
    df["winner"] = df.apply(
        lambda row: row["home_team"]
        if row.get("homePoints", 0) > row.get("awayPoints", 0)
        else row["away_team"],
        axis=1,
    )

    df["is_upset"] = False  # placeholder for later analysis

    output_path = "data/games.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully saved {len(df):,} FBS vs FBS games to {output_path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    fetch_fbs_games_for_seasons()

