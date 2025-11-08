#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd
from datetime import datetime
import os
import time

# -----------------------------
# CONFIGURATION AND API KEYS
# -----------------------------

# NOTE: The CFB Data API key must be passed via the Authorization header,
# but the CFB games endpoint has a 'division' parameter we can use for filtering.
# Also, we use a different API key format for The Odds API.

# Replace YOUR_CFBD_API_KEY and YOUR_ODDS_API_KEY with your actual keys
CFB_API_KEY = "22b49b98a8b3f5734caad78b6bd9dbf5" # Assuming this is the CFBD key
ODDS_API_KEY = "22b49b98a8b3f5734caad78b6bd9dbf5" # Assuming this is The Odds API key

# Calculate the starting year (10 years ago, including the current year)
CURRENT_YEAR = datetime.now().year
START_YEAR = CURRENT_YEAR - 9 # For 10 years of data (e.g., 2016-2025)

# CFBD base URL for games
CFB_GAMES_URL = "https://api.collegefootballdata.com/games"
ODDS_API_URL = f"https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds?regions=us&markets=h2h&apiKey={ODDS_API_KEY}"


# -----------------------------
# Fetch college football games (Updated)
# -----------------------------
def fetch_cfb_games_for_seasons():
    """
    Fetches all FBS games from the past 10 seasons (regular and postseason).
    It filters the results to ensure both teams are from the FBS division.
    """
    all_games_data = []
    
    # CFBD API requires the key in the Authorization header
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    
    print(f"🏈 Starting data fetch from {START_YEAR} to {CURRENT_YEAR}...")

    # Loop through each year
    for year in range(START_YEAR, CURRENT_YEAR + 1):
        # Fetch Regular Season Games
        params_reg = {'year': year, 'seasonType': 'regular', 'division': 'fbs'}
        r_reg = requests.get(CFB_GAMES_URL, headers=headers, params=params_reg)
        
        # Fetch Postseason Games
        params_post = {'year': year, 'seasonType': 'postseason', 'division': 'fbs'}
        r_post = requests.get(CFB_GAMES_URL, headers=headers, params=params_post)
        
        # Check for API errors
        if r_reg.status_code != 200 or r_post.status_code != 200:
            print(f"🚨 Error fetching data for {year}. Status: {r_reg.status_code} / {r_post.status_code}")
            continue

        games = r_reg.json() + r_post.json()
        
        # Check for API message (e.g., rate limit)
        if isinstance(games, dict) and "message" in games:
            print(f"🚨 API returned message for {year}: {games['message']}")
            time.sleep(5) # Wait before next attempt
            continue

        # --- FBS vs. FBS Filtering ---
        # The 'division: fbs' parameter ensures one team is FBS, 
        # but we must filter to ensure the opponent is *also* FBS.
        fbs_games = [
            g for g in games 
            if g.get('homeDivision', '').lower() == 'fbs' and g.get('awayDivision', '').lower() == 'fbs'
        ]

        print(f"   ✅ Fetched {len(fbs_games):,} FBS games for {year}")
        all_games_data.extend(fbs_games)
        
        # Be mindful of rate limits
        time.sleep(0.5) 

    if not all_games_data:
        print("❌ Failed to retrieve any game data.")
        return

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(all_games_data)
    
    # --- Data Cleaning/Renaming for Consistency ---
    
    # Rename for consistency (your original script used home_team/away_team)
    if "homeTeam" in df.columns:
        df = df.rename(columns={"homeTeam": "home_team", "awayTeam": "away_team"})
    
    # Create simple win/upset columns (can be refined in a separate process script)
    df["winner"] = df.apply(
        lambda row: row.get("home_team") if row.get("homePoints", 0) > row.get("awayPoints", 0) else row.get("away_team"), 
        axis=1
    )
    df["is_upset"] = False # Placeholder for now
    
    df.to_csv("data/games.csv", index=False)
    print(f"\n✅ Successfully saved {len(df):,} total FBS vs. FBS games to data/games.csv")


# -----------------------------
# Fetch odds (Unchanged logic, updated URL variable)
# -----------------------------
def fetch_odds():
    """Fetches real-time odds from The Odds API for currently scheduled NCAAF games."""
    print("\n💰 Fetching real-time odds for current games...")
    r = requests.get(ODDS_API_URL)
    
    if r.status_code != 200:
        print(f"🚨 Error fetching odds. Status: {r.status_code}. Response: {r.text}")
        return

    odds = r.json()
    
    if isinstance(odds, dict) and "message" in odds:
        print("🚨 Odds API returned message:", odds["message"])
        return
        
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(odds)
    df.to_csv("data/odds.csv", index=False)
    print(f"✅ Saved {len(df):,} odds records to data/odds.csv")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    fetch_cfb_games_for_seasons()
    # fetch_odds() # Odds are often expensive/rate-limited, keep separate for historical data
