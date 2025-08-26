#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from datetime import datetime
import os

# -----------------------------
# APIs (replace YOUR_CFBD_API_KEY)
# -----------------------------
CFB_API = "https://api.collegefootballdata.com/games?year=2024&seasonType=regular&week=1"
CFB_API_KEY = "22b49b98a8b3f5734caad78b6bd9dbf5" 
ODDS_API = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds?regions=us&markets=h2h&apiKey=22b49b98a8b3f5734caad78b6bd9dbf5"

# -----------------------------
# Fetch college football games
# -----------------------------
def fetch_cfb_games():
    headers = {"Authorization": f"Bearer {CFB_API_KEY}"}
    r = requests.get(CFB_API, headers=headers)
    games = r.json()

    # Check for API message
    if isinstance(games, dict) and "message" in games:
        print("API returned message:", games["message"])
        return

    if isinstance(games, dict):
        games = [games]

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(games)

    # Rename columns if they exist, create winner and is_upset placeholders
    if "homeTeam" in df.columns and "awayTeam" in df.columns:
        df = df.rename(columns={"homeTeam": "home_team", "awayTeam": "away_team"})
        df["winner"] = df.apply(lambda row: row["home_team"] if row.get("homeScore", 0) > row.get("awayScore", 0) else row["away_team"], axis=1)
        df["is_upset"] = False  # placeholder
    else:
        # Fallback dummy row for testing
        df["home_team"] = ["Team A"]
        df["away_team"] = ["Team B"]
        df["winner"] = ["Team A"]
        df["is_upset"] = [False]

    df.to_csv("data/games.csv", index=False)
    print("✅ Saved games.csv")

# -----------------------------
# Fetch odds
# -----------------------------
def fetch_odds():
    r = requests.get(ODDS_API)
    odds = r.json()
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(odds)
    df.to_csv("data/odds.csv", index=False)
    print("✅ Saved odds.csv")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    fetch_cfb_games()
    fetch_odds()

