#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from datetime import datetime

# APIs (replace API_KEY if needed for The Odds API)
CFB_API = "https://api.collegefootballdata.com/games?year=2024&seasonType=regular&week=1"
ODDS_API = f"https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds?regions=us&markets=h2h&apiKey=YOUR_API_KEY"

def fetch_cfb_games():
    r = requests.get(CFB_API)
    games = r.json()
    df = pd.DataFrame(games)
    df.to_csv("data/games.csv", index=False)
    print("✅ Saved games.csv")

def fetch_odds():
    r = requests.get(ODDS_API)
    odds = r.json()
    df = pd.DataFrame(odds)
    df.to_csv("data/odds.csv", index=False)
    print("✅ Saved odds.csv")

if __name__ == "__main__":
    fetch_cfb_games()
    fetch_odds()


# In[ ]:




