#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def process_upsets():
    games = pd.read_csv("data/games.csv")
    odds = pd.read_csv("data/odds.csv")

    # Example simplification: mark upset if underdog won
    # (You’d refine this later with spread logic)
    if "home_points" in games.columns:
        games["winner"] = games.apply(
            lambda row: row["home_team"] if row["home_points"] > row["away_points"] else row["away_team"],
            axis=1
        )
    else:
        games["winner"] = None

    games["is_upset"] = False
    # Stub rule: if odds existed, mark True when lowest implied probability wins
    # (for now just mark random few as True to demonstrate)
    if len(games) > 0:
        games.loc[::5, "is_upset"] = True  # every 5th game flagged

    games.to_csv("data/upsets.csv", index=False)
    print("✅ Saved upsets.csv")

if __name__ == "__main__":
    process_upsets()
    
# Example schema for processed_data.csv
processed = df.copy()
processed["winner"] = processed.apply(
    lambda row: row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"], axis=1
)
processed["is_upset"] = processed["winner"] == processed["underdog"]

# Save
processed.to_csv("data/processed_data.csv", index=False)

# In[ ]:




