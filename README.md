# FBS Weekly Predictor

A Streamlit app that predicts outcomes of all scheduled FBS college football games using machine learning trained on historical season data.

## Overview

- Training data includes 2022, 2023, 2024, and the current 2025 season so far
- Predictions cover all scheduled FBS games for the week
- Model features include rolling 3-game averages for yards-per-play, points-per-minute, turnovers, and opponent defensive strength
- The app does not depend on a single team's historical record before this season; it uses broad team trend features across recent seasons

## Quick Start

```bash
pip install -r requirements.txt
python data_fetch.py
python feature_engineering.py
python train_model.py
streamlit run app.py
```

## Data Flow

1. `data_fetch.py`
   - Pulls historical game results from the College Football Data API for 2022-2025
   - Pulls currently scheduled FBS games for this week
2. `feature_engineering.py`
   - Builds rolling 3-game team metrics and efficiency stats
   - Saves `data/team_lookup.csv` and `data/ml_ready_features.csv`
3. `train_model.py`
   - Trains a RandomForest classifier on engineered matchup features
   - Saves `models/ncaa_model.pkl`
4. `app.py`
   - Loads the trained model and scheduled games
   - Generates predictions and projected scores for each matchup

## Files

```text
.
├── app.py
├── data_fetch.py
├── feature_engineering.py
├── train_model.py
├── train_pytorch.py
├── requirements.txt
├── data/
│   ├── detailed_stats.csv
│   ├── ml_ready_features.csv
│   ├── team_lookup.csv
│   └── scheduled_games.csv
├── models/
│   └── ncaa_model.pkl
└── README.md
```

## Notes

This version is a general FBS weekly predictor rather than a fixed mini-playoff bracket. The same model can still be used to predict all FBS games, then you can narrow to a subset if desired.
