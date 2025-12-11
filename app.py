import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone # Import datetime

# --- 1. CONFIGURATION ---
# IMPORTANT: Adjust this path based on where you run app.py relative to your artifacts
OUT_DIR = Path("ml_model/pytorch_training_winner")

MODEL_PATH = OUT_DIR / "winner_model.pth"
SCALER_PATH = OUT_DIR / "scaler.joblib"
FEATURE_COLUMNS_PATH = OUT_DIR / "feature_columns.json"
LABEL_ENCODERS_PATH = OUT_DIR / "label_encoders.joblib"

DEVICE = torch.device("cpu") # Use CPU for deployment

# --- 2. MODEL DEFINITION (Must match the architecture used in training) ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- 3. ARTIFACT LOADING (Runs ONCE, Cached by Streamlit) ---
# Use st.cache_resource to load heavy models and objects only once
@st.cache_resource
def load_all_artifacts():
    try:
        # Load feature columns, scaler, label encoders
        with open(FEATURE_COLUMNS_PATH, "r") as f:
            feature_columns = json.load(f)
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)

        # Initialize and load model
        INPUT_DIM = len(feature_columns)
        model = MLP(input_dim=INPUT_DIM).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        return scaler, label_encoders, feature_columns, model, INPUT_DIM
    except Exception as e:
        # In a Streamlit app, st.error and st.stop() would be used.
        # In a non-Streamlit context (like a Colab cell), st.stop() might not work as expected,
        # leading to an implicit return None and a TypeError on unpacking.
        # Explicitly raise an error to provide a clearer failure message.
        raise RuntimeError(f"Error loading model artifacts from {OUT_DIR}. Please ensure all model files exist. Error: {e}") from e

# Load everything globally
try:
    scaler, label_encoders, feature_columns, model, INPUT_DIM = load_all_artifacts()
except RuntimeError as e:
    # Catch the RuntimeError and set variables to None to avoid program crash
    # The main function will then display an error message in Streamlit if it runs.
    print(f"Failed to load model artifacts: {e}")
    scaler, label_encoders, feature_columns, model, INPUT_DIM = None, None, None, None, None

def predict_winner(raw_game_data: dict) -> dict:
    """
    Processes raw game data using the SAME preprocessing steps used in training.
    This ensures correct feature alignment and correct predictions.
    """

    # --- SAFETY: Ensure model artifacts loaded ---
    if scaler is None or label_encoders is None or feature_columns is None or model is None:
        raise RuntimeError("Model artifacts are not loaded. Cannot make predictions.")

    # --- 1. Convert incoming data to DataFrame ---
    df = pd.DataFrame([raw_game_data]).copy()

    # --- 2. Match training preprocessing ---

    # Convert booleans → ints
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    # Convert startDate to numeric (training treated ALL numeric via pd.to_numeric)
    if "startDate" in df.columns:
        df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
        df["startDate"] = df["startDate"].view("int64")  # EXACT match to training numeric conversion

    # Identify columns for label encoding vs one-hot vs numeric
    # Based on training:
    # - label_encoders keys = label-encoded columns
    # - feature_columns with prefix_* = one-hot columns
    # - everything else numeric
    le_cols = list(label_encoders.keys())

    # Determine one-hot columns by checking prefixes in feature_columns
    # e.g. "homeConference_ACC" → prefix "homeConference"
    ohe_prefixes = set()
    for col in feature_columns:
        if "_" in col:
            prefix = col.split("_")[0]
            ohe_prefixes.add(prefix)

    # Some prefixes in training had format like "homeConference_*"
    # Identify original OHE columns by looking at raw features
    ohe_cols = [c for c in df.columns if c in ohe_prefixes]

    # Numeric columns = everything else
    numeric_cols = [c for c in df.columns if c not in le_cols and c not in ohe_cols]

    # --- 3. Label Encoding (high-cardinality columns) ---
    df_le = df.copy()
    for col in le_cols:
        if col in df_le.columns:
            series = df_le[col].astype(str)
            df_le[col] = series.map(label_encoders[col]).fillna(-1)

    # --- 4. One-Hot Encoding using same PREFIX as training ---
    df_ohe = pd.DataFrame(index=df.index)
    for col in ohe_cols:
        temp = pd.get_dummies(df[col].fillna(""), prefix=col)
        df_ohe = pd.concat([df_ohe, temp], axis=1)

    # --- 5. Numeric columns (training used pd.to_numeric + fillna median) ---
    df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # --- 6. Combine everything EXACTLY like training ---
    X_df_new = pd.concat([df_num, df_ohe, df_le[le_cols]], axis=1)

    # --- 7. Align to training feature columns ---
    X_aligned = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    for col in X_df_new.columns:
        if col in X_aligned.columns:
            X_aligned[col] = X_df_new[col].values[0]

    # --- 8. Scale ---
    X_scaled = scaler.transform(X_aligned.values.astype(np.float32))

    # --- 9. Predict with PyTorch model ---
    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        logits = model(x_tensor).cpu().numpy().ravel()[0]
        prob = 1 / (1 + np.exp(-logits))

    # --- 10. Return prediction ---
    return {
        "home_team_win_prob": float(prob),
        "prediction": "Home Team Wins" if prob >= 0.5 else "Away Team Wins",
    }

# --- 5. STREAMLIT UI ---
def main():
    st.set_page_config(page_title="CFB Winner Predictor", layout="wide")
    st.title("🏈 College Football Winner Prediction for Upcoming Games")
    st.markdown("Loading games from `data/games.csv` and predicting outcomes for uncompleted games.")

    # Check if artifacts were loaded successfully. If not, display an error and stop.
    if scaler is None or label_encoders is None or feature_columns is None or model is None or INPUT_DIM is None:
        st.error("Model artifacts could not be loaded. Please ensure the `ml_model/pytorch_training_winner` directory and its contents (`winner_model.pth`, `scaler.joblib`, `feature_columns.json`, `label_encoders.joblib`) exist relative to the app's execution directory.")
        st.stop() # Stop the Streamlit app if crucial artifacts are missing

    # --- CSV LOADING AND DUMMY CREATION ---
    csv_path = 'data/games.csv'

    # Ensure the directory exists
    Path('data').mkdir(parents=True, exist_ok=True)

    if not Path(csv_path).exists():
        st.info("`data/games.csv` not found. Creating a dummy file for demonstration.")
        dummy_data = [
            {'season': 2024, 'week': 1, 'completed': False, 'neutralSite': False, 'conferenceGame': True,
             'attendance': 0, 'venueId': '1', 'homeId': '52', 'home_points': 0, 'homePregameElo': 1800,
             'awayId': '230', 'away_points': 0, 'awayPregameElo': 1600, 'excitementIndex': 0.0,
             'startDate': '2024-09-01', 'homeConference': 'SEC', 'awayConference': 'ACC',
             'homeClassification': 'fbs', 'awayClassification': 'fbs', 'seasonType': 'regular'},
            {'season': 2024, 'week': 1, 'completed': False, 'neutralSite': True, 'conferenceGame': False,
             'attendance': 0, 'venueId': '2', 'homeId': '15', 'home_points': 0, 'homePregameElo': 1700,
             'awayId': '123', 'away_points': 0, 'awayPregameElo': 1750, 'excitementIndex': 0.0,
             'startDate': '2024-09-02', 'homeConference': 'Big Ten', 'awayConference': 'Big 12',
             'homeClassification': 'fbs', 'awayClassification': 'fbs', 'seasonType': 'regular'},
            {'season': 2023, 'week': 10, 'completed': True, 'neutralSite': False, 'conferenceGame': True,
             'attendance': 80000, 'venueId': '3', 'homeId': '52', 'home_points': 28, 'homePregameElo': 1850,
             'awayId': '15', 'away_points': 24, 'awayPregameElo': 1700, 'excitementIndex': 7.5,
             'startDate': '2023-11-04', 'homeConference': 'SEC', 'awayConference': 'Big Ten',
             'homeClassification': 'fbs', 'awayClassification': 'fbs', 'seasonType': 'regular'}
        ]
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(csv_path, index=False)
        st.success("Dummy `data/games.csv` created!")

    try:
        games_df = pd.read_csv(csv_path)
        # Convert 'startDate' to datetime objects and filter for future games
        games_df['startDate'] = pd.to_datetime(games_df['startDate'])
        
        # Get the current time as a timezone-aware object (UTC)
        now_utc = datetime.now(timezone.utc) 
    
        # Now the comparison is between two timezone-aware objects (datetime64[ns, UTC] vs. datetime[UTC])
        future_games_df = games_df[games_df['startDate'] > now_utc].copy()

    except Exception as e:
        st.error(f"Error loading or processing games.csv: {e}")
        st.stop()

    st.subheader("Loaded Games Data (Filtered for Future Games)")
    st.dataframe(future_games_df.head())

    # --- FILTER AND PREDICT FOR UNCOMPLETED GAMES ---
    st.header("Predictions for Upcoming (Uncompleted) Games")
    predictions = []

    if future_games_df.empty:
        st.warning("No future games found in `data/games.csv` to predict.")
    else:
        progress_text = "Making predictions for upcoming games. Please wait..."
        my_bar = st.progress(0, text=progress_text)

        for i, (index, row) in enumerate(future_games_df.iterrows()):
            # Convert row to dictionary for predict_winner function
            # Ensure all required keys are present and correctly named
            raw_game_data = {
                "season": row['season'],
                "week": row['week'],
                "completed": False, # Set to False for future games
                "neutralSite": row['neutralSite'],
                "conferenceGame": row['conferenceGame'],
                "attendance": row['attendance'],
                "venueId": str(row['venueId']), # Ensure IDs are strings if label encoded
                "homeId": str(row['homeId']),
                "home_points": 0, # Set to 0 for pre-game prediction
                "homePregameElo": row['homePregameElo'],
                "awayId": str(row['awayId']),
                "away_points": 0, # Set to 0 for pre-game prediction
                "awayPregameElo": row['awayPregameElo'],
                "excitementIndex": row['excitementIndex'],
                "startDate": row['startDate'],
                "homeConference": row['homeConference'],
                "awayConference": row['awayConference'],
                "homeClassification": row['homeClassification'],
                "awayClassification": row['awayClassification'],
                "seasonType": row['seasonType'],
            }

            game_info = row.to_dict()
            try:
                result = predict_winner(raw_game_data)
                # Add original game info and prediction to results
                game_info.update(result)
            except Exception as e:
                st.error(f"Error predicting for game at index {index}: {e}")
                game_info['prediction_error'] = str(e)
                game_info['home_team_win_prob'] = None # Ensure key exists
                game_info['prediction'] = "Error"
            finally:
                predictions.append(game_info)

            # Update progress bar
            my_bar.progress((i + 1) / len(future_games_df), text=f"Predicting game {i+1} of {len(future_games_df)}")

        my_bar.empty() # Clear the progress bar after completion

        # Display predictions
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            # Select and reorder columns for display
            display_cols = [
                'homeId', 'awayId', 'homePregameElo', 'awayPregameElo',
                'home_team_win_prob', 'prediction', 'season', 'week', 'startDate'
            ]
            # Only format if 'home_team_win_prob' is not None
            format_dict = {'home_team_win_prob': lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A'}
            st.dataframe(predictions_df[display_cols].style.format(format_dict))
        else:
            st.info("No predictions could be generated for uncompleted games.")

if __name__ == "__main__":
    main()
