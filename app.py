import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

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
        st.error(f"Error loading model artifacts. Check if files exist in {OUT_DIR}. Error: {e}")
        st.stop() # Stop the app if crucial files are missing

# Load everything globally
scaler, label_encoders, feature_columns, model, INPUT_DIM = load_all_artifacts()

# --- 4. PREDICTION FUNCTION (The core inference logic) ---
def predict_winner(raw_game_data: dict) -> dict:
    """
    Processes raw game data using saved artifacts and makes a prediction.

    """

    # 1. Convert raw input dict to a one-row DataFrame
    raw_df = pd.DataFrame([raw_game_data])
    df_features_new = raw_df.copy()

    # 2. Handle boolean columns
    for c in df_features_new.columns:
        if df_features_new[c].dtype == "bool":
            df_features_new[c] = df_features_new[c].astype(int)

    # 3. Label Encoding (for high-cardinality columns like IDs)
    df_le = df_features_new.copy()
    for col, mapping in label_encoders.items():
        if col in df_le.columns:
            series = df_le[col].fillna("").astype(str)
            # Map known values, fill unknown/unseen values with -1 or a defined default.
            df_le[col] = series.map(mapping).fillna(-1).astype(float)

    # 4. One-Hot Encoding (for low-cardinality columns)
    # Get dummies for all string/object columns that weren't label-encoded

    # Determine which columns to one-hot encode based on saved features
    ohe_cols_in_input = [col for col in df_features_new.columns if df_features_new[col].dtype == 'object' and col not in label_encoders]

    # Use training features list to create OHE columns
    # We must know the exact columns used in training (feature_columns) to align the data.
    # The safest way is to let get_dummies run on the input and then align.

    # Filter df_le to only include columns that are NOT object/string (already label encoded or numeric)
    # Then append one-hot encoded columns

    # Separate numeric/LE columns from those that need OHE
    numeric_le_cols = [c for c in df_le.columns if c not in ohe_cols_in_input]
    df_ohe = pd.get_dummies(df_features_new[ohe_cols_in_input].fillna(""), dummy_na=False)

    # 5. Combine and Align (CRITICAL STEP)
    # Concatenate the processed parts
    X_df_new = pd.concat([df_le[numeric_le_cols].fillna(0), df_ohe], axis=1)

    # Align the new data to the feature column order from training
    X_df_aligned = pd.DataFrame(0.0, index=X_df_new.index, columns=feature_columns)

    # Fill the aligned frame with values from the new data
    for col in X_df_new.columns:
        if col in X_df_aligned.columns:
             X_df_aligned[col] = X_df_new[col].values[0] # Transfer the single value

    if X_df_aligned.shape[1] != INPUT_DIM:
         raise ValueError(f"Feature count mismatch: Expected {INPUT_DIM}, got {X_df_aligned.shape[1]}")

    # 6. Scale and Predict
    X_new = X_df_aligned.values.astype(np.float32)
    X_scaled = scaler.transform(X_new)

    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor).cpu().numpy().ravel()[0]

        # Convert logit to probability (Sigmoid)
        probability = 1.0 / (1.0 + np.exp(-logits))

    # 7. Interpret Output
    prediction = int(probability >= 0.5)

    return {
        "home_team_win_prob": float(probability),
        "prediction": "Home Team Wins" if prediction == 1 else "Away Team Wins",
    }

# --- 5. STREAMLIT UI ---
def main():
    st.set_page_config(page_title="CFB Winner Predictor", layout="wide")
    st.title("🏈 College Football Winner Prediction")
    st.markdown("Use the input features below to predict the outcome of a game.")

    # --- INPUT WIDGETS ---
    st.header("Game & Team Statistics")

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Home Team Inputs")
            home_elo = st.number_input("Home Pre-Game ELO", min_value=1000, max_value=2500, value=1700, key="h_elo")
            home_id = st.text_input("Home Team ID (e.g., '52')", value='52', key="h_id")
            home_classification = st.selectbox("Home Classification", ["fbs", "fcs", "ii", "iii"], key="h_class", index=0)
            home_conference = st.text_input("Home Conference", value="SEC", key="h_conf")

        with col2:
            st.subheader("Away Team Inputs")
            away_elo = st.number_input("Away Pre-Game ELO", min_value=1000, max_value=2500, value=1500, key="a_elo")
            away_id = st.text_input("Away Team ID (e.g., '230')", value='230', key="a_id")
            away_classification = st.selectbox("Away Classification", ["fbs", "fcs", "ii", "iii"], key="a_class", index=0)
            away_conference = st.text_input("Away Conference", value="ACC", key="a_conf")

        with col3:
            st.subheader("Game Context")
            game_location_select = st.selectbox("Game Location", ["Home", "Neutral"], key="location_select")
            is_conference = st.checkbox("Is a Conference Game?", value=True, key="conf_game")

            with st.expander("More Game Details (All features for model alignment)"):
                season = st.number_input("Season Year", min_value=2000, max_value=2030, value=2024, key="season")
                week = st.number_input("Week Number", min_value=1, max_value=16, value=1, key="week")
                completed = st.checkbox("Game Completed (Assume True for pre-game prediction)", value=True, key="completed")
                attendance = st.number_input("Attendance", min_value=0, value=0, key="attendance")
                venue_id = st.text_input("Venue ID", value='0', key="venue_id")
                excitement_index = st.number_input("Excitement Index", min_value=0.0, value=0.0, format="%.2f", key="excitement_index")
                start_date = st.text_input("Start Date (YYYY-MM-DD)", value='2024-09-01', key="start_date")
                season_type = st.selectbox("Season Type", ["regular", "postseason"], key="season_type", index=0)

                # These are typically game outcomes, but if the model expects them as input for alignment, provide defaults
                # Their presence in feature_columns for a pre-game predictor is unusual and suggests potential model training considerations.
                st.markdown("--- *Points are typically game outcomes, setting to 0 for pre-game prediction* ---")
                home_points = st.number_input("Home Points (Default 0)", value=0, key="h_pts")
                away_points = st.number_input("Away Points (Default 0)", value=0, key="a_pts")


    # --- PREDICTION TRIGGER ---
    st.markdown("---currentState")
    if st.button("Calculate Prediction", use_container_width=True, type="primary"):
        # 1. Assemble the raw input dictionary.
        # The keys MUST match the column names of your training data (before pre-processing/scaling).

        raw_input = {
            "homePregameElo": home_elo, # Corrected key name
            "awayPregameElo": away_elo, # Corrected key name
            "homeId": home_id,
            "awayId": away_id,
            "conferenceGame": is_conference,

            # New features added
            "season": season,
            "week": week,
            "completed": completed,
            "attendance": attendance,
            "venueId": venue_id,
            "excitementIndex": excitement_index,
            "startDate": start_date,
            "homeConference": home_conference,
            "awayConference": away_conference,
            "homeClassification": home_classification,
            "awayClassification": away_classification,
            "seasonType": season_type,

            # Derived features
            "neutralSite": True if game_location_select == "Neutral" else False,

            # Problematic 'outcome' features, defaulted to 0
            "home_points": home_points,
            "away_points": away_points,
        }

        # --- Debug Information (temporarily added for diagnosis) ---
        with st.expander("💡 Debug Information (Click to expand)"):
            st.write("**Features Expected by Model (`feature_columns`):**")
            st.json(feature_columns)
            st.write("**Raw Input provided from UI:**")
            st.json(raw_input)
            missing_features_in_input = [f for f in feature_columns if f not in raw_input and f not in [c for c in raw_input if raw_input[c] in [home_classification, away_classification, season_type]] # Exclude base categorical features that get OHE
            ] # Adjusting check for features that are base for OHE
            if missing_features_in_input:
                st.warning(f"🚨 The following features expected by the model are MISSING from the UI input and will be set to 0: {missing_features_in_input}")
            else:
                st.success("✅ All model-expected features seem to be present in UI input or will be derived by the prediction function.")
        # --- End Debug Information ---

        # 2. Call the prediction function
        try:
            with st.spinner('Running PyTorch inference...'):
                prediction_result = predict_winner(raw_input)

            # 3. Display Results
            st.subheader("Prediction Result")
            st.info(f"Predicted Winner: **{prediction_result['prediction']}**")

            prob_home = prediction_result['home_team_win_prob']
            prob_away = 1.0 - prob_home

            st.progress(prob_home, text=f"Home Win Probability: {prob_home:.2%}")
            st.markdown(f"Away Win Probability: {prob_away:.2%}")

        except ValueError as e:
            st.error(f"Prediction failed due to data alignment error. Please check feature inputs. Details: {e}")
        except Exception as e:
            st.exception(e)

if __name__ == "__main__":
    main()
