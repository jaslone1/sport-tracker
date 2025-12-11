import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone # Import datetime
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder to check type

# --- 1. CONFIGURATION ---
OUT_DIR = Path("ml_model/pytorch_training_winner")

MODEL_PATH = OUT_DIR / "winner_model.pth"
SCALER_PATH = OUT_DIR / "scaler.joblib"
FEATURE_COLUMNS_PATH = OUT_DIR / "feature_columns.json"
LABEL_ENCODERS_PATH = OUT_DIR / "label_encoders.joblib"

DEVICE = torch.device("cpu")  # Use CPU for deployment

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
    except FileNotFoundError as e:
        raise RuntimeError(f"One or more artifact files not found in {OUT_DIR}. Expected files: winner_model.pth, scaler.joblib, feature_columns.json, label_encoders.joblib. Error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading model artifacts from {OUT_DIR}. Please ensure all model files exist and are valid. Error: {e}") from e

# Load everything globally
try:
    scaler, label_encoders, feature_columns, model, INPUT_DIM = load_all_artifacts()
except RuntimeError as e:
    # Print to console and keep variables None for the UI to handle
    print(f"Failed to load model artifacts: {e}")
    scaler, label_encoders, feature_columns, model, INPUT_DIM = None, None, None, None, None

def predict_winner(raw_game_data: dict) -> dict:
    """
    Processes raw game data using the SAME preprocessing steps used in training.
    Includes verbose debugging output to Streamlit to help diagnose feature-mismatch issues.
    """
    try:
        # --- SAFETY: Ensure model artifacts loaded ---
        if scaler is None or label_encoders is None or feature_columns is None or model is None:
            raise RuntimeError("Model artifacts are not loaded. Cannot make predictions. Check that ml_model/pytorch_training_winner contains the trained model, scaler, label encoders, and feature_columns.json.")

        # --- 1. Convert incoming data to DataFrame ---
        df = pd.DataFrame([raw_game_data]).copy()

        # --- 2. Match training preprocessing ---
        # Convert booleans → ints
        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(int)

        # Convert startDate to numeric (training treated ALL numeric via pd.to_numeric)
        if "startDate" in df.columns:
            # coerce errors to NaT, then convert to int64 epoch (ns); if NaT remains, will be NaT -> handle below
            df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
            if df["startDate"].isna().any():
                # st.warning("startDate could not be parsed for one or more records; converted to NaT and then to numeric 0.")
                pass
            # convert to int64. If NaT, .view will produce a large negative; coerce that to 0 to be safe.
            try:
                df["startDate"] = df["startDate"].view("int64")
                # convert possible NaT sentinel to 0
                df["startDate"] = df["startDate"].replace({pd.NaT: 0})
                df["startDate"] = df["startDate"].fillna(0).astype(np.int64)
            except Exception:
                # fallback: convert to unix epoch seconds
                df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
                df["startDate"] = df["startDate"].astype('int64', errors='ignore').fillna(0).astype(np.int64)

        # Identify columns for label encoding vs one-hot vs numeric
        le_cols = list(label_encoders.keys()) if isinstance(label_encoders, dict) else []

        # --- 3. Label Encoding ---
        df_le = pd.DataFrame(index=df.index)
        for col in le_cols:
            if col in df.columns:
                encoder = label_encoders[col]
                # Check if it's a LabelEncoder object (preferred) or a dictionary mapping
                if isinstance(encoder, LabelEncoder):
                    # It's a LabelEncoder
                    df_le[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0)
                elif isinstance(encoder, dict):
                    # It's a dictionary, assume it's a direct mapping (e.g., if LabelEncoder was saved as a dict representation)
                    df_le[col] = df[col].apply(lambda x: encoder.get(x, 0)) # Use .get with default 0 for unseen categories
                else:
                    # Fallback for unexpected encoder type
                    # st.warning(f"Unexpected encoder type for column '{col}'. Expected LabelEncoder or dict, got {type(encoder)}. Assigning 0.")
                    df_le[col] = 0
            else:
                # If the column is completely missing from the input, fill with 0
                df_le[col] = 0

        # --- Training used ONE-HOT for these base categorical columns ---
        OHE_BASE_COLS = [
            "homeClassification",
            "awayClassification",
            "seasonType"
        ]

        # Determine which OHE columns are actually present in the input
        ohe_cols = [c for c in OHE_BASE_COLS if c in df.columns]

        # Numeric columns = everything else that is NOT label-encoded and NOT in OHE_BASE_COLS
        numeric_cols = [c for c in df.columns if c not in le_cols and c not in ohe_cols]

        # --- 4. One-Hot Encoding using the training prefixes ---
        df_ohe = pd.DataFrame(index=df.index)
        for col in ohe_cols:
            # Training used prefix="<col>"
            temp = pd.get_dummies(df[col].fillna(""), prefix=col)

            # Ensure we only keep categories that existed during training
            # The model expects columns that exist in feature_columns.json
            allowed_cols = [c for c in temp.columns if c in feature_columns]

            # Missing training-era categories must still exist in correct format → add them as zeros
            for m in [c for c in feature_columns if c.startswith(col + "_") and c not in allowed_cols]:
                temp[m] = 0

            # Keep only training categories
            temp = temp[[c for c in temp.columns if c in feature_columns]]

            df_ohe = pd.concat([df_ohe, temp], axis=1)

        # --- 5. Numeric columns (training used pd.to_numeric + fillna median) ---
        # We'll coerce numeric columns; fill NaNs with 0 for inference (training filled with medians)
        df_num = pd.DataFrame(index=df.index)
        for col in numeric_cols:
            try:
                df_num[col] = pd.to_numeric(df[col], errors="coerce")
                # fillna with 0 for safety; training used medians so distributions may differ
                df_num[col] = df_num[col].fillna(0)
            except Exception:
                df_num[col] = 0

        # --- 6. Combine everything EXACTLY like training ---
        # Ensure label encoded df_le only contains the le_cols that exist
        df_le_subset = df_le[[c for c in le_cols if c in df_le.columns]] if le_cols else pd.DataFrame(index=df.index)
        X_df_new = pd.concat([df_num, df_ohe, df_le_subset], axis=1).fillna(0)

        # --- 7. Align to training feature columns ---
        # Use reindex for robust alignment, filling missing columns with 0 and dropping extra columns
        X_aligned = X_df_new.reindex(columns=feature_columns, fill_value=0.0)

        # --- CRITICAL FIX: Neutralize the 'completed' feature ---
        # The training scaler has an unstable std dev for 'completed' (mostly 1s).
        # We must set the prediction value (which is 0) to the training mean (mu) so it scales to 0.
        # This prevents the severe negative outlier (-58.82) that causes 100% confidence errors.
        
        try:
            completed_idx = feature_columns.index("completed")
            completed_mean = scaler.mean_[completed_idx]
            
            # Replace the input value (which is 0) with the training mean.
            X_aligned.loc[:, "completed"] = completed_mean
        except ValueError:
            # If 'completed' is not in feature_columns, ignore this fix.
            pass
        except IndexError:
            # If scaler.mean_ is too short (shouldn't happen if alignment is correct), raise an error.
            st.warning("Scaler mean array is shorter than expected. Cannot apply 'completed' fix.")
            
        # --- 8. Sanity-check feature count matches model input dim ---
        if X_aligned.shape[1] != len(feature_columns):
            raise ValueError(f"Feature count mismatch: Expected {len(feature_columns)}, got {X_aligned.shape[1]}")

        # --- 9. Scale ---
        X_scaled = scaler.transform(X_aligned.values.astype(np.float32))
            st.write("DEBUG — Full Scaled Input Vector (Check for extreme values):", X_scaled[0, :])
        except Exception as e:
            raise

        # --- 10. Predict with PyTorch model ---
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
            logits = model(x_tensor).cpu().numpy().ravel()[0]
            prob = 1.0 / (1.0 + np.exp(-logits))

        # final interpretation
        # Calculate the probability of the home team winning
        home_win_prob = prob 
        
        # Determine the winner based on the home team's probability
        prediction_text = "Home Team Wins" if home_win_prob >= 0.5 else "Away Team Wins"
        
        return {
            "home_team_win_prob": float(home_win_prob), 
            "prediction": prediction_text,
        }

    except Exception as e:
        # Provide a thorough error message to Streamlit and to the caller
        err_text = f"Prediction failed: {e}"
        st.error(err_text)
        # Also print debug info to console
        print(err_text)
        # Return an error-like response (keeps keys consistent)
        return {
            "home_team_win_prob": None,
            "prediction": "Error",
            "prediction_error": str(e),
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
    st.dataframe(future_games_df.head()) # Display the filtered future games

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
            # Ensure all required keys are present and correctly named, with sensible defaults
            raw_game_data = {
                "season": row['season'],
                "week": row['week'],
                "completed": False, # Set to False for future games
                "neutralSite": row.get('neutralSite', False), # Default to False if missing
                "conferenceGame": row.get('conferenceGame', False), # Default to False if missing
                "attendance": row.get('attendance', 0), # Default to 0 if missing
                "venueId": row.get('venueId', '0'), # Default to '0' if missing
                "homeId": row.get('homeId', '0'), # Default to '0' if missing
                "home_points": 0, # Set to 0 for pre-game prediction
                "homePregameElo": row.get('homePregameElo', 1500), # Default to 1500 (average Elo) if missing
                "awayId": row.get('awayId', '0'), # Default to '0' if missing
                "away_points": 0, # Set to 0 for pre-game prediction
                "awayPregameElo": row.get('awayPregameElo', 1500), # Default to 1500 (average Elo) if missing
                "excitementIndex": row.get('excitementIndex', 0.0), # Default to 0.0 if missing
                "startDate": row.get('startDate', datetime.now(timezone.utc).isoformat()), # Default to current UTC time if missing
                "homeConference": row.get('homeConference', None),
                "awayConference": row.get('awayConference', None),
                "homeClassification": row.get('homeClassification', None),
                "awayClassification": row.get('awayClassification', None),
                "seasonType": row.get('seasonType', None),
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
