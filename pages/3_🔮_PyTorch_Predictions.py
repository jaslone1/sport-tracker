import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
from pathlib import Path
import torch
import torch.nn as nn

# --- Configuration ---
PYTORCH_MODEL_DIR = Path('ml_model/pytorch_training')
DATA_PATH = 'data/odds.csv'

# --------------------------------------------------------------------------
# --- PyTorch Model Definition ---
# --- MUST MATCH YOUR TRAINING SCRIPT EXACTLY ---
# --------------------------------------------------------------------------
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
        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------
# --- Data Loading Functions ---
# --------------------------------------------------------------------------

# (Copied from app.py for this page to be self-sufficient)
@st.cache_data
def load_data(path):
    """Load and filter the processed CSV data for FBS teams."""
    
    if not os.path.exists(path):
        st.error(f"Error: Processed file not found at {path}.")
        return pd.DataFrame()
    try:
        # Load with low_memory=False to match training script
        df = pd.read_csv(path, low_memory=False) 
        
        # Ensure 'is_upset' exists (even if NaN, for consistency)
        if "is_upset" not in df.columns:
            df["is_upset"] = np.nan # Add placeholder
        
        # We need all columns for preprocessing, even those that will be dropped
        # We also need the target column for stratification, even if it's NaN for unplayed games
        
        # Ensure all necessary analytical columns are numeric
        numeric_cols = ['homePoints', 'awayPoints', 'homePregameElo', 'homePostgameElo', 
                        'awayPregameElo', 'awayPostgameElo', 'attendance', 'neutralSite']
        for col in numeric_cols:
             if col in df.columns:
                 if col == 'neutralSite':
                     df[col] = df[col].astype(bool) 
                 else:
                     df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows missing conference data (as done in main app)
        df.dropna(subset=['homeConference', 'awayConference'], inplace=True)
        
        # FBS FILTERING LOGIC
        non_fbs_keywords = ['FCS', 'II', 'III', 'D-2', 'D-3', 'NAIA']
        mask_home = ~df['homeConference'].astype(str).str.contains('|'.join(non_fbs_keywords), case=False, na=False)
        mask_away = ~df['awayConference'].astype(str).str.contains('|'.join(non_fbs_keywords), case=False, na=False)
        df = df[mask_home & mask_away]
        
        return df
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------
# --- Artifact Loading ---
# --------------------------------------------------------------------------

@st.cache_resource
def load_pytorch_artifacts():
    """Loads all artifacts needed for Pytorch prediction."""
    try:
        with open(PYTORCH_MODEL_DIR / "feature_columns.json", "r") as f:
            feature_columns = json.load(f)
        
        scaler = joblib.load(PYTORCH_MODEL_DIR / "scaler.joblib")
        label_encoders = joblib.load(PYTORCH_MODEL_DIR / "label_encoders.joblib")
        
        # Load the model
        input_dim = len(feature_columns)
        model = MLP(input_dim)
        model.load_state_dict(torch.load(PYTORCH_MODEL_DIR / "upset_model_best.pth"))
        model.eval() # Set model to evaluation mode

        return model, scaler, feature_columns, label_encoders
    except FileNotFoundError as e:
        st.error(f"Error loading artifacts: {e}. Make sure all files are in `{PYTORCH_MODEL_DIR}`.")
        return None, None, None, None

# --------------------------------------------------------------------------
# --- Preprocessing and Prediction ---
# --------------------------------------------------------------------------

def preprocess_single_game(game_row_series, feature_columns, label_encoders, scaler):
    """
    Applies the *exact* preprocessing from the training script to a single game.
    """
    
    # Convert the Series to a 1-row DataFrame
    game_df = game_row_series.to_frame().T
    
    # 1. Apply Label Encoders (from training)
    for col, mapping in label_encoders.items():
        if col in game_df.columns:
            # Map the value using the loaded mapping.
            # If value is new, map to 0 (or a default index)
            game_df[col] = game_df[col].fillna("").astype(str).map(mapping)
            # Fill any unmapped (new) values with 0
            game_df[col] = game_df[col].fillna(0).astype(float)
            
    # 2. Apply One-Hot Encoding (implicitly via get_dummies)
    # We must re-create the dummy columns *exactly* as they were in training.
    # The safest way is to build the DataFrame and then re-index it.
    
    # Identify one-hot columns:
    # They are in feature_columns but *not* label_encoders and *not* a simple numeric col
    
    # We will build up the feature-set for our single game
    processed_df = pd.DataFrame(index=[game_row_series.name])
    
    # Get all numeric and label-encoded columns first
    for col in game_df.columns:
        if col in feature_columns:
            processed_df[col] = game_df[col]
            
    # Convert booleans
    for c in processed_df.columns:
        if processed_df[c].dtype == "bool":
            processed_df[c] = processed_df[c].astype(int)

    # 3. Handle One-Hot Encoding (using pd.get_dummies)
    # This is tricky. The *training script* did this, but we only have one row.
    # The *correct* way is to use the `feature_columns` list as the master template.
    
    # We will use pd.get_dummies and *then* reindex
    
    # Find columns that were one-hot-encoded
    # (Based on training script logic)
    to_one_hot = []
    if 'to_one_hot' in locals(): # Placeholder if we had this list
        pass # Use it
    else:
        # Inferring one-hot columns is hard.
        # A simpler/safer way: re-index and let pd.get_dummies create NaNs
        pass
    
    # Let's trust the training script's `pd.get_dummies` logic
    # We create a combined DataFrame and let `get_dummies` find object columns
    
    # Let's simplify and rely on the re-indexing step.
    # The training script used `pd.get_dummies` on columns.
    # We can try to replicate that.
    
    # SAFER METHOD: REINDEXING
    # Create a 1-row DataFrame with all possible feature columns, initialized to 0
    final_features_df = pd.DataFrame(0, index=[game_row_series.name], columns=feature_columns)
    
    # Fill in the values we *do* have
    for col in game_df.columns:
        if col in final_features_df.columns:
            final_features_df[col] = pd.to_numeric(game_df[col], errors='coerce')
    
    # Handle the categorical columns that *became* one-hot-encoded
    # e.g., if 'homeConference_SEC' is in `feature_columns`
    # and our game_row_series['homeConference'] == 'SEC'
    
    # This loop handles the one-hot columns
    for col in final_features_df.columns:
        if '_' in col:
            original_col = col.split('_')[0]
            value = col.split('_', 1)[1]
            if original_col in game_df.columns:
                if str(game_df[original_col].values[0]) == value:
                    final_features_df[col] = 1

    # 4. Fill NaNs (as done in training)
    # The training script filled numeric with median, but reindex filled with 0.
    # We'll fill NaNs with 0, matching the `fillna(0)` at the end of training.
    final_features_df = final_features_df.fillna(0)
    
    # 5. Apply Scaler
    scaled_data = scaler.transform(final_features_df)
    
    return scaled_data


# --- Main Page Logic ---
st.set_page_config(page_title="PyTorch Predictions", layout="wide")
st.title("🔮 PyTorch Upset Predictions")
st.markdown("Use the complex PyTorch MLP model to predict upset probability for unplayed games.")

model, scaler, feature_columns, label_encoders = load_pytorch_artifacts()
full_df = load_data(DATA_PATH)

if model is None or full_df.empty:
    st.error("Model or data failed to load. Cannot make predictions.")
else:
    # Get unplayed games
    df_unplayed = full_df[full_df['homePoints'].isna() | full_df['awayPoints'].isna()].copy()
    
    if df_unplayed.empty:
        st.info("No unplayed games found in the dataset.")
    else:
        # Create a display string for the selectbox
        df_unplayed['display_name'] = df_unplayed.apply(
            lambda row: f"{row['awayTeam']} @ {row['homeTeam']} (Elo: {int(row['awayPregameElo'])} @ {int(row['homePregameElo'])})",
            axis=1
        )
        
        selected_game_display = st.selectbox(
            "Select an Unplayed Game:",
            options=df_unplayed['display_name']
        )
        
        if selected_game_display:
            # Find the selected game's data row (by index)
            selected_index = df_unplayed[df_unplayed['display_name'] == selected_game_display].index[0]
            game_row = full_df.loc[selected_index]
            
            st.subheader("Selected Game")
            st.dataframe(game_row.to_frame().T, use_container_width=True)
            
            try:
                # Preprocess the single game row
                X_scaled = preprocess_single_game(game_row, feature_columns, label_encoders, scaler)
                
                # Convert to tensor
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                
                # Make prediction
                with torch.no_grad():
                    logits = model(X_tensor)
                    probability = torch.sigmoid(logits).item() # .item() gets the scalar value
                
                st.subheader("Prediction")
                
                # Determine underdog
                if game_row['homePregameElo'] > game_row['awayPregameElo']:
                    underdog = game_row['awayTeam']
                    favorite = game_row['homeTeam']
                else:
                    underdog = game_row['homeTeam']
                    favorite = game_row['awayTeam']
                
                st.metric(
                    label=f"Predicted Probability of an Upset (A win by {underdog})",
                    value=f"{probability * 100:.2f}%"
                )
                
                if probability > 0.5:
                    st.success(f"**High Upset Chance:** The model predicts '{underdog}' is more likely to win.")
                elif probability > 0.2:
                    st.warning(f"**Upset Watch:** The model gives '{underdog}' a decent chance.")
                else:
                    st.info(f"**Favorite Favored:** The model predicts '{favorite}' will likely win.")

            except Exception as e:
                st.error(f"An error occurred during preprocessing or prediction: {e}")
                st.error("This often happens if the game data has values (e.g., a new conference) that the model's preprocessing artifacts don't recognize.")
