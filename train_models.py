# train_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# --- Configuration ---
DATA_PATH = 'data/odds.csv'
MODEL_DIR = 'ml_model'
import os
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_data(df):
    """
    Prepares features and target for upset prediction.
    An upset (Target=1) occurs when the low-Elo team wins.
    """
    
    # Ensure necessary columns exist and handle potential NaNs from the start
    required_cols = ['homePregameElo', 'awayPregameElo', 'home_win', 'homeTeam', 'awayTeam', 'neutralSite']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' is missing from the DataFrame. Please check your data.")
            
    # 1. Calculate features
    df['elo_diff'] = df['homePregameElo'] - df['awayPregameElo']
    df['home_advantage'] = np.where(df['neutralSite'] == True, 0, 1)

    # 2. Define the Target (Y) and Features (X)
    
    # Determine which team is the 'Dog' (Lower Elo)
    df['dog_team'] = np.where(df['elo_diff'] >= 0, df['awayTeam'], df['homeTeam'])
    
    # Calculate dog_win (1 if low-Elo team wins, 0 otherwise)
    df['dog_win'] = np.where(
        df['dog_team'] == df['homeTeam'], 
        df['home_win'], 
        1 - df['home_win']
    )
    
    # Target (Y): 1 = Upset (Dog Win), 0 = Expected Result (Favorite Win)
    Y = df['dog_win'].astype(int)
    
    # Features (X): Magnitude of Elo difference and whether the game is at home
    df['abs_elo_diff'] = np.abs(df['elo_diff'])
    X = df[['abs_elo_diff', 'home_advantage']] 
    
    # ----------------------------------------------------
    # 3. CRUCIAL STEP: Drop rows where features or target are NaN
    # We combine X and Y, drop rows with NaNs, and then separate them again.
    data_combined = pd.concat([X, Y], axis=1).dropna()
    X_cleaned = data_combined[['abs_elo_diff', 'home_advantage']]
    Y_cleaned = data_combined['dog_win']
    # ----------------------------------------------------

    # Report how many games were dropped due to NaNs
    if len(df) != len(X_cleaned):
        print(f"Warning: Dropped {len(df) - len(X_cleaned)} rows due to missing values.")
        
    return X_cleaned, Y_cleaned

# Load the data (simplified, assumes cleaning is done in app.py logic)
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Cannot train models.")
    exit()

X, Y = prepare_data(df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Total Games Used for Training: {len(X_train)}")
print(f"Upset Rate in Training Data: {Y_train.mean():.2%}")

# train_models.py (Continuation)

def train_logistic_regression(X_train, Y_train):
    """Trains and saves a Logistic Regression Model."""
    print("\n--- Training Logistic Regression ---")
    
    # 1. Initialize and Train
    log_reg = LogisticRegression(random_state=42, solver='liblinear')
    log_reg.fit(X_train, Y_train)
    
    # 2. Evaluation
    Y_pred = log_reg.predict(X_test)
    Y_proba = log_reg.predict_proba(X_test)[:, 1]
    
    print("LogReg Classification Report:\n", classification_report(Y_test, Y_pred, zero_division=0))
    print(f"LogReg AUC Score: {roc_auc_score(Y_test, Y_proba):.4f}")
    
    # 3. Save the model
    model_path = os.path.join(MODEL_DIR, 'logistic_regression_upset_model.pkl')
    joblib.dump(log_reg, model_path)
    print(f"LogReg Model saved to {model_path}")
    
    return log_reg

# Run the training
log_reg_model = train_logistic_regression(X_train, Y_train)

# train_models.py (Continuation)

def train_xgboost(X_train, Y_train):
    """Trains and saves an XGBoost Classifier."""
    print("\n--- Training XGBoost Classifier ---")
    
    # 1. Initialize and Train
    # A simplified configuration for a fast example
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    xgb_clf.fit(X_train, Y_train)
    
    # 2. Evaluation
    Y_pred = xgb_clf.predict(X_test)
    Y_proba = xgb_clf.predict_proba(X_test)[:, 1]
    
    print("XGBoost Classification Report:\n", classification_report(Y_test, Y_pred, zero_division=0))
    print(f"XGBoost AUC Score: {roc_auc_score(Y_test, Y_proba):.4f}")
    
    # 3. Save the model
    model_path = os.path.join(MODEL_DIR, 'xgboost_upset_model.pkl')
    joblib.dump(xgb_clf, model_path)
    print(f"XGBoost Model saved to {model_path}")
    
    return xgb_clf

# Run the training
xgb_model = train_xgboost(X_train, Y_train)