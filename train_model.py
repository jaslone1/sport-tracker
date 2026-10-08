import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import norm


def train_model():
    file_path = "data/ml_ready_features.csv"
    if not os.path.exists(file_path):
        print("❌ Feature file not found.")
        return

    df = pd.read_csv(file_path)

    features = [
        "neutral_site", "h_roll_pts_scored", "h_roll_ypp", "h_roll_ppm", "h_roll_turnovers", "h_sos",
        "a_roll_pts_scored", "a_roll_ypp", "a_roll_ppm", "a_roll_turnovers", "a_sos"
    ]

    targets = ["home_points", "away_points", "home_win"]
    df = df.dropna(subset=features + targets)

    # Compute target representations
    df["spread"] = df["home_points"] - df["away_points"]  # Positive = Home Win
    df["total"] = df["home_points"] + df["away_points"]

    # Temporal split by season
    max_year = df["year"].max()
    train_df = df[df["year"] < max_year]
    test_df = df[df["year"] == max_year]

    X_train, X_test = train_df[features], test_df[features]

    # --- 1. TRAIN SCORE MODELS ---
    spread_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42)
    spread_model.fit(X_train, train_df["spread"])

    total_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42)
    total_model.fit(X_train, train_df["total"])

    # --- 2. TRAIN MARGIN-TO-PROBABILITY CALIBRATOR ---
    # Predict margins on training set to fit the logistic calibrator
    train_pred_spreads = spread_model.predict(X_train)

    calibrator = LogisticRegression()
    # Reshape array to 2D for scikit-learn fit
    calibrator.fit(train_pred_spreads.reshape(-1, 1), train_df["home_win"])

    # --- 3. EVALUATE ON TEST SET ---
    pred_spreads = spread_model.predict(X_test)
    pred_totals = total_model.predict(X_test)

    # Derived Scores
    pred_home_scores = (pred_totals + pred_spreads) / 2
    pred_away_scores = (pred_totals - pred_spreads) / 2

    # Derived Win Probabilities (Guaranteed 100% consistent with score line)
    pred_win_probs = calibrator.predict_proba(pred_spreads.reshape(-1, 1))[:, 1]
    pred_winners = pred_spreads > 0  # True if Home win predicted

    # Metrics
    acc = accuracy_score(test_df["home_win"], pred_winners)
    h_mae = mean_absolute_error(test_df["home_points"], pred_home_scores)
    a_mae = mean_absolute_error(test_df["away_points"], pred_away_scores)

    print("\n--- Coherent Model Performance ---")
    print(f"✅ Winner Accuracy (Derived from Score): {acc:.2%}")
    print(f"✅ Home Score MAE: {h_mae:.2f} pts")
    print(f"✅ Away Score MAE: {a_mae:.2f} pts")

    # Example single game prediction printout
    sample_spread = pred_spreads[0]
    sample_h_score = pred_home_scores[0]
    sample_a_score = pred_away_scores[0]
    sample_prob = pred_win_probs[0]

    print("\n--- Example Prediction ---")
    print(f"Scoreline: Home {sample_h_score:.1f} - Away {sample_a_score:.1f}")
    print(f"Projected Margin: {sample_spread:+.1f} pts")
    print(f"Home Win Probability: {sample_prob:.1%}")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "features": features,
        "spread_model": spread_model,
        "total_model": total_model,
        "calibrator": calibrator,
    }, "models/ncaa_model.pkl")


if __name__ == "__main__":
    train_model()