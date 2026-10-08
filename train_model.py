import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error


def train_model():
    file_path = "data/ml_ready_features.csv"
    if not os.path.exists(file_path):
        print("❌ Feature file not found.")
        return

    df = pd.read_csv(file_path)

    raw_features = [
        "neutral_site",
        "h_roll_pts_scored",
        "h_roll_ypp",
        "h_roll_ppm",
        "h_roll_turnovers",
        "h_sos",
        "a_roll_pts_scored",
        "a_roll_ypp",
        "a_roll_ppm",
        "a_roll_turnovers",
        "a_sos",
    ]

    targets = ["home_points", "away_points", "home_win"]
    df = df.dropna(subset=raw_features + targets)

    # Compute differential features (Home minus Away)
    df["diff_pts_scored"] = df["h_roll_pts_scored"] - df["a_roll_pts_scored"]
    df["diff_ypp"] = df["h_roll_ypp"] - df["a_roll_ypp"]
    df["diff_ppm"] = df["h_roll_ppm"] - df["a_roll_ppm"]
    df["diff_turnovers"] = df["h_roll_turnovers"] - df["a_roll_turnovers"]
    df["diff_sos"] = df["h_sos"] - df["a_sos"]

    features = raw_features + [
        "diff_pts_scored",
        "diff_ypp",
        "diff_ppm",
        "diff_turnovers",
        "diff_sos",
    ]

    # Target representations
    df["spread"] = df["home_points"] - df["away_points"]  # Positive = Home Win
    df["total"] = df["home_points"] + df["away_points"]

    # Temporal split by season
    max_year = df["year"].max()
    train_df = df[df["year"] < max_year]
    test_df = df[df["year"] == max_year]

    X_train, X_test = train_df[features], test_df[features]

    # --- 1. TRAIN SCORE MODELS ---
    spread_model = RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42
    )
    spread_model.fit(X_train, train_df["spread"])

    total_model = RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42
    )
    total_model.fit(X_train, train_df["total"])

    # --- 2. TRAIN MARGIN-TO-PROBABILITY CALIBRATOR ---
    train_pred_spreads = spread_model.predict(X_train)

    # FIX: fit_intercept=False guarantees 0 spread maps strictly to 50% probability,
    # eliminating any potential misalignment between scoreline and winner.
    calibrator = LogisticRegression(fit_intercept=False)
    calibrator.fit(train_pred_spreads.reshape(-1, 1), train_df["home_win"])

    # --- 3. EVALUATE ON TEST SET ---
    pred_spreads = spread_model.predict(X_test)
    pred_totals = total_model.predict(X_test)

    # Derived Scores
    pred_home_scores = (pred_totals + pred_spreads) / 2
    pred_away_scores = (pred_totals - pred_spreads) / 2

    # Derived Win Probabilities
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

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "features": features,
            "spread_model": spread_model,
            "total_model": total_model,
            "calibrator": calibrator,
        },
        "models/ncaa_model.pkl",
    )
    print("\n🚀 Model saved to models/ncaa_model.pkl")


if __name__ == "__main__":
    train_model()