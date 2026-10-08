import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error


def train_model():
    file_path = "data/ml_ready_features.csv"
    if not os.path.exists(file_path):
        print("❌ Feature file not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv(file_path)

    # FIX 1: Don't exclude the max week upfront; we use temporal splitting below instead.
    # We also keep raw features here for reference or feature computation.
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

    targets = ["home_win", "home_points", "away_points"]

    # FIX 2: Drop missing values across BOTH features AND target variables
    # to prevent ValueError during model fitting.
    df = df.dropna(subset=raw_features + targets)

    # OPTIONAL ENHANCEMENT: Create differential features (home minus away)
    # Differential features generalize much better than raw separate team stats.
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

    # FIX 3: Replace random train_test_split with a Temporal Split.
    # Using the most recent year/season as the test set avoids lookahead bias.
    max_year = df["year"].max()
    train_mask = df["year"] < max_year
    test_mask = df["year"] == max_year

    # Fallback if your dataset only spans 1 single year: split on recent weeks instead
    if train_mask.sum() == 0:
        max_week = df["week"].max()
        train_mask = df["week"] < (max_week - 3)
        test_mask = df["week"] >= (max_week - 3)

    train_df = df[train_mask]
    test_df = df[test_mask]

    X_train, X_test = train_df[features], test_df[features]
    y_win_train, y_win_test = train_df["home_win"], test_df["home_win"]
    y_h_train, y_h_test = train_df["home_points"], test_df["home_points"]
    y_a_train, y_a_test = train_df["away_points"], test_df["away_points"]

    print(f"🧠 Training models on {len(X_train)} games...")
    print(f"🧪 Testing on {len(X_test)} out-of-time games (Season/Year: {max_year})...")

    # --- MODEL TRAINING ---
    winner_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    winner_model.fit(X_train, y_win_train)

    h_score_model = RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42
    )
    h_score_model.fit(X_train, y_h_train)

    a_score_model = RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=10, random_state=42
    )
    a_score_model.fit(X_train, y_a_train)

    # --- EVALUATION ---
    y_pred = winner_model.predict(X_test)

    accuracy = accuracy_score(y_win_test, y_pred)
    h_mae = mean_absolute_error(y_h_test, h_score_model.predict(X_test))
    a_mae = mean_absolute_error(y_a_test, a_score_model.predict(X_test))

    print("\n--- Model Performance (Out-of-Time Test Set) ---")
    print(f"✅ Winner Model Accuracy: {accuracy:.2%}")
    print(f"✅ Home Score MAE: {h_mae:.2f} pts")
    print(f"✅ Away Score MAE: {a_mae:.2f} pts")

    # --- SAVE ARTIFACTS ---
    os.makedirs("models", exist_ok=True)
    models = {
        "features": features,  # Included feature list so inference script knows exact inputs
        "winner": winner_model,
        "h_score": h_score_model,
        "a_score": a_score_model,
    }
    joblib.dump(models, "models/ncaa_model.pkl")
    print("\n🚀 Models and metadata saved to models/ncaa_model.pkl")


if __name__ == "__main__":
    train_model()