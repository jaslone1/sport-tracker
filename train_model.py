import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def train_model():
    file_path = "data/ml_ready_features.csv"
    if not os.path.exists(file_path):
        print("❌ Feature file not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv(file_path)

    # Exclude the most recent week to prevent data leakage
    max_year = df["year"].max()
    max_week = df[df["year"] == max_year]["week"].max()
    df = df[~((df["year"] == max_year) & (df["week"] == max_week))]

    features = [
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

    df = df.dropna(subset=features)
    X = df[features]
    y_win = df["home_win"]
    y_h_score = df["home_points"]
    y_a_score = df["away_points"]

    X_train, X_test, y_win_train, y_win_test, y_h_train, y_h_test, y_a_train, y_a_test = train_test_split(
        X, y_win, y_h_score, y_a_score, test_size=0.2, random_state=42
    )

    print(f"🧠 Training models on {len(X_train)} games...")

    winner_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    winner_model.fit(X_train, y_win_train)

    h_score_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    h_score_model.fit(X_train, y_h_train)

    a_score_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    a_score_model.fit(X_train, y_a_train)

    y_pred = winner_model.predict(X_test)
    y_probs = winner_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_win_test, y_pred)
    h_mae = mean_absolute_error(y_h_test, h_score_model.predict(X_test))
    a_mae = mean_absolute_error(y_a_test, a_score_model.predict(X_test))

    print("\n--- Model Performance ---")
    print(f"✅ Winner Model Accuracy: {accuracy:.2%}")
    print(f"✅ Home Score MAE: {h_mae:.2f}")
    print(f"✅ Away Score MAE: {a_mae:.2f}")

    os.makedirs("models", exist_ok=True)
    models = {
        "winner": winner_model,
        "h_score": h_score_model,
        "a_score": a_score_model,
    }
    joblib.dump(models, "models/ncaa_model.pkl")
    print("\n🚀 Models saved to models/ncaa_model.pkl")


if __name__ == "__main__":
    train_model()