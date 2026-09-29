import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model():
    file_path = "data/ml_ready_features.csv"
    if not os.path.exists(file_path):
        print("❌ Feature file not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv(file_path)

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
    y = df["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"🧠 Training model on {len(X_train)} games...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,  # Reduced max_depth to prevent over-confident probabilities
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Performance ---")
    print(f"✅ Test Accuracy: {accuracy:.2%}")
    print(
        f"📊 Sample Probabilities (First 5 test games): {y_probs[:5].round(3)}"
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/ncaa_model.pkl")
    print("\n🚀 Model saved to models/ncaa_model.pkl")


if __name__ == "__main__":
    train_model()