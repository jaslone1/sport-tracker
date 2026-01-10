import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

def train_model():
    file_path = "data/ml_ready_features.csv"
    if not os.path.exists(file_path):
        print("❌ Feature file not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv(file_path)

    # 1. MATCH THE COLUMNS EXACTLY (Fixed names from 'to' to 'turnovers')
    features = [
        'neutral_site', 
        'h_roll_pts_scored', 'h_roll_ypp', 'h_roll_ppm', 'h_roll_turnovers', 'h_sos',
        'a_roll_pts_scored', 'a_roll_ypp', 'a_roll_ppm', 'a_roll_turnovers', 'a_sos'
    ]
    
    # Check if features exist before dropping NaNs to avoid a crash
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"❌ Error: Missing columns in CSV: {missing}")
        return

    df = df.dropna(subset=features)
    
    X = df[features]
    y = df['home_win']
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🧠 Training model on {len(X_train)} games...")

    # 2. Optimized Random Forest
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. ANALYSIS: Feature Importance
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n📊 Feature Importance:")
    print(importances)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Performance ---")
    print(f"✅ Accuracy: {accuracy:.2%}")

    # 5. Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/ncaa_model.pkl")
    print("\n🚀 Model saved to models/ncaa_model.pkl")

if __name__ == "__main__":
    train_model()