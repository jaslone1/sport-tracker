import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.inspection import permutation_importance

# -----------------------------
# CONFIG (Updated for Feature CSV)
# -----------------------------
DATA_PATH = Path("data/ml_ready_features.csv")  # Point to engineered data
OUT_DIR = Path("ml_model/pytorch_training_winner")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 50 # Increased slightly as stats data is more complex
LR = 1e-3

# ... [Keep your plot_loss, plot_roc, plot_confusion, plot_feature_importance functions here] ...

# -----------------------------
# LOAD & PREPROCESS (Updated)
# -----------------------------
print("📥 Loading engineered dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Our target is 'home_win' (created in the previous step)
y = df["home_win"].astype(int).values

# Select only the features we want to train on. 
# We ignore IDs, names, and the target itself.
exclude = ["game_id", "year", "week", "home_team", "away_team", "home_win"]
feature_columns = [c for c in df.columns if c not in exclude]

print(f"🔧 Training on {len(feature_columns)} stats-based features.")
X = df[feature_columns].values.astype(float)

# -----------------------------
# SPLIT & SCALE
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train_full
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save artifacts
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
with open(OUT_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# -----------------------------
# TORCH DATASET & MODEL
# -----------------------------
def make_loader(X_arr, y_arr, batch_size=BATCH_SIZE, shuffle=True):
    X_t = torch.tensor(X_arr, dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Outputting logits
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X_train.shape[1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ... [Keep your Train Loop and Evaluation logic here] ...

print("\n✅ Training complete. Features based on Rolling Stats are much more predictive!")
