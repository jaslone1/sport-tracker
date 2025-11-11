"""
Train a PyTorch MLP to predict 'is_upset' from data/games.csv.

Outputs saved to ml_model/pytorch_training/
- upset_model.pth (PyTorch weights)
- scaler.joblib (StandardScaler)
- feature_columns.json (ordered feature list)
- history.csv (epoch, train_loss, val_loss)
- test_metrics.csv (accuracy, roc_auc, precision, recall, f1)
- classification_report.txt
- feature_importance.csv
- PNG plots: loss_curve.png, roc_curve.png, confusion_matrix.png, feature_importance.png
"""

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
# CONFIG
# -----------------------------
DATA_PATH = Path("data/games.csv")             # your existing CSV
OUT_DIR = Path("ml_model/pytorch_training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
VAL_SIZE = 0.15  # fraction of train set used for validation

# -----------------------------
# UTIL: plots
# -----------------------------
def plot_loss(history_df, path):
    """Plot the training and validation loss curves."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    
    # Convert to numpy arrays to ensure 1D numeric
    epochs = history_df["epoch"].to_numpy()
    train_loss = history_df["train_loss"].astype(float).to_numpy()
    val_loss = history_df["val_loss"].astype(float).to_numpy()
    
    plt.plot(epochs, train_loss, label="train_loss", linewidth=2)
    plt.plot(epochs, val_loss, label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_roc(y_true, y_probs, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion(cm, out_path):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Confusion matrix")
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["not upset (0)", "upset (1)"], rotation=45)
    plt.yticks(tick_marks, ["not upset (0)", "upset (1)"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path)
    plt.close()

def plot_feature_importance(feat_df, out_path, top_n=20):
    top = feat_df.sort_values("importance_mean", ascending=True).tail(top_n)
    plt.figure(figsize=(8, max(4, 0.25 * top_n)))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title("Permutation Feature Importance (mean decrease in score)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------------
# LOAD & PREPROCESS
# -----------------------------
print("📥 Loading dataset:", DATA_PATH)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Ensure data/games.csv exists.")

# use low_memory=False to avoid dtype mixed warnings and parse all columns, then handle dtypes
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Loaded {len(df):,} rows and {len(df.columns):,} columns")

# Ensure target column exists
if "is_upset" not in df.columns:
    raise KeyError("Column 'is_upset' not found in games.csv. Script expects 'is_upset' as binary target.")

# Drop rows where target is missing
df = df.dropna(subset=["is_upset"])
df["is_upset"] = df["is_upset"].astype(int)

# We will use only pre-game/non-leaky fields. Keep safe columns and transform others.
# Columns to explicitly drop (leaky, large text, or irrelevant)
drop_columns = [
    "id", "winner", "highlights", "notes", "homeLineScores", "awayLineScores",
    "homePoints", "awayPoints", "homePostgameWinProbability", "awayPostgameWinProbability",
    "homePostgameElo", "awayPostgameElo", "startTimeTBD", "venue", "home_team", "away_team"
]
# only drop those present
drop_columns = [c for c in drop_columns if c in df.columns]
df_features = df.drop(columns=drop_columns)

# Convert boolean-ish columns to integers
for c in df_features.columns:
    if df_features[c].dtype == "bool":
        df_features[c] = df_features[c].astype(int)

# We will convert object dtype categorical columns with get_dummies but limit explosion:
# - For columns with high cardinality (like team names), better to use frequency cutoff.
# We'll treat 'homeId' and 'awayId' as numeric ids when present; if object, we'll label-encode them.

MAX_ONE_HOT_UNIQUE = 60  # if cardinality <= this, one-hot; else label-encode to avoid too-wide matrix

to_one_hot = []
to_label_encode = []
to_numeric = []

for col in df_features.columns:
    if col == "is_upset":
        continue
    dtype = df_features[col].dtype
    if dtype == "object":
        nunique = df_features[col].nunique(dropna=True)
        if nunique <= MAX_ONE_HOT_UNIQUE:
            to_one_hot.append(col)
        else:
            to_label_encode.append(col)
    elif np.issubdtype(dtype, np.number):
        to_numeric.append(col)
    else:
        # fallback to one-hot
        to_one_hot.append(col)

print(f"→ Numeric columns: {len(to_numeric)} | one-hot columns: {len(to_one_hot)} | label-encode columns: {len(to_label_encode)}")

# Apply one-hot encoding for small-cardinality object columns
if to_one_hot:
    df_ohe = pd.get_dummies(df_features[to_one_hot].fillna(""), prefix=to_one_hot, dummy_na=False)
else:
    df_ohe = pd.DataFrame(index=df_features.index)

# Label-encode high-cardinality object columns by mapping to integers (and save map)
label_encoders = {}
for col in to_label_encode:
    series = df_features[col].fillna("").astype(str)
    uniques = series.unique().tolist()
    mapping = {v: i for i, v in enumerate(uniques)}
    df_features[col] = series.map(mapping).astype(float)
    label_encoders[col] = mapping

# Numeric columns (fillna with median)
df_numeric = df_features[to_numeric].copy() if to_numeric else pd.DataFrame(index=df_features.index)
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")
    med = df_numeric[col].median()
    df_numeric[col] = df_numeric[col].fillna(med)

# Combine all processed columns
X_df = pd.concat([df_numeric, df_ohe] + ([df_features[to_label_encode]] if to_label_encode else []), axis=1).fillna(0)

# Ensure no column names duplicate and column order is fixed
X_df = X_df.loc[:, ~X_df.columns.duplicated()]

feature_columns = X_df.columns.tolist()
print(f"🔧 Final feature matrix shape: {X_df.shape}")

# Save feature columns & label encoder maps for inference later
joblib.dump(feature_columns, OUT_DIR / "feature_columns.json")
joblib.dump(label_encoders, OUT_DIR / "label_encoders.joblib")

# Target
y = df["is_upset"].values.astype(int)
X = X_df.values.astype(float)

# -----------------------------
# TRAIN / VAL / TEST SPLIT
# -----------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_train_full
)
print(f"📊 Splits: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

# -----------------------------
# SCALER
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
# Save feature names as JSON (joblib earlier saved list)
with open(OUT_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# -----------------------------
# TORCH DATASET / DATALOADER
# -----------------------------
def make_loader(X_arr, y_arr, batch_size=BATCH_SIZE, shuffle=True):
    X_t = torch.tensor(X_arr, dtype=torch.float32)
    y_t = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)

# -----------------------------
# MODEL (MLP) - uses logits + BCEWithLogitsLoss
# -----------------------------
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

model = MLP(input_dim=X_train.shape[1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAIN LOOP
# -----------------------------
history = {"epoch": [], "train_loss": [], "val_loss": []}
best_val_loss = float("inf")
best_model_path = OUT_DIR / "upset_model_best.pth"

print("🚀 Starting training...")
start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = float(np.mean(batch_losses))

    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(DEVICE), yv.to(DEVICE)
            v_logits = model(Xv)
            v_loss = criterion(v_logits, yv)
            val_losses.append(v_loss.item())
    val_loss = float(np.mean(val_losses))

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch:02d}/{EPOCHS} — train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

    # save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✅ Saved best model (val_loss improved to {val_loss:.4f})")

elapsed = time.time() - start_time
print(f"🏁 Training complete in {elapsed/60:.2f} minutes. Best val loss: {best_val_loss:.4f}")

# Save final model as well
final_model_path = OUT_DIR / "upset_model_final.pth"
torch.save(model.state_dict(), final_model_path)

# Save history to CSV
hist_df = pd.DataFrame(history)
hist_df.to_csv(OUT_DIR / "history.csv", index=False)
plot_loss(hist_df, OUT_DIR / "loss_curve.png")

# -----------------------------
# WRAPPER FOR sklearn perm import
# -----------------------------
class TorchEstimatorWrapper:
    """
    Minimal sklearn-style wrapper exposing fit (no-op), predict, predict_proba, score.
    sklearn.permutation_importance will use .score if scoring is None.
    """
    def __init__(self, torch_model, device):
        self.model = torch_model
        self.device = device

    def fit(self, X, y=None):
        # model already trained externally
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
            logits = self.model(xt).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
            logits = self.model(xt).cpu().numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            # sklearn expects shape (n_samples, n_classes)
            return np.vstack([1 - probs, probs]).T

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

# load best model weights into a clean model instance
best_model = MLP(input_dim=X_train.shape[1]).to(DEVICE)
best_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
best_model.eval()

# -----------------------------
# EVALUATE ON TEST SET
# -----------------------------
# get test probabilities and labels
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    logits_test = best_model(X_test_tensor).cpu().numpy().ravel()
    probs_test = 1.0 / (1.0 + np.exp(-logits_test))
    preds_test = (probs_test >= 0.5).astype(int)

acc = accuracy_score(y_test, preds_test)
roc_auc = roc_auc_score(y_test, probs_test)
cm = confusion_matrix(y_test, preds_test)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds_test, average="binary", zero_division=0)

print("\n📈 Test Results:")
print(f"  Accuracy: {acc:.4f}")
print(f"  ROC AUC:  {roc_auc:.4f}")
print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
print("  Confusion matrix:\n", cm)

# Save metrics
metrics = {
    "accuracy": float(acc),
    "roc_auc": float(roc_auc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "n_test": int(len(y_test))
}
pd.DataFrame([metrics]).to_csv(OUT_DIR / "test_metrics.csv", index=False)

# Save classification report text
cls_report = classification_report(y_test, preds_test)
with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(cls_report)
print("\nClassification report saved to", OUT_DIR / "classification_report.txt")

# Plot ROC and confusion matrix
plot_roc(y_test, probs_test, OUT_DIR / "roc_curve.png")
plot_confusion(cm, OUT_DIR / "confusion_matrix.png")

# -----------------------------
# PERMUTATION IMPORTANCE
# -----------------------------
print("\n🔎 Calculating permutation importance (this may take a while)...")
wrapped = TorchEstimatorWrapper(best_model, DEVICE)
# permutation_importance will call wrapped.score(X, y) repeatedly (we implemented score)
perm = permutation_importance(wrapped, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, scoring=None)

feat_imp_df = pd.DataFrame({
    "feature": feature_columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

feat_imp_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
print("Feature importance saved to", OUT_DIR / "feature_importance.csv")

# Save top-n figure
plot_feature_importance(feat_imp_df, OUT_DIR / "feature_importance.png", top_n=25)

# -----------------------------
# SAVE ARTIFACTS
# -----------------------------
# Save final model weights (already saved best & final) - keep a standard name
torch.save(best_model.state_dict(), OUT_DIR / "upset_model.pth")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")
joblib.dump(feature_columns, OUT_DIR / "feature_columns_list.joblib")
joblib.dump(label_encoders, OUT_DIR / "label_encoders.joblib")

print("\n✅ All artifacts saved to:", OUT_DIR)
print("You can now use these files in your Streamlit app (scaler + feature_columns + model + CSV metrics + PNG plots).")

