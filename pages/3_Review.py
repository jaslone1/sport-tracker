import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Model Review", layout="wide")

st.title("Model Review")
st.markdown("Review of the **PyTorch model performance**, feature importance, and training history.")

# --- Load artifacts ---
@st.cache_data
def load_artifacts(model_dir='ml_model/pytorch_training'):
    dir_path = Path(model_dir)
    data = {}
    try:
        data['metrics'] = pd.read_csv(dir_path / "test_metrics.csv")
    except:
        data['metrics'] = None
    try:
        data['history'] = pd.read_csv(dir_path / "history.csv")
    except:
        data['history'] = None
    try:
        data['importance'] = pd.read_csv(dir_path / "feature_importance.csv")
    except:
        data['importance'] = None
    try:
        with open(dir_path / "classification_report.txt") as f:
            data['report'] = f.read()
    except:
        data['report'] = None
    data['plots'] = {
        "loss": dir_path / "loss_curve.png",
        "roc": dir_path / "roc_curve.png",
        "confusion": dir_path / "confusion_matrix.png"
    }
    return data

artifacts = load_artifacts()

# --- Metrics ---
st.header("Performance Metrics")
if artifacts['metrics'] is not None:
    m = artifacts['metrics'].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC AUC", f"{m.get('roc_auc', 0):.3f}")
    c2.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
    c3.metric("F1", f"{m.get('f1', 0):.3f}")
    c4.metric("Precision", f"{m.get('precision', 0):.3f}")
    c5.metric("Recall", f"{m.get('recall', 0):.3f}")
else:
    st.warning("Test metrics file not found.")

# --- Plots ---
st.header("Evaluation Plots")
col1, col2, col3 = st.columns(3)
for col, (name, path) in zip([col1, col2, col3], artifacts['plots'].items()):
    if path.exists():
        col.image(str(path), caption=name.capitalize(), use_column_width=True)
    else:
        col.info(f"{name.capitalize()} plot missing.")

# --- Feature Importance ---
st.header("Feature Importance")
if artifacts['importance'] is not None:
    top = artifacts['importance'].sort_values(by="importance_mean", ascending=False).head(25)
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X('importance_mean', title='Importance'),
        y=alt.Y('feature', sort='-x'),
        tooltip=['feature', 'importance_mean']
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Feature importance file not found.")

# --- Training History ---
st.header("Training History")
if artifacts['history'] is not None:
    melted = artifacts['history'].melt('epoch', var_name='metric', value_name='loss')
    chart = alt.Chart(melted).mark_line().encode(
        x='epoch', y='loss', color='metric', tooltip=['epoch', 'loss']
    ).properties(title='Loss over Epochs')
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Training history file missing.")

# --- Classification Report ---
st.header("Classification Report")
if artifacts['report']:
    st.code(artifacts['report'])
else:
    st.info("Classification report not found.")
