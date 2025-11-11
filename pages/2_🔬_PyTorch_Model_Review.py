import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="PyTorch Model Review",
    layout="wide"
)

PYTORCH_MODEL_DIR = 'ml_model/pytorch_training' # Path from your training script

# --------------------------------------------------------------------------
# --- PyTorch Model Artifact Loading Function ---
# --------------------------------------------------------------------------

@st.cache_data
def load_pytorch_artifacts(model_dir):
    """
    Loads all *available* artifacts from the PyTorch training run,
    based on the user-provided file list.
    """
    PYTORCH_DIR = Path(model_dir)
    artifacts = {}

    # Load test_metrics.json
    try:
        # Try reading as line-delimited first
        artifacts['test_metrics'] = pd.read_json(PYTORCH_DIR / "test_metrics.json", lines=True)
        if artifacts['test_metrics'].empty:
             # Fallback for non-line-delimited (e.g., list of dicts)
            artifacts['test_metrics'] = pd.read_json(PYTORCH_DIR / "test_metrics.json")
    except Exception:
         artifacts['test_metrics'] = None
    
    # Load history.csv
    try:
        artifacts['history'] = pd.read_csv(PYTORCH_DIR / "history.csv")
    except Exception:
        artifacts['history'] = None

    # Store path to loss_curve.png
    artifacts['loss_curve_png'] = PYTORCH_DIR / "loss_curve.png"
    
    return artifacts

# --- Load Artifacts ---
pytorch_artifacts = load_pytorch_artifacts(PYTORCH_MODEL_DIR)

# -------------------------------------------------------------------------
# --- PYTORCH MODEL PERFORMANCE REVIEW ---
# -------------------------------------------------------------------------
st.header("🔬 PyTorch Model Performance Review")
st.markdown(f"Displaying available training and evaluation artifacts from the PyTorch MLP model, loaded from `{PYTORCH_MODEL_DIR}`.")

# --- Test Metrics KPIs ---
st.subheader("Test Set Performance Metrics")
metrics_df = pytorch_artifacts.get('test_metrics')

if metrics_df is not None and not metrics_df.empty:
    # Extract first row of metrics
    metrics = metrics_df.iloc[0]
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    # Use .get() for safety in case a column is missing
    m_col1.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
    m_col2.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    m_col3.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
    m_col4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    m_col5.metric("Recall", f"{metrics.get('recall', 0):.4f}")
else:
    st.warning(f"Could not load test metrics from `{PYTORCH_MODEL_DIR}/test_metrics.json`.")

# --- Training History ---
st.subheader("Training & Validation Loss History")

history_df = pytorch_artifacts.get('history')

if history_df is not None:
    st.markdown("**Interactive Training History (from `history.csv`)**")
    # Melt dataframe for st.line_chart
    history_melted = history_df.melt(
        'epoch', 
        var_name='metric', 
        value_name='loss'
    )
    # Create Altair chart for better tooltips
    chart = alt.Chart(history_melted).mark_line().encode(
        x=alt.X('epoch', title='Epoch'),
        y=alt.Y('loss', title='Loss', scale=alt.Scale(type='log')), # Use log scale for loss
        color='metric',
        tooltip=['epoch', 'metric', 'loss']
    ).properties(
        title='Training History (loss vs. val_loss)'
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    # Fallback to the static image if history.csv is missing
    st.markdown("**Static Loss Curve (from `loss_curve.png`)**")
    if pytorch_artifacts['loss_curve_png'].exists():
        st.image(str(pytorch_artifacts['loss_curve_png']), use_column_width=True)
    else:
        st.warning(f"Could not load training history. Neither `history.csv` nor `loss_curve.png` was found in `{PYTORCH_MODEL_DIR}`.")

st.markdown("---")
st.info("The following artifacts were not found in your folder and are not displayed: `roc_curve.png`, `confusion_matrix.png`, `feature_importance.csv`, `classification_report.txt`.")
