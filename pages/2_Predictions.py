import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

st.set_page_config(page_title="Predictions", layout="wide")

st.title("Predictions")
st.markdown("Model-based **upset probability predictions** for upcoming FBS games.")

# --- Load data and models ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/odds.csv')
    if 'elo_diff' not in df.columns:
        df['elo_diff'] = df['homePregameElo'] - df['awayPregameElo']
    return df

@st.cache_data
def predict(df):
    LOGREG_MODEL_PATH = 'ml_model/logistic_regression_upset_model.pkl'
    XGBOOST_MODEL_PATH = 'ml_model/xgboost_classifier_upset_model.pkl'

    df['neutralSite'] = df['neutralSite'].fillna(0).astype(int)
    X = df[['elo_diff', 'neutralSite']].fillna(0)

    try:
        logreg = joblib.load(LOGREG_MODEL_PATH)
        df['logreg_upset_proba'] = logreg.predict_proba(X)[:, 1]
    except:
        df['logreg_upset_proba'] = np.nan

    try:
        xgb = joblib.load(XGBOOST_MODEL_PATH)
        df['xgb_upset_proba'] = xgb.predict_proba(X)[:, 1]
    except:
        df['xgb_upset_proba'] = np.nan

    return df

df = predict(load_data())
df_unplayed = df[df['homePoints'].isna() | df['awayPoints'].isna()].copy()

if df_unplayed.empty:
    st.info("No upcoming games available for predictions.")
    st.stop()

df_unplayed['Underdog'] = np.where(df_unplayed['elo_diff'] < 0, df_unplayed['homeTeam'], df_unplayed['awayTeam'])
base_cols = ['homeTeam', 'awayTeam', 'homePregameElo', 'awayPregameElo', 'Underdog']

# --- Logistic Regression ---
st.header("📊 Logistic Regression Predictions")
top_logreg = df_unplayed.sort_values(by='logreg_upset_proba', ascending=False).head(25)
st.dataframe(top_logreg[base_cols + ['logreg_upset_proba']], use_container_width=True)

chart_logreg = alt.Chart(top_logreg).mark_bar().encode(
    x=alt.X('logreg_upset_proba', bin=alt.Bin(maxbins=10), title='Upset Probability'),
    y='count()',
    tooltip=['logreg_upset_proba', 'count()']
)
st.altair_chart(chart_logreg, use_container_width=True)

# --- XGBoost ---
st.header("🌳 XGBoost Predictions")
top_xgb = df_unplayed.sort_values(by='xgb_upset_proba', ascending=False).head(25)
st.dataframe(top_xgb[base_cols + ['xgb_upset_proba']], use_container_width=True)

chart_xgb = alt.Chart(top_xgb).mark_bar(color='#008080').encode(
    x=alt.X('xgb_upset_proba', bin=alt.Bin(maxbins=10), title='Upset Probability'),
    y='count()',
    tooltip=['xgb_upset_proba', 'count()']
)
st.altair_chart(chart_xgb, use_container_width=True)
