#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd

df = pd.read_csv("data/upsets.csv")
st.write("Columns in dataframe:", df.columns.tolist())
st.set_page_config(page_title="Sports Upset Tracker", layout="wide")

st.title("🏈 Sports Upset Tracker (MVP)")
st.markdown("This site highlights potential and past upsets based on odds vs. results.")

# Load processed data
try:
    df = pd.read_csv("data/upsets.csv")
except:
    st.error("No data found. Run data_fetch.py and process.py first.")
    st.stop()

# Load data
df = pd.read_csv("data/upsets.csv")

st.title("🏈 Sports Upset Tracker")

# Show what columns exist
st.write("Columns in dataset:", df.columns.tolist())

# Safe selection for Past Upsets
if "is_upset" in df.columns:
    past_upsets = df[df["is_upset"] == True]
else:
    past_upsets = df.copy()  # fallback

# Pick only columns that exist
columns_to_show = [c for c in ["home_team", "away_team", "winner", "is_upset"] if c in past_upsets.columns]

st.subheader("Past Upsets")
st.dataframe(past_upsets[columns_to_show] if columns_to_show else past_upsets)
# Summary chart
st.subheader("Upsets by Week (demo)")
if "week" in df.columns:
    upset_counts = df.groupby("week")["is_upset"].sum().reset_index()
    st.bar_chart(upset_counts.set_index("week"))

# Upcoming games placeholder
st.subheader("Upcoming Games (with odds)")
if "start_date" in df.columns:
    upcoming = df[df["winner"].isna()]
    st.dataframe(upcoming[["home_team", "away_team", "start_date"]])

