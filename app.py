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

# Past upsets
st.subheader("Past Upsets")
st.write("Columns in dataset:", df.columns.tolist())
past_upsets = df[df["is_upset"] == True]
st.dataframe(past_upsets[["home_team", "away_team", "winner", "is_upset"]])

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

