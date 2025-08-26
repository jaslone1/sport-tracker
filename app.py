{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1b3457d-e11c-4507-8386-e8c154dabecb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "\n",
    "st.set_page_config(page_title=\"Sports Upset Tracker\", layout=\"wide\")\n",
    "\n",
    "st.title(\"🏈 Sports Upset Tracker (MVP)\")\n",
    "st.markdown(\"This site highlights potential and past upsets based on odds vs. results.\")\n",
    "\n",
    "# Load processed data\n",
    "try:\n",
    "    df = pd.read_csv(\"data/upsets.csv\")\n",
    "except:\n",
    "    st.error(\"No data found. Run data_fetch.py and process.py first.\")\n",
    "    st.stop()\n",
    "\n",
    "# Past upsets\n",
    "st.subheader(\"Past Upsets\")\n",
    "past_upsets = df[df[\"is_upset\"] == True]\n",
    "st.dataframe(past_upsets[[\"home_team\", \"away_team\", \"winner\", \"is_upset\"]])\n",
    "\n",
    "# Summary chart\n",
    "st.subheader(\"Upsets by Week (demo)\")\n",
    "if \"week\" in df.columns:\n",
    "    upset_counts = df.groupby(\"week\")[\"is_upset\"].sum().reset_index()\n",
    "    st.bar_chart(upset_counts.set_index(\"week\"))\n",
    "\n",
    "# Upcoming games placeholder\n",
    "st.subheader(\"Upcoming Games (with odds)\")\n",
    "if \"start_date\" in df.columns:\n",
    "    upcoming = df[df[\"winner\"].isna()]\n",
    "    st.dataframe(upcoming[[\"home_team\", \"away_team\", \"start_date\"]])\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "anaconda-2024.02-py310",
   "language": "python",
   "name": "conda-env-anaconda-2024.02-py310-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
