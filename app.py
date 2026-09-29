# Exact feature order required by the model
feature_cols = [
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


def predict_matchup(h_team, a_team, model, lookup_df):
    h_s = lookup_df[lookup_df["team"] == h_team]
    a_s = lookup_df[lookup_df["team"] == a_team]

    if h_s.empty or a_s.empty:
        return None, None

    h_s = h_s.iloc[0]
    a_s = a_s.iloc[0]

    h_sos_val = h_s.get("opp_def_strength", h_s.get("roll_pts_allowed", 0))
    a_sos_val = a_s.get("opp_def_strength", a_s.get("roll_pts_allowed", 0))

    input_df = pd.DataFrame(
        [
            {
                "neutral_site": 0,
                "h_roll_pts_scored": h_s.get("roll_pts_scored", 0),
                "h_roll_ypp": h_s.get("roll_ypp", 0),
                "h_roll_ppm": h_s.get("roll_ppm", 0),
                "h_roll_turnovers": h_s.get("roll_turnovers", 0),
                "h_sos": h_sos_val,
                "a_roll_pts_scored": a_s.get("roll_pts_scored", 0),
                "a_roll_ypp": a_s.get("roll_ypp", 0),
                "a_roll_ppm": a_s.get("roll_ppm", 0),
                "a_roll_turnovers": a_s.get("roll_turnovers", 0),
                "a_sos": a_sos_val,
            }
        ]
    )[feature_cols]

    # Use predict_proba to get actual probabilities [prob_away_win, prob_home_win]
    probs = model.predict_proba(input_df)[0]
    home_win_prob = probs[1]
    away_win_prob = probs[0]

    if home_win_prob >= away_win_prob:
        return h_team, home_win_prob
    else:
        return a_team, away_win_prob