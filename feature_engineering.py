import os
import pandas as pd


def create_ml_features():
    # 1. Load Data
    df = pd.read_csv(
        "data/detailed_stats.csv",
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )
    df.columns = df.columns.str.strip()

    # Ensure neutral_site exists
    if "neutral_site" not in df.columns:
        df["neutral_site"] = 0
    else:
        df["neutral_site"] = df["neutral_site"].fillna(0).astype(int)

    cols_to_fix = [
        "home_points",
        "away_points",
        "h_yds",
        "a_yds",
        "h_to",
        "a_to",
        "h_pos_sec",
        "a_pos_sec",
        "h_pen_yds",
        "a_pen_yds",
        "h_rushingAttempts",
        "h_completionAttempts",
        "a_rushingAttempts",
        "a_completionAttempts",
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Target: Home Win
    df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)

    # Play counts & Per-play / Per-minute stats
    df["h_plays"] = df["h_rushingAttempts"] + df["h_completionAttempts"]
    df["a_plays"] = df["a_rushingAttempts"] + df["a_completionAttempts"]
    df["h_ypp"] = df["h_yds"] / df["h_plays"].replace(0, 1)
    df["a_ypp"] = df["a_yds"] / df["a_plays"].replace(0, 1)

    df["h_pos_sec"] = df["h_pos_sec"].replace(0, 1800)
    df["a_pos_sec"] = df["a_pos_sec"].replace(0, 1800)
    df["h_ppm"] = df["home_points"] / (df["h_pos_sec"] / 60)
    df["a_ppm"] = df["away_points"] / (df["a_pos_sec"] / 60)

    # Unpivot home and away performances into single team stream
    h_cols = [
        "year",
        "week",
        "h_team",
        "h_yds",
        "h_to",
        "h_ypp",
        "h_ppm",
        "h_pen_yds",
        "home_points",
        "away_points",
    ]
    a_cols = [
        "year",
        "week",
        "a_team",
        "a_yds",
        "a_to",
        "a_ypp",
        "a_ppm",
        "a_pen_yds",
        "away_points",
        "home_points",
    ]
    shared_cols = [
        "year",
        "week",
        "team",
        "yards",
        "turnovers",
        "ypp",
        "ppm",
        "pen_yds",
        "pts_scored",
        "pts_allowed",
    ]

    home_df = df[h_cols].copy()
    away_df = df[a_cols].copy()
    home_df.columns = shared_cols
    away_df.columns = shared_cols

    perf_df = pd.concat([home_df, away_df]).sort_values(
        ["team", "year", "week"]
    )

    stats_to_roll = [
        "yards",
        "turnovers",
        "ypp",
        "ppm",
        "pen_yds",
        "pts_scored",
        "pts_allowed",
    ]

    # Calculate rolling averages shifted by 1 game (no data leakage)
    for stat in stats_to_roll:
        perf_df[f"roll_{stat}"] = (
            perf_df.groupby(["team", "year"])[stat]
            .transform(
                lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
            )
        )

    perf_df["opp_def_strength"] = perf_df["roll_pts_allowed"]

    # Save latest lookup data sorted by year AND week
    latest_stats = perf_df.sort_values(["year", "week"]).groupby("team").tail(1)
    os.makedirs("data", exist_ok=True)
    latest_stats.to_csv("data/team_lookup.csv", index=False)

    rolling_cols = ["year", "week", "team"] + [
        f"roll_{s}" for s in stats_to_roll
    ]
    rolling_lookup = perf_df[rolling_cols]

    # Merge Home rolling stats
    df = df.merge(
        rolling_lookup,
        left_on=["year", "week", "h_team"],
        right_on=["year", "week", "team"],
        how="left",
    )
    df = df.rename(
        columns={f"roll_{s}": f"h_roll_{s}" for s in stats_to_roll}
    ).drop(columns=["team"])
    df = df.rename(columns={"h_roll_pts_allowed": "h_sos"})

    # Merge Away rolling stats
    df = df.merge(
        rolling_lookup,
        left_on=["year", "week", "a_team"],
        right_on=["year", "week", "team"],
        how="left",
    )
    df = df.rename(
        columns={f"roll_{s}": f"a_roll_{s}" for s in stats_to_roll}
    ).drop(columns=["team"])
    df = df.rename(columns={"a_roll_pts_allowed": "a_sos"})

    feature_cols = [
        "year",
        "week",
        "h_team",
        "a_team",
        "home_win",
        "neutral_site",
        "home_points",
        "away_points",
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

    df_ml_clean = df[feature_cols].dropna()
    df_ml_clean.to_csv("data/ml_ready_features.csv", index=False)

    print(
        f"🚀 Success! Created clean ML dataset with {len(df_ml_clean)} games."
    )


if __name__ == "__main__":
    create_ml_features()