import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
import pickle
from rapidfuzz import process
import plotly.graph_objects as go
from config import API_KEY  # store API key separately

# -----------------------------
# Load historical data
# -----------------------------
files = ["PL_2021_2022.csv", "PL_2022_2023.csv", "PL_2023_2024.csv", "PL_2024_2025.csv"]
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

csv_teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())

# -----------------------------
# Load trained model
# -----------------------------
with open("premier_league_model.pkl", "rb") as f:
    clf, le_team = pickle.load(f)

# -----------------------------
# Fuzzy mapping
# -----------------------------
def fuzzy_map(api_name, choices=csv_teams, threshold=80):
    match, score, _ = process.extractOne(api_name, choices)
    if score >= threshold:
        return match
    return api_name

# -----------------------------
# Stats
# -----------------------------
def team_last_5_stats(team, date, n=5):
    past_home = df[(df["HomeTeam"] == team) & (df["Date"] < date)].sort_values("Date", ascending=False).head(n)
    past_away = df[(df["AwayTeam"] == team) & (df["Date"] < date)].sort_values("Date", ascending=False).head(n)

    recent = pd.concat([
        past_home[["FTHG", "FTAG", "FTR"]],
        past_away[["FTAG", "FTHG", "FTR"]].rename(columns={"FTAG": "FTHG", "FTHG": "FTAG"})
    ])

    goals_scored = recent["FTHG"].mean() if not recent.empty else 0
    goals_conceded = recent["FTAG"].mean() if not recent.empty else 0
    form_points = recent["FTR"].apply(lambda x: 3 if x == "H" else 1 if x == "D" else 0).sum() if not recent.empty else 0

    return goals_scored, goals_conceded, form_points

# -----------------------------
# Last 5 games form as colors
# -----------------------------
def last_5_form_colors(team, date):
    past_home = df[(df["HomeTeam"] == team) & (df["Date"] < date)].sort_values("Date", ascending=False).head(5)
    past_away = df[(df["AwayTeam"] == team) & (df["Date"] < date)].sort_values("Date", ascending=False).head(5)

    # Normalize FTR for away games
    past_away = past_away.copy()
    past_away["FTR"] = past_away["FTR"].apply(lambda x: "H" if x == "A" else "A" if x == "H" else "D")

    recent = pd.concat([past_home[["Date", "FTR"]], past_away[["Date", "FTR"]]])
    recent = recent.sort_values("Date", ascending=False).head(5)

    colors = []
    for r in recent["FTR"]:
        if r == "H":  # Win
            colors.append("green")
        elif r == "D":  # Draw
            colors.append("yellow")
        else:  # Loss
            colors.append("red")
    return colors

# -----------------------------
# Predictor
# -----------------------------
def predict_match(home_team, away_team, date):
    home_norm = fuzzy_map(home_team)
    away_norm = fuzzy_map(away_team)

    home_stats = team_last_5_stats(home_norm, date)
    away_stats = team_last_5_stats(away_norm, date)

    features = pd.DataFrame([home_stats + away_stats],
                            columns=["HG_Scored", "HG_Conceded", "H_Form",
                                     "AG_Scored", "AG_Conceded", "A_Form"])

    probs = clf.predict_proba(features)[0]
    return {"Home Win": probs[0], "Draw": probs[1], "Away Win": probs[2]}, home_stats, away_stats

# -----------------------------
# Fixtures
# -----------------------------
def get_next_matchday():
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers)
    matches = response.json().get("matches", [])

    now = datetime.now(timezone.utc)
    future_matches = [m for m in matches if datetime.fromisoformat(m["utcDate"][:-1]).replace(tzinfo=timezone.utc) > now]

    if not future_matches:
        return []

    next_matchday = min(m["matchday"] for m in future_matches)
    return [m for m in future_matches if m["matchday"] == next_matchday]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Premier League Predictor")

if st.button("🔮 Predict Next Fixture Odds"):
    matches = get_next_matchday()

    if not matches:
        st.warning("No upcoming matches found.")
    else:
        st.subheader(f"Next Matchday: {matches[0]['matchday']}")

        for match in matches:
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            date = datetime.fromisoformat(match["utcDate"][:-1])

            # Run prediction
            probs, home_stats, away_stats = predict_match(home, away, date)

            st.markdown(f"### {home} vs {away}")
            st.markdown(f"📅 {date.strftime('%d/%m/%Y %H:%M')}")

            # Show stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"{home} (last 5)", value=f"{home_stats[2]} pts",
                          delta=f"{home_stats[0]:.1f} GS / {home_stats[1]:.1f} GC")
                # Form heatmap
                fig_form = go.Figure(go.Bar(
                    x=list(range(1, 6)),
                    y=[1]*5,
                    marker_color=last_5_form_colors(fuzzy_map(home), date),
                    orientation="h",
                    text=["W" if c=="green" else "D" if c=="yellow" else "L" for c in last_5_form_colors(fuzzy_map(home), date)],
                    textposition="inside",
                    showlegend=False
                ))
                fig_form.update_layout(height=50, yaxis=dict(showticklabels=False))
                st.plotly_chart(fig_form, use_container_width=True)
            with col2:
                st.metric(label=f"{away} (last 5)", value=f"{away_stats[2]} pts",
                          delta=f"{away_stats[0]:.1f} GS / {away_stats[1]:.1f} GC")
                # Form heatmap
                fig_form2 = go.Figure(go.Bar(
                    x=list(range(1, 6)),
                    y=[1]*5,
                    marker_color=last_5_form_colors(fuzzy_map(away), date),
                    orientation="h",
                    text=["W" if c=="green" else "D" if c=="yellow" else "L" for c in last_5_form_colors(fuzzy_map(away), date)],
                    textposition="inside",
                    showlegend=False
                ))
                fig_form2.update_layout(height=50, yaxis=dict(showticklabels=False))
                st.plotly_chart(fig_form2, use_container_width=True)

            # Odds chart
            fig = go.Figure(go.Bar(
                x=list(probs.values()),
                y=list(probs.keys()),
                orientation="h",
                marker_color=["green", "gray", "red"],
                text=[f"{p*100:.1f}%" for p in probs.values()],
                textposition="outside"
            ))
            fig.update_layout(
                title="Match Outcome Probabilities",
                xaxis=dict(range=[0, 1], title="Probability"),
                yaxis=dict(title="Outcome"),
                margin=dict(l=60, r=60, t=50, b=50),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
