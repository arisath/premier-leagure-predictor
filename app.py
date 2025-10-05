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
files = ["data/PL_2021_2022.csv", "data/PL_2022_2023.csv", "data/PL_2023_2024.csv", "data/PL_2024_2025.csv"]
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

    # Normalize away FTR
    past_away = past_away.copy()
    past_away["FTR"] = past_away["FTR"].apply(lambda x: "H" if x == "A" else "A" if x == "H" else "D")

    recent = pd.concat([
        past_home[["Date", "FTHG", "FTAG", "FTR"]],
        past_away[["Date", "FTHG", "FTAG", "FTR"]]
    ]).sort_values("Date", ascending=False).head(n)

    goals_scored = recent["FTHG"].mean() if not recent.empty else 0
    goals_conceded = recent["FTAG"].mean() if not recent.empty else 0
    form_points = recent["FTR"].apply(lambda x: 3 if x=="H" else 1 if x=="D" else 0).sum() if not recent.empty else 0

    # Ensure exactly n FTRs
    ftr_list = recent["FTR"].tolist()
    if len(ftr_list) < n:
        ftr_list += [""] * (n - len(ftr_list))

    return goals_scored, goals_conceded, form_points, ftr_list

# -----------------------------
# Predictor
# -----------------------------
def predict_match(home_team, away_team, date):
    home_norm = fuzzy_map(home_team)
    away_norm = fuzzy_map(away_team)

    home_stats = team_last_5_stats(home_norm, date)
    away_stats = team_last_5_stats(away_norm, date)

    features = pd.DataFrame([home_stats[:3] + away_stats[:3]],
                            columns=["HG_Scored", "HG_Conceded", "H_Form",
                                     "AG_Scored", "AG_Conceded", "A_Form"])

    probs = clf.predict_proba(features)[0]
    # Return as a dict (explicit order: Home Win, Draw, Away Win)
    return {"Home Win": probs[0], "Draw": probs[1], "Away Win": probs[2]}, home_stats, away_stats

# -----------------------------
# Fixtures
# -----------------------------
def get_next_matches(n=10):
    url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(url, headers=headers)
    matches = response.json().get("matches", [])

    now = datetime.now(timezone.utc)
    future_matches = [m for m in matches if datetime.fromisoformat(m["utcDate"][:-1]).replace(tzinfo=timezone.utc) > now]

    future_matches_sorted = sorted(future_matches, key=lambda x: x["utcDate"])
    return future_matches_sorted[:n]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Premier League Predictor")

if st.button("🔮 Predict Next 10 Fixture Odds"):
    matches = get_next_matches(10)

    if not matches:
        st.warning("No upcoming matches found.")
    else:
        for i, match in enumerate(matches):
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            home_icon = match["homeTeam"].get("crest")
            away_icon = match["awayTeam"].get("crest")
            date = datetime.fromisoformat(match["utcDate"][:-1])

            # Run prediction (probs is computed here and is in scope for the following code)
            probs, home_stats, away_stats = predict_match(home, away, date)

            # Header with icons (fixed size)
            st.markdown(f"""
            <h3>
                <img src="{home_icon}" height="40" style="vertical-align:middle; margin-right:5px;"> {home} vs
                <img src="{away_icon}" height="40" style="vertical-align:middle; margin-left:5px;"> {away}
            </h3>
            """, unsafe_allow_html=True)
            st.markdown(f"📅 {date.strftime('%d/%m/%Y %H:%M')}")

            # Stats & Form
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"{home} (last 5)", value=f"{home_stats[2]} pts",
                          delta=f"{home_stats[0]:.1f} GS / {home_stats[1]:.1f} GC")
                # Heatmap with 5 equal segments
                fig_form = go.Figure()
                for idx, ftr in enumerate(home_stats[3]):
                    color = "green" if ftr=="H" else "yellow" if ftr=="D" else "red" if ftr=="A" else "lightgray"
                    fig_form.add_trace(go.Bar(
                        x=[1/5],
                        y=[1],
                        orientation="h",
                        marker_color=color,
                        base=[idx/5],
                        showlegend=False,
                        text=ftr,
                        textposition="inside"
                    ))
                fig_form.update_layout(height=50, yaxis=dict(showticklabels=False), xaxis=dict(showticklabels=False, range=[0,1]), margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_form, use_container_width=True, key=f"{home}_form_{i}")

            with col2:
                st.metric(label=f"{away} (last 5)", value=f"{away_stats[2]} pts",
                          delta=f"{away_stats[0]:.1f} GS / {away_stats[1]:.1f} GC")
                fig_form2 = go.Figure()
                for idx, ftr in enumerate(away_stats[3]):
                    color = "green" if ftr=="H" else "yellow" if ftr=="D" else "red" if ftr=="A" else "lightgray"
                    fig_form2.add_trace(go.Bar(
                        x=[1/5],
                        y=[1],
                        orientation="h",
                        marker_color=color,
                        base=[idx/5],
                        showlegend=False,
                        text=ftr,
                        textposition="inside"
                    ))
                fig_form2.update_layout(height=50, yaxis=dict(showticklabels=False), xaxis=dict(showticklabels=False, range=[0,1]), margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_form2, use_container_width=True, key=f"{away}_form_{i}")

            # -----------------------------
            # Decimal odds (implied from probabilities)
            # -----------------------------
            # probs is a dict with keys in the order: Home Win, Draw, Away Win
            outcomes = list(probs.keys())
            prob_values = list(probs.values())

            # Compute decimal odds; guard against zero probability
            odds_values = [(1.0 / p) if p > 0 else float('inf') for p in prob_values]
            odds_texts = [f"{o:.2f}" if o != float('inf') else '∞' for o in odds_values]

            # Build a small table for clarity
            odds_df = pd.DataFrame({
                "Outcome": outcomes,
                "Probability": [f"{p*100:.1f}%" for p in prob_values],
                "Decimal Odds": odds_texts
            })
            st.table(odds_df)

            # Also show a horizontal bar chart annotated with probability + decimal odds
            fig = go.Figure(go.Bar(
                x=prob_values,
                y=outcomes,
                orientation="h",
                marker_color=["green", "gray", "red"],
                text=[f"{p*100:.1f}% | {odds_texts[idx]}" for idx, p in enumerate(prob_values)],
                textposition="outside"
            ))
            fig.update_layout(
                title="Match Outcome Probabilities (with Decimal Odds)",
                xaxis=dict(range=[0, 1], title="Probability"),
                yaxis=dict(title="Outcome"),
                margin=dict(l=60, r=60, t=50, b=50),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True, key=f"odds_chart_{i}")
            st.divider()
