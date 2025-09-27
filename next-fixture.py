import pandas as pd
import requests
from datetime import datetime, timezone
import pickle
from rapidfuzz import process
from config import API_KEY  # Your API key in a separate file

# -----------------------------
# Load historical data
# -----------------------------
files = ["PL_2021_2022.csv", "PL_2022_2023.csv", "PL_2023_2024.csv", "PL_2024_2025.csv"]
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Unique team names from historical CSVs
csv_teams = pd.unique(df[["HomeTeam","AwayTeam"]].values.ravel())

# -----------------------------
# Load trained model
# -----------------------------
with open("premier_league_model.pkl", "rb") as f:
    clf, le_team = pickle.load(f)

# -----------------------------
# Fuzzy mapping for team names
# -----------------------------
def fuzzy_map(api_name, choices=csv_teams, threshold=80):
    """
    Maps an API team name to the closest match in historical CSV teams.
    Returns the matched CSV name if similarity > threshold, else returns original.
    """
    match, score, _ = process.extractOne(api_name, choices)
    if score >= threshold:
        return match
    return api_name

# -----------------------------
# Compute last 5 games stats
# -----------------------------
def team_last_5_stats(team, date, n=5):
    past_home = df[(df["HomeTeam"] == team) & (df["Date"] < date)].sort_values("Date", ascending=False).head(n)
    past_away = df[(df["AwayTeam"] == team) & (df["Date"] < date)].sort_values("Date", ascending=False).head(n)
    
    recent = pd.concat([
        past_home[["FTHG","FTAG","FTR"]],
        past_away[["FTAG","FTHG","FTR"]].rename(columns={"FTAG":"FTHG","FTHG":"FTAG"})
    ])
    
    goals_scored = recent["FTHG"].mean() if not recent.empty else 0
    goals_conceded = recent["FTAG"].mean() if not recent.empty else 0
    form_points = recent["FTR"].apply(lambda x: 3 if x=="H" else 1 if x=="D" else 0).sum() if not recent.empty else 0
    
    return goals_scored, goals_conceded, form_points

# -----------------------------
# Predict function
# -----------------------------
def predict_match(home_team, away_team, date):
    home_norm = fuzzy_map(home_team)
    away_norm = fuzzy_map(away_team)
    
    home_stats = team_last_5_stats(home_norm, date)
    away_stats = team_last_5_stats(away_norm, date)
    
    features = pd.DataFrame([home_stats + away_stats],
                            columns=["HG_Scored","HG_Conceded","H_Form","AG_Scored","AG_Conceded","A_Form"])
    
    probs = clf.predict_proba(features)[0]  # [Home Win, Draw, Away Win]
    return {"Home Win": probs[0], "Draw": probs[1], "Away Win": probs[2]}

# -----------------------------
# Fetch scheduled matches
# -----------------------------
url = "https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED"
headers = {"X-Auth-Token": API_KEY}
response = requests.get(url, headers=headers)
matches = response.json().get("matches", [])

# -----------------------------
# Determine next matchday automatically
# -----------------------------
now = datetime.now(timezone.utc)
future_matches = [m for m in matches if datetime.fromisoformat(m["utcDate"][:-1]).replace(tzinfo=timezone.utc) > now]

if not future_matches:
    print("No upcoming matches found.")
    exit()

next_matchday = min(m["matchday"] for m in future_matches)
next_matchday_matches = [m for m in future_matches if m["matchday"] == next_matchday]

# -----------------------------
# Predict probabilities for each match
# -----------------------------
print(f"Next Premier League matchday: {next_matchday}\n")
for match in next_matchday_matches:
    home = match["homeTeam"]["name"]
    away = match["awayTeam"]["name"]
    date = datetime.fromisoformat(match["utcDate"][:-1])
    
    probs = predict_match(home, away, date)
    probs_percent = {k: f"{v*100:.1f}%" for k,v in probs.items()}
    
    print(f"{home} vs {away} on {date.strftime('%d/%m/%Y %H:%M')}")
    print(f"Predicted probabilities: {probs_percent}\n")
