import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------
# Load multiple seasons
# -----------------------------
files = ["PL_2021_2022.csv", "PL_2022_2023.csv", "PL_2023_2024.csv", "PL_2024_2025.csv"]
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# -----------------------------
# Encode team names
# -----------------------------
teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
le_team = LabelEncoder()
le_team.fit(teams)

# -----------------------------
# Feature: last 5 games stats
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
# Prepare training data
# -----------------------------
feature_rows = []
labels = []

for idx, row in df.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    date = row["Date"]
    
    home_stats = team_last_5_stats(home, date)
    away_stats = team_last_5_stats(away, date)
    
    features = home_stats + away_stats
    feature_rows.append(features)
    labels.append(row["FTR"])  # H, D, A

X = pd.DataFrame(feature_rows, columns=["HG_Scored","HG_Conceded","H_Form",
                                        "AG_Scored","AG_Conceded","A_Form"])
y = labels

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train classifier
# -----------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
with open("premier_league_model.pkl", "wb") as f:
    pickle.dump((clf, le_team), f)

print("Model retrained and saved to premier_league_model.pkl")
