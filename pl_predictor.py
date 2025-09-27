import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Load dataset
# ----------------------------
data_file = "matches.csv"  # replace with your dataset path
df = pd.read_csv(data_file)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')

# ----------------------------
# 2. Encode teams and results
# ----------------------------
teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
le_team = LabelEncoder()
le_team.fit(teams)

df['HomeTeamEnc'] = le_team.transform(df['HomeTeam']).astype(int)
df['AwayTeamEnc'] = le_team.transform(df['AwayTeam']).astype(int)
df['FTREnc'] = df['FTR'].map({'H':0,'D':1,'A':2})

# ----------------------------
# 3. Last 5 games form
# ----------------------------
df['HomeGFLast5'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['HomeGALast5'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['AwayGFLast5'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df['AwayGALast5'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

# ----------------------------
# 4. Head-to-head last 5 matches
# ----------------------------
def compute_h2h(row, df, n=5):
    home = row['HomeTeam']
    away = row['AwayTeam']
    mask = (((df['HomeTeam']==home) & (df['AwayTeam']==away)) |
            ((df['HomeTeam']==away) & (df['AwayTeam']==home)))
    last_matches = df[mask].loc[df['Date'] < row['Date']].tail(n)
    if last_matches.empty:
        return pd.Series([0.0, 0.0])
    home_goals = last_matches.apply(lambda x: x['FTHG'] if x['HomeTeam']==home else x['FTAG'], axis=1).mean()
    away_goals = last_matches.apply(lambda x: x['FTAG'] if x['HomeTeam']==home else x['FTHG'], axis=1).mean()
    return pd.Series([home_goals, away_goals])

df[['H2HHome','H2HAway']] = df.apply(lambda row: compute_h2h(row, df), axis=1)

# ----------------------------
# 5. Prepare features and target
# ----------------------------
features = ['HomeTeamEnc','AwayTeamEnc','HomeGFLast5','HomeGALast5','AwayGFLast5','AwayGALast5','H2HHome','H2HAway']
df_model = df.dropna(subset=features+['FTREnc'])

X = df_model[features]
y = df_model['FTREnc']

# ----------------------------
# 6. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# ----------------------------
# 7. Train Random Forest
# ----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
clf = CalibratedClassifierCV(rf)
clf.fit(X_train, y_train)

# ----------------------------
# 8. Function to predict future match
# ----------------------------
def predict_match(home_team, away_team, df, model, le_team, n_form=5):
    home_enc = le_team.transform([home_team])[0]
    away_enc = le_team.transform([away_team])[0]

    # Last N games form
    home_past = df[df['HomeTeam']==home_team].sort_values('Date').tail(n_form)
    home_gf_last5 = home_past['FTHG'].mean() if not home_past.empty else 0
    home_ga_last5 = home_past['FTAG'].mean() if not home_past.empty else 0

    away_past = df[df['AwayTeam']==away_team].sort_values('Date').tail(n_form)
    away_gf_last5 = away_past['FTAG'].mean() if not away_past.empty else 0
    away_ga_last5 = away_past['FTHG'].mean() if not away_past.empty else 0

    # Head-to-head last N games
    mask = (((df['HomeTeam']==home_team) & (df['AwayTeam']==away_team)) |
            ((df['HomeTeam']==away_team) & (df['AwayTeam']==home_team)))
    h2h = df[mask].sort_values('Date').tail(n_form)
    h2h_home = h2h.apply(lambda x: x['FTHG'] if x['HomeTeam']==home_team else x['FTAG'], axis=1).mean() if not h2h.empty else 0
    h2h_away = h2h.apply(lambda x: x['FTAG'] if x['HomeTeam']==home_team else x['FTHG'], axis=1).mean() if not h2h.empty else 0

    # Feature vector
    X_new = np.array([[home_enc, away_enc, home_gf_last5, home_ga_last5,
                       away_gf_last5, away_ga_last5, h2h_home, h2h_away]])
    
    prob = model.predict_proba(X_new)[0]
    result_map = {0:'Home Win',1:'Draw',2:'Away Win'}
    return {result_map[i]: round(p,2) for i,p in enumerate(prob)}

# ----------------------------
# 9. Interactive example
# ----------------------------
while True:
    home = input("Enter Home Team (or 'exit' to quit): ")
    if home.lower() == 'exit':
        break
    away = input("Enter Away Team: ")
    probs = predict_match(home, away, df, clf, le_team)
    print(f"Predicted probabilities: {probs}\n")
