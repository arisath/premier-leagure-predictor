import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
import pickle
from rapidfuzz import process
import plotly.graph_objects as go
from config import API_KEY, ODDS_API_KEY  # store API keys separately

# -----------------------------
# Load historical data
# -----------------------------
files = [
    "data/PL_2021_2022.csv",
    "data/PL_2022_2023.csv",
    "data/PL_2023_2024.csv",
    "data/PL_2024_2025.csv"
]
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

    past_away = past_away.copy()
    past_away["FTR"] = past_away["FTR"].apply(lambda x: "H" if x == "A" else "A" if x == "H" else "D")

    recent = pd.concat([
        past_home[["Date", "FTHG", "FTAG", "FTR"]],
        past_away[["Date", "FTHG", "FTAG", "FTR"]]
    ]).sort_values("Date", ascending=False).head(n)

    goals_scored = recent["FTHG"].mean() if not recent.empty else 0
    goals_conceded = recent["FTAG"].mean() if not recent.empty else 0
    form_points = recent["FTR"].apply(lambda x: 3 if x=="H" else 1 if x=="D" else 0).sum() if not recent.empty else 0

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
# The Odds API - Fetch upcoming events
# -----------------------------
def fetch_upcoming_events(sport="soccer_epl", regions="uk"):
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/events"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching events: {e}")
        return []

# -----------------------------
# Find event ID for a match
# -----------------------------
def find_event_id(home, away, events, threshold=75):
    """
    Find the Odds API eventId for a given home/away team using fuzzy matching.
    """
    best_score = 0
    best_id = None
    for e in events:
        home_score = process.extractOne(home, [e["home_team"]])[1]
        away_score = process.extractOne(away, [e["away_team"]])[1]
        avg_score = (home_score + away_score) / 2
        if avg_score > best_score and avg_score >= threshold:
            best_score = avg_score
            best_id = e["id"]
    return best_id

# -----------------------------
# Fetch odds for a specific event
# -----------------------------
def fetch_odds_for_event(event_id, sport="soccer_epl", regions="uk", markets="h2h"):
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching odds for event {event_id}: {e}")
        return []

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Premier League Predictor")

FIXED_ORDER = ["Home Win", "Draw", "Away Win"]

def format_model_odds(probs):
    """Return a DataFrame for model odds in fixed order."""
    data = []
    for outcome in FIXED_ORDER:
        prob = probs.get(outcome, 0)
        odds = 1/prob if prob > 0 else float('inf')
        data.append({
            "Outcome": outcome,
            "Probability": f"{prob*100:.1f}%",
            "Model Odds": f"{odds:.2f}".rstrip('0').rstrip('.') if odds != float('inf') else "∞"
        })
    return pd.DataFrame(data)

def format_bookmaker_odds_fixed(bookmaker):
    """
    Return a DataFrame with fixed order Home/Draw/Away for a bookmaker.
    Ignores the outcome names and just uses the order in the 'h2h' market:
    [Home team, Draw, Away team].
    """
    market = next((mk for mk in bookmaker.get("markets", []) if mk.get("key") == "h2h"), None)
    if not market:
        return pd.DataFrame({
            "Outcome": ["Home Win", "Draw", "Away Win"],
            "Decimal Odds": ["N/A", "N/A", "N/A"]
        })

    # The Odds API 'outcomes' list is usually in order: Home, Draw, Away
    outcomes_list = market.get("outcomes", [])
    decimal_odds = []
    for idx in range(3):
        try:
            price = outcomes_list[idx]["price"]
            # Format without trailing zeros
            price = f"{price:.2f}".rstrip('0').rstrip('.') if isinstance(price, float) else price
            decimal_odds.append(price)
        except IndexError:
            decimal_odds.append("N/A")

    return pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Decimal Odds": decimal_odds
    })

if st.button("🔮 Predict Next 10 Fixture Odds"):
    matches = get_next_matches(10)
    events = fetch_upcoming_events()

    if not matches:
        st.warning("No upcoming matches found.")
    else:
        for i, match in enumerate(matches):
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            home_icon = match["homeTeam"].get("crest")
            away_icon = match["awayTeam"].get("crest")
            date = datetime.fromisoformat(match["utcDate"][:-1])

            # Model prediction
            probs, home_stats, away_stats = predict_match(home, away, date)

            st.markdown(f"""
            <h3>
                <img src="{home_icon}" height="40" style="vertical-align:middle; margin-right:5px;"> {home} vs
                <img src="{away_icon}" height="40" style="vertical-align:middle; margin-left:5px;"> {away}
            </h3>
            """, unsafe_allow_html=True)
            st.markdown(f"📅 {date.strftime('%d/%m/%Y %H:%M')}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"{home} (last 5)", value=f"{home_stats[2]} pts",
                          delta=f"{home_stats[0]:.1f} GS / {home_stats[1]:.1f} GC")
            with col2:
                st.metric(label=f"{away} (last 5)", value=f"{away_stats[2]} pts",
                          delta=f"{away_stats[0]:.1f} GS / {away_stats[1]:.1f} GC")

            # --- Combine model and bookmaker odds for grouped bar chart ---
            odds_sources = ["Model"]  # start with model
            odds_df = format_model_odds(probs)
            values = [ [float(row["Model Odds"]) if row["Model Odds"] != "∞" else 0 for row in odds_df.to_dict("records")] ]

            # Get bookmaker odds using event ID
            event_id = find_event_id(home, away, events)
            if event_id:
                event_odds = fetch_odds_for_event(event_id)
                if event_odds:
                    top_bookmakers = event_odds.get("bookmakers", [])[:3]
                    for bk in top_bookmakers:
                        bk_df = format_bookmaker_odds_fixed(bk)
                        odds_sources.append(bk.get("title", "Bookmaker"))
                        values.append([float(x) if x != "N/A" else 0 for x in bk_df["Decimal Odds"]])
                else:
                    st.info("No bookmaker odds available yet for this event.")
            else:
                st.info("No matching event found for this fixture.")

            # --- Plot grouped bar chart ---
            fig = go.Figure()
            bar_width = 0.15
            n_sources = len(odds_sources)
            offsets = [(-bar_width*n_sources/2)+(bar_width/2)+bar_width*i for i in range(n_sources)]

            for idx, source in enumerate(odds_sources):
                fig.add_trace(go.Bar(
                    x=FIXED_ORDER,
                    y=values[idx],
                    name=source,
                    width=bar_width,
                    offset=offsets[idx]
                ))

            fig.update_layout(
                title="Model Odds vs Bookmaker Odds",
                yaxis=dict(title="Decimal Odds"),
                barmode="group",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

