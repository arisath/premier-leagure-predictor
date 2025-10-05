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
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
import pickle
from rapidfuzz import process
import plotly.graph_objects as go
import base64

# -----------------------------
# Load bookmaker icons as base64
# -----------------------------
def load_icon_base64(path):
    """Load local PNG and return a base64 string suitable for Plotly."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"

BOOKMAKER_ICONS = {
    "Smarkets": load_icon_base64("img/smarkets.png"),
    "Betfair": load_icon_base64("img/betfair.png"),
    "Paddy Power": load_icon_base64("img/paddy_power.png")
}

# -----------------------------
# Constants
# -----------------------------
FIXED_ORDER = ["Home Win", "Draw", "Away Win"]

# -----------------------------
# Helper functions
# -----------------------------
def format_model_odds(probs):
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
    Returns fixed order Home/Draw/Away for a bookmaker.
    """
    market = next((mk for mk in bookmaker.get("markets", []) if mk.get("key") == "h2h"), None)
    if not market:
        return pd.DataFrame({
            "Outcome": FIXED_ORDER,
            "Decimal Odds": ["N/A"]*3
        })

    outcomes_list = market.get("outcomes", [])
    decimal_odds = []
    for idx in range(3):
        try:
            price = outcomes_list[idx]["price"]
            price = f"{price:.2f}".rstrip('0').rstrip('.') if isinstance(price, float) else price
            decimal_odds.append(price)
        except IndexError:
            decimal_odds.append("N/A")

    return pd.DataFrame({
        "Outcome": FIXED_ORDER,
        "Decimal Odds": decimal_odds
    })

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Premier League Predictor")

FIXED_ORDER = ["Home Win", "Draw", "Away Win"]

# Bookmaker icons folder
BOOKMAKER_ICONS = {
    "Smarkets": "img/smarkets.png",
    "Betfair": "img/betfair.png",
    "Paddy Power": "img/paddy_power.png"
}

def format_model_odds(probs):
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
    market = next((mk for mk in bookmaker.get("markets", []) if mk.get("key") == "h2h"), None)
    if not market:
        return pd.DataFrame({
            "Outcome": FIXED_ORDER,
            "Decimal Odds": ["N/A"]*3
        })

    outcomes_list = market.get("outcomes", [])
    decimal_odds = []
    for idx in range(3):
        try:
            price = outcomes_list[idx]["price"]
            price = f"{price:.2f}".rstrip('0').rstrip('.') if isinstance(price, float) else price
            decimal_odds.append(price)
        except IndexError:
            decimal_odds.append("N/A")

    return pd.DataFrame({
        "Outcome": FIXED_ORDER,
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

            probs, home_stats, away_stats = predict_match(home, away, date)

            # Header
            st.markdown(f"""
            <h3>
                <img src="{home_icon}" height="40" style="vertical-align:middle; margin-right:5px;"> {home} vs
                <img src="{away_icon}" height="40" style="vertical-align:middle; margin-left:5px;"> {away}
            </h3>
            """, unsafe_allow_html=True)
            st.markdown(f"📅 {date.strftime('%d/%m/%Y %H:%M')}")

            # Last 5 games form
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"{home} (last 5)", value=f"{home_stats[2]} pts",
                          delta=f"{home_stats[0]:.1f} GS / {home_stats[1]:.1f} GC")
                fig_form = go.Figure()
                for idx_ftr, ftr in enumerate(home_stats[3]):
                    color = "green" if ftr=="H" else "yellow" if ftr=="D" else "red" if ftr=="A" else "lightgray"
                    fig_form.add_trace(go.Bar(
                        x=[1/5],
                        y=[1],
                        orientation="h",
                        marker_color=color,
                        base=[idx_ftr/5],
                        showlegend=False,
                        text=ftr,
                        textposition="inside"
                    ))
                fig_form.update_layout(height=50, yaxis=dict(showticklabels=False), xaxis=dict(showticklabels=False, range=[0,1]), margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_form, use_container_width=True)
            with col2:
                st.metric(label=f"{away} (last 5)", value=f"{away_stats[2]} pts",
                          delta=f"{away_stats[0]:.1f} GS / {away_stats[1]:.1f} GC")
                fig_form2 = go.Figure()
                for idx_ftr, ftr in enumerate(away_stats[3]):
                    color = "green" if ftr=="H" else "yellow" if ftr=="D" else "red" if ftr=="A" else "lightgray"
                    fig_form2.add_trace(go.Bar(
                        x=[1/5],
                        y=[1],
                        orientation="h",
                        marker_color=color,
                        base=[idx_ftr/5],
                        showlegend=False,
                        text=ftr,
                        textposition="inside"
                    ))
                fig_form2.update_layout(height=50, yaxis=dict(showticklabels=False), xaxis=dict(showticklabels=False, range=[0,1]), margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_form2, use_container_width=True)

            # Model odds
            odds_df = format_model_odds(probs)

            # Bookmaker odds
            event_id = find_event_id(home, away, events)
            bookmaker_dfs = []
            bk_names = []
            colors = ["green", "orange", "red"]
            if event_id:
                event_odds = fetch_odds_for_event(event_id)
                if event_odds:
                    for bk in event_odds.get("bookmakers", [])[:3]:
                        bk_df = format_bookmaker_odds_fixed(bk)
                        bookmaker_dfs.append(bk_df)
                        bk_names.append(bk.get("title", f"Bookmaker {len(bk_names)+1}"))

            # Plot grouped bar chart
            fig = go.Figure()
            # Model
            fig.add_trace(go.Bar(
                x=FIXED_ORDER,
                y=[float(row["Model Odds"]) for row in odds_df.to_dict('records')],
                name="Model",
                marker_color="blue"
            ))
            # Bookmakers
            for idx, bk_df in enumerate(bookmaker_dfs):
                fig.add_trace(go.Bar(
                    x=FIXED_ORDER,
                    y=[float(o) if o != "N/A" else 0 for o in bk_df["Decimal Odds"]],
                    name=bk_names[idx],
                    marker_color=colors[idx % len(colors)]
                ))

            fig.update_layout(
                barmode='group',
                title="Model vs Bookmaker Odds",
                xaxis_title="Outcome",
                yaxis_title="Decimal Odds",
                margin=dict(l=60, r=60, t=50, b=50),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show bookmaker icons and names below plot, aligned
            if bk_names:
                cols_icons = st.columns(len(bk_names))
                cols_names = st.columns(len(bk_names))
                for idx, bk_name in enumerate(bk_names):
                    color = colors[idx % len(colors)]
                    icon_path = BOOKMAKER_ICONS.get(bk_name)
                    # Icon row
                    if icon_path:
                        cols_icons[idx].image(icon_path, width=50)
                    # Name row, centered, colored same as bar
                    cols_names[idx].markdown(f"<div style='text-align:center; color:{color}'>{bk_name}</div>", unsafe_allow_html=True)

            st.divider()
