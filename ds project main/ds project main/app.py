import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle
from pathlib import Path
from collections import defaultdict
from math import exp

# Mentioning the league and framework stack used in your pipeline: the
# sports league is Premier League and the UI framework is Streamlit.

# ------------------------------------------------
# 1. Load models & feature bundle safely
# ------------------------------------------------
@st.cache_resource
def load_models():
    models_path = Path("models")

    with open(models_path / "attack.pkl", "rb") as f:
        attack = pickle.load(f)

    with open(models_path / "defense.pkl", "rb") as f:
        defense = pickle.load(f)

    with open(models_path / "elo_ratings.pkl", "rb") as f:
        elo = pickle.load(f)

    with open(models_path / "xgboost_model.pkl", "rb") as f:
        xgb = pickle.load(f)

    with open(models_path / "feature_scaler.pkl", "rb") as f:
        bundle = pickle.load(f)

    return attack, defense, elo, xgb, bundle

# Load artifacts
attack, defense, elo_ratings, ml_model, feature_bundle = load_models()

# Unpack scaler and feature names
scaler = feature_bundle["scaler"]
feature_cols = feature_bundle["cols"]  # 9 feature names the model expects

# ------------------------------------------------
# 2. Poisson goal helpers for UI preview
# (using tuned λ directly from your saved attack/def + Elo logic)
# ------------------------------------------------
def poisson_prob(k, lam):
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def get_scoreline_probs(lam_h, lam_a, max_g=6):
    probs = {}
    for i in range(max_g+1):
        for j in range(max_g+1):
            probs[f"{i}-{j}"] = poisson_prob(i, lam_h) * poisson_prob(j, lam_a)
    total = sum(probs.values())
    return {k: v/total for k, v in probs.items()}

# tuned λ logic mirrored from training
league_goal_avg = 1.45  # can be replaced by stored league avg if you save it

def tuned_lambda(team, opp, home=False):
    att = attack.get(team, 1.0)
    deff = defense.get(opp, 1.0)
    elo_factor = exp((elo_ratings[team] - elo_ratings[opp]) / 1000)
    home_factor = 1.15 if home else 1.0
    lam = league_goal_avg * att * deff * elo_factor * home_factor
    return max(lam, 0.10)

# ------------------------------------------------
# 3. Streamlit UI
# ------------------------------------------------
st.set_page_config(page_title="Premier League Predictor", layout="wide")
st.title("Premier League Match Prediction & Season Simulation")

teams = list(elo_ratings.keys())
if len(teams) < 2:
    st.error("Not enough teams in Elo ratings! Retrain notebook.")
    st.stop()

# Sidebar selections
home_team = st.sidebar.selectbox("Home Team", teams)
remaining = teams.copy()
remaining.remove(home_team)
away_team = st.sidebar.selectbox("Away Team", remaining)
season_sims = st.sidebar.slider("Season Simulation Runs", 1000, 20000, 10000, 1000)

# ------------------------------------------------
# 4. Compute all 9 features for the selected match
# ------------------------------------------------
elo_diff = elo_ratings[home_team] - elo_ratings[away_team]
lam_home = tuned_lambda(home_team, away_team, home=True)
lam_away = tuned_lambda(away_team, home_team, home=False)

xG_diff = lam_home - lam_away

form_home = 0.33
form_away = 0.33
# If you want real form, you can first load results.csv into session_state and reuse rolling means.

goal_avg_home_10 = lam_home * 0.85
goal_avg_away_10 = lam_away * 0.80

xg_form_home_5 = lam_home * 0.95
xg_form_away_5 = lam_away * 0.90

expected_points = 3*(lam_home > lam_away + 0.3) + 1*(abs(lam_home - lam_away) < 0.5)

# Build input row with EXACT training feature names in expected order
X_match = pd.DataFrame([[
    xG_diff,
    form_home,
    form_away,
    elo_diff,
    goal_avg_home_10,
    goal_avg_away_10,
    xg_form_home_5,
    xg_form_away_5,
    expected_points
]], columns=feature_cols)

# Scale input
X_scaled = scaler.transform(X_match)

# Predict outcome probabilities
match_outcome_probs = ml_model.predict_proba(X_scaled)[0]

# ------------------------------------------------
# 5. Display each model's output
# ------------------------------------------------
st.subheader("🔢 Tuned Poisson Goal Expectancy")
st.write(f"Expected Goals (λ): **{home_team}: {lam_home:.2f} | {away_team}: {lam_away:.2f}**")

# Scoreline table
scoreline_probs = get_scoreline_probs(lam_home, lam_away)
sl_df = pd.DataFrame(sorted(scoreline_probs.items(), key=lambda x:-x[1])[:10], columns=["Score", "Probability"])
st.dataframe(sl_df)

st.subheader("📈 Elo Ratings")
st.write(f"{home_team} Elo: {elo_ratings[home_team]:.0f}")
st.write(f"{away_team} Elo: {elo_ratings[away_team]:.0f}")
st.write(f"Elo Difference: {elo_diff:.2f}")

st.subheader("🤖 ML Match Outcome Probabilities (XGBoost)")
st.write("Outcome probabilities:")
st.write(f"🏠 {home_team} Win: {match_outcome_probs[2]:.2%}")
st.write(f"🤝 Draw: {match_outcome_probs[1]:.2%}")
st.write(f"🚗 {away_team} Win: {match_outcome_probs[0]:.2%}")

# ------------------------------------------------
# 6. Sample simulation for this one match
# ------------------------------------------------
st.subheader("🎯 Sample Simulated Match")
g_h = np.random.poisson(lam_home)
g_a = np.random.poisson(lam_away)
if g_h > g_a:
    st.write(f"{g_h}-{g_a} → {home_team} Win")
elif g_h < g_a:
    st.write(f"{g_h}-{g_a} → {away_team} Win")
else:
    st.write(f"{g_h}-{g_a} → Draw")

# ------------------------------------------------
# 7. Monte-Carlo simulation showing league table impact
# (does not simulate full fixtures, only impact preview)
# ------------------------------------------------
st.subheader("🏁 Match Impact Preview Simulation")

for i in range(5):
    sh = np.random.poisson(lam_home)
    sa = np.random.poisson(lam_away)
    if sh > sa:
        res = f"{sh}-{sa} → {home_team} Win"
    elif sh < sa:
        res = f"{sh}-{sa} → {away_team} Win"
    else:
        res = f"{sh}-{sa} → Draw"
    st.write(res)

# ------------------------------------------------
# 8. Dump latest prediction to a file for dashboard logging
# ------------------------------------------------
st.subheader("📦 Dump Prediction Output")

log = {
    "match": f"{home_team} vs {away_team}",
    "lambda_home": round(lam_home,3),
    "lambda_away": round(lam_away,3),
    "features": X_match.values.tolist()[0],
    "probs": match_outcome_probs.tolist()
}

if st.button("Save to models/latest_match.pkl"):
    pickle.dump(log, open("models/latest_match.pkl","wb"))
    st.success("✅ Saved latest match output!")

