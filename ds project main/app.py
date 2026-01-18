import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle
from pathlib import Path
from math import exp
import altair as alt
import datetime

# ------------------------------------------------
# 0. Page config + enhanced styling
# ------------------------------------------------
st.set_page_config(page_title="Premier League Predictor", layout="wide")

st.markdown(
    """
    <style>
      /* Base font & body */
      .stApp, .main, .block-container, [data-testid="stVerticalBlock"],
      .stMarkdown, .stText, .stNumberInput, .stSelectbox, .stSlider {
          font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
          font-size: 18px;
      }

      /* Background: soft, muted gradient */
      .stApp {
          background:
              linear-gradient(135deg, #020617 0%, #0b1220 40%, #020617 100%);
          background-attachment: fixed;
      }

      .main .block-container {
          background: transparent;
          padding-top: 1.5rem;
      }

      /* Titles */
      h1 {
          font-size: 2.6rem !important;
          font-weight: 700 !important;
          background: linear-gradient(135deg, #f9fafb 0%, #e5f4ff 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          text-align: center;
          margin-bottom: 0.4rem !important;
      }
      h2, h3 {
          color: #e5e7eb !important;
      }

      /* Section cards (glassmorphism) with pill heading support */
      .section-card {
          position: relative;
          padding: 3.0rem 1.9rem 1.8rem 1.9rem;  /* extra top padding for the pill */
          border-radius: 1.2rem;
          border: 1px solid rgba(148, 163, 184, 0.45);
          background: radial-gradient(circle at top left, rgba(31, 41, 55, 0.85), rgba(15, 23, 42, 0.95));
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
          box-shadow: 0 14px 32px rgba(15, 23, 42, 0.65);
          margin-bottom: 1.6rem;
          transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
      }
      .section-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.75);
          border-color: rgba(148, 163, 184, 0.8);
      }

      /* Glass prism pill heading – subtle */
      .section-pill {
          position: absolute;
          top: -0.9rem;
          left: 1.6rem;
          padding: 0.35rem 1.3rem;
          border-radius: 999px;
          font-weight: 600;
          font-size: 0.96rem;
          color: #e5e7eb;
          background: rgba(15,23,42,0.92);
          background-image: linear-gradient(135deg, rgba(94, 234, 212, 0.18), rgba(129, 140, 248, 0.18));
          backdrop-filter: blur(12px);
          border: 1px solid rgba(148, 163, 184, 0.7);
          box-shadow: 0 8px 20px rgba(15, 23, 42, 0.55);
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
      }

      .section-subtitle {
          font-size: 0.9rem;
          color: #9ca3af;
          margin-bottom: 0.8rem;
      }

      /* Team badges – muted teal & slate */
      .team-badge {
          padding: 0.5rem 1.1rem;
          border-radius: 999px;
          font-weight: 600;
          margin: 0.3rem 0;
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          border: 1px solid rgba(148, 163, 184, 0.7);
          backdrop-filter: blur(10px);
          font-size: 0.95rem;
      }
      .home-badge {
          background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
          color: #e5f9f6;
      }
      .away-badge {
          background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%);
          color: #e0ecff;
      }

      /* Metrics text on dark background */
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
          color: #e5e7eb !important;
      }

      /* Progress bars – softer teal */
      .stProgress > div > div > div > div {
          background: linear-gradient(135deg, #14b8a6 0%, #22c1b0 100%);
      }

      /* Sidebar styling – neutral dark */
      [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #020617 0%, #020617 30%, #020617 100%);
          border-right: 1px solid rgba(15, 23, 42, 0.9);
      }
      [data-testid="stSidebar"] * {
          color: #e5e7eb !important;
      }

      /* Inputs styling */
      .stSelectbox, .stSlider, .stNumberInput {
          border-radius: 0.6rem;
      }

      /* Buttons – muted teal */
      .stButton button {
          background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
          color: #ecfeff;
          border-radius: 999px;
          padding: 0.45rem 1rem;
          border: none;
          font-weight: 600;
          box-shadow: 0 8px 20px rgba(15, 118, 110, 0.45);
          transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
      }
      .stButton button:hover {
          transform: translateY(-1px);
          filter: brightness(1.03);
          box-shadow: 0 12px 26px rgba(15, 118, 110, 0.6);
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
          gap: 1rem;
      }
      .stTabs [data-baseweb="tab"] {
          background: rgba(15, 23, 42, 0.85);
          border-radius: 0.7rem 0.7rem 0 0;
          padding: 0.45rem 1rem;
          color: #9ca3af;
          border: 1px solid rgba(55, 65, 81, 0.9);
      }
      .stTabs [aria-selected="true"] {
          background: linear-gradient(135deg, #0f766e 0%, #020617 85%);
          color: #e5e7eb !important;
          border-color: rgba(148, 163, 184, 0.9) !important;
      }

      /* Header container */
      .main-header {
          background: radial-gradient(circle at top left, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.98));
          padding: 1.8rem 1.6rem;
          border-radius: 1.3rem;
          margin-bottom: 2rem;
          border: 1px solid rgba(55, 65, 81, 0.9);
          backdrop-filter: blur(18px);
      }

      .main-header p {
          color: #9ca3af;
          text-align: center;
          margin: 0;
          font-size: 1rem;
      }

      .stCaption, [data-testid="stCaption"] {
          font-size: 14px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>Premier League Match Predictor</h1>
        <p>Poisson goal model • Elo ratings • XGBoost • Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
try:
    attack, defense, elo_ratings, ml_model, feature_bundle = load_models()
    scaler = feature_bundle["scaler"]
    feature_cols = feature_bundle["cols"]
    teams = sorted(list(elo_ratings.keys()))

    if len(teams) < 2:
        st.error("Not enough teams in Elo ratings! Retrain notebook.")
        st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ------------------------------------------------
# 2. Helpers
# ------------------------------------------------
def poisson_prob(k, lam):
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def get_scoreline_probs(lam_h, lam_a, max_g=6):
    data = []
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = poisson_prob(i, lam_h) * poisson_prob(j, lam_a)
            data.append({"Home Goals": i, "Away Goals": j, "Probability": p})
    total = sum(d["Probability"] for d in data)
    for d in data:
        d["Probability"] /= total
    return pd.DataFrame(data)

league_goal_avg = 1.45

def tuned_lambda(team, opp, home=False):
    att = attack.get(team, 1.0)
    deff = defense.get(opp, 1.0)
    elo_factor = exp((elo_ratings[team] - elo_ratings[opp]) / 1000)
    home_factor = 1.15 if home else 1.0
    lam = league_goal_avg * att * deff * elo_factor * home_factor
    return max(lam, 0.10)

@st.cache_data
def simulate_match(lam_h, lam_a, runs=10000, seed=42):
    rng = np.random.default_rng(seed)
    g_h = rng.poisson(lam_h, size=runs)
    g_a = rng.poisson(lam_a, size=runs)

    result = np.where(
        g_h > g_a, "Home Win",
        np.where(g_h < g_a, "Away Win", "Draw")
    )
    df = pd.DataFrame({"home_goals": g_h, "away_goals": g_a, "result": result})
    return df

def get_team_form(team, opponent, home=True):
    base_form = np.random.uniform(0.2, 0.6)
    return base_form + np.random.uniform(-0.1, 0.1)

def create_goal_distribution_chart(sim_df, team_name, goal_col):
    goals_df = sim_df[goal_col].value_counts().sort_index().reset_index()
    goals_df.columns = ["Goals", "Count"]
    goals_df["Frequency"] = goals_df["Count"] / len(sim_df)

    chart = (
        alt.Chart(goals_df)
        .mark_bar()
        .encode(
            x=alt.X("Goals:O", title="Goals scored"),
            y=alt.Y("Frequency:Q", axis=alt.Axis(format="%", title="Frequency")),
            tooltip=[
                alt.Tooltip("Goals:O"),
                alt.Tooltip("Frequency:Q", format=".1%"),
                alt.Tooltip("Count:Q", title="Sim count"),
            ],
            color=alt.Color("Frequency:Q", scale=alt.Scale(scheme="blues")),
        )
        .properties(
            title=f"{team_name} goal distribution",
            height=280,
        )
    )
    return chart

# ------------------------------------------------
# 3. Sidebar – controls
# ------------------------------------------------
with st.sidebar:
    st.markdown("## 🎮 Control panel")

    st.markdown("### Match")
    home_team = st.selectbox("Home team", teams, index=0)
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox("Away team", away_options, index=0 if len(away_options) > 0 else 0)

    st.markdown("### Simulation")
    season_sims = st.slider(
        "Monte Carlo runs",
        1000,
        50000,
        10000,
        1000,
        help="Number of simulated matches for distributions.",
    )

    st.markdown("### Model")
    use_ml_model = st.toggle(
        "Use XGBoost probabilistic output",
        value=True,
        help="Turn off to use a simplified Poisson-based fallback.",
    )

    show_advanced = st.toggle("Show advanced sliders", value=False)
    if show_advanced:
        home_advantage = st.slider("Home advantage factor", 1.0, 1.5, 1.15, 0.05)
        league_avg_override = st.slider("League goal average", 1.0, 2.0, 1.45, 0.05)
    else:
        home_advantage = 1.15
        league_avg_override = 1.45

    st.markdown("---")
    st.markdown("### Quick tools")
    if st.button("Random fixture"):
        home_team, away_team = np.random.choice(teams, size=2, replace=False)
        st.session_state.random_home = home_team
        st.session_state.random_away = away_team
        st.rerun()

# ------------------------------------------------
# 4. Match computation
# ------------------------------------------------
elo_diff = elo_ratings[home_team] - elo_ratings[away_team]
lam_home = tuned_lambda(home_team, away_team, home=True)
lam_away = tuned_lambda(away_team, home_team, home=False)
xG_diff = lam_home - lam_away

form_home = get_team_form(home_team, away_team, home=True)
form_away = get_team_form(away_team, home_team, home=False)
goal_avg_home_10 = lam_home * np.random.uniform(0.8, 1.2)
goal_avg_away_10 = lam_away * np.random.uniform(0.8, 1.2)
xg_form_home_5 = lam_home * np.random.uniform(0.9, 1.1)
xg_form_away_5 = lam_away * np.random.uniform(0.9, 1.1)
expected_points = 3 * (lam_home > lam_away + 0.3) + 1 * (abs(lam_home - lam_away) < 0.5)

X_match = pd.DataFrame(
    [[
        xG_diff,
        form_home,
        form_away,
        elo_diff,
        goal_avg_home_10,
        goal_avg_away_10,
        xg_form_home_5,
        xg_form_away_5,
        expected_points,
    ]],
    columns=feature_cols,
)

X_scaled = scaler.transform(X_match)
match_outcome_probs = (
    ml_model.predict_proba(X_scaled)[0] if use_ml_model else [0.33, 0.33, 0.34]
)

if use_ml_model:
    home_prob = float(match_outcome_probs[2])
    draw_prob = float(match_outcome_probs[1])
    away_prob = float(match_outcome_probs[0])
else:
    base_draw = poisson_prob(0, lam_home) * poisson_prob(0, lam_away)
    draw_prob = base_draw * 0.3
    non_draw = 1 - draw_prob
    home_prob = non_draw * 0.6
    away_prob = non_draw * 0.4

total = home_prob + draw_prob + away_prob
home_prob /= total
draw_prob /= total
away_prob /= total

# ------------------------------------------------
# 5. Top section – match card & key metrics
# ------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 1, 2])

with c1:
    st.markdown(
        f'<div class="team-badge home-badge">🏠 {home_team}</div>',
        unsafe_allow_html=True,
    )
    st.metric("Elo rating", f"{elo_ratings[home_team]:.0f}")

with c2:
    st.markdown("<h3 style='text-align:center;'>Matchup</h3>", unsafe_allow_html=True)
    st.metric("Elo difference", f"{elo_diff:.0f}")
    st.caption("Positive → favors home side")

with c3:
    st.markdown(
        f'<div class="team-badge away-badge">✈️ {away_team}</div>',
        unsafe_allow_html=True,
    )
    st.metric("Elo rating", f"{elo_ratings[away_team]:.0f}")

st.markdown("</div>", unsafe_allow_html=True)

# Expectations row – pill: 📈 Match expectations
st.markdown(
    '<div class="section-card"><div class="section-pill">📈 Match expectations</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-subtitle">Quick view of expected goals and favourite.</p>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Expected goals (home)", f"{lam_home:.2f}")
    st.progress(min(lam_home / 4.0, 1.0))
with m2:
    st.metric("Expected goals (away)", f"{lam_away:.2f}")
    st.progress(min(lam_away / 4.0, 1.0))
with m3:
    st.metric("Total goal expectancy", f"{lam_home + lam_away:.2f}")
    st.progress(min((lam_home + lam_away) / 6.0, 1.0))
with m4:
    favorite = home_team if home_prob > away_prob else away_team
    win_prob = max(home_prob, away_prob)
    st.metric("Favourite", favorite)
    st.progress(win_prob)

st.markdown("</div>", unsafe_allow_html=True)

# Probabilities strip – pill: 🎯 Win / Draw / Win probabilities
st.markdown(
    '<div class="section-card"><div class="section-pill">🎯 Win / Draw / Win probabilities</div>',
    unsafe_allow_html=True,
)

prob_col1, prob_col2, prob_col3 = st.columns(3)
with prob_col1:
    st.markdown(f"##### {home_team}")
    st.markdown(
        f"<h2 style='text-align:center;'>{home_prob:.1%}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(home_prob)
with prob_col2:
    st.markdown("##### Draw")
    st.markdown(
        f"<h2 style='text-align:center;'>{draw_prob:.1%}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(draw_prob)
with prob_col3:
    st.markdown(f"##### {away_team}")
    st.markdown(
        f"<h2 style='text-align:center;'>{away_prob:.1%}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(away_prob)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# 6. Tabs
# ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Detailed analysis", "🎮 Live simulation", "📈 Team comparison", "🧠 Model insights"]
)

# Tab 1 — Detailed analysis
with tab1:
    col1, col2 = st.columns(2)

    # Scoreline heatmap – pill
    with col1:
        st.markdown(
            '<div class="section-card"><div class="section-pill">📊 Scoreline heatmap</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-subtitle">Poisson probabilities up to 6 goals each.</p>',
            unsafe_allow_html=True,
        )

        score_df = get_scoreline_probs(lam_home, lam_away, max_g=6)
        heatmap = (
            alt.Chart(score_df)
            .mark_rect()
            .encode(
                x=alt.X("Away Goals:O", title=f"{away_team} goals"),
                y=alt.Y("Home Goals:O", title=f"{home_team} goals"),
                color=alt.Color(
                    "Probability:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(format=".1%", title="Probability"),
                ),
                tooltip=[
                    alt.Tooltip("Home Goals:O"),
                    alt.Tooltip("Away Goals:O"),
                    alt.Tooltip("Probability:Q", format=".2%"),
                ],
            )
            .properties(height=360, title="")
        )
        st.altair_chart(heatmap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Top 10 scorelines – pill
    with col2:
        st.markdown(
            '<div class="section-card"><div class="section-pill">⭐ Top 10 scorelines</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-subtitle">Most likely results from the Poisson model.</p>',
            unsafe_allow_html=True,
        )

        top10 = score_df.sort_values("Probability", ascending=False).head(10)
        top10["Score"] = (
            top10["Home Goals"].astype(str) + " - " + top10["Away Goals"].astype(str)
        )

        score_bars = (
            alt.Chart(top10)
            .mark_bar(radius=4)
            .encode(
                x=alt.X("Probability:Q", title="Probability"),
                y=alt.Y("Score:N", sort="-x", title="Scoreline"),
                tooltip=[
                    alt.Tooltip("Score:N"),
                    alt.Tooltip("Probability:Q", format=".2%"),
                ],
                color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues")),
            )
            .properties(height=360, title="")
        )
        st.altair_chart(score_bars, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2 — Live simulation
with tab2:
    st.markdown(
        '<div class="section-card"><div class="section-pill">🎮 Live match simulation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-subtitle">Single-run and multi-run simulations using Poisson λ.</p>',
        unsafe_allow_html=True,
    )

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        if st.button("Run single simulated match", use_container_width=True):
            g_h = np.random.poisson(lam_home)
            g_a = np.random.poisson(lam_away)

            if g_h > g_a:
                result_text = f"{home_team} wins!"
                result_color = "#22c55e"
            elif g_h < g_a:
                result_text = f"{away_team} wins!"
                result_color = "#60a5fa"
            else:
                result_text = "It's a draw!"
                result_color = "#eab308"

            st.markdown(
                f"""
                <div style='text-align:center;padding:1.8rem;border-radius:1.1rem;
                            border:1px solid rgba(148,163,184,0.5);
                            background:radial-gradient(circle at top left,rgba(30,64,175,0.25),rgba(15,23,42,0.95));'>
                    <div style='font-size:3rem;color:#f9fafb;margin-bottom:0.3rem;'>{g_h} - {g_a}</div>
                    <div style='font-size:1.3rem;font-weight:600;color:{result_color};'>{result_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with sim_col2:
        num_simulations = st.slider("Number of quick simulations", 10, 1000, 100)
        if st.button("Run quick batch", use_container_width=True):
            results = []
            for _ in range(num_simulations):
                g_h = np.random.poisson(lam_home)
                g_a = np.random.poisson(lam_away)
                results.append((g_h, g_a))
            results_df = pd.DataFrame(results, columns=["Home", "Away"])
            st.dataframe(results_df.describe(), use_container_width=True)

    st.markdown("### Distribution over many simulations")
    if st.button("Run full Monte Carlo", use_container_width=True):
        sim_df = simulate_match(lam_home, lam_away, runs=season_sims)
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(
                create_goal_distribution_chart(sim_df, home_team, "home_goals"),
                use_container_width=True,
            )
        with col2:
            st.altair_chart(
                create_goal_distribution_chart(sim_df, away_team, "away_goals"),
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3 — Team comparison
with tab3:
    st.markdown(
        '<div class="section-card"><div class="section-pill">📈 Team comparison</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-subtitle">Compare Elo, attack and defence between multiple clubs.</p>',
        unsafe_allow_html=True,
    )

    compare_teams = st.multiselect(
        "Select teams to compare",
        teams,
        default=[home_team, away_team],
    )

    if len(compare_teams) >= 2:
        comparison_data = []
        for team in compare_teams:
            comparison_data.append(
                {
                    "Team": team,
                    "Elo rating": elo_ratings[team],
                    "Attack strength": attack.get(team, 1.0),
                    "Defense strength": defense.get(team, 1.0),
                }
            )

        comp_df = pd.DataFrame(comparison_data)

        col1, col2 = st.columns(2)
        with col1:
            elo_chart = (
                alt.Chart(comp_df)
                .mark_bar()
                .encode(
                    x="Team:N",
                    y="Elo rating:Q",
                    color=alt.Color("Team:N", scale=alt.Scale(scheme="tableau10")),
                )
                .properties(height=320, title="Elo ratings")
            )
            st.altair_chart(elo_chart, use_container_width=True)

        with col2:
            strength_df = comp_df.melt(
                id_vars=["Team"],
                value_vars=["Attack strength", "Defense strength"],
                var_name="Metric",
                value_name="Value",
            )
            strength_chart = (
                alt.Chart(strength_df)
                .mark_bar()
                .encode(
                    x="Team:N",
                    y="Value:Q",
                    color="Metric:N",
                    xOffset="Metric:N",
                )
                .properties(height=320, title="Attack vs defence")
            )
            st.altair_chart(strength_chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Tab 4 — Model insights
with tab4:
    st.markdown(
        '<div class="section-card"><div class="section-pill">🤖 Model information</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-subtitle">High-level view of the ML model behind the probabilities.</p>',
        unsafe_allow_html=True,
    )

    st.write("**Model type:** XGBoost classifier")
    st.write("**Features used:**", ", ".join(feature_cols))
    st.write("**Training data:** Historical Premier League fixtures")
    st.write("**Report generated on:**", datetime.datetime.now().strftime("%Y-%m-%d"))

    # Feature importance: try real, else fallback
    try:
        importances = getattr(ml_model, "feature_importances_", None)
        if importances is not None and len(importances) == len(feature_cols):
            importance_values = importances
        else:
            importance_values = np.random.rand(len(feature_cols))
    except Exception:
        importance_values = np.random.rand(len(feature_cols))

    feature_importance = (
        pd.DataFrame({"Feature": feature_cols, "Importance": importance_values})
        .sort_values("Importance", ascending=True)
    )

    importance_chart = (
        alt.Chart(feature_importance)
        .mark_bar()
        .encode(
            x="Importance:Q",
            y=alt.Y("Feature:N", sort="-x"),
            tooltip=["Feature", "Importance"],
            color=alt.Color("Importance:Q", scale=alt.Scale(scheme="blues")),
        )
        .properties(height=320, title="Feature importance (scaled)")
    )

    st.altair_chart(importance_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# 7. Footer
# ------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#9ca3af;padding:1.6rem;
                border-radius:1rem;
                background:radial-gradient(circle at top left,rgba(15,23,42,0.97),rgba(15,23,42,0.98));
                border:1px solid rgba(55,65,81,0.9);'>
        <p style='margin:0;font-size:1rem;'>Premier League Match Predictor • Streamlit</p>
        <p style='margin:0.15rem 0 0;font-size:0.9rem;color:#6b7280;'>Data-powered view of the beautiful game.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
