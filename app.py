import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import pickle
import plotly.graph_objects as go

# —————————————————————————————
# 1. Title and sidebar controls
# —————————————————————————————
st.title("Shot Map Interactive xG")

# ——— Charger les données pour obtenir la liste des équipes et joueurs ———
@st.cache_data
def load_teams_and_players():
    df = pd.read_csv("data/psg_marseille_shots_enriched.csv")
    teams = list(set(df["team"]))
    return df, teams

df_full, all_teams = load_teams_and_players()

# ——— Initialisation des filtres dans session_state ———
def reset_filters(selected_team):
    st.session_state["min_xg"] = 0.0
    st.session_state["goals"] = False
    st.session_state["non_goals"] = False
    st.session_state["team"] = selected_team
    st.session_state["player_list"] = list(set(df_full[df_full["team"] == selected_team]["player"]))

if "team" not in st.session_state:
    default_team = all_teams[0]
    reset_filters(default_team)

st.sidebar.header("Filters & Settings")

# Sélecteur d'équipe
selected_team = st.sidebar.selectbox("Select team", options=all_teams, index=all_teams.index(st.session_state["team"]))
if selected_team != st.session_state["team"]:
    reset_filters(selected_team)

if st.sidebar.button("Reset filters"):
    reset_filters(selected_team)

min_xg = st.sidebar.slider(
    "Minimum predicted xG to display",
    float(0.0), float(1.0), st.session_state["min_xg"], step=0.01, key="min_xg"
)

goals_state = st.sidebar.checkbox("Show only goals", value=st.session_state["goals"], key="goals", disabled=False)
non_goals_state = st.sidebar.checkbox("Show only non-goals", value=st.session_state["non_goals"], key="non_goals", disabled=goals_state)
if non_goals_state:
    goals_state = False
if goals_state:
    non_goals_state = False

# Liste des joueurs de l'équipe sélectionnée
team_players = list(set(df_full[df_full["team"] == selected_team]["player"]))
player_list = st.sidebar.multiselect(
    "Select player(s)",
    options=team_players,
    default=team_players,
    key="player_list"
)

# —————————————————————————————
# 2. Load model & data (cached)
# —————————————————————————————
@st.cache_resource
def load_model():
    with open("models/xgboost.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data/psg_marseille_shots_enriched.csv")
    # compute predicted xG
    model = load_model()
    features = [
        "x", "y",
        "shot_body_part", "shot_technique",
        "under_pressure", "shot_first_time",
        "distance_to_goal", "angle_to_goal",
        "is_central_zone", "is_left_side", "is_right_side"
    ]
    df["y_proba"] = model.predict_proba(df[features])[:, 1]
    return df

df = load_data()

# —————————————————————————————
# 3. Apply filters
# —————————————————————————————
mask = (df["team"] == selected_team) & (df["y_proba"] >= min_xg) & (df["player"].isin(player_list))
if goals_state:
    mask &= (df["goal"] == 1)
if non_goals_state:
    mask &= (df["goal"] == 0)

filtered = df[mask]

st.sidebar.write(f"Displaying {len(filtered)} shots")

# —————————————————————————————
# 4. Draw pitch and shots (Plotly version corrigée)
# —————————————————————————————

# Définir les dimensions du terrain complet StatsBomb (x: 0-120, y: 0-80)
pitch_xmin, pitch_xmax = 0, 120
pitch_ymin, pitch_ymax = 0, 80

fig = go.Figure()

# Lignes du terrain complet
fig.add_shape(type="rect", x0=pitch_xmin, y0=pitch_ymin, x1=pitch_xmax, y1=pitch_ymax, line=dict(color="black", width=2))
fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="black", width=1))  # Surface de réparation
fig.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="black", width=1))  # 6m
#fig.add_shape(type="circle", x0=108-8, y0=40-8, x1=108+8, y1=40+8, line=dict(color="black", width=1))  # Point de penalty
#fig.add_shape(type="circle", x0=120-18, y0=40-18, x1=120+18, y1=40+18, line=dict(color="black", width=1))  # Arc de cercle

# Marqueurs (taille réduite)
markers = pd.Series(filtered['goal']).map({1: 'circle', 0: 'x'})
fig.add_trace(go.Scatter(
    x=filtered['x'],
    y=filtered['y'],
    mode='markers',
    marker=dict(
        size=filtered['y_proba']*40 + 8,  # Taille encore réduite
        color=filtered['y_proba'],
        colorscale='Plasma',
        colorbar=dict(title='Predicted xG'),
        line=dict(color='black', width=1),
        symbol=markers
    ),
    text=[
        f"Joueur: {row['player']}<br>Minute: {row['minute']}<br>xG: {row['y_proba']:.2f}<br>But: {'Oui' if row['goal']==1 else 'Non'}"
        for _, row in filtered.iterrows()
    ],
    hoverinfo='text',
    name='Shots'
))

fig.update_layout(
    title=f"Shot Map with Predicted xG – {selected_team}",
    xaxis=dict(range=[pitch_xmin, pitch_xmax], showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(range=[pitch_ymin, pitch_ymax], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
    plot_bgcolor='#228B22',
    width=900,
    height=600,
    margin=dict(l=10, r=10, t=40, b=10)
)

# —————————————————————————————
# 5. Render in Streamlit
# —————————————————————————————
st.plotly_chart(fig)
