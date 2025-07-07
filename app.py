import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import pickle

# —————————————————————————————
# 1. Title and sidebar controls
# —————————————————————————————
st.title("PSG vs. Marseille (2015/16) – Shot Map Interactive xG")

# Sidebar filters
st.sidebar.header("Filters & Settings")
min_xg = st.sidebar.slider(
    "Minimum predicted xG to display",
    float(0.0), float(1.0), 0.0, step=0.01
)
show_goals = st.sidebar.checkbox("Show only goals", value=False)
show_non_goals = st.sidebar.checkbox("Show only non-goals", value=False)
player_list = st.sidebar.multiselect(
    "Select player(s)",
    options=[],
    default=[]
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

# populate player_list if empty
if not player_list:
    player_list = df["player"].unique().tolist()

# —————————————————————————————
# 3. Apply filters
# —————————————————————————————
mask = (df["y_proba"] >= min_xg) & (df["player"].isin(player_list))
if show_goals:
    mask &= (df["goal"] == 1)
if show_non_goals:
    mask &= (df["goal"] == 0)

filtered = df[mask]

st.sidebar.write(f"Displaying {len(filtered)} shots")

# —————————————————————————————
# 4. Draw pitch and shots
# —————————————————————————————
pitch = VerticalPitch(
    pitch_type='statsbomb',
    pitch_color='grass',
    half=True,
    pad_bottom=-11
)
fig, ax = pitch.draw(figsize=(12, 10))

norm = plt.Normalize(filtered['y_proba'].min(), filtered['y_proba'].max())
cmap = plt.cm.plasma

for _, shot in filtered.iterrows():
    color = cmap(norm(shot['y_proba']))
    marker = 'o' if shot['goal'] == 1 else 'X'
    size = 200 * shot['y_proba'] + 50
    pitch.scatter(
        shot['x'], shot['y'],
        ax=ax, s=size,
        marker=marker, color=color,
        edgecolors='black', linewidth=0.5, alpha=0.8
    )

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label('Predicted xG')

# legend
ax.scatter([], [], c='white', edgecolors='black', marker='o', s=100, label='Goal')
ax.scatter([], [], c='white', edgecolors='black', marker='X', s=100, label='No goal')
ax.legend(loc='upper right', facecolor='white', framealpha=0.8, edgecolor='black')

ax.set_title(
    "Shot Map with Predicted xG",
    fontsize=18,
    pad=20
)

# —————————————————————————————
# 5. Render in Streamlit
# —————————————————————————————
st.pyplot(fig)
