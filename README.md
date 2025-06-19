# Expected Goals Model & Football Match Analysis

This project delivers a complete pipeline to:

* Build an **Expected Goals (xG) model** using real football match data (StatsBomb)
* Visualize shots, heatmaps and at terms player performance
* [In coming ...] Deploy an interactive **Streamlit application** for tactical and scouting analysis

## Objective

To simulate the contribution of a **Football Data Scientist** within a professional club such as **Paris Saint-Germain**, by turning raw match events into actionable, visual, and predictive insights.

## Features

* Preprocessing of StatsBomb event data (shots, players, match context)
* XGBoost models to predict shot success probability (xG)
* Visualization: shot maps, xG overlays
* [In coming...] Streamlit app for match

## Use Cases

* **Performance analysis**: Identify high-xG zones
* **Coaching insights**: Understand shot quality over quantity

## Tech Stack

* Python (Pandas, Scikit-learn, XGBoost)
* StatsBomb Open Data
* mplsoccer, Seaborn, Matplotlib

## Project Structure

```
football-xg-analysis/
├── data/                # Cleaned StatsBomb datasets
├── notebooks/          # EDA and model training
├── models/             # Saved xG models
├── visuals/            # Static charts and Pitch shotmaps 
└── README.md
```

## About

This project was built as part of my goal to work as a **Sport Data Scientist** within a top club environment. It reflects my passion for tactical data, player development, and high-performance analytics.

---

*Julian — aspiring Football Data Scientist, combining ML, strategy and passion for the game.*
