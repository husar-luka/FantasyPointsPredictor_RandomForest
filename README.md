## Fantasy Points Predictor
This project builds a machine learning model to predict weekly Fantasy Points (PPR) for NFL players based on past performance, usage patterns, and Vegas betting lines.

The goal is to provide interpretable projections, and to analyze model performance across different player positions (QB, RB, WR, TE).

## Features
**Pulls historical NFL weekly data**

Using nfl_data_py, seasons 2020–2024 are imported:

- Passing stats

- Rushing stats

- Receiving stats

- Touchdowns

- Team information

- Fantasy scoring (standard + PPR)

**Adds betting-based contextual features**

From import_schedules():

- Spread Line

- Total Line

- Home/Away indicators

- These variables provide market expectations, which improve prediction stability.

**Rolling window player stats**

Rolling averages of the last 3 and 5 games:

- Passing yards / TDs

- Rushing yards / TDs

- Receiving yards / TDs

- Attempts, targets, completions, etc.

**Interpretable ML model**

Uses a Random Forest Regressor instead of deep learning to maintain explainability.

**Performance breakdown by player position**

Evaluated separately for:

- QB

- RB

- WR

- TE

**Tableau visualizations**

Created dashboards to:

- Compare actual vs predicted OOP

- Analyze error distributions per position

- Identify outliers and volatility

## Requirements
pip install nfl-data-py pandas scikit-learn numpy matplotlib joblib

