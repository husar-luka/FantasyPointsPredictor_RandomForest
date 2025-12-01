# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import preprocessing as p
import numpy as np

feature_cols = [col for col in p.table.columns if col not in p.drop_cols]

train_data = p.table[p.table['season'] < 2024]
test_data = p.table[p.table['season'] == 2024]

X_train = train_data[feature_cols]
y_train = train_data[p.target_col]

X_test = test_data[feature_cols]
y_test = test_data[p.target_col]

model = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=2, 
    random_state=42,
    n_jobs=-1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0,max(y_test)], [0,max(y_test)], color="red")  # diagonal baseline
plt.xlabel("Actual Fantasy Points")
plt.ylabel("Predicted Fantasy Points")
plt.title("Predicted vs Actual Fantasy Points (2024)")
plt.show()


# %% Per position evaluation
pos_results = {}

for pos in ["QB", "RB", "WR", "TE"]:
    subset = test_data[test_data[f"position_group_{pos}"] == 1]
    preds_pos = model.predict(subset[feature_cols])
    pos_results[pos] = {
        "MAE": mean_absolute_error(subset[p.target_col], preds_pos),
        "RMSE": np.sqrt(mean_squared_error(subset[p.target_col], preds_pos)),
        "R2": r2_score(subset[p.target_col], preds_pos)
    }

pos_results

# %% Attach predictions back to test_data for Tableau export
test_data = test_data.copy()
test_data["predicted_points"] = y_pred
test_data["error"] = test_data[p.target_col] - test_data["predicted_points"]
test_data["absolute_error"] = test_data["error"].abs()

# Decode position from one-hot columns into a single string
ohe_cols = ["position_group_QB", "position_group_RB", "position_group_WR", "position_group_TE"]
test_data["position"] = (
    test_data[ohe_cols]
    .idxmax(axis=1)
    .str.replace("position_group_", "")
)

# %% Build a Tableau-ready export DataFrame
tableau_export = test_data[
    [
        "player_name",
        "recent_team",
        "position",
        "season",
        "week",
        "fantasy_points_ppr",
        "predicted_points",
        "absolute_error",
        "spread_line",
        "total_line",
        "is_home",
        "is_away"
    ]
].copy()

# Save to CSV for Tableau / PowerBI
tableau_export.to_csv("fantasy_predictions_2024.csv", index=False)
print("Exported Tableau data to fantasy_predictions_2024.csv")
