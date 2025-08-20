# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from GetLeafModelData import GetLeafModelData

num_base_width_segments = 3

X, y = GetLeafModelData(num_base_width_segments)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Area")
plt.ylabel("Predicted Area")
plt.title(f"Random Forest ({num_base_width_segments} Widths + Length)")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"Result/RandomForest/RandomForest_{num_base_width_segments}_With_Length.png")

print(f"RF MSE: {mse}, R2: {r2}")
results = {
    "model": "Random Forest",
    "segments": num_base_width_segments,
    "mse": mse,
    "r2": r2
}
with open(f'Result/RandomForest/RandomForest_{num_base_width_segments}_With_Length.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save the model to a file
joblib.dump(model, "SavedModels/random_forest_model.pkl")

plt.show()
