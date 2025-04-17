# Author: Kevyn Angueira Irizarry
# Created: 2025-04-17
# Last Modified: 2025-04-17

import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from GetLeafModelData import GetLeafModelData

num_base_width_segments = 3

X, y = GetLeafModelData(num_base_width_segments)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Area")
plt.ylabel("Predicted Area")
plt.title(f"Gradient Boosting ({num_base_width_segments} Widths + Length)")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"Result/GradientBoosting/GradientBoosting_{num_base_width_segments}_With_Length.png")

print(f"GB MSE: {mse}, R2: {r2}")
results = {
    "model": "Gradient Boosting",
    "segments": num_base_width_segments,
    "mse": mse,
    "r2": r2
}
with open(f'Result/GradientBoosting/GradientBoosting_{num_base_width_segments}_With_Length.json', 'w') as f:
    json.dump(results, f, indent=4)

plt.show()
