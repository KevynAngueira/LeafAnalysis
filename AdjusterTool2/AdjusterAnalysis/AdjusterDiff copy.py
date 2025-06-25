import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy import trapezoid

# Raw leaf loss adjustment table as a dictionary {Growth Stage: [Yield loss at defol %]}
growth_stages = [
    "7 Leaf", "8 Leaf", "9 Leaf", "10 Leaf", "11 Leaf", "12 Leaf", "13 Leaf", "14 Leaf", "15 Leaf", "16 Leaf",
    "17 Leaf", "18 Leaf", "19-21 Leaf", "Tasseled", "Silked", "Silks Brown", "Pre-Blister", "Blister",
    "Early Milk", "Milk", "Late Milk", "Soft Dough", "Early Dent", "Dent", "Late Dent", "Nearly Mature", "Mature"
]

# Defoliation percent values as column headers
defol_percents = np.arange(10, 101, 5)

# Leaf loss values from the user's table (must be aligned with growth_stages and defol_percents)
yield_loss_data = [
    [0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9, 9],
    [0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11],
    [0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 9, 10, 11, 12, 13],
    [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 11, 13, 14, 15, 16],
    [0, 0, 1, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22],
    [0, 0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 15, 16, 18, 20, 23, 26, 28],
    [0, 1, 1, 2, 3, 4, 6, 8, 10, 11, 13, 15, 17, 19, 22, 25, 28, 31, 34],
    [0, 1, 2, 3, 4, 6, 8, 10, 13, 15, 17, 20, 22, 25, 28, 32, 36, 40, 44],
    [1, 1, 2, 3, 5, 7, 9, 12, 15, 17, 20, 23, 26, 30, 34, 38, 42, 46, 51],
    [1, 2, 3, 4, 6, 8, 11, 14, 18, 20, 23, 27, 31, 36, 40, 44, 49, 55, 61],
    [2, 3, 4, 5, 7, 9, 13, 17, 21, 24, 28, 32, 37, 43, 48, 53, 59, 65, 72],
    [2, 3, 5, 7, 9, 11, 15, 19, 24, 28, 33, 38, 44, 50, 56, 62, 69, 76, 84],
    [3, 4, 6, 8, 11, 14, 18, 22, 27, 32, 38, 43, 51, 57, 64, 71, 79, 87, 96],
    [3, 5, 7, 9, 13, 17, 21, 26, 31, 36, 42, 48, 55, 62, 68, 75, 83, 91, 100],
    [3, 5, 7, 9, 12, 16, 20, 24, 29, 34, 39, 45, 51, 58, 65, 72, 80, 88, 97],
    [2, 4, 6, 8, 11, 15, 18, 22, 27, 31, 36, 41, 47, 54, 60, 66, 74, 81, 90],
    [2, 3, 5, 7, 10, 13, 16, 20, 24, 28, 32, 37, 43, 49, 54, 60, 66, 73, 81],
    [2, 3, 5, 7, 10, 13, 16, 19, 22, 26, 30, 34, 39, 45, 50, 55, 60, 66, 73],
    [2, 3, 4, 6, 8, 11, 14, 17, 20, 24, 28, 32, 36, 41, 45, 50, 55, 60, 66],
    [1, 2, 3, 5, 7, 9, 12, 15, 18, 21, 24, 28, 32, 37, 41, 45, 49, 54, 59],
    [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 35, 38, 42, 46, 50],
    [1, 1, 2, 2, 4, 6, 8, 10, 12, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41],
    [0, 0, 1, 1, 2, 3, 5, 7, 9, 11, 13, 15, 18, 21, 23, 25, 27, 29, 32],
    [0, 0, 0, 1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 15, 17, 19, 20, 21, 23],
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

question_arr = []

data = pd.read_csv("KahootResults.csv")
for question_idx in data['Question'].unique():
    question_data = data[data['Question']==question_idx].copy()

    expanded_rows = []
    for _, row in question_data.iterrows():
        for _ in range(int(row['Number'])):
            expanded_rows.append({
                'Answer': row['Answer'],
                'Expected': row['Expected']
            })
    question_data = pd.DataFrame(expanded_rows)

    # Calculate MAE for each adjuster
    question_data['mae'] = np.abs(question_data['Answer'] - question_data['Expected'])
    question_arr.append(question_data)

#print(question_arr)

mae_percentage_arr = []
auc_equal_arr = []
auc_linear_arr = []

for high_mae_percentage in range(10, 101, 10):

    total_mae = 0
    total_samples = 0

    for question_data in question_arr:
        question_samples = int(len(question_data)*(high_mae_percentage/100))
        question_mae = np.sum(question_data.nlargest(question_samples, 'mae')['mae']) 
        #print(question_mae)
        #print(question_samples)

        total_mae += question_mae
        total_samples += question_samples

    mean_mae = total_mae/total_samples

    print(mean_mae)

    # Interpolated defoliation percentages (integer range)
    defol_full_range = np.arange(10, 101)

    # Initialize containers for results
    mean_diff_equal = []
    mean_diff_linear = []

    # Define linear weights decreasing from 1 at 10% to 0.1 at 100%
    linear_weights = np.linspace(1.0, 0.1, len(defol_full_range))
    linear_weights /= linear_weights.sum()  # Normalize

    # Main calculation loop
    for stage_idx, stage in enumerate(growth_stages):
        yield_losses = yield_loss_data[stage_idx]
        interp_func = interp1d(defol_percents, yield_losses, kind='linear', bounds_error=False, fill_value="extrapolate")

        diffs_equal = []
        diffs_linear = []

        for true_defol in defol_full_range:
            for delta in [-mean_mae, mean_mae]:
                est_defol = true_defol + delta
                if est_defol < 10 or est_defol > 100:
                    continue
                y_true = float(interp_func(true_defol))
                y_est = float(interp_func(est_defol))
                diff = y_est - y_true
                diffs_equal.append(diff)
                weight_idx = np.where(defol_full_range == true_defol)[0][0]
                weight = linear_weights[weight_idx]
                diffs_linear.append(diff * weight)

        mean_diff_equal.append(np.mean(np.abs(diffs_equal)))
        mean_diff_linear.append(np.sum(np.abs(diffs_linear)))  # already weighted

    # Area under the curve (AUC)
    #auc_equal = trapezoid(mean_diff_equal, dx=1)
    #auc_linear = trapezoid(mean_diff_linear, dx=1)
    #auc_equal = trapezoid(mean_diff_equal, dx=1) * (high_mae_percentage/100)
    #auc_linear = trapezoid(mean_diff_linear, dx=1) * (high_mae_percentage/100)
    auc_linear = np.mean(mean_diff_equal) * (high_mae_percentage/100)
    auc_equal = np.mean(mean_diff_linear) * (high_mae_percentage/100)
    
    print(high_mae_percentage/100)
    print(f"Mean Yield Loss Distortion (Equal Weighting): {auc_equal:.2f}")
    print(f"Mean Yield Loss Distortion (Linear Weighting): {auc_linear:.2f}")

    auc_equal_arr.append(auc_equal)
    auc_linear_arr.append(auc_linear)
    mae_percentage_arr.append(high_mae_percentage)

    '''
    # Plotting
    plt.figure(figsize=(14, 6))
    x = np.arange(len(growth_stages))

    plt.plot(x, mean_diff_equal, label="Equal Weighting", marker='o')
    plt.plot(x, mean_diff_linear, label="Linear Weighting", marker='x')
    plt.xticks(x, growth_stages, rotation=45, ha='right')
    plt.ylabel("Mean Yield Loss Difference (%)")
    plt.title(f"Mean Yield Loss Overestimation from {mean_mae}% Defoliation Error\n(Top {high_mae_percentage}% Worst Adjuster Estimates)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    '''

# Plotting
plt.figure(figsize=(14, 6))
x = np.arange(len(mae_percentage_arr))

plt.plot(x, auc_equal_arr, label="Equal Weighting", marker='o')
plt.plot(x, auc_linear_arr, label="Linear Weighting", marker='x')
plt.xticks(x, mae_percentage_arr, rotation=45, ha='right')
plt.ylabel("Mean Yield Loss Difference (%)")
plt.title(f"Yield Loss per Top X% Highest Error Estimates")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



