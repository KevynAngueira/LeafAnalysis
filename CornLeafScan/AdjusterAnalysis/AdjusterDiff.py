# Author: Kevyn Angueira Irizarry
# Created: 2025-08-20
# Last Modified: 2025-08-20

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


# Load the Kahoot survey, separate out answers by questions, and add MAE column
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


# Calculate the cumulative yield loss distortion component explained by the X% of highest error answers
defol_full_range = np.arange(10, 101)
yield_loss_lookup_table = {}

components_equal_yield_loss = []
components_linear_yield_loss = []

for high_error_percentage in range(10, 101, 10):

    # Get the errors of the X% highest error answers in each question
    defoliation_error_arr = []
    for question_data in question_arr:
        question_samples = int(len(question_data)*(high_error_percentage/100))
        question_errors = question_data.nlargest(question_samples, 'mae')['mae']
        defoliation_error_arr.extend(question_errors.tolist())

    # Translate defoliation errors to expected yield loss
    expected_equal_yield_losses = []
    expected_linear_yield_losses = []
    for defoliation_error in defoliation_error_arr:

        # Edge Case: Skip instances of 0 error
        #if defoliation_error == 0:
        #    continue

        # Check if translation exists in the fast lookup table, else save it
        if defoliation_error not in yield_loss_lookup_table:

            # Get the expected yield loss at each growth stage
            all_growth_equal_yield_losses = []
            all_growth_linear_yield_losses = []
            for stage_idx, stage in enumerate(growth_stages):
                single_growth_equal_yield_loss = 0
                single_growth_linear_yield_loss = 0

                yield_losses = yield_loss_data[stage_idx]
                interp_func = interp1d(defol_percents, yield_losses, kind='linear', bounds_error=False, fill_value="extrapolate")

                # Get the yield loss of each defoliation error at a given growth stage
                for true_defol in range(10, 101, 10):
                    for delta in [-defoliation_error, defoliation_error]:
                        est_defol = true_defol + delta
                        
                        # Esge Case: Outside of bounds
                        if est_defol < 10 or est_defol > 100:
                            continue
                        
                        # Translate true and estimated defoliation to yield loss
                        y_true = float(interp_func(true_defol))
                        y_est = float(interp_func(est_defol))

                        # The yield loss error
                        y_error = np.abs(y_est - y_true)

                        # Calculate expected equal weight yield loss for single growth stage
                        equal_weight = 0.1
                        equal_y_error = y_error*equal_weight
                        single_growth_equal_yield_loss += equal_y_error

                        # Calculate expected linear weight yield loss for single growth stage
                        linear_weight = (110-true_defol)/550
                        linear_y_error = y_error*linear_weight
                        single_growth_linear_yield_loss += linear_y_error

                # Add single stage growth yield loss error to all growth array
                all_growth_equal_yield_losses.append(single_growth_equal_yield_loss)
                all_growth_linear_yield_losses.append(single_growth_linear_yield_loss)
            
            # Calculate expected yield loss error across all growth stages for given defoliation error
            expected_equal_yield_loss = np.mean(all_growth_equal_yield_losses)
            expected_linear_yield_loss = np.mean(all_growth_linear_yield_losses)

            yield_loss_lookup_table[defoliation_error] = {'equal':expected_equal_yield_loss, 'linear':expected_linear_yield_loss}
        
        # Append the expected yield loss error of each translated defoliation error
        expected_equal_yield_losses.append(yield_loss_lookup_table[defoliation_error]['equal'])
        expected_linear_yield_losses.append(yield_loss_lookup_table[defoliation_error]['linear'])

    # Calculate the component of the total yield error explained by the given X% of high error estimates
    component_equal_error = np.mean(expected_equal_yield_losses)*(high_error_percentage/100)
    component_linear_error = np.mean(expected_linear_yield_losses)*(high_error_percentage/100)
    
    components_equal_yield_loss.append(component_equal_error)
    components_linear_yield_loss.append(component_linear_error)

#print(yield_loss_lookup_table)
print("----component_equal_error----")
print(components_equal_yield_loss)
print("----component_linear_error----")
print(components_linear_yield_loss)

# Plotting
plt.figure(figsize=(14, 6))
defol_percents = np.arange(10, 91, 10)
x = np.arange(len(defol_percents))

plt.plot(x, components_equal_yield_loss[:-1], label="Equal Weighting", marker='o')
plt.plot(x, components_linear_yield_loss[:-1], label="Linear Weighting", marker='x')
plt.xticks(x, defol_percents, rotation=45, ha='right')
plt.ylabel("Distortion (# fields)")
plt.title(f"Yield Loss Distortion per Top X% Highest Error Estimates")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
