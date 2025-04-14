# Author: Kevyn Angueira Irizarry
# Created: 2025-04-14
# Last Modified: 2025-04-14

import numpy as np

count = 0
values = list(range(0, 61, 5))  # [0, 5, ..., 60]

'''
for a in values:
    for b in values:
        for c in values:
            if a + b + c <= 60:
                count += 1
'''

combinations = []

for a in values:
    for b in values:
        if a + b <= 60:
            count += 1
            combinations.append((a, b))

num_leaves = 18
x = len(combinations) / num_leaves
scaled_indices = np.arange(0, len(combinations), x).astype(int)
selected_attempts = [combinations[i] for i in scaled_indices]

print(combinations)
print(len(selected_attempts))
print(selected_attempts)
print("Total valid combinations:", count)
