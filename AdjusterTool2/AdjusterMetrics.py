# Author: Kevyn Angueira Irizarry
# Created: 2025-04-21
# Last Modified: 2025-04-21

import pandas as pd
import numpy as np

FILE_PATH = "KahootResults.xlsx"  # update this if needed

def main():
    df = pd.read_excel(FILE_PATH)

    # Ensure columns are correct
    if not {'Number', 'Answer', 'Expected'}.issubset(df.columns):
        raise ValueError("Excel file must contain columns: 'Number', 'Answer', 'Expected'")

    # Clean and convert data
    df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
    df['Answer'] = pd.to_numeric(df['Answer'], errors='coerce')
    df['Expected'] = pd.to_numeric(df['Expected'], errors='coerce')

    df = df.dropna(subset=['Number', 'Answer', 'Expected'])

    # Expand the dataset to individual rows based on 'Number'
    answers = np.repeat(df['Answer'].values, df['Number'].astype(int))
    expected = np.repeat(df['Expected'].values, df['Number'].astype(int))

    # Compute errors
    errors = answers - expected
    abs_errors = np.abs(errors)
    squared_errors = errors**2

    # Compute statistics
    mae = np.mean(abs_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    bias = np.mean(errors)
    variance = np.var(errors)
    std_dev = np.std(errors)
    count = len(errors)

    # Print results
    print("
📊 --- Adjuster Defoliation Accuracy Stats ---")
    print(f"👥 Total Responses: {count}")
    print(f"🔹 MAE (Mean Absolute Error): {mae:.2f}")
    print(f"🔸 MSE (Mean Squared Error): {mse:.2f}")
    print(f"🔻 RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"⚖️  Bias (Mean Error): {bias:.2f}")
    print(f"📈 Variance: {variance:.2f}")
    print(f"📉 Std Dev: {std_dev:.2f}")
    print()

if __name__ == "__main__":
    main()
