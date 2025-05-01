# Author: Kevyn Angueira Irizarry
# Created: 2025-04-21
# Last Modified: 2025-05-01

import pandas as pd
import numpy as np

import scipy.stats as stats
import statsmodels.api as sm

FILE_PATH = "KahootResults.xlsx"  # update this if needed

def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)
    return 100 * np.mean(smape)

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
    smape = calculate_smape(expected, answers)
    count = len(errors)

    # Print results
    print("📊 --- Adjuster Defoliation Accuracy Stats ---")
    print(f"👥 Total Responses: {count}")
    print(f"🔹 MAE (Mean Absolute Error): {mae:.2f}")
    print(f"🔸 MSE (Mean Squared Error): {mse:.2f}")
    print(f"🔻 RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"⚖️  Bias (Mean Error): {bias:.2f}")
    print(f"📈 Variance: {variance:.2f}")
    print(f"📉 Std Dev: {std_dev:.2f}")
    print(f"🌓 SMAPE (Symmetric MAPE): {smape:.2f}%")
    print()

    # Group into quartiles by expected value
    df_expanded = pd.DataFrame({'Answer': answers, 'Expected': expected})
    df_expanded['abs_error'] = np.abs(df_expanded['Answer'] - df_expanded['Expected'])
    df_expanded['quartile'] = pd.qcut(df_expanded['Expected'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # MAE and SMAPE by quartile
    group_stats = df_expanded.groupby('quartile', observed=False).agg(
        MAE=('abs_error', 'mean'),
        SMAPE=('abs_error', lambda x: np.mean(
            200 * x / (np.abs(df_expanded.loc[x.index, 'Answer']) + np.abs(df_expanded.loc[x.index, 'Expected']) + 1e-8)
        ))
    )

    # Correlation between expected and abs_error
    corr = np.corrcoef(df_expanded['Expected'], df_expanded['abs_error'])[0, 1]

    # Linear regression of abs_error ~ Expected
    X = sm.add_constant(df_expanded['Expected'])
    model = sm.OLS(df_expanded['abs_error'], X).fit()
    slope = model.params.iloc[1]
    pval = model.pvalues.iloc[1]

    # Print results
    print("📊 --- Error Behavior by Quartile of Expected ---")
    print(group_stats)
    print()

    print("📉 --- Error Scale Dependence Analysis ---")
    print(f"📈 Correlation(Expected, Absolute Error): {corr:.3f}")
    print(f"📉 Regression Slope: {slope:.3f} (p = {pval:.4f})")
    print()

    # Simple written interpretation
    print("🧠 --- Interpretation ---")
    if abs(corr) < 0.2:
        print("✅ Errors appear largely scale-invariant (no strong relationship to defoliation level).")
    elif corr > 0.5:
        print("⚠️ Errors increase notably with defoliation level (scale-dependent).")
    elif corr < -0.5:
        print("⚠️ Errors decrease as defoliation increases (inverse scale-dependence).")
    else:
        print("ℹ️ Moderate relationship detected between error size and defoliation level.")
    print()

if __name__ == "__main__":
    main()
