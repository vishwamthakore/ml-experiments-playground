import pandas as pd

def profile_dataset(df: pd.DataFrame, target_col: str):
    info = {}

    # ---------------- Basic shape ----------------
    info["n_rows"] = df.shape[0]
    info["n_columns"] = df.shape[1]
    info["columns"] = list(df.columns)
    info["target_column_name"] = target_col

    # ---------------- datatypes ----------------
    info["dtypes"] = df.dtypes.astype(str)

    # ---------------- null values ----------------
    info["total_nulls"] = df.isnull().sum().sum()
    info["nulls_per_column"] = df.isnull().sum()

    # ---------------- target analysis ----------------
    target = df[target_col]

    if target.dtype == "O" or target.nunique() < 15:
        info["problem_type"] = "classification"
    else:
        info["problem_type"] = "regression"

    info["target_unique_values"] = target.nunique()
    info["target_distribution"] = target.value_counts()

    # ---------------- feature types ----------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    info["numeric_columns"] = numeric_cols
    info["categorical_columns"] = categorical_cols

    # ---------------- numeric statistics ----------------
    info["numeric_summary"] = df[numeric_cols].describe().T

    # ---------------- correlations ----------------
    if len(numeric_cols) > 1:
        info["correlation"] = df[numeric_cols].corr()

    return info