import pandas as pd
import numpy as np


def get_numerical_columns(df):
    """Return columns containing numerical data, excluding booleans."""
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def get_numerical_summary(df):
    """Calculate descriptive statistics for all numerical columns."""
    numerical_columns = get_numerical_columns(df)

    if not numerical_columns:
        return pd.DataFrame()

    records = []

    for column in numerical_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        modes = series.mode()
        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50)
        q3 = series.quantile(0.75)

        records.append({
            "Variable": str(column),
            "Count": int(series.count()),
            "Mean": series.mean(),
            "Median": series.median(),
            "Mode": modes.iloc[0] if not modes.empty else np.nan,
            "Minimum": series.min(),
            "Maximum": series.max(),
            "Range": series.max() - series.min(),
            "Variance": series.var(),
            "Standard Deviation": series.std(),
            "Q1 (25%)": q1,
            "Q2 (50%)": q2,
            "Q3 (75%)": q3,
            "IQR": q3 - q1,
            "Skewness": series.skew(),
            "Kurtosis": series.kurt()
        })

    return pd.DataFrame(records)


def get_variable_statistics(df, column):
    """Calculate detailed statistics for one numerical variable."""
    if column not in df.columns:
        raise ValueError("Selected column does not exist.")

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        return {}

    modes = series.mode()
    q1 = series.quantile(0.25)
    q2 = series.quantile(0.50)
    q3 = series.quantile(0.75)

    return {
        "Count": int(series.count()),
        "Missing Values": int(df[column].isna().sum()),
        "Mean": series.mean(),
        "Median": series.median(),
        "Mode": modes.iloc[0] if not modes.empty else np.nan,
        "Minimum": series.min(),
        "Maximum": series.max(),
        "Range": series.max() - series.min(),
        "Variance": series.var(),
        "Standard Deviation": series.std(),
        "First Quartile (Q1)": q1,
        "Second Quartile (Q2)": q2,
        "Third Quartile (Q3)": q3,
        "Interquartile Range (IQR)": q3 - q1,
        "Skewness": series.skew(),
        "Kurtosis": series.kurt()
    }


def get_numerical_distribution(df, column):
    """Return cleaned numerical values for distribution visualizations."""
    if column not in df.columns:
        raise ValueError("Selected column does not exist.")

    return pd.to_numeric(df[column], errors="coerce").dropna()