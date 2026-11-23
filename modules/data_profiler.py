
import pandas as pd


def classify_column(series):
    """Identify the general type of a dataset column."""

    if pd.api.types.is_bool_dtype(series):
        return "Categorical"

    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"

    return "Categorical"


def get_column_profile(df):
    """Generate information about every column."""

    profile = []

    for column in df.columns:
        series = df[column]

        profile.append({
            "Column Name": str(column),
            "Data Type": str(series.dtype),
            "Variable Type": classify_column(series),
            "Non-Null Values": int(series.notna().sum()),
            "Missing Values": int(series.isna().sum()),
            "Missing (%)": round(series.isna().mean() * 100, 2),
            "Unique Values": int(series.nunique())
        })

    return pd.DataFrame(profile)


def get_dataset_summary(df):
    """Generate overall dataset information."""

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Duplicate Rows": int(df.duplicated().sum()),
        "Missing Cells": int(df.isna().sum().sum()),
        "Numerical Columns": sum(
            classify_column(df[col]) == "Numerical"
            for col in df.columns
        ),
        "Categorical Columns": sum(
            classify_column(df[col]) == "Categorical"
            for col in df.columns
        ),
        "Memory Usage (KB)": round(
            df.memory_usage(deep=True).sum() / 1024, 2
        )
    }
