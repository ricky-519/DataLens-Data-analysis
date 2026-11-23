
import pandas as pd
import numpy as np
from scipy import stats


# --------------------------------------------------
# 1. VARIABLE TYPE DETECTION
# --------------------------------------------------

def detect_variable_type(series):
    """Classify a column as Numerical, Categorical or Date/Time."""

    if pd.api.types.is_bool_dtype(series):
        return "Categorical"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"

    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"

    values = series.dropna()

    if values.empty:
        return "Categorical"

    # Avoid treating ordinary numeric strings as dates.
    if not pd.api.types.is_numeric_dtype(series):
        try:
            parsed = pd.to_datetime(
                values,
                errors="coerce",
                format="mixed"
            )

            valid_ratio = parsed.notna().mean()

            if valid_ratio >= 0.80:
                return "Date/Time"

        except (ValueError, TypeError):
            pass

    return "Categorical"


def get_variable_type_report(df):
    """Return the detected type of every dataset column."""

    results = []

    for column in df.columns:
        results.append({
            "Column": str(column),
            "Data Type": str(df[column].dtype),
            "Variable Type": detect_variable_type(df[column]),
            "Missing Values": int(df[column].isna().sum()),
            "Unique Values": int(df[column].nunique())
        })

    return pd.DataFrame(results)


def get_columns_by_type(df, variable_type):
    """Get columns belonging to a selected variable type."""

    return [
        column for column in df.columns
        if detect_variable_type(df[column]) == variable_type
    ]


# --------------------------------------------------
# 2. NUMERICAL DISTRIBUTION
# --------------------------------------------------

def get_numerical_columns(df):
    return get_columns_by_type(df, "Numerical")


def get_distribution_statistics(df, column):
    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return {}

    skewness = float(values.skew())
    kurtosis = float(values.kurtosis())

    if skewness > 0.5:
        shape = "Positively Skewed (Right-Skewed)"
    elif skewness < -0.5:
        shape = "Negatively Skewed (Left-Skewed)"
    else:
        shape = "Approximately Symmetrical"

    if kurtosis > 0.5:
        tail_shape = "Heavy-tailed (Leptokurtic)"
    elif kurtosis < -0.5:
        tail_shape = "Light-tailed (Platykurtic)"
    else:
        tail_shape = "Similar to Normal (Mesokurtic)"

    return {
        "Count": int(values.count()),
        "Mean": float(values.mean()),
        "Median": float(values.median()),
        "Mode": float(values.mode().iloc[0]),
        "Standard Deviation": float(values.std()),
        "Variance": float(values.var()),
        "Minimum": float(values.min()),
        "Maximum": float(values.max()),
        "Range": float(values.max() - values.min()),
        "Q1": float(values.quantile(0.25)),
        "Q3": float(values.quantile(0.75)),
        "Skewness": skewness,
        "Kurtosis (Excess)": kurtosis,
        "Distribution Shape": shape,
        "Tail Characteristics": tail_shape
    }


def test_normality(df, column):
    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    results = []

    if len(values) < 8:
        return pd.DataFrame([{
            "Test": "Normality Tests",
            "Statistic": None,
            "P-Value": None,
            "Conclusion": "At least 8 valid observations are required."
        }])

    shapiro_values = values

    if len(shapiro_values) > 5000:
        shapiro_values = shapiro_values.sample(
            n=5000,
            random_state=42
        )

    shapiro_stat, shapiro_p = stats.shapiro(shapiro_values)

    results.append({
        "Test": "Shapiro-Wilk",
        "Statistic": float(shapiro_stat),
        "P-Value": float(shapiro_p),
        "Conclusion": (
            "Evidence against normality"
            if shapiro_p < 0.05
            else "Insufficient evidence against normality"
        )
    })

    dagostino_stat, dagostino_p = stats.normaltest(values)

    results.append({
        "Test": "D'Agostino's K-squared",
        "Statistic": float(dagostino_stat),
        "P-Value": float(dagostino_p),
        "Conclusion": (
            "Evidence against normality"
            if dagostino_p < 0.05
            else "Insufficient evidence against normality"
        )
    })

    return pd.DataFrame(results)


def compare_with_normal_distribution(df, column):
    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(values) < 2:
        return pd.DataFrame()

    mean = values.mean()
    std = values.std()

    if pd.isna(std) or std == 0:
        return pd.DataFrame()

    x_values = np.linspace(
        values.min(),
        values.max(),
        200
    )

    return pd.DataFrame({
        "Value": x_values,
        "Normal Density": stats.norm.pdf(
            x_values,
            loc=mean,
            scale=std
        )
    })


# --------------------------------------------------
# 3. CATEGORICAL DISTRIBUTION
# --------------------------------------------------

def get_categorical_distribution(df, column):
    values = df[column].dropna()

    if values.empty:
        return pd.DataFrame(
            columns=["Category", "Count", "Percentage"]
        )

    counts = values.astype(str).value_counts()

    result = pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
        "Percentage": (
            counts.values / len(values) * 100
        ).round(2)
    })

    return result.reset_index(drop=True)


def get_categorical_statistics(df, column):
    values = df[column].dropna()

    if values.empty:
        return {}

    counts = values.astype(str).value_counts()

    return {
        "Total Values": int(len(values)),
        "Unique Categories": int(values.nunique()),
        "Most Frequent Category": str(counts.index[0]),
        "Most Frequent Count": int(counts.iloc[0]),
        "Least Frequent Category": str(counts.index[-1]),
        "Least Frequent Count": int(counts.iloc[-1])
    }


# --------------------------------------------------
# 4. DATE/TIME DISTRIBUTION
# --------------------------------------------------

def get_datetime_values(df, column):
    """Convert a date/time column into datetime values."""

    return pd.to_datetime(
        df[column],
        errors="coerce",
        format="mixed"
    )


def get_datetime_distribution(df, column, frequency="Month"):
    dates = get_datetime_values(df, column).dropna()

    if dates.empty:
        return pd.DataFrame(
            columns=["Period", "Count"]
        )

    if frequency == "Year":
        grouped = dates.dt.to_period("Y").value_counts().sort_index()

    elif frequency == "Month":
        grouped = dates.dt.to_period("M").value_counts().sort_index()

    elif frequency == "Week":
        grouped = dates.dt.to_period("W").value_counts().sort_index()

    elif frequency == "Day":
        grouped = dates.dt.date.value_counts().sort_index()

    elif frequency == "Day of Week":
        grouped = dates.dt.day_name().value_counts()

        day_order = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]

        grouped = grouped.reindex(day_order, fill_value=0)

    elif frequency == "Hour":
        grouped = dates.dt.hour.value_counts().sort_index()

    else:
        raise ValueError("Unsupported date frequency.")

    result = pd.DataFrame({
        "Period": grouped.index.astype(str),
        "Count": grouped.values
    })

    return result.reset_index(drop=True)


def get_datetime_statistics(df, column):
    dates = get_datetime_values(df, column).dropna()

    if dates.empty:
        return {}

    return {
        "Valid Dates": int(len(dates)),
        "Earliest Date": str(dates.min()),
        "Latest Date": str(dates.max()),
        "Unique Dates": int(dates.nunique()),
        "Missing or Invalid Dates": int(
            get_datetime_values(df, column).isna().sum()
        ),
        "Date Range (Days)": int(
            (dates.max() - dates.min()).days
        )
    }


# --------------------------------------------------
# 5. OVERALL DATASET DISTRIBUTION
# --------------------------------------------------

def get_dataset_distribution_summary(df):
    type_report = get_variable_type_report(df)

    type_counts = type_report["Variable Type"].value_counts()

    return {
        "Total Rows": int(len(df)),
        "Total Columns": int(len(df.columns)),
        "Numerical Columns": int(type_counts.get("Numerical", 0)),
        "Categorical Columns": int(type_counts.get("Categorical", 0)),
        "Date/Time Columns": int(type_counts.get("Date/Time", 0)),
        "Total Missing Cells": int(df.isna().sum().sum()),
        "Total Unique Rows": int(df.drop_duplicates().shape[0])
    }


def get_distribution_report(df):
    """Generate a combined report for all variable types."""

    results = []

    for column in df.columns:
        variable_type = detect_variable_type(df[column])

        if variable_type == "Numerical":
            statistics = get_distribution_statistics(df, column)

        elif variable_type == "Categorical":
            statistics = get_categorical_statistics(df, column)

        else:
            statistics = get_datetime_statistics(df, column)

        results.append({
            "Column": str(column),
            "Variable Type": variable_type,
            **statistics
        })

    return pd.DataFrame(results)
