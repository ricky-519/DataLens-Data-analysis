
import pandas as pd
import numpy as np
from scipy import stats


# --------------------------------------------------
# 1. NUMERICAL UNIVARIATE ANALYSIS
# --------------------------------------------------

def analyze_numerical_variable(df, column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return {}

    result = {
        "Variable": column,
        "Variable Type": "Numerical",
        "Valid Values": int(values.count()),
        "Missing Values": int(df[column].isna().sum()),
        "Unique Values": int(values.nunique()),
        "Mean": float(values.mean()),
        "Median": float(values.median()),
        "Mode": float(values.mode().iloc[0]),
        "Minimum": float(values.min()),
        "Maximum": float(values.max()),
        "Range": float(values.max() - values.min()),
        "Variance": float(values.var()),
        "Standard Deviation": float(values.std()),
        "Q1": float(values.quantile(0.25)),
        "Q2": float(values.quantile(0.50)),
        "Q3": float(values.quantile(0.75)),
        "IQR": float(
            values.quantile(0.75) - values.quantile(0.25)
        ),
        "Skewness": float(values.skew()),
        "Kurtosis": float(values.kurtosis())
    }

    return result


def get_numerical_observations(df, column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return ["No valid numerical observations found."]

    observations = []

    observations.append(
        f"The variable contains {len(values)} valid numerical observations."
    )

    observations.append(
        f"The average value is {values.mean():.2f}, "
        f"while the median is {values.median():.2f}."
    )

    if values.mean() > values.median():
        observations.append(
            "The mean is greater than the median, suggesting possible "
            "right skewness."
        )
    elif values.mean() < values.median():
        observations.append(
            "The mean is lower than the median, suggesting possible "
            "left skewness."
        )
    else:
        observations.append(
            "The mean and median are approximately equal."
        )

    if values.skew() > 0.5:
        observations.append("The distribution is positively skewed.")
    elif values.skew() < -0.5:
        observations.append("The distribution is negatively skewed.")
    else:
        observations.append("The distribution is approximately symmetrical.")

    observations.append(
        f"The observed values range from {values.min():.2f} "
        f"to {values.max():.2f}."
    )

    return observations


# --------------------------------------------------
# 2. CATEGORICAL UNIVARIATE ANALYSIS
# --------------------------------------------------

def analyze_categorical_variable(df, column):

    values = df[column].dropna()

    if values.empty:
        return {}

    frequencies = values.astype(str).value_counts()

    return {
        "Variable": column,
        "Variable Type": "Categorical",
        "Valid Values": int(len(values)),
        "Missing Values": int(df[column].isna().sum()),
        "Unique Categories": int(values.nunique()),
        "Most Frequent Category": str(frequencies.index[0]),
        "Most Frequent Count": int(frequencies.iloc[0]),
        "Least Frequent Category": str(frequencies.index[-1]),
        "Least Frequent Count": int(frequencies.iloc[-1])
    }


def get_categorical_observations(df, column):

    values = df[column].dropna()

    if values.empty:
        return ["No valid categorical observations found."]

    frequencies = values.astype(str).value_counts()

    total = len(values)
    most_common = frequencies.index[0]
    least_common = frequencies.index[-1]

    observations = [
        f"The variable contains {values.nunique()} unique categories.",
        (
            f"'{most_common}' is the most frequent category, "
            f"appearing {frequencies.iloc[0]} times "
            f"({frequencies.iloc[0] / total * 100:.2f}%)."
        ),
        (
            f"'{least_common}' is the least frequent category, "
            f"appearing {frequencies.iloc[-1]} times."
        )
    ]

    return observations


def get_categorical_frequency(df, column):

    values = df[column].dropna().astype(str)

    counts = values.value_counts()

    if counts.empty:
        return pd.DataFrame(
            columns=["Category", "Count", "Percentage"]
        )

    return pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
        "Percentage": (
            counts.values / len(values) * 100
        ).round(2)
    }).reset_index(drop=True)


# --------------------------------------------------
# 3. DATE/TIME UNIVARIATE ANALYSIS
# --------------------------------------------------

def analyze_datetime_variable(df, column):

    dates = pd.to_datetime(
        df[column],
        errors="coerce",
        format="mixed"
    ).dropna()

    if dates.empty:
        return {}

    return {
        "Variable": column,
        "Variable Type": "Date/Time",
        "Valid Dates": int(len(dates)),
        "Missing or Invalid Dates": int(
            pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed"
            ).isna().sum()
        ),
        "Unique Dates": int(dates.nunique()),
        "Earliest Date": str(dates.min()),
        "Latest Date": str(dates.max()),
        "Date Range (Days)": int(
            (dates.max() - dates.min()).days
        )
    }


def get_datetime_observations(df, column):

    dates = pd.to_datetime(
        df[column],
        errors="coerce",
        format="mixed"
    ).dropna()

    if dates.empty:
        return ["No valid date/time observations found."]

    return [
        f"The variable contains {len(dates)} valid date/time values.",
        f"The earliest recorded date is {dates.min()}.",
        f"The latest recorded date is {dates.max()}.",
        (
            f"The dataset covers a period of "
            f"{(dates.max() - dates.min()).days} days."
        )
    ]


# --------------------------------------------------
# 4. UNIVARIATE ANALYSIS REPORT
# --------------------------------------------------

def generate_univariate_report(df, column, variable_type):

    if variable_type == "Numerical":
        result = analyze_numerical_variable(df, column)

    elif variable_type == "Categorical":
        result = analyze_categorical_variable(df, column)

    elif variable_type == "Date/Time":
        result = analyze_datetime_variable(df, column)

    else:
        raise ValueError("Unsupported variable type.")

    if not result:
        return pd.DataFrame()

    return pd.DataFrame(
        result.items(),
        columns=["Metric", "Value"]
    )
