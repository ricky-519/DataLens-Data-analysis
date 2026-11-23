
import pandas as pd
import numpy as np


def _parse_datetime(series):
    """Safely parse a column as datetime."""
    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed"
        )
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def _add_insight(insights, category, severity, title, description, recommendation):
    insights.append({
        "Category": category,
        "Severity": severity,
        "Title": title,
        "Description": description,
        "Recommendation": recommendation
    })


def get_dataset_overview(df):
    """Generate basic dataset information."""

    if df is None or df.empty:
        return {}

    return {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Total Cells": int(df.size),
        "Missing Cells": int(df.isna().sum().sum()),
        "Complete Cells": int(df.notna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Numerical Columns": int(
            len(df.select_dtypes(include=np.number).columns)
        ),
        "Categorical Columns": int(
            len(df.select_dtypes(include=["object", "category", "bool"]).columns)
        )
    }


def analyze_missing_values(df):
    """Identify columns with missing values."""

    results = []

    for column in df.columns:
        missing = int(df[column].isna().sum())

        if missing > 0:
            percentage = missing / len(df) * 100

            results.append({
                "Column": column,
                "Missing Count": missing,
                "Missing Percentage": round(percentage, 2)
            })

    return pd.DataFrame(results)


def analyze_numerical_insights(df):
    """Generate insights about numerical variables."""

    insights = []

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numerical_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            continue

        mean_value = values.mean()
        median_value = values.median()
        std_value = values.std()
        unique_count = values.nunique()

        if unique_count <= 1:
            _add_insight(
                insights,
                "Numerical Analysis",
                "Warning",
                f"Constant variable: {column}",
                f"{column} has only one unique numerical value.",
                "Consider whether this column provides useful analytical information."
            )
            continue

        # Skewness
        skewness = values.skew()

        if pd.notna(skewness) and abs(skewness) > 1:
            direction = "right" if skewness > 0 else "left"

            _add_insight(
                insights,
                "Distribution",
                "Info",
                f"Skewed distribution: {column}",
                f"{column} is strongly {direction}-skewed (skewness: {skewness:.2f}).",
                "Consider examining the distribution and whether a transformation is appropriate."
            )

        # Mean and median comparison
        if mean_value > median_value * 1.1 and median_value >= 0:
            _add_insight(
                insights,
                "Numerical Analysis",
                "Info",
                f"Mean exceeds median: {column}",
                f"The mean ({mean_value:.2f}) is notably higher than the median ({median_value:.2f}).",
                "Investigate high-value observations that may influence the mean."
            )

        elif median_value > mean_value * 1.1 and mean_value >= 0:
            _add_insight(
                insights,
                "Numerical Analysis",
                "Info",
                f"Median exceeds mean: {column}",
                f"The median ({median_value:.2f}) is notably higher than the mean ({mean_value:.2f}).",
                "Investigate low-value observations and the overall distribution."
            )

        # IQR outlier detection
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = values[
                (values < lower_bound) |
                (values > upper_bound)
            ]

            if len(outliers) > 0:
                percentage = len(outliers) / len(values) * 100

                _add_insight(
                    insights,
                    "Outliers",
                    "Warning" if percentage > 5 else "Info",
                    f"Potential outliers: {column}",
                    f"Detected {len(outliers)} potential outliers ({percentage:.2f}%) using the IQR method.",
                    "Review these observations before deciding whether to retain, remove or cap them."
                )

        # High variability
        coefficient_of_variation = (
            std_value / abs(mean_value)
            if pd.notna(std_value) and mean_value != 0
            else np.nan
        )

        if pd.notna(coefficient_of_variation) and coefficient_of_variation > 1:
            _add_insight(
                insights,
                "Variability",
                "Info",
                f"High relative variability: {column}",
                f"The coefficient of variation is {coefficient_of_variation:.2f}.",
                "Investigate why values vary substantially relative to the mean."
            )

    return insights


def analyze_categorical_insights(df):
    """Generate insights about categorical variables."""

    insights = []

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:

        values = df[column].dropna()

        if values.empty:
            continue

        frequencies = values.value_counts()
        unique_count = len(frequencies)

        if unique_count == 1:
            _add_insight(
                insights,
                "Categorical Analysis",
                "Warning",
                f"Single-category variable: {column}",
                f"All non-missing observations in {column} have the same value.",
                "Check whether this column is necessary for analysis."
            )
            continue

        dominant_category = frequencies.index[0]
        dominant_percentage = frequencies.iloc[0] / len(values) * 100

        if dominant_percentage >= 80:
            _add_insight(
                insights,
                "Categorical Analysis",
                "Info",
                f"Dominant category: {column}",
                f"'{dominant_category}' accounts for {dominant_percentage:.2f}% of non-missing observations.",
                "Check whether the category imbalance affects your analysis."
            )

        if unique_count > 50:
            _add_insight(
                insights,
                "Categorical Analysis",
                "Info",
                f"High cardinality: {column}",
                f"{column} contains {unique_count} unique categories.",
                "Consider grouping rare categories or treating this column as an identifier where appropriate."
            )

    return insights


def analyze_correlation_insights(df, threshold=0.7):
    """Find strong numerical correlations."""

    insights = []

    numerical_df = df.select_dtypes(include=np.number)
    numerical_df = numerical_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if numerical_df.shape[1] < 2:
        return insights

    correlation = numerical_df.corr(method="pearson")

    columns = correlation.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            corr_value = correlation.iloc[i, j]

            if pd.isna(corr_value):
                continue

            if abs(corr_value) >= threshold:

                direction = (
                    "positive" if corr_value > 0 else "negative"
                )

                strength = (
                    "very strong" if abs(corr_value) >= 0.9
                    else "strong"
                )

                _add_insight(
                    insights,
                    "Correlation",
                    "Info",
                    f"Strong correlation: {columns[i]} and {columns[j]}",
                    f"{columns[i]} and {columns[j]} have a {strength} {direction} correlation (r = {corr_value:.3f}).",
                    "Investigate this relationship further. Correlation does not establish causation."
                )

    return insights


def analyze_datetime_insights(df):
    """Identify date columns and basic chronological patterns."""

    insights = []
    detected_columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
        elif series.dtype == "object":
            values = series.dropna()

            if values.empty:
                continue

            parsed_values = _parse_datetime(values)

            if parsed_values.notna().mean() < 0.80:
                continue

            parsed = pd.to_datetime(series, errors="coerce")
        else:
            continue

        valid_dates = parsed.dropna()

        if valid_dates.empty:
            continue

        detected_columns.append(column)

        _add_insight(
            insights,
            "Date/Time Analysis",
            "Info",
            f"Date column detected: {column}",
            f"{column} contains {len(valid_dates)} valid dates, ranging from {valid_dates.min()} to {valid_dates.max()}.",
            "Use this column for chronological trend and time-series analysis."
        )

    return insights, detected_columns


def analyze_data_quality_insights(df):
    """Identify general data quality issues."""

    insights = []

    missing_total = int(df.isna().sum().sum())
    total_cells = int(df.size)

    if missing_total > 0:
        missing_percentage = missing_total / total_cells * 100

        _add_insight(
            insights,
            "Data Quality",
            "Warning",
            "Missing values detected",
            f"The dataset contains {missing_total} missing cells ({missing_percentage:.2f}% of all cells).",
            "Review the missing value report and choose suitable cleaning methods."
        )

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        _add_insight(
            insights,
            "Data Quality",
            "Warning",
            "Duplicate records detected",
            f"The dataset contains {duplicate_count} duplicate rows.",
            "Review duplicate records and remove them only if they are unintended."
        )

    infinite_count = 0

    numerical_df = df.select_dtypes(include=np.number)

    if not numerical_df.empty:
        infinite_count = int(
            np.isinf(numerical_df.to_numpy(dtype=float)).sum()
        )

    if infinite_count > 0:
        _add_insight(
            insights,
            "Data Quality",
            "Warning",
            "Infinite values detected",
            f"The dataset contains {infinite_count} positive or negative infinite values.",
            "Investigate the affected columns and replace or handle infinite values."
        )

    for column in df.columns:

        if df[column].nunique(dropna=True) == 1:
            _add_insight(
                insights,
                "Data Quality",
                "Info",
                f"Constant column: {column}",
                f"{column} contains only one distinct non-missing value.",
                "Consider removing the column if it does not contribute to analysis."
            )

    return insights


def generate_all_insights(df, correlation_threshold=0.7):
    """Generate a complete automated insights report."""

    if df is None or df.empty:
        raise ValueError("Cannot generate insights from an empty dataset.")

    insights = []

    insights.extend(
        analyze_data_quality_insights(df)
    )

    insights.extend(
        analyze_numerical_insights(df)
    )

    insights.extend(
        analyze_categorical_insights(df)
    )

    insights.extend(
        analyze_correlation_insights(
            df,
            threshold=correlation_threshold
        )
    )

    datetime_insights, datetime_columns = analyze_datetime_insights(df)

    insights.extend(datetime_insights)

    return {
        "overview": get_dataset_overview(df),
        "missing_values": analyze_missing_values(df),
        "insights": insights,
        "datetime_columns": datetime_columns,
        "total_insights": len(insights)
    }


def get_insights_dataframe(insights):
    """Convert insight results into a DataFrame for export."""

    return pd.DataFrame(insights)


def get_insights_summary(insights):
    """Count insights by severity and category."""

    if not insights:
        return {
            "Total": 0,
            "Warnings": 0,
            "Info": 0,
            "Categories": 0
        }

    insights_df = pd.DataFrame(insights)

    return {
        "Total": len(insights_df),
        "Warnings": int(
            (insights_df["Severity"] == "Warning").sum()
        ),
        "Info": int(
            (insights_df["Severity"] == "Info").sum()
        ),
        "Categories": int(
            insights_df["Category"].nunique()
        )
    }
