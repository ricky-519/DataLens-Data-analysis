import pandas as pd
import numpy as np


def get_missing_value_report(df):
    """Report missing values in every column."""
    report = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = (
            df[column].isna().mean() * 100
            if len(df) > 0 else 0
        )

        report.append({
            "Column": str(column),
            "Missing Values": missing_count,
            "Present Values": int(df[column].notna().sum()),
            "Missing Percentage": round(missing_percentage, 2)
        })

    return pd.DataFrame(report)


def get_duplicate_report(df):
    """Report completely duplicated rows."""
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())

    return {
        "Duplicate Rows": duplicate_count,
        "Duplicate Percentage": round(
            duplicate_count / len(df) * 100, 2
        ) if len(df) > 0 else 0,
        "Duplicate Row Indices": [
            int(index) for index in df.index[duplicate_mask]
            if isinstance(index, (int, np.integer))
        ]
    }


def get_infinite_value_report(df):
    """Find positive and negative infinity in numerical columns."""
    report = []

    for column in df.columns:
        if (
            pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        ):
            values = pd.to_numeric(df[column], errors="coerce")
            positive_infinity = int(np.isposinf(values).sum())
            negative_infinity = int(np.isneginf(values).sum())

            report.append({
                "Column": str(column),
                "Positive Infinity": positive_infinity,
                "Negative Infinity": negative_infinity,
                "Total Infinite Values": (
                    positive_infinity + negative_infinity
                )
            })

    return pd.DataFrame(report)


def get_constant_columns(df):
    """Identify columns with zero or one distinct non-missing value."""
    constant_columns = []

    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)

        if unique_count <= 1:
            constant_columns.append({
                "Column": str(column),
                "Unique Values": int(unique_count),
                "Status": (
                    "Entirely Missing"
                    if df[column].isna().all()
                    else "Constant"
                )
            })

    return pd.DataFrame(constant_columns)


def get_consistency_report(df):
    """Detect basic potential consistency issues in columns."""
    report = []

    for column in df.columns:
        series = df[column].dropna()

        mixed_types = (
            series.map(lambda value: type(value).__name__).nunique() > 1
            if not series.empty else False
        )

        whitespace_values = 0
        case_variants = 0

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):
            text_values = series.astype(str)

            whitespace_values = int(
                text_values.ne(text_values.str.strip()).sum()
            )

            normalized = text_values.str.strip().str.casefold()
            case_variants = int(
                normalized.duplicated(keep=False).sum()
                - text_values.duplicated(keep=False).sum()
            )
            case_variants = max(case_variants, 0)

        if mixed_types or whitespace_values > 0 or case_variants > 0:
            report.append({
                "Column": str(column),
                "Mixed Data Types": mixed_types,
                "Values With Extra Whitespace": whitespace_values,
                "Potential Case Variations": case_variants
            })

    return pd.DataFrame(report)


def get_data_quality_summary(df):
    """Generate overall data quality metrics."""
    total_cells = df.shape[0] * df.shape[1]

    missing_cells = int(df.isna().sum().sum())
    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells > 0 else 0
    )

    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = (
        duplicate_count / len(df) * 100
        if len(df) > 0 else 0
    )

    infinite_count = 0

    for column in df.columns:
        if (
            pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        ):
            infinite_count += int(
                np.isinf(
                    pd.to_numeric(df[column], errors="coerce")
                ).sum()
            )

    infinite_percentage = (
        infinite_count / total_cells * 100
        if total_cells > 0 else 0
    )

    # A transparent indicator, not a universal data-quality standard.
    score = max(
        0,
        round(
            100
            - missing_percentage
            - duplicate_percentage
            - infinite_percentage,
            2
        )
    )

    if score >= 95:
        status = "Good"
    elif score >= 80:
        status = "Moderate"
    else:
        status = "Needs Attention"

    return {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Missing Cells": missing_cells,
        "Missing Percentage": round(missing_percentage, 2),
        "Duplicate Rows": duplicate_count,
        "Duplicate Percentage": round(duplicate_percentage, 2),
        "Infinite Values": infinite_count,
        "Quality Score": score,
        "Status": status
    }