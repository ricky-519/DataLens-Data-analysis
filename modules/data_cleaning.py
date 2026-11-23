import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """Remove rows containing missing values in selected columns."""
    if columns:
        return df.dropna(subset=columns).copy()

    return df.dropna().copy()


def fill_missing_values(df, column, method, custom_value=None):
    """Fill missing values in a selected column."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        raise ValueError("Selected column does not exist.")

    if method == "Mean":
        if not pd.api.types.is_numeric_dtype(cleaned_df[column]):
            raise ValueError("Mean can only be used with numerical columns.")

        value = cleaned_df[column].mean()

    elif method == "Median":
        if not pd.api.types.is_numeric_dtype(cleaned_df[column]):
            raise ValueError("Median can only be used with numerical columns.")

        value = cleaned_df[column].median()

    elif method == "Mode":
        modes = cleaned_df[column].mode(dropna=True)

        if modes.empty:
            raise ValueError("No non-missing value is available to use as mode.")

        value = modes.iloc[0]

    elif method == "Custom Value":
        value = custom_value

    else:
        raise ValueError("Unsupported missing-value method.")

    cleaned_df[column] = cleaned_df[column].fillna(value)

    return cleaned_df


def remove_duplicate_rows(df):
    """Remove completely duplicated rows."""
    return df.drop_duplicates().copy()


def trim_text_values(df, columns=None):
    """Remove leading and trailing whitespace from text columns."""
    cleaned_df = df.copy()

    if columns is None:
        columns = [
            column for column in cleaned_df.columns
            if (
                pd.api.types.is_object_dtype(cleaned_df[column])
                or pd.api.types.is_string_dtype(cleaned_df[column])
            )
        ]

    for column in columns:
        if (
            pd.api.types.is_object_dtype(cleaned_df[column])
            or pd.api.types.is_string_dtype(cleaned_df[column])
        ):
            cleaned_df[column] = cleaned_df[column].map(
                lambda value: value.strip()
                if isinstance(value, str) else value
            )

    return cleaned_df


def standardise_text_case(df, column, case):
    """Convert text values in a column to upper or lower case."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        raise ValueError("Selected column does not exist.")

    if case not in ["Uppercase", "Lowercase", "Title Case"]:
        raise ValueError("Unsupported text case.")

    def transform(value):
        if not isinstance(value, str):
            return value

        if case == "Uppercase":
            return value.upper()
        if case == "Lowercase":
            return value.lower()
        return value.title()

    cleaned_df[column] = cleaned_df[column].map(transform)

    return cleaned_df


def convert_column_type(df, column, target_type):
    """Convert a column to a supported data type."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        raise ValueError("Selected column does not exist.")

    try:
        if target_type == "Integer":
            numeric = pd.to_numeric(cleaned_df[column], errors="raise")

            if numeric.notna().any() and (
                (numeric.dropna() % 1) != 0
            ).any():
                raise ValueError(
                    "Some values contain decimals and cannot be "
                    "converted to integers without losing information."
                )

            cleaned_df[column] = numeric.astype("Int64")

        elif target_type == "Float":
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column], errors="raise"
            )

        elif target_type == "String":
            cleaned_df[column] = cleaned_df[column].astype("string")

        elif target_type == "Date/Time":
            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column], errors="raise"
            )

        elif target_type == "Category":
            cleaned_df[column] = cleaned_df[column].astype("category")

        else:
            raise ValueError("Unsupported target data type.")

    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Conversion failed: {error}") from error

    return cleaned_df


def rename_column(df, old_name, new_name):
    """Rename a column after validating the new name."""
    cleaned_df = df.copy()

    new_name = new_name.strip()

    if old_name not in cleaned_df.columns:
        raise ValueError("Selected column does not exist.")

    if not new_name:
        raise ValueError("New column name cannot be empty.")

    if new_name != old_name and new_name in cleaned_df.columns:
        raise ValueError("A column with that name already exists.")

    return cleaned_df.rename(columns={old_name: new_name})


def get_cleaning_summary(original_df, cleaned_df):
    """Compare the original and cleaned datasets."""
    return {
        "Original Rows": len(original_df),
        "Cleaned Rows": len(cleaned_df),
        "Rows Removed": len(original_df) - len(cleaned_df),
        "Original Columns": len(original_df.columns),
        "Cleaned Columns": len(cleaned_df.columns),
        "Original Missing Cells": int(original_df.isna().sum().sum()),
        "Cleaned Missing Cells": int(cleaned_df.isna().sum().sum()),
        "Original Duplicate Rows": int(original_df.duplicated().sum()),
        "Cleaned Duplicate Rows": int(cleaned_df.duplicated().sum())
    }
def scan_dataset_for_cleaning(df):
    """
    Scan a dataset and generate recommended automatic cleaning actions.
    Recommendations are conservative and do not modify the input DataFrame.
    """
    recommendations = []

    # 1. Detect duplicate rows
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        recommendations.append({
            "id": "remove_duplicates",
            "title": "Remove exact duplicate rows",
            "description": (
                f"Found {duplicate_count} completely duplicated rows. "
                "Keep the first occurrence of each row."
            ),
            "affected": duplicate_count,
            "risk": "Review",
            "default_selected": True,
            "columns": []
        })

    # 2. Detect extra whitespace in text columns
    text_columns = [
        column for column in df.columns
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        )
    ]

    whitespace_columns = []

    for column in text_columns:
        values = df[column].dropna().astype(str)
        count = int(values.ne(values.str.strip()).sum())

        if count > 0:
            whitespace_columns.append(column)

    if whitespace_columns:
        affected = sum(
            int(
                df[column].dropna().astype(str).ne(
                    df[column].dropna().astype(str).str.strip()
                ).sum()
            )
            for column in whitespace_columns
        )

        recommendations.append({
            "id": "trim_whitespace",
            "title": "Trim extra whitespace",
            "description": (
                f"Found extra leading or trailing spaces in "
                f"{len(whitespace_columns)} text columns."
            ),
            "affected": affected,
            "risk": "Low",
            "default_selected": True,
            "columns": whitespace_columns
        })

    # 3. Detect infinite numerical values
    numeric_columns = [
        column for column in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        )
    ]

    infinite_columns = []
    infinite_count = 0

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        count = int(np.isinf(values).sum())

        if count > 0:
            infinite_columns.append(column)
            infinite_count += count

    if infinite_columns:
        recommendations.append({
            "id": "replace_infinite",
            "title": "Replace infinite values with missing values",
            "description": (
                f"Found {infinite_count} positive or negative "
                "infinite values. Replace them with missing values "
                "so they can be handled separately."
            ),
            "affected": infinite_count,
            "risk": "Review",
            "default_selected": True,
            "columns": infinite_columns
        })

    # 4. Detect missing values in numerical columns
    numeric_missing_columns = [
        column for column in numeric_columns
        if df[column].isna().any()
        or column in infinite_columns
    ]

    if numeric_missing_columns:
        affected = sum(
            int(df[column].isna().sum())
            + int(
                np.isinf(
                    pd.to_numeric(df[column], errors="coerce")
                ).sum()
            )
            for column in numeric_missing_columns
        )

        recommendations.append({
            "id": "fill_numeric_missing",
            "title": "Fill missing numerical values with median",
            "description": (
                f"Fill missing values in {len(numeric_missing_columns)} "
                "numerical columns using each column's median. "
                "This is generally more resistant to extreme values than "
                "the mean, but may not suit every dataset."
            ),
            "affected": affected,
            "risk": "Review",
            "default_selected": True,
            "columns": numeric_missing_columns
        })

    # 5. Detect missing values in categorical/text columns
    categorical_columns = [
        column for column in df.columns
        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]

    categorical_missing_columns = [
        column for column in categorical_columns
        if df[column].isna().any()
        and df[column].dropna().nunique() > 0
    ]

    if categorical_missing_columns:
        affected = sum(
            int(df[column].isna().sum())
            for column in categorical_missing_columns
        )

        recommendations.append({
            "id": "fill_categorical_missing",
            "title": "Fill missing categorical values with mode",
            "description": (
                f"Fill missing values in {len(categorical_missing_columns)} "
                "categorical columns using each column's most frequent value. "
                "Review this choice because it can increase the apparent "
                "frequency of the most common category."
            ),
            "affected": affected,
            "risk": "Review",
            "default_selected": False,
            "columns": categorical_missing_columns
        })

    # 6. Detect columns containing mixed Python value types
    mixed_type_columns = []

    for column in df.columns:
        values = df[column].dropna()

        if not values.empty:
            type_count = values.map(
                lambda value: type(value).__name__
            ).nunique()

            if type_count > 1:
                mixed_type_columns.append(str(column))

    if mixed_type_columns:
        recommendations.append({
            "id": "mixed_types_review",
            "title": "Review columns with mixed data types",
            "description": (
                "Potentially mixed Python value types were detected in: "
                + ", ".join(mixed_type_columns)
                + ". Review these columns manually; no automatic conversion "
                "is applied."
            ),
            "affected": len(mixed_type_columns),
            "risk": "Manual review",
            "default_selected": False,
            "columns": mixed_type_columns
        })

    return recommendations


def apply_automatic_cleaning(df, selected_action_ids, recommendations):
    """
    Apply approved cleaning actions to a copy of the DataFrame.
    The original DataFrame is never modified.
    """
    cleaned_df = df.copy(deep=True)

    selected = set(selected_action_ids)

    actions = {
        action["id"]: action
        for action in recommendations
    }

    # Remove exact duplicate rows
    if "remove_duplicates" in selected:
        cleaned_df = cleaned_df.drop_duplicates().copy()

    # Trim text whitespace
    if "trim_whitespace" in selected:
        columns = actions["trim_whitespace"]["columns"]

        for column in columns:
            if column in cleaned_df.columns:
                cleaned_df[column] = cleaned_df[column].map(
                    lambda value: value.strip()
                    if isinstance(value, str) else value
                )

    # Replace positive and negative infinity with missing values
    if "replace_infinite" in selected:
        columns = actions["replace_infinite"]["columns"]

        for column in columns:
            if column in cleaned_df.columns:
                cleaned_df[column] = cleaned_df[column].replace(
                    [np.inf, -np.inf], np.nan
                )

    # Fill numerical missing values using the median
    if "fill_numeric_missing" in selected:
        columns = actions["fill_numeric_missing"]["columns"]

        for column in columns:
            if column in cleaned_df.columns:
                median_value = cleaned_df[column].median()

                if pd.notna(median_value):
                    cleaned_df[column] = cleaned_df[column].fillna(
                        median_value
                    )

    # Fill categorical missing values using the mode
    if "fill_categorical_missing" in selected:
        columns = actions["fill_categorical_missing"]["columns"]

        for column in columns:
            if column in cleaned_df.columns:
                modes = cleaned_df[column].mode(dropna=True)

                if not modes.empty:
                    cleaned_df[column] = cleaned_df[column].fillna(
                        modes.iloc[0]
                    )

    return cleaned_df