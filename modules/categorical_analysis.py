import pandas as pd


def get_categorical_columns(df):
    """Identify categorical columns, excluding booleans from numerical data."""
    categorical_columns = []

    for column in df.columns:
        series = df[column]

        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        ):
            categorical_columns.append(column)

    return categorical_columns


def get_frequency_distribution(df, column):
    """Calculate category frequencies and percentages for a column."""
    if column not in df.columns:
        raise ValueError("Selected column does not exist.")

    series = df[column].dropna()

    if series.empty:
        return pd.DataFrame(
            columns=["Category", "Frequency", "Percentage"]
        )

    counts = series.value_counts(dropna=True)

    result = pd.DataFrame({
        "Category": counts.index.astype(str),
        "Frequency": counts.values
    })

    total = result["Frequency"].sum()

    result["Percentage"] = (
        result["Frequency"] / total * 100
    ).round(2)

    return result.reset_index(drop=True)


def get_categorical_summary(df):
    """Generate summary statistics for all categorical columns."""
    categorical_columns = get_categorical_columns(df)
    records = []

    for column in categorical_columns:
        series = df[column]
        non_null = series.dropna()
        modes = non_null.mode()

        records.append({
            "Variable": str(column),
            "Total Records": len(series),
            "Non-Null Values": int(non_null.count()),
            "Missing Values": int(series.isna().sum()),
            "Unique Categories": int(non_null.nunique()),
            "Most Frequent Category": (
                str(modes.iloc[0]) if not modes.empty else "N/A"
            ),
            "Most Frequent Count": (
                int(non_null.value_counts().iloc[0])
                if not non_null.empty else 0
            )
        })

    return pd.DataFrame(records)


def get_category_statistics(df, column):
    """Calculate detailed statistics for a selected categorical variable."""
    if column not in df.columns:
        raise ValueError("Selected column does not exist.")

    series = df[column]
    non_null = series.dropna()
    modes = non_null.mode()

    return {
        "Total Records": len(series),
        "Non-Null Values": int(non_null.count()),
        "Missing Values": int(series.isna().sum()),
        "Unique Categories": int(non_null.nunique()),
        "Most Frequent Category": (
            str(modes.iloc[0]) if not modes.empty else "N/A"
        ),
        "Most Frequent Count": (
            int(non_null.value_counts().iloc[0])
            if not non_null.empty else 0
        ),
        "Missing Percentage": round(
            series.isna().mean() * 100, 2
        )
    }


def get_category_observations(df, column):
    """Generate basic observations about a categorical variable."""
    frequency = get_frequency_distribution(df, column)
    statistics = get_category_statistics(df, column)

    observations = []

    if frequency.empty:
        return ["This variable has no non-missing category values."]

    observations.append(
        f"The variable contains {statistics['Unique Categories']} "
        "unique categories."
    )

    observations.append(
        f"The most frequent category is "
        f"'{statistics['Most Frequent Category']}' "
        f"with {statistics['Most Frequent Count']} records."
    )

    if statistics["Missing Values"] > 0:
        observations.append(
            f"{statistics['Missing Values']} values are missing "
            f"({statistics['Missing Percentage']:.2f}%)."
        )
    else:
        observations.append("No missing values were found.")

    if len(frequency) == 1:
        observations.append(
            "Only one category is present in this variable."
        )
    else:
        observations.append(
            f"The least frequent category is "
            f"'{frequency.iloc[-1]['Category']}' "
            f"with {frequency.iloc[-1]['Frequency']} records."
        )

    return observations