
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. AVAILABLE COLUMN IDENTIFICATION
# --------------------------------------------------

def get_groupable_columns(df):
    """Return columns that can be used to create groups."""
    return df.columns.tolist()


def get_numerical_columns(df):
    """Return numerical columns, excluding Boolean columns."""
    return df.select_dtypes(
        include="number"
    ).columns.difference(
        df.select_dtypes(include="bool").columns
    ).tolist()


def get_categorical_columns(df):
    """Return categorical and Boolean columns."""
    return df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()


# --------------------------------------------------
# 2. GROUP-WISE NUMERICAL SUMMARY
# --------------------------------------------------

def get_group_summary(
    df,
    group_columns,
    numerical_column,
    include_missing_groups=False
):
    """
    Calculate descriptive statistics for a numerical variable
    across one or more grouping columns.
    """

    if isinstance(group_columns, str):
        group_columns = [group_columns]

    required_columns = group_columns + [numerical_column]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("One or more selected columns do not exist.")

    data = df[required_columns].copy()

    data[numerical_column] = pd.to_numeric(
        data[numerical_column],
        errors="coerce"
    )

    data[numerical_column] = data[numerical_column].replace(
        [np.inf, -np.inf], np.nan
    )

    if not include_missing_groups:
        data = data.dropna(subset=group_columns)

    data = data.dropna(subset=[numerical_column])

    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby(
            group_columns,
            observed=True,
            dropna=not include_missing_groups
        )[numerical_column]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Standard_Deviation="std",
            Minimum="min",
            Maximum="max",
            Sum="sum"
        )
        .reset_index()
    )

    return summary


# --------------------------------------------------
# 3. GROUP-WISE CATEGORICAL FREQUENCY
# --------------------------------------------------

def get_group_frequency(
    df,
    group_column,
    category_column,
    include_missing_groups=False
):
    """
    Calculate category counts and percentages within each group.
    """

    data = df[[group_column, category_column]].copy()

    if not include_missing_groups:
        data = data.dropna(
            subset=[group_column, category_column]
        )

    if data.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                category_column,
                "Count",
                "Percentage"
            ]
        )

    data[group_column] = data[group_column].astype("string")
    data[category_column] = data[category_column].astype("string")

    counts = (
        data.groupby(
            [group_column, category_column],
            dropna=not include_missing_groups,
            observed=True
        )
        .size()
        .reset_index(name="Count")
    )

    counts["Percentage"] = (
        counts["Count"]
        / counts.groupby(group_column)["Count"].transform("sum")
        * 100
    ).round(2)

    return counts


# --------------------------------------------------
# 4. MULTIPLE GROUP COMPARISON
# --------------------------------------------------

def get_multi_group_summary(
    df,
    group_columns,
    numerical_columns
):
    """
    Summarize multiple numerical variables across multiple groups.
    """

    if isinstance(group_columns, str):
        group_columns = [group_columns]

    if isinstance(numerical_columns, str):
        numerical_columns = [numerical_columns]

    selected_columns = group_columns + numerical_columns

    data = df[selected_columns].copy()

    for column in numerical_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        data[column] = data[column].replace(
            [np.inf, -np.inf], np.nan
        )

    data = data.dropna(subset=group_columns)

    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby(
            group_columns,
            observed=True
        )[numerical_columns]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )

    summary.columns = [
        f"{column}_{statistic}"
        for column, statistic in summary.columns
    ]

    return summary.reset_index()


# --------------------------------------------------
# 5. GROUP RANKING
# --------------------------------------------------

def rank_groups(
    summary_df,
    group_column,
    metric_column,
    ascending=False
):
    """
    Rank groups according to a selected metric.
    """

    if summary_df.empty:
        return pd.DataFrame()

    if group_column not in summary_df.columns:
        raise ValueError("Group column not found in summary.")

    if metric_column not in summary_df.columns:
        raise ValueError("Metric column not found in summary.")

    ranking = summary_df.copy()

    ranking[metric_column] = pd.to_numeric(
        ranking[metric_column],
        errors="coerce"
    )

    ranking = ranking.dropna(subset=[metric_column])

    ranking = ranking.sort_values(
        metric_column,
        ascending=ascending
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1)
    )

    return ranking


# --------------------------------------------------
# 6. GROUP SIZE DISTRIBUTION
# --------------------------------------------------

def get_group_sizes(
    df,
    group_columns,
    include_missing_groups=False
):
    """
    Calculate the number of records in each group.
    """

    if isinstance(group_columns, str):
        group_columns = [group_columns]

    data = df.copy()

    if not include_missing_groups:
        data = data.dropna(subset=group_columns)

    if data.empty:
        return pd.DataFrame()

    sizes = (
        data.groupby(
            group_columns,
            observed=True,
            dropna=not include_missing_groups
        )
        .size()
        .reset_index(name="Record Count")
    )

    sizes["Percentage"] = (
        sizes["Record Count"] / sizes["Record Count"].sum() * 100
    ).round(2)

    return sizes.sort_values(
        "Record Count",
        ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------
# 7. GROUP-WISE OBSERVATIONS
# --------------------------------------------------

def get_group_observations(
    summary_df,
    group_column,
    metric_column
):
    """
    Generate basic observations from a group summary.
    """

    if summary_df.empty:
        return ["No valid group data is available."]

    if group_column not in summary_df.columns:
        return ["The selected grouping column was not found."]

    if metric_column not in summary_df.columns:
        return ["The selected metric was not found."]

    data = summary_df.copy()

    data[metric_column] = pd.to_numeric(
        data[metric_column],
        errors="coerce"
    )

    data = data.dropna(subset=[metric_column])

    if data.empty:
        return ["No valid metric values are available."]

    highest_index = data[metric_column].idxmax()
    lowest_index = data[metric_column].idxmin()

    highest_group = data.loc[highest_index, group_column]
    lowest_group = data.loc[lowest_index, group_column]

    highest_value = data.loc[highest_index, metric_column]
    lowest_value = data.loc[lowest_index, metric_column]

    observations = [
        f"{len(data)} groups are included in this comparison.",
        (
            f"The highest {metric_column} is {highest_value:.2f} "
            f"for {group_column} '{highest_group}'."
        ),
        (
            f"The lowest {metric_column} is {lowest_value:.2f} "
            f"for {group_column} '{lowest_group}'."
        )
    ]

    if len(data) > 1:
        observations.append(
            f"The difference between the highest and lowest "
            f"{metric_column} is {highest_value - lowest_value:.2f}."
        )

    return observations


# --------------------------------------------------
# 8. GROUP-WISE ANALYSIS REPORT
# --------------------------------------------------

def generate_group_report(summary_df):
    """
    Convert a group summary into a downloadable report format.
    """

    if summary_df.empty:
        return pd.DataFrame()

    return summary_df.copy()
