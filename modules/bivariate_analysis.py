
import pandas as pd
import numpy as np
from scipy import stats


# --------------------------------------------------
# 1. NUMERICAL VS NUMERICAL
# --------------------------------------------------

def analyze_numerical_relationship(df, column_x, column_y):

    data = df[[column_x, column_y]].copy()

    data[column_x] = pd.to_numeric(
        data[column_x], errors="coerce"
    )
    data[column_y] = pd.to_numeric(
        data[column_y], errors="coerce"
    )

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 2:
        return {}

    x = data[column_x]
    y = data[column_y]

    pearson_r, pearson_p = stats.pearsonr(x, y)

    spearman_r, spearman_p = stats.spearmanr(x, y)

    covariance = x.cov(y)

    if abs(pearson_r) >= 0.7:
        strength = "Strong"
    elif abs(pearson_r) >= 0.4:
        strength = "Moderate"
    elif abs(pearson_r) >= 0.2:
        strength = "Weak"
    else:
        strength = "Very weak or negligible"

    if pearson_r > 0:
        direction = "Positive"
    elif pearson_r < 0:
        direction = "Negative"
    else:
        direction = "No linear direction"

    return {
        "Variable X": column_x,
        "Variable Y": column_y,
        "Valid Observations": int(len(data)),
        "Pearson Correlation": float(pearson_r),
        "Pearson P-value": float(pearson_p),
        "Spearman Correlation": float(spearman_r),
        "Spearman P-value": float(spearman_p),
        "Covariance": float(covariance),
        "Relationship Strength": strength,
        "Relationship Direction": direction
    }


def get_correlation_matrix(df, columns=None, method="pearson"):

    if columns is None:
        columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

    if not columns:
        return pd.DataFrame()

    numeric_df = df[columns].apply(
        pd.to_numeric, errors="coerce"
    )

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf], np.nan
    )

    return numeric_df.corr(method=method)


# --------------------------------------------------
# 2. CATEGORICAL VS CATEGORICAL
# --------------------------------------------------

def analyze_categorical_relationship(
    df, column_x, column_y
):

    data = df[[column_x, column_y]].dropna()

    if data.empty:
        return {}

    table = pd.crosstab(
        data[column_x].astype(str),
        data[column_y].astype(str)
    )

    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "Observations": len(data),
            "Chi-square Statistic": None,
            "P-value": None,
            "Degrees of Freedom": None,
            "Cramer's V": None,
            "Interpretation": (
                "At least two categories are required "
                "in each variable."
            )
        }, table

    chi2, p_value, dof, expected = stats.chi2_contingency(
        table
    )

    n = table.to_numpy().sum()
    minimum_dimension = min(table.shape) - 1

    cramers_v = (
        np.sqrt(chi2 / (n * minimum_dimension))
        if n > 0 and minimum_dimension > 0
        else np.nan
    )

    if cramers_v < 0.1:
        strength = "Very weak association"
    elif cramers_v < 0.3:
        strength = "Weak association"
    elif cramers_v < 0.5:
        strength = "Moderate association"
    else:
        strength = "Strong association"

    if p_value < 0.05:
        significance = (
            "Statistically significant association "
            "(at the 5% significance level)."
        )
    else:
        significance = (
            "No statistically significant association "
            "detected at the 5% significance level."
        )

    result = {
        "Observations": int(n),
        "Chi-square Statistic": float(chi2),
        "P-value": float(p_value),
        "Degrees of Freedom": int(dof),
        "Cramer's V": float(cramers_v),
        "Association Strength": strength,
        "Interpretation": significance
    }

    return result, table


# --------------------------------------------------
# 3. NUMERICAL VS CATEGORICAL
# --------------------------------------------------

def analyze_numerical_by_category(
    df, numerical_column, categorical_column
):

    data = df[
        [numerical_column, categorical_column]
    ].copy()

    data[numerical_column] = pd.to_numeric(
        data[numerical_column], errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if data.empty:
        return pd.DataFrame(), {}

    summary = (
        data.groupby(categorical_column, observed=True)[
            numerical_column
        ]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Standard_Deviation="std",
            Minimum="min",
            Maximum="max"
        )
        .reset_index()
    )

    groups = [
        group[numerical_column].values
        for _, group in data.groupby(
            categorical_column, observed=True
        )
        if len(group) > 0
    ]

    test_results = {}

    if len(groups) >= 2:

        if all(len(group) >= 2 for group in groups):
            f_stat, anova_p = stats.f_oneway(*groups)

            test_results["ANOVA F-statistic"] = float(f_stat)
            test_results["ANOVA P-value"] = float(anova_p)
        else:
            test_results["ANOVA F-statistic"] = None
            test_results["ANOVA P-value"] = None

        try:
            h_stat, kruskal_p = stats.kruskal(*groups)

            test_results["Kruskal-Wallis Statistic"] = float(h_stat)
            test_results["Kruskal-Wallis P-value"] = float(
                kruskal_p
            )
        except ValueError:
            test_results["Kruskal-Wallis Statistic"] = None
            test_results["Kruskal-Wallis P-value"] = None

    else:
        test_results["Message"] = (
            "At least two non-empty categories are required."
        )

    return summary, test_results


# --------------------------------------------------
# 4. DATETIME VS NUMERICAL
# --------------------------------------------------

def analyze_datetime_numerical(
    df, datetime_column, numerical_column, frequency="Month"
):

    data = df[
        [datetime_column, numerical_column]
    ].copy()

    data[datetime_column] = pd.to_datetime(
        data[datetime_column],
        errors="coerce",
        format="mixed"
    )

    data[numerical_column] = pd.to_numeric(
        data[numerical_column],
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if data.empty:
        return pd.DataFrame()

    if frequency == "Year":
        data["Period"] = data[datetime_column].dt.to_period(
            "Y"
        ).astype(str)

    elif frequency == "Quarter":
        data["Period"] = data[datetime_column].dt.to_period(
            "Q"
        ).astype(str)

    elif frequency == "Month":
        data["Period"] = data[datetime_column].dt.to_period(
            "M"
        ).astype(str)

    elif frequency == "Day":
        data["Period"] = data[datetime_column].dt.date.astype(
            str
        )

    else:
        raise ValueError("Unsupported time frequency.")

    summary = (
        data.groupby("Period")[numerical_column]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Minimum="min",
            Maximum="max"
        )
        .reset_index()
    )

    return summary


# --------------------------------------------------
# 5. DATETIME VS CATEGORICAL
# --------------------------------------------------

def analyze_datetime_categorical(
    df, datetime_column, categorical_column, frequency="Month"
):

    data = df[
        [datetime_column, categorical_column]
    ].copy()

    data[datetime_column] = pd.to_datetime(
        data[datetime_column],
        errors="coerce",
        format="mixed"
    )

    data = data.dropna()

    if data.empty:
        return pd.DataFrame()

    if frequency == "Year":
        data["Period"] = data[datetime_column].dt.year.astype(
            str
        )

    elif frequency == "Month":
        data["Period"] = data[datetime_column].dt.to_period(
            "M"
        ).astype(str)

    elif frequency == "Day of Week":
        data["Period"] = data[
            datetime_column
        ].dt.day_name()

    else:
        raise ValueError("Unsupported time frequency.")

    result = pd.crosstab(
        data["Period"],
        data[categorical_column].astype(str)
    )

    return result


# --------------------------------------------------
# 6. AUTOMATIC RELATIONSHIP OBSERVATIONS
# --------------------------------------------------

def get_relationship_observations(result):

    observations = []

    if not result:
        return ["Insufficient data to analyse this relationship."]

    if "Pearson Correlation" in result:

        correlation = result["Pearson Correlation"]
        direction = result["Relationship Direction"]
        strength = result["Relationship Strength"]

        observations.append(
            f"The relationship is {strength.lower()} "
            f"and {direction.lower()}."
        )

        observations.append(
            f"Pearson correlation coefficient: {correlation:.3f}."
        )

        if result["Pearson P-value"] < 0.05:
            observations.append(
                "The linear correlation is statistically significant "
                "at the 5% significance level."
            )
        else:
            observations.append(
                "The linear correlation is not statistically "
                "significant at the 5% significance level."
            )

        observations.append(
            "Correlation does not establish causation."
        )

    elif "Cramer's V" in result:

        observations.append(
            f"Association strength: {result['Association Strength']}."
        )

        observations.append(result["Interpretation"])

    return observations
