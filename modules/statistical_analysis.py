
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


# ==================================================
# COMMON HELPERS
# ==================================================

def numeric_values(series):
    values = pd.to_numeric(series, errors="coerce")
    return (
        values.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )


def interpret_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return (
            f"Statistically significant at alpha={alpha}. "
            "Reject the null hypothesis."
        )
    return (
        f"Not statistically significant at alpha={alpha}. "
        "Fail to reject the null hypothesis."
    )


def make_test_result(name, statistic, p_value, alpha=0.05, **details):
    return {
        "Test": name,
        "Test Statistic": float(statistic),
        "P-value": float(p_value),
        "Significance Level": float(alpha),
        "Interpretation": interpret_p_value(p_value, alpha),
        **details
    }


# ==================================================
# ASSUMPTION CHECKS
# ==================================================

def check_normality(df, columns):
    """
    Shapiro-Wilk normality test for selected numerical columns.
    For more than 5000 observations, uses a reproducible
    random sample of 5000 observations.
    """
    results = []

    for column in columns:
        values = numeric_values(df[column])

        if len(values) < 3:
            results.append({
                "Variable": column,
                "Sample Size": len(values),
                "Test": "Shapiro-Wilk",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Interpretation": "At least 3 observations are required."
            })
            continue

        if values.nunique() < 2:
            results.append({
                "Variable": column,
                "Sample Size": len(values),
                "Test": "Shapiro-Wilk",
                "Statistic": np.nan,
                "P-value": np.nan,
                "Interpretation": "Normality cannot be tested on constant data."
            })
            continue

        if len(values) > 5000:
            values = values.sample(5000, random_state=42)

        statistic, p_value = stats.shapiro(values)

        results.append({
            "Variable": column,
            "Sample Size": len(values),
            "Test": "Shapiro-Wilk",
            "Statistic": float(statistic),
            "P-value": float(p_value),
            "Interpretation": (
                "Evidence against normality."
                if p_value < 0.05
                else "No significant evidence against normality."
            )
        })

    return pd.DataFrame(results)


def check_equal_variance(df, numerical_column, group_column):
    """Levene's test for equality of group variances."""
    data = df[[numerical_column, group_column]].copy()
    data[numerical_column] = pd.to_numeric(
        data[numerical_column], errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    groups = [
        group[numerical_column].to_numpy()
        for _, group in data.groupby(group_column, observed=True)
    ]

    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        raise ValueError(
            "At least two groups with two observations each are required."
        )

    statistic, p_value = stats.levene(*groups, center="median")

    return make_test_result(
        "Levene's Test",
        statistic,
        p_value,
        Equal_Variance=bool(p_value >= 0.05)
    )


# ==================================================
# EFFECT SIZES
# ==================================================

def cohens_d(group_a, group_b):
    """Cohen's d using pooled standard deviation."""
    a = numeric_values(pd.Series(group_a))
    b = numeric_values(pd.Series(group_b))

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least two observations.")

    pooled_variance = (
        ((len(a) - 1) * a.var(ddof=1))
        + ((len(b) - 1) * b.var(ddof=1))
    ) / (len(a) + len(b) - 2)

    if pooled_variance <= 0:
        return np.nan

    return float((a.mean() - b.mean()) / np.sqrt(pooled_variance))


def eta_squared_from_f(f_statistic, df_between, df_within):
    """Eta-squared calculated from an ANOVA F statistic."""
    denominator = (
        f_statistic * df_between + df_within
    )

    if denominator <= 0:
        return np.nan

    return float(
        (f_statistic * df_between) / denominator
    )


def cramers_v(contingency_table):
    """Cramer's V for a contingency table."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)

    n = contingency_table.to_numpy().sum()
    denominator = min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1
    )

    if n <= 0 or denominator <= 0:
        return np.nan

    return float(np.sqrt(chi2 / (n * denominator)))


# ==================================================
# PARAMETRIC HYPOTHESIS TESTS
# ==================================================

def one_sample_t_test(df, column, population_mean, alpha=0.05):
    values = numeric_values(df[column])

    if len(values) < 2:
        raise ValueError("At least two valid observations are required.")

    statistic, p_value = stats.ttest_1samp(
        values,
        popmean=float(population_mean)
    )

    return make_test_result(
        "One-Sample T-Test",
        statistic,
        p_value,
        alpha,
        **{
            "Sample Mean": float(values.mean()),
            "Population Mean": float(population_mean),
            "Sample Size": int(len(values))
        }
    )


def independent_t_test(
    df, numerical_column, group_column,
    group_a, group_b, alpha=0.05
):
    data = df[[numerical_column, group_column]].copy()
    data[numerical_column] = pd.to_numeric(
        data[numerical_column], errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    a = data.loc[data[group_column] == group_a, numerical_column]
    b = data.loc[data[group_column] == group_b, numerical_column]

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least two observations.")

    # Welch's t-test does not assume equal group variances.
    statistic, p_value = stats.ttest_ind(
        a, b, equal_var=False
    )

    return make_test_result(
        "Welch's Independent T-Test",
        statistic,
        p_value,
        alpha,
        **{
            "Group A": str(group_a),
            "Group B": str(group_b),
            "Mean A": float(a.mean()),
            "Mean B": float(b.mean()),
            "Cohen's d": cohens_d(a, b),
            "Sample Size A": int(len(a)),
            "Sample Size B": int(len(b))
        }
    )


def paired_t_test(df, column_a, column_b, alpha=0.05):
    data = df[[column_a, column_b]].copy()

    for column in [column_a, column_b]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 2:
        raise ValueError("At least two complete pairs are required.")

    differences = data[column_a] - data[column_b]

    if np.allclose(differences, 0):
        raise ValueError("All paired differences are zero.")

    statistic, p_value = stats.ttest_rel(
        data[column_a], data[column_b]
    )

    return make_test_result(
        "Paired T-Test",
        statistic,
        p_value,
        alpha,
        **{
            "Mean A": float(data[column_a].mean()),
            "Mean B": float(data[column_b].mean()),
            "Mean Difference": float(differences.mean()),
            "Sample Size": int(len(data))
        }
    )


# ==================================================
# NON-PARAMETRIC TESTS
# ==================================================

def mann_whitney_test(
    df, numerical_column, group_column,
    group_a, group_b, alpha=0.05
):
    data = df[[numerical_column, group_column]].copy()
    data[numerical_column] = pd.to_numeric(
        data[numerical_column], errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    a = data.loc[data[group_column] == group_a, numerical_column]
    b = data.loc[data[group_column] == group_b, numerical_column]

    if len(a) < 1 or len(b) < 1:
        raise ValueError("Both groups need valid observations.")

    statistic, p_value = stats.mannwhitneyu(
        a, b, alternative="two-sided"
    )

    return make_test_result(
        "Mann-Whitney U Test",
        statistic,
        p_value,
        alpha,
        **{
            "Group A": str(group_a),
            "Group B": str(group_b),
            "Median A": float(a.median()),
            "Median B": float(b.median())
        }
    )


def wilcoxon_test(df, column_a, column_b, alpha=0.05):
    data = df[[column_a, column_b]].copy()

    for column in [column_a, column_b]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 2:
        raise ValueError("At least two complete pairs are required.")

    differences = data[column_a] - data[column_b]

    if np.allclose(differences, 0):
        raise ValueError("All paired differences are zero.")

    statistic, p_value = stats.wilcoxon(
        data[column_a], data[column_b]
    )

    return make_test_result(
        "Wilcoxon Signed-Rank Test",
        statistic,
        p_value,
        alpha,
        Sample_Size=int(len(data))
    )


# ==================================================
# CONFIDENCE INTERVALS
# ==================================================

def confidence_interval_mean(
    df, column, confidence_level=0.95
):
    values = numeric_values(df[column])

    if len(values) < 2:
        raise ValueError("At least two observations are required.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    mean = float(values.mean())
    standard_error = stats.sem(values)

    lower, upper = stats.t.interval(
        confidence_level,
        df=len(values) - 1,
        loc=mean,
        scale=standard_error
    )

    return {
        "Variable": column,
        "Sample Size": int(len(values)),
        "Mean": mean,
        "Confidence Level": float(confidence_level),
        "Lower Bound": float(lower),
        "Upper Bound": float(upper),
        "Margin of Error": float(upper - mean)
    }


def confidence_interval_difference(
    df, numerical_column, group_column,
    group_a, group_b, confidence_level=0.95
):
    """Welch confidence interval for the difference in group means."""
    data = df[[numerical_column, group_column]].copy()
    data[numerical_column] = pd.to_numeric(
        data[numerical_column], errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    a = data.loc[data[group_column] == group_a, numerical_column]
    b = data.loc[data[group_column] == group_b, numerical_column]

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least two observations.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    difference = float(a.mean() - b.mean())

    se_squared = (
        a.var(ddof=1) / len(a)
        + b.var(ddof=1) / len(b)
    )

    standard_error = np.sqrt(se_squared)

    if standard_error == 0:
        raise ValueError("Cannot calculate an interval with zero standard error.")

    numerator = se_squared ** 2
    denominator = (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )

    degrees_freedom = numerator / denominator
    critical_value = stats.t.ppf(
        (1 + confidence_level) / 2,
        degrees_freedom
    )

    margin = critical_value * standard_error

    return {
        "Group A": str(group_a),
        "Group B": str(group_b),
        "Mean Difference (A - B)": difference,
        "Confidence Level": float(confidence_level),
        "Lower Bound": float(difference - margin),
        "Upper Bound": float(difference + margin),
        "Degrees of Freedom": float(degrees_freedom)
    }


# ==================================================
# SIMPLE AND MULTIPLE LINEAR REGRESSION
# ==================================================

def linear_regression(df, predictors, target):
    """
    OLS regression using one or more numerical predictors.
    Returns model metrics, coefficients, predictions and diagnostics.
    """
    if isinstance(predictors, str):
        predictors = [predictors]

    predictors = list(dict.fromkeys(predictors))

    if not predictors:
        raise ValueError("Select at least one predictor.")

    if target in predictors:
        raise ValueError("The target cannot also be a predictor.")

    selected = predictors + [target]

    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate column names are not supported.")

    data = df[selected].copy()

    for column in selected:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) <= len(predictors) + 1:
        raise ValueError(
            "Not enough complete observations for the selected predictors."
        )

    if any(data[column].nunique() < 2 for column in predictors):
        raise ValueError("Each predictor must contain at least two distinct values.")

    x = sm.add_constant(data[predictors], has_constant="add")
    y = data[target]

    model = sm.OLS(y, x).fit()

    predictions = model.predict(x)
    residuals = y - predictions

    coefficients = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Standard Error": model.bse.values,
        "T-statistic": model.tvalues.values,
        "P-value": model.pvalues.values,
        "CI Lower": model.conf_int()[0].values,
        "CI Upper": model.conf_int()[1].values
    })

    # Residual normality: Shapiro-Wilk, sampled if needed.
    normality_p = np.nan
    if len(residuals) >= 3 and residuals.nunique() > 1:
        residual_sample = residuals
        if len(residual_sample) > 5000:
            residual_sample = residual_sample.sample(
                5000, random_state=42
            )
        normality_p = float(
            stats.shapiro(residual_sample).pvalue
        )

    # Heteroscedasticity: Breusch-Pagan test.
    bp_p = np.nan
    try:
        bp_result = het_breuschpagan(model.resid, model.model.exog)
        bp_p = float(bp_result[1])
    except (ValueError, np.linalg.LinAlgError):
        pass

    # Autocorrelation indicator; interpretation depends on data ordering.
    dw_statistic = float(durbin_watson(residuals))

    prediction_table = pd.DataFrame({
        "Actual": y,
        "Predicted": predictions,
        "Residual": residuals
    })

    return {
        "R-squared": float(model.rsquared),
        "Adjusted R-squared": float(model.rsquared_adj),
        "F-statistic": float(model.fvalue),
        "F-test P-value": float(model.f_pvalue),
        "Observations": int(len(data)),
        "Predictors": predictors,
        "Target": target,
        "Coefficients": coefficients,
        "Predictions": prediction_table,
        "Residual Normality P-value": normality_p,
        "Breusch-Pagan P-value": bp_p,
        "Durbin-Watson Statistic": dw_statistic,
        "Model Summary": model.summary().as_text(),
        "Model": model
    }
