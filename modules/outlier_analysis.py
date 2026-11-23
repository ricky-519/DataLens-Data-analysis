
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest


def get_numerical_columns(df):
    """Return numerical columns, excluding boolean columns."""
    return [
        col for col in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
        )
    ]


def detect_iqr_outliers(df, column):
    """Detect outliers using the IQR method."""
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()

    if valid.empty:
        return pd.Series(False, index=df.index)

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return values.lt(lower).fillna(False) | values.gt(upper).fillna(False)


def detect_zscore_outliers(df, column, threshold=3):
    """Detect outliers using the standard Z-score method."""
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()

    if len(valid) < 2:
        return pd.Series(False, index=df.index)

    std = valid.std()

    if std == 0 or pd.isna(std):
        return pd.Series(False, index=df.index)

    z_scores = (values - valid.mean()) / std

    return z_scores.abs().gt(threshold).fillna(False)


def detect_modified_zscore_outliers(df, column, threshold=3.5):
    """Detect outliers using the Modified Z-score method."""
    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()

    if valid.empty:
        return pd.Series(False, index=df.index)

    median = valid.median()
    mad = (valid - median).abs().median()

    if mad == 0 or pd.isna(mad):
        return pd.Series(False, index=df.index)

    modified_z = 0.6745 * (values - median) / mad

    return modified_z.abs().gt(threshold).fillna(False)


def detect_isolation_forest_outliers(df, column, contamination=0.05):
    """Detect outliers using Isolation Forest."""
    values = pd.to_numeric(df[column], errors="coerce")

    result = pd.Series(False, index=df.index)
    valid = values.dropna()

    if len(valid) < 5 or valid.nunique() < 2:
        return result

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(valid.to_frame())

    result.loc[valid.index] = predictions == -1

    return result


def get_outlier_summary(df, method="IQR"):
    """Generate outlier counts for all numerical columns."""
    numerical_columns = get_numerical_columns(df)
    results = []

    for column in numerical_columns:

        if method == "IQR":
            mask = detect_iqr_outliers(df, column)

        elif method == "Z-Score":
            mask = detect_zscore_outliers(df, column)

        elif method == "Modified Z-Score":
            mask = detect_modified_zscore_outliers(df, column)

        elif method == "Isolation Forest":
            mask = detect_isolation_forest_outliers(df, column)

        else:
            raise ValueError("Unsupported outlier detection method.")

        count = int(mask.sum())
        total = int(df[column].notna().sum())

        results.append({
            "Column": column,
            "Total Values": total,
            "Outliers": count,
            "Outlier Percentage": round(
                count / total * 100, 2
            ) if total else 0
        })

    return pd.DataFrame(results)


def get_outlier_details(df, column, method="IQR"):
    """Return rows identified as outliers for a selected column."""

    if method == "IQR":
        mask = detect_iqr_outliers(df, column)

    elif method == "Z-Score":
        mask = detect_zscore_outliers(df, column)

    elif method == "Modified Z-Score":
        mask = detect_modified_zscore_outliers(df, column)

    elif method == "Isolation Forest":
        mask = detect_isolation_forest_outliers(df, column)

    else:
        raise ValueError("Unsupported outlier detection method.")

    return df.loc[mask].copy()


def treat_outliers(df, column, method="IQR", treatment="Cap"):
    """Apply an outlier treatment to a copy of the dataset."""

    cleaned_df = df.copy(deep=True)

    if method == "IQR":
        mask = detect_iqr_outliers(cleaned_df, column)

        values = pd.to_numeric(cleaned_df[column], errors="coerce")
        valid = values.dropna()

        if valid.empty:
            return cleaned_df

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

    elif method == "Z-Score":
        mask = detect_zscore_outliers(cleaned_df, column)

        values = pd.to_numeric(cleaned_df[column], errors="coerce")
        valid = values.dropna()

        if len(valid) < 2:
            return cleaned_df

        mean = valid.mean()
        std = valid.std()

        if std == 0 or pd.isna(std):
            return cleaned_df

        lower = mean - 3 * std
        upper = mean + 3 * std

    elif method == "Modified Z-Score":
        mask = detect_modified_zscore_outliers(cleaned_df, column)

        values = pd.to_numeric(cleaned_df[column], errors="coerce")
        valid = values.dropna()

        if valid.empty:
            return cleaned_df

        median = valid.median()
        mad = (valid - median).abs().median()

        if mad == 0 or pd.isna(mad):
            return cleaned_df

        deviation = 3.5 * mad / 0.6745
        lower = median - deviation
        upper = median + deviation

    else:
        raise ValueError(
            "Treatment currently supports IQR, Z-Score, "
            "and Modified Z-Score."
        )

    if treatment == "Remove":
        return cleaned_df.loc[~mask].copy()

    elif treatment == "Cap":
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column], errors="coerce"
        ).clip(lower=lower, upper=upper)

    elif treatment == "Keep":
        return cleaned_df

    else:
        raise ValueError("Unsupported treatment.")

    return cleaned_df
