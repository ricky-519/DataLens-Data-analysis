
"""
DataLens ML preprocessing utilities.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)


def inspect_dataset(df):
    if df is None or df.empty:
        raise ValueError("Please provide a non-empty dataset.")

    numerical = df.select_dtypes(include="number").columns.tolist()

    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "datetime_columns": datetime,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2), 3
        )
    }


def suggest_ml_task(df, target):
    if target not in df.columns:
        raise ValueError("Selected target column does not exist.")

    series = df[target].dropna()

    if series.empty:
        return "Invalid target"

    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return "Classification or Regression"
        return "Regression"

    return "Classification"


def prepare_features(
    df,
    target,
    scaler="StandardScaler",
    encoding="OneHotEncoder"
):
    if target not in df.columns:
        raise ValueError("Target column not found.")

    if df[target].isna().all():
        raise ValueError("Target column contains only missing values.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Convert datetime features into numeric components.
    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    for column in datetime_columns:
        X[column + "_year"] = X[column].dt.year
        X[column + "_month"] = X[column].dt.month
        X[column + "_day"] = X[column].dt.day
        X[column + "_dayofweek"] = X[column].dt.dayofweek
        X.drop(columns=[column], inplace=True)

    # Remove rows with missing target values.
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    scaler_options = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler
    }

    if scaler not in scaler_options:
        raise ValueError("Unsupported scaler.")

    encoder_options = {
        "OneHotEncoder": OneHotEncoder(
            handle_unknown="ignore"
        ),
        "OrdinalEncoder": OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    }

    if encoding not in encoder_options:
        raise ValueError("Unsupported encoding method.")

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_options[scaler]())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder_options[encoding])
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return X, y, preprocessor


def get_dataset_readiness(df):
    info = inspect_dataset(df)

    warnings = []

    if info["missing_values"] > 0:
        warnings.append("Dataset contains missing values.")

    if info["duplicate_rows"] > 0:
        warnings.append("Dataset contains duplicate rows.")

    if not info["numerical_columns"]:
        warnings.append("No numerical columns found.")

    if len(df.columns) < 2:
        warnings.append("At least two columns are needed for supervised ML.")

    return {
        "ready": len(warnings) == 0,
        "warnings": warnings,
        "information": info
    }
