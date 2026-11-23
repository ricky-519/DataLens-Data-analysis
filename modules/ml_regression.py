
"""
DataLens ML-02: Regression Module

Provides regression model training, evaluation, visualization data,
and model export.
"""

from io import BytesIO

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# REGRESSION ALGORITHM REGISTRY
# =========================================================

def get_regression_models(random_state=42):
    """
    Return supported regression estimators.

    Each call creates fresh model instances.
    """

    return {
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(alpha=1.0),

        "Lasso Regression": Lasso(
            alpha=0.01,
            max_iter=10000
        ),

        "Elastic Net": ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000
        ),

        "Decision Tree Regression": DecisionTreeRegressor(
            max_depth=None,
            random_state=random_state
        ),

        "Random Forest Regression": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),

        "Extra Trees Regression": ExtraTreesRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ),

        "Gradient Boosting Regression": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state
        ),

        "Support Vector Regression": SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1
        ),

        "KNN Regression": KNeighborsRegressor(
            n_neighbors=5
        ),
    }


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_regression_data(df, target):
    """
    Separate features and target.

    Removes rows with missing target values, replaces infinite
    feature values with missing values, and removes columns
    containing no usable values.
    """

    if df is None or df.empty:
        raise ValueError("Please load a non-empty dataset.")

    if target not in df.columns:
        raise ValueError("Selected target column does not exist.")

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(
            "Regression requires a numerical target column."
        )

    working_df = df.copy()

    working_df = working_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    working_df = working_df.dropna(subset=[target])

    if len(working_df) < 5:
        raise ValueError(
            "At least 5 rows with valid target values are required."
        )

    X = working_df.drop(columns=[target])
    y = working_df[target].astype(float)

    # Remove columns containing only missing values.
    usable_columns = X.columns[X.notna().any()].tolist()
    X = X[usable_columns]

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after removing "
            "the target and empty columns."
        )

    # Convert datetime columns into numerical components.
    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    for column in datetime_columns:
        X[f"{column}_year"] = X[column].dt.year
        X[f"{column}_month"] = X[column].dt.month
        X[f"{column}_day"] = X[column].dt.day
        X[f"{column}_dayofweek"] = X[column].dt.dayofweek

        X.drop(columns=[column], inplace=True)

    # Detect feature types.
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Exclude unsupported remaining data types.
    supported_columns = numerical_columns + categorical_columns
    X = X[supported_columns]

    if X.empty or len(supported_columns) == 0:
        raise ValueError(
            "No supported numerical or categorical features found."
        )

    return X, y, numerical_columns, categorical_columns


# =========================================================
# PREPROCESSING PIPELINE
# =========================================================

def build_preprocessor(numerical_columns, categorical_columns):

    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# MODEL TRAINING
# =========================================================

def train_regression_model(
    df,
    target,
    model_name,
    test_size=0.2,
    random_state=42
):
    """
    Train a regression model using a preprocessing pipeline.

    Preprocessing is fitted only on training data to prevent
    information leakage.
    """

    X, y, numerical_columns, categorical_columns = (
        prepare_regression_data(df, target)
    )

    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")

    if len(X) < 5:
        raise ValueError("Insufficient samples for training.")

    models = get_regression_models(random_state)

    if model_name not in models:
        raise ValueError("Unsupported regression algorithm.")

    # KNN requires enough training samples.
    if model_name == "KNN Regression":
        estimated_train_size = int(len(X) * (1 - test_size))

        if estimated_train_size < 2:
            raise ValueError(
                "Not enough training samples for KNN Regression."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    estimator = models[model_name]

    # Ensure KNN does not request more neighbors than training rows.
    if model_name == "KNN Regression":
        estimator.set_params(
            n_neighbors=min(5, len(X_train))
        )

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, predictions)

    results = {
        "model": pipeline,
        "model_name": model_name,
        "target": target,
        "metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
        },
        "actual": y_test.reset_index(drop=True),
        "predicted": pd.Series(
            predictions,
            name="Predicted"
        ),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "feature_count": X.shape[1],
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "test_size": test_size,
    }

    return results


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_regression_feature_importance(trained_pipeline):

    model = trained_pipeline.named_steps["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]

    if not hasattr(preprocessor, "get_feature_names_out"):
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_

    elif hasattr(model, "coef_"):
        importance_values = np.asarray(model.coef_).ravel()

    else:
        return None

    if len(feature_names) != len(importance_values):
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_df["Absolute Importance"] = (
        importance_df["Importance"].abs()
    )

    return importance_df.sort_values(
        "Absolute Importance",
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# MODEL EXPORT
# =========================================================

def export_regression_model(trained_pipeline):

    buffer = BytesIO()

    joblib.dump(trained_pipeline, buffer)

    buffer.seek(0)

    return buffer.getvalue()
    