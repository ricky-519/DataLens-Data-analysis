
"""
DataLens ML-06: Anomaly Detection

Algorithms:
- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- Elliptic Envelope
- DBSCAN-based anomaly detection

Features:
- Numerical and categorical preprocessing
- Missing-value handling
- Feature scaling
- Anomaly identification
- Anomaly scores
- Evaluation summary
- Feature-level analysis
- CSV export
- Model export

Label convention:
  1  = Normal
 -1  = Anomaly
"""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN


# =========================================================
# ALGORITHM REGISTRY
# =========================================================

def get_anomaly_algorithms():

    return {
        "Isolation Forest": {
            "description": (
                "Detects anomalies by isolating observations "
                "through random partitioning."
            ),
            "type": "Ensemble"
        },

        "Local Outlier Factor": {
            "description": (
                "Detects observations with substantially lower "
                "local density than their neighbors."
            ),
            "type": "Density-based"
        },

        "One-Class SVM": {
            "description": (
                "Learns a boundary around normal observations "
                "and identifies observations outside it."
            ),
            "type": "Boundary-based"
        },

        "Elliptic Envelope": {
            "description": (
                "Uses robust covariance estimation to identify "
                "observations outside an elliptical data boundary."
            ),
            "type": "Statistical"
        },

        "DBSCAN": {
            "description": (
                "Identifies observations in low-density regions "
                "as noise."
            ),
            "type": "Density-based"
        }
    }


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_anomaly_data(df, excluded_columns=None):

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Please provide a non-empty pandas DataFrame.")

    excluded_columns = excluded_columns or []

    invalid_columns = [
        column for column in excluded_columns
        if column not in df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Unknown excluded columns: {invalid_columns}"
        )

    X = df.drop(columns=excluded_columns).copy()

    if X.empty:
        raise ValueError("No features remain after excluding columns.")

    X = X.replace([np.inf, -np.inf], np.nan)

    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    for column in datetime_columns:

        X[f"{column}_year"] = X[column].dt.year
        X[f"{column}_month"] = X[column].dt.month
        X[f"{column}_day"] = X[column].dt.day
        X[f"{column}_dayofweek"] = X[column].dt.dayofweek

        X.drop(columns=[column], inplace=True)

    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("All selected features are empty.")

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    supported_columns = numerical_columns + categorical_columns

    X = X[supported_columns]

    if X.empty:
        raise ValueError(
            "No supported numerical or categorical features found."
        )

    # Retain original row positions so duplicate DataFrame indexes
    # cannot accidentally duplicate observations in the output.
    original_positions = np.arange(len(X))

    valid_rows = X.notna().any(axis=1)

    X = X.loc[valid_rows]
    original_positions = original_positions[valid_rows.to_numpy()]

    if len(X) < 5:
        raise ValueError(
            "At least 5 usable observations are required."
        )

    numerical_columns = [
        column for column in numerical_columns
        if column in X.columns
    ]

    categorical_columns = [
        column for column in categorical_columns
        if column in X.columns
    ]

    return (
        X,
        numerical_columns,
        categorical_columns,
        original_positions
    )


# =========================================================
# PREPROCESSING
# =========================================================

def build_anomaly_preprocessor(
    numerical_columns,
    categorical_columns
):

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

    if not transformers:
        raise ValueError("No usable features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# MAIN ANOMALY DETECTION ENGINE
# =========================================================

def run_anomaly_detection(
    df,
    algorithm="Isolation Forest",
    excluded_columns=None,
    contamination=0.05,
    n_neighbors=20,
    nu=0.05,
    kernel="rbf",
    gamma="scale",
    eps=0.5,
    min_samples=5,
    random_state=42
):

    algorithms = get_anomaly_algorithms()

    if algorithm not in algorithms:
        raise ValueError(
            f"Unsupported anomaly detection algorithm: {algorithm}"
        )

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "Contamination must be greater than 0 and at most 0.5."
        )

    if n_neighbors < 2:
        raise ValueError("Number of neighbors must be at least 2.")

    if not 0 < nu <= 1:
        raise ValueError("One-Class SVM nu must be in (0, 1].")

    if eps <= 0:
        raise ValueError("DBSCAN epsilon must be greater than zero.")

    if min_samples < 2:
        raise ValueError("Minimum samples must be at least 2.")

    (
        X,
        numerical_columns,
        categorical_columns,
        original_positions
    ) = prepare_anomaly_data(
        df,
        excluded_columns=excluded_columns
    )

    if n_neighbors >= len(X):
        n_neighbors = len(X) - 1

    preprocessor = build_anomaly_preprocessor(
        numerical_columns,
        categorical_columns
    )

    X_transformed = preprocessor.fit_transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = np.asarray(
        X_transformed,
        dtype=np.float64
    )

    if not np.isfinite(X_transformed).all():
        raise ValueError(
            "Preprocessing generated invalid numerical values."
        )

    # -----------------------------------------------------
    # MODEL CONFIGURATION
    # -----------------------------------------------------

    if algorithm == "Isolation Forest":

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            n_jobs=-1
        )

        labels = model.fit_predict(X_transformed)

        # Higher scores indicate more normal observations.
        anomaly_scores = -model.decision_function(X_transformed)

    elif algorithm == "Local Outlier Factor":

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
            n_jobs=-1
        )

        labels = model.fit_predict(X_transformed)

        anomaly_scores = -model.negative_outlier_factor_

    elif algorithm == "One-Class SVM":

        model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )

        model.fit(X_transformed)

        labels = model.predict(X_transformed)

        anomaly_scores = -model.decision_function(
            X_transformed
        ).ravel()

    elif algorithm == "Elliptic Envelope":

        if X_transformed.shape[1] >= len(X_transformed):
            raise ValueError(
                "Elliptic Envelope requires fewer features than "
                "observations. Exclude columns or use another algorithm."
            )

        model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state,
            support_fraction=None
        )

        labels = model.fit_predict(X_transformed)

        anomaly_scores = -model.decision_function(
            X_transformed
        )

    elif algorithm == "DBSCAN":

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )

        labels = model.fit_predict(X_transformed)

        # DBSCAN identifies noise as -1.
        # Convert its labels into the common anomaly convention.
        labels = np.where(labels == -1, -1, 1)

        # Estimate local density using nearest-neighbor distances.
        neighbors = NearestNeighbors(
            n_neighbors=min(min_samples, len(X))
        )

        neighbors.fit(X_transformed)

        distances, _ = neighbors.kneighbors(X_transformed)

        anomaly_scores = distances[:, -1]

    else:
        raise ValueError("Invalid anomaly detection algorithm.")

    labels = np.asarray(labels, dtype=int)
    anomaly_scores = np.asarray(
        anomaly_scores,
        dtype=float
    ).reshape(-1)

    # -----------------------------------------------------
    # ANOMALY SUMMARY
    # -----------------------------------------------------

    anomaly_mask = labels == -1
    normal_mask = labels == 1

    anomaly_count = int(anomaly_mask.sum())
    normal_count = int(normal_mask.sum())

    total_count = len(labels)

    anomaly_percentage = (
        anomaly_count / total_count * 100
    )

    # -----------------------------------------------------
    # OUTPUT DATASET
    # -----------------------------------------------------

    output_df = df.iloc[original_positions].copy()

    output_df["Anomaly_Label"] = labels
    output_df["Anomaly_Status"] = np.where(
        anomaly_mask,
        "Anomaly",
        "Normal"
    )
    output_df["Anomaly_Score"] = anomaly_scores

    # -----------------------------------------------------
    # SCORE SUMMARY
    # -----------------------------------------------------

    score_summary = pd.DataFrame({
        "Category": ["Normal", "Anomaly"],
        "Observations": [normal_count, anomaly_count]
    })

    score_summary["Percentage"] = (
        score_summary["Observations"] / total_count * 100
    ).round(2)

    # -----------------------------------------------------
    # FEATURE ANALYSIS
    # -----------------------------------------------------

    anomaly_feature_summary = []

    numeric_original = X.select_dtypes(
        include="number"
    )

    if not numeric_original.empty and anomaly_count > 0:

        normal_data = numeric_original.loc[normal_mask]
        anomaly_data = numeric_original.loc[anomaly_mask]

        for column in numeric_original.columns:

            normal_mean = (
                normal_data[column].mean()
                if not normal_data.empty
                else np.nan
            )

            anomaly_mean = (
                anomaly_data[column].mean()
                if not anomaly_data.empty
                else np.nan
            )

            anomaly_feature_summary.append({
                "Feature": column,
                "Normal Mean": normal_mean,
                "Anomaly Mean": anomaly_mean,
                "Mean Difference": anomaly_mean - normal_mean
            })

    feature_summary_df = pd.DataFrame(
        anomaly_feature_summary
    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {
        "model": model,
        "preprocessor": preprocessor,
        "algorithm": algorithm,
        "labels": labels,
        "anomaly_scores": anomaly_scores,
        "output_dataset": output_df,
        "score_summary": score_summary,
        "feature_summary": feature_summary_df,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "total_count": total_count,
        "anomaly_percentage": float(anomaly_percentage),
        "normal_percentage": float(
            normal_count / total_count * 100
        ),
        "original_features": X.shape[1],
        "transformed_features": X_transformed.shape[1],
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "excluded_columns": excluded_columns or []
    }


# =========================================================
# MODEL EXPORT
# =========================================================

def export_anomaly_model(results):

    export_object = {
        "algorithm": results["algorithm"],
        "model": results["model"],
        "preprocessor": results["preprocessor"],
        "excluded_columns": results["excluded_columns"]
    }

    buffer = BytesIO()

    joblib.dump(export_object, buffer)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# DATASET EXPORT
# =========================================================

def export_anomaly_dataset(results):

    return results["output_dataset"].to_csv(
        index=False
    ).encode("utf-8")


# =========================================================
# SUMMARY EXPORT
# =========================================================

def export_anomaly_summary(results):

    return results["score_summary"].to_csv(
        index=False
    ).encode("utf-8")
