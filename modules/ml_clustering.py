
"""
DataLens ML-04: Clustering Module

Algorithms:
1. K-Means
2. Mini-Batch K-Means
3. DBSCAN
4. OPTICS
5. Agglomerative Clustering
6. Birch
7. Mean Shift
8. Gaussian Mixture Model
9. Spectral Clustering

Includes:
- Automatic preprocessing
- Numerical and categorical feature handling
- Cluster evaluation
- PCA visualization
- Cluster summaries
- CSV export
- Model export
"""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    Birch,
    MeanShift,
    SpectralClustering,
    estimate_bandwidth
)

from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# =========================================================
# ALGORITHM REGISTRY
# =========================================================

def get_clustering_models(random_state=42):

    return {
        "K-Means": KMeans(
            n_clusters=3,
            random_state=random_state,
            n_init=10
        ),

        "Mini-Batch K-Means": MiniBatchKMeans(
            n_clusters=3,
            random_state=random_state,
            n_init=10
        ),

        "DBSCAN": DBSCAN(
            eps=0.5,
            min_samples=5
        ),

        "OPTICS": OPTICS(
            min_samples=5
        ),

        "Agglomerative Clustering": AgglomerativeClustering(
            n_clusters=3
        ),

        "Birch": Birch(
            n_clusters=3
        ),

        "Mean Shift": MeanShift(),

        "Gaussian Mixture Model": GaussianMixture(
            n_components=3,
            random_state=random_state
        ),

        "Spectral Clustering": SpectralClustering(
            n_clusters=3,
            random_state=random_state,
            affinity="nearest_neighbors",
            assign_labels="kmeans"
        )
    }


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_clustering_data(df, excluded_columns=None):

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Please provide a non-empty pandas dataset.")

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
        raise ValueError("No features remain for clustering.")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain only missing values.
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("All selected features contain missing values.")

    # Convert date/time columns into numerical components.
    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    for column in datetime_columns:

        X[f"{column}_year"] = X[column].dt.year
        X[f"{column}_month"] = X[column].dt.month
        X[f"{column}_day"] = X[column].dt.day
        X[f"{column}_dayofweek"] = X[column].dt.dayofweek

        X.drop(columns=[column], inplace=True)

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

    # Remove rows where every selected feature is missing.
    valid_rows = X.notna().any(axis=1)

    X = X.loc[valid_rows]

    if len(X) < 3:
        raise ValueError(
            "At least 3 usable observations are required."
        )

    numerical_columns = [
        column for column in numerical_columns
        if column in X.columns
    ]

    categorical_columns = [
        column for column in categorical_columns
        if column in X.columns
    ]

    return X, numerical_columns, categorical_columns


# =========================================================
# PREPROCESSING
# =========================================================

def build_clustering_preprocessor(
    numerical_columns,
    categorical_columns
):

    transformers = []

    if numerical_columns:

        numeric_pipeline = Pipeline([
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
            ("numerical", numeric_pipeline, numerical_columns)
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
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# MODEL TRAINING
# =========================================================

def train_clustering_model(
    df,
    model_name,
    excluded_columns=None,
    n_clusters=3,
    eps=0.5,
    min_samples=5,
    random_state=42
):

    X, numerical_columns, categorical_columns = (
        prepare_clustering_data(
            df,
            excluded_columns=excluded_columns
        )
    )

    models = get_clustering_models(random_state)

    if model_name not in models:
        raise ValueError("Unsupported clustering algorithm.")

    if n_clusters < 2:
        raise ValueError("Number of clusters must be at least 2.")

    if n_clusters >= len(X):
        raise ValueError(
            "Number of clusters must be smaller than "
            "the number of observations."
        )

    if eps <= 0:
        raise ValueError("DBSCAN epsilon must be greater than zero.")

    if min_samples < 2:
        raise ValueError("Minimum samples must be at least 2.")

    preprocessor = build_clustering_preprocessor(
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
            "Preprocessing produced invalid numerical values."
        )

    model = models[model_name]

    # Configure algorithms that accept a cluster count.
    if model_name == "Gaussian Mixture Model":
        model.set_params(n_components=n_clusters)

    elif model_name in [
        "K-Means",
        "Mini-Batch K-Means",
        "Agglomerative Clustering",
        "Birch",
        "Spectral Clustering"
    ]:
        model.set_params(n_clusters=n_clusters)

    elif model_name == "DBSCAN":
        model.set_params(
            eps=eps,
            min_samples=min_samples
        )

    elif model_name == "OPTICS":
        model.set_params(
            min_samples=min_samples
        )

    elif model_name == "Mean Shift":

        bandwidth = estimate_bandwidth(
            X_transformed,
            quantile=0.2,
            n_samples=min(500, len(X_transformed))
        )

        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = None

        model.set_params(
            bandwidth=bandwidth
        )

    model.fit(X_transformed)

    if model_name == "Gaussian Mixture Model":
        labels = model.predict(X_transformed)
    else:
        labels = model.labels_

    labels = np.asarray(labels, dtype=int)

    # Noise points have the label -1.
    cluster_labels = [
        label for label in np.unique(labels)
        if label != -1
    ]

    number_of_clusters = len(cluster_labels)
    noise_count = int(np.sum(labels == -1))

    # -----------------------------------------------------
    # CLUSTER EVALUATION
    # -----------------------------------------------------

    metrics = {
        "Silhouette Score": None,
        "Davies-Bouldin Index": None,
        "Calinski-Harabasz Index": None
    }

    valid_mask = labels != -1

    valid_features = X_transformed[valid_mask]
    valid_labels = labels[valid_mask]

    valid_cluster_count = len(np.unique(valid_labels))

    if (
        valid_cluster_count >= 2
        and valid_cluster_count < len(valid_labels)
    ):

        try:
            metrics["Silhouette Score"] = float(
                silhouette_score(
                    valid_features,
                    valid_labels
                )
            )
        except Exception:
            pass

        try:
            metrics["Davies-Bouldin Index"] = float(
                davies_bouldin_score(
                    valid_features,
                    valid_labels
                )
            )
        except Exception:
            pass

        try:
            metrics["Calinski-Harabasz Index"] = float(
                calinski_harabasz_score(
                    valid_features,
                    valid_labels
                )
            )
        except Exception:
            pass

    # -----------------------------------------------------
    # PCA VISUALIZATION
    # -----------------------------------------------------

    if X_transformed.shape[1] >= 2:

        pca = PCA(
            n_components=2,
            random_state=random_state
        )

        reduced_data = pca.fit_transform(X_transformed)

        explained_variance = float(
            pca.explained_variance_ratio_.sum() * 100
        )

    else:

        pca = None

        reduced_data = np.column_stack([
            X_transformed[:, 0],
            np.zeros(len(X_transformed))
        ])

        explained_variance = None

    cluster_df = pd.DataFrame({
        "Component 1": reduced_data[:, 0],
        "Component 2": reduced_data[:, 1],
        "Cluster": labels.astype(str)
    })

    # -----------------------------------------------------
    # CLUSTERED DATASET
    # -----------------------------------------------------

    clustered_df = df.loc[X.index].copy()

    clustered_df["Cluster"] = labels

    distribution = (
        pd.Series(labels, name="Cluster")
        .value_counts()
        .sort_index()
    )

    distribution_df = pd.DataFrame({
        "Cluster": distribution.index.astype(str),
        "Observations": distribution.values
    })

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    return {
        "model": model,
        "preprocessor": preprocessor,
        "model_name": model_name,
        "cluster_labels": labels,
        "number_of_clusters": number_of_clusters,
        "noise_count": noise_count,
        "metrics": metrics,
        "cluster_data": cluster_df,
        "clustered_dataset": clustered_df,
        "distribution": distribution_df,
        "training_rows": len(X),
        "feature_count": X.shape[1],
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "excluded_columns": excluded_columns or [],
        "explained_variance": explained_variance,
        "pca_model": pca
    }


# =========================================================
# CLUSTER SUMMARY
# =========================================================

def get_cluster_summary(results):

    clustered_df = results["clustered_dataset"]

    if clustered_df.empty:
        return pd.DataFrame()

    summary = (
        clustered_df.groupby("Cluster")
        .size()
        .reset_index(name="Observations")
    )

    summary["Percentage"] = (
        summary["Observations"] / len(clustered_df) * 100
    ).round(2)

    return summary


# =========================================================
# MODEL EXPORT
# =========================================================

def export_clustering_model(results):

    export_object = {
        "preprocessor": results["preprocessor"],
        "model": results["model"],
        "model_name": results["model_name"],
        "excluded_columns": results["excluded_columns"]
    }

    buffer = BytesIO()

    joblib.dump(export_object, buffer)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# CLUSTERED DATASET EXPORT
# =========================================================

def export_clustered_dataset(results):

    return results["clustered_dataset"].to_csv(
        index=False
    ).encode("utf-8")
