
"""
DataLens ML-05: Dimensionality Reduction

Algorithms:
- PCA
- Kernel PCA
- Truncated SVD
- t-SNE
- UMAP (optional)

Features:
- Numerical and categorical preprocessing
- Missing-value handling
- Feature scaling
- 2D and 3D dimensionality reduction
- Explained variance where supported
- Reduced dataset export
- Model export
"""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# OPTIONAL UMAP
# =========================================================

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP = None
    UMAP_AVAILABLE = False


# =========================================================
# ALGORITHM REGISTRY
# =========================================================

def get_dimensionality_algorithms():

    algorithms = {
        "PCA": {
            "description": (
                "Linear dimensionality reduction that maximizes "
                "explained variance."
            ),
            "type": "Linear",
            "supports_explained_variance": True
        },

        "Kernel PCA": {
            "description": (
                "Non-linear dimensionality reduction using kernel functions."
            ),
            "type": "Non-linear",
            "supports_explained_variance": False
        },

        "Truncated SVD": {
            "description": (
                "Dimensionality reduction using truncated singular "
                "value decomposition."
            ),
            "type": "Linear",
            "supports_explained_variance": True
        },

        "t-SNE": {
            "description": (
                "Non-linear embedding technique useful for visualizing "
                "high-dimensional data."
            ),
            "type": "Non-linear",
            "supports_explained_variance": False
        }
    }

    if UMAP_AVAILABLE:
        algorithms["UMAP"] = {
            "description": (
                "Non-linear manifold learning technique that preserves "
                "local and global structure."
            ),
            "type": "Non-linear",
            "supports_explained_variance": False
        }

    return algorithms


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare_dimensionality_data(
    df,
    excluded_columns=None
):

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

    # Convert datetime features into numerical components.
    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    for column in datetime_columns:

        X[f"{column}_year"] = X[column].dt.year
        X[f"{column}_month"] = X[column].dt.month
        X[f"{column}_day"] = X[column].dt.day
        X[f"{column}_dayofweek"] = X[column].dt.dayofweek

        X.drop(columns=[column], inplace=True)

    # Remove entirely empty columns.
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
            "No numerical or categorical features were found."
        )

    # Remove rows that have no usable values.
    X = X.loc[X.notna().any(axis=1)]

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
# PREPROCESSOR
# =========================================================

def build_dimensionality_preprocessor(
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
        raise ValueError("No usable features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# MAIN DIMENSIONALITY REDUCTION
# =========================================================

def run_dimensionality_reduction(
    df,
    algorithm="PCA",
    n_components=2,
    excluded_columns=None,
    kernel="rbf",
    gamma=None,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1
):

    algorithms = get_dimensionality_algorithms()

    if algorithm not in algorithms:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    X, numerical_columns, categorical_columns = (
        prepare_dimensionality_data(
            df,
            excluded_columns=excluded_columns
        )
    )

    if n_components not in [2, 3]:
        raise ValueError(
            "Number of components must be either 2 or 3."
        )

    if n_components >= len(X):
        raise ValueError(
            "Number of components must be smaller than "
            "the number of observations."
        )

    preprocessor = build_dimensionality_preprocessor(
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

    if X_transformed.shape[1] < 2:
        raise ValueError(
            "At least two transformed features are required."
        )

    if X_transformed.shape[0] <= n_components:
        raise ValueError(
            "Not enough observations for the selected components."
        )

    # -----------------------------------------------------
    # MODEL SELECTION
    # -----------------------------------------------------

    if algorithm == "PCA":

        model = PCA(
            n_components=n_components,
            random_state=random_state
        )

        reduced_data = model.fit_transform(X_transformed)

        explained_variance = (
            model.explained_variance_ratio_ * 100
        )

        total_explained_variance = float(
            explained_variance.sum()
        )

        cumulative_variance = np.cumsum(
            explained_variance
        )

    elif algorithm == "Kernel PCA":

        model = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            eigen_solver="arpack",
            random_state=random_state
        )

        reduced_data = model.fit_transform(X_transformed)

        explained_variance = None
        total_explained_variance = None
        cumulative_variance = None

    elif algorithm == "Truncated SVD":

        max_components = min(
            X_transformed.shape[0] - 1,
            X_transformed.shape[1] - 1
        )

        if n_components > max_components:
            raise ValueError(
                "Selected component count is too high for "
                "Truncated SVD and this dataset."
            )

        model = TruncatedSVD(
            n_components=n_components,
            random_state=random_state
        )

        reduced_data = model.fit_transform(X_transformed)

        explained_variance = (
            model.explained_variance_ratio_ * 100
        )

        total_explained_variance = float(
            explained_variance.sum()
        )

        cumulative_variance = np.cumsum(
            explained_variance
        )

    elif algorithm == "t-SNE":

        if len(X_transformed) <= perplexity:
            raise ValueError(
                "t-SNE perplexity must be smaller than "
                "the number of observations."
            )

        model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,
            init="pca",
            random_state=random_state
        )

        reduced_data = model.fit_transform(X_transformed)

        explained_variance = None
        total_explained_variance = None
        cumulative_variance = None

    elif algorithm == "UMAP":

        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP is not installed. Install it using "
                "pip install umap-learn."
            )

        if n_neighbors >= len(X_transformed):
            raise ValueError(
                "UMAP n_neighbors must be smaller than "
                "the number of observations."
            )

        model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )

        reduced_data = model.fit_transform(X_transformed)

        explained_variance = None
        total_explained_variance = None
        cumulative_variance = None

    else:
        raise ValueError("Invalid dimensionality algorithm.")

    # -----------------------------------------------------
    # REDUCED DATAFRAME
    # -----------------------------------------------------

    component_columns = [
        f"Component_{i + 1}"
        for i in range(n_components)
    ]

    reduced_df = pd.DataFrame(
        reduced_data,
        columns=component_columns,
        index=X.index
    )

    # -----------------------------------------------------
    # COMPONENT INFORMATION
    # -----------------------------------------------------

    component_info = None

    if explained_variance is not None:

        component_info = pd.DataFrame({
            "Component": component_columns,
            "Explained Variance (%)": explained_variance,
            "Cumulative Variance (%)": cumulative_variance
        })

    # -----------------------------------------------------
    # MERGED OUTPUT
    # -----------------------------------------------------

    output_df = df.loc[X.index].copy()

    for column in component_columns:
        output_df[column] = reduced_df[column]

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    return {
        "algorithm": algorithm,
        "model": model,
        "preprocessor": preprocessor,
        "reduced_data": reduced_df,
        "output_dataset": output_df,
        "component_info": component_info,
        "explained_variance": explained_variance,
        "total_explained_variance": total_explained_variance,
        "cumulative_variance": cumulative_variance,
        "number_of_components": n_components,
        "training_rows": len(X),
        "original_features": X.shape[1],
        "transformed_features": X_transformed.shape[1],
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "excluded_columns": excluded_columns or [],
        "transformed_data": X_transformed
    }


# =========================================================
# MODEL EXPORT
# =========================================================

def export_dimensionality_model(results):

    export_object = {
        "algorithm": results["algorithm"],
        "model": results["model"],
        "preprocessor": results["preprocessor"],
        "excluded_columns": results["excluded_columns"],
        "number_of_components": results["number_of_components"]
    }

    buffer = BytesIO()

    joblib.dump(export_object, buffer)

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# DATASET EXPORT
# =========================================================

def export_reduced_dataset(results):

    return results["output_dataset"].to_csv(
        index=False
    ).encode("utf-8")


# =========================================================
# COMPONENT SUMMARY
# =========================================================

def get_component_summary(results):

    if results["component_info"] is None:
        return None

    return results["component_info"].copy()
