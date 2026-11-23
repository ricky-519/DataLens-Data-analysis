
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


# --------------------------------------------------
# 1. GET NUMERICAL COLUMNS
# --------------------------------------------------

def get_multivariate_columns(df):
    """Return numerical columns suitable for multivariate analysis."""

    return df.select_dtypes(
        include="number"
    ).columns.difference(
        df.select_dtypes(include="bool").columns
    ).tolist()


# --------------------------------------------------
# 2. PREPARE NUMERICAL DATA
# --------------------------------------------------

def prepare_multivariate_data(df, columns):
    """
    Prepare numerical data:
    - Convert values to numeric
    - Replace infinite values
    - Remove columns that contain no valid values
    - Impute missing values using column medians
    """

    if not columns:
        raise ValueError("Select at least one numerical column.")

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate columns were selected.")

    if df.columns.duplicated().any():
        raise ValueError(
            "Dataset contains duplicate column names. "
            "Rename duplicate columns before analysis."
        )

    missing_columns = [
        col for col in columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found: {missing_columns}"
        )

    data = df[columns].copy()

    for column in columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace([np.inf, -np.inf], np.nan)

    # Remove columns with no valid numerical values
    data = data.dropna(axis=1, how="all")

    if data.empty or data.shape[1] == 0:
        raise ValueError("No valid numerical data is available.")

    # Remove constant columns
    constant_columns = [
        col for col in data.columns
        if data[col].nunique(dropna=True) <= 1
    ]

    data = data.drop(columns=constant_columns)

    if data.shape[1] == 0:
        raise ValueError(
            "All selected columns are constant or invalid."
        )

    # Fill missing values using median
    data = data.fillna(data.median())

    if data.isna().any().any():
        raise ValueError(
            "Unable to prepare the selected numerical data."
        )

    return data


# --------------------------------------------------
# 3. CORRELATION MATRIX
# --------------------------------------------------

def get_multivariate_correlation(
    df,
    columns,
    method="pearson"
):
    """Calculate correlation between multiple variables."""

    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError("Unsupported correlation method.")

    data = prepare_multivariate_data(df, columns)

    if data.shape[1] < 2:
        raise ValueError(
            "Select at least two valid numerical variables."
        )

    return data.corr(method=method)


# --------------------------------------------------
# 4. VARIANCE INFLATION FACTOR
# --------------------------------------------------

def get_vif_analysis(df, columns):
    """
    Calculate VIF to identify multicollinearity
    among numerical variables.
    """

    data = prepare_multivariate_data(df, columns)

    if data.shape[1] < 2:
        raise ValueError(
            "VIF requires at least two valid numerical variables."
        )

    if len(data) <= data.shape[1]:
        raise ValueError(
            "VIF requires more observations than variables."
        )

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    results = []

    for index, column in enumerate(data.columns):

        try:
            vif_value = variance_inflation_factor(
                scaled_data,
                index
            )

            if np.isinf(vif_value):
                interpretation = "Perfect multicollinearity"

            elif vif_value >= 10:
                interpretation = "High multicollinearity"

            elif vif_value >= 5:
                interpretation = "Moderate multicollinearity"

            else:
                interpretation = "Low multicollinearity"

            results.append({
                "Variable": column,
                "VIF": round(float(vif_value), 3),
                "Interpretation": interpretation
            })

        except (ValueError, np.linalg.LinAlgError):
            results.append({
                "Variable": column,
                "VIF": np.nan,
                "Interpretation": "Unable to calculate"
            })

    return pd.DataFrame(results)


# --------------------------------------------------
# 5. PRINCIPAL COMPONENT ANALYSIS
# --------------------------------------------------

def perform_pca(df, columns, n_components=None):
    """
    Perform PCA on standardized numerical variables.

    Returns:
    - Transformed component scores
    - Explained variance report
    - Component loadings
    - Fitted PCA model
    """

    data = prepare_multivariate_data(df, columns)

    if data.shape[1] < 2:
        raise ValueError(
            "PCA requires at least two valid numerical variables."
        )

    if len(data) < 2:
        raise ValueError(
            "PCA requires at least two observations."
        )

    if n_components is None:
        n_components = min(data.shape[1], data.shape[0])

    if (
        not isinstance(n_components, int)
        or n_components < 1
        or n_components > min(data.shape)
    ):
        raise ValueError(
            "Invalid number of principal components."
        )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)

    transformed_data = pca.fit_transform(scaled_data)

    component_names = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    scores = pd.DataFrame(
        transformed_data,
        columns=component_names,
        index=data.index
    )

    explained_variance = pd.DataFrame({
        "Component": component_names,
        "Explained Variance Ratio": (
            pca.explained_variance_ratio_
        ),
        "Explained Variance (%)": (
            pca.explained_variance_ratio_ * 100
        ).round(2),
        "Cumulative Variance (%)": (
            np.cumsum(pca.explained_variance_ratio_) * 100
        ).round(2)
    })

    loadings = pd.DataFrame(
        pca.components_.T,
        index=data.columns,
        columns=component_names
    )

    loadings.index.name = "Variable"

    return {
        "scores": scores,
        "explained_variance": explained_variance,
        "loadings": loadings,
        "model": pca,
        "scaler": scaler
    }


# --------------------------------------------------
# 6. PAIRWISE RELATIONSHIP DATA
# --------------------------------------------------

def get_pairwise_data(df, columns):
    """Prepare data for pairwise relationship visualization."""

    data = prepare_multivariate_data(df, columns)

    if data.shape[1] < 2:
        raise ValueError(
            "Pairwise analysis requires at least two variables."
        )

    return data


# --------------------------------------------------
# 7. MULTIVARIATE OBSERVATIONS
# --------------------------------------------------

def get_multivariate_observations(correlation_matrix):
    """Generate observations from a correlation matrix."""

    observations = []

    if correlation_matrix.shape[1] < 2:
        return ["At least two variables are required."]

    columns = correlation_matrix.columns

    strongest_positive = None
    strongest_negative = None

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            value = correlation_matrix.iloc[i, j]

            if pd.isna(value):
                continue

            pair = (
                columns[i],
                columns[j],
                float(value)
            )

            if value > 0:
                if (
                    strongest_positive is None
                    or value > strongest_positive[2]
                ):
                    strongest_positive = pair

            elif value < 0:
                if (
                    strongest_negative is None
                    or value < strongest_negative[2]
                ):
                    strongest_negative = pair

    if strongest_positive is not None:

        observations.append(
            f"The strongest positive correlation is between "
            f"{strongest_positive[0]} and {strongest_positive[1]} "
            f"({strongest_positive[2]:.3f})."
        )

    if strongest_negative is not None:

        observations.append(
            f"The strongest negative correlation is between "
            f"{strongest_negative[0]} and {strongest_negative[1]} "
            f"({strongest_negative[2]:.3f})."
        )

    if not observations:
        observations.append(
            "No meaningful pairwise correlations were identified."
        )

    observations.append(
        "Correlation does not establish causation."
    )

    return observations


# --------------------------------------------------
# 8. PCA SUMMARY
# --------------------------------------------------

def get_pca_observations(pca_result):
    """Generate observations from PCA explained variance."""

    variance_df = pca_result["explained_variance"]

    if variance_df.empty:
        return ["No PCA results are available."]

    first_component = variance_df.iloc[0]

    total_variance = variance_df.iloc[-1][
        "Cumulative Variance (%)"
    ]

    observations = [
        (
            f"{first_component['Component']} explains "
            f"{first_component['Explained Variance (%)']:.2f}% "
            "of the total variance."
        ),
        (
            f"The selected components collectively explain "
            f"{total_variance:.2f}% of the total variance."
        )
    ]

    return observations
