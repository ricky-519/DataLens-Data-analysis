
import pandas as pd
import numpy as np


def detect_variable_type(series):
    """Identify a column's analytical variable type."""

    if pd.api.types.is_bool_dtype(series):
        return "Categorical"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"

    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique()

        # Low-cardinality numeric columns may represent categories
        if unique_count <= 10:
            return "Numerical (Low Cardinality)"

        return "Numerical"

    # Attempt to identify dates stored as strings
    if series.notna().any():
        sample = series.dropna().astype(str).head(100)

        try:
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed"
            )

            if parsed.notna().mean() >= 0.8:
                return "Date/Time"
        except (ValueError, TypeError):
            pass

    unique_count = series.nunique()
    non_null_count = series.notna().sum()

    # Identify text-like columns
    if non_null_count > 0:
        average_length = (
            series.dropna().astype(str).str.len().mean()
        )

        if unique_count > 20 and average_length > 40:
            return "Text"

    return "Categorical"


def generate_dataset_description(df):
    """Generate an automatic description of a dataset."""

    rows, columns = df.shape

    numerical_count = 0
    categorical_count = 0
    datetime_count = 0
    text_count = 0

    for column in df.columns:
        variable_type = detect_variable_type(df[column])

        if variable_type.startswith("Numerical"):
            numerical_count += 1
        elif variable_type == "Categorical":
            categorical_count += 1
        elif variable_type == "Date/Time":
            datetime_count += 1
        elif variable_type == "Text":
            text_count += 1

    description = (
        f"The uploaded dataset contains {rows:,} rows "
        f"and {columns:,} columns. "
        f"It has {numerical_count} numerical variables, "
        f"{categorical_count} categorical variables, "
        f"{datetime_count} date/time variables and "
        f"{text_count} text variables."
    )

    if rows == 0 or columns == 0:
        description += (
            " The dataset is empty or has no usable columns."
        )

    return description


def generate_variable_profile(df):
    """Generate detailed information for each variable."""

    profile = []

    for column in df.columns:

        series = df[column]
        variable_type = detect_variable_type(series)

        record = {
            "Variable": str(column),
            "Data Type": str(series.dtype),
            "Variable Type": variable_type,
            "Non-Null Count": int(series.notna().sum()),
            "Missing Count": int(series.isna().sum()),
            "Missing (%)": round(
                series.isna().mean() * 100, 2
            ),
            "Unique Values": int(series.nunique())
        }

        if variable_type.startswith("Numerical"):

            numeric_series = pd.to_numeric(
                series, errors="coerce"
            ).dropna()

            if not numeric_series.empty:

                record["Mean"] = round(
                    numeric_series.mean(), 3
                )

                record["Median"] = round(
                    numeric_series.median(), 3
                )

                record["Std Dev"] = round(
                    numeric_series.std(), 3
                )

                record["Minimum"] = numeric_series.min()
                record["Maximum"] = numeric_series.max()

            else:
                record["Mean"] = None
                record["Median"] = None
                record["Std Dev"] = None
                record["Minimum"] = None
                record["Maximum"] = None

        elif variable_type == "Categorical":

            modes = series.mode(dropna=True)

            record["Most Frequent Value"] = (
                str(modes.iloc[0]) if not modes.empty else "N/A"
            )

        elif variable_type == "Text":

            record["Average Text Length"] = (
                round(
                    series.dropna().astype(str).str.len().mean(),
                    2
                )
                if series.notna().any()
                else 0
            )

        profile.append(record)

    return pd.DataFrame(profile)


def generate_initial_observations(df):
    """Generate basic, rule-based analytical observations."""

    observations = []

    rows, columns = df.shape

    if rows == 0 or columns == 0:
        return ["The dataset is empty or has no usable columns."]

    total_cells = rows * columns
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells > 0 else 0
    )

    observations.append(
        f"The dataset contains {rows:,} records "
        f"and {columns:,} variables."
    )

    observations.append(
        f"Overall missing data: {missing_percentage:.2f}% "
        f"({missing_cells:,} missing cells)."
    )

    if duplicate_rows > 0:
        observations.append(
            f"{duplicate_rows:,} duplicate rows were detected."
        )
    else:
        observations.append(
            "No completely duplicate rows were detected."
        )

    # Missing value observations
    missing_columns = df.columns[df.isna().any()].tolist()

    if missing_columns:
        for column in missing_columns[:5]:
            percentage = df[column].isna().mean() * 100

            observations.append(
                f"'{column}' contains {percentage:.2f}% missing values."
            )
    else:
        observations.append(
            "All columns have complete values."
        )

    # Constant columns
    constant_columns = [
        column for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        observations.append(
            "Constant or entirely empty columns detected: "
            + ", ".join(map(str, constant_columns))
        )

    # Numerical observations
    numerical_df = df.select_dtypes(include="number")

    for column in numerical_df.columns[:5]:

        series = numerical_df[column].dropna()

        if len(series) < 2:
            continue

        if series.nunique() <= 1:
            continue

        skewness = series.skew()

        if pd.notna(skewness):

            if skewness > 1:
                shape = "positively skewed"
            elif skewness < -1:
                shape = "negatively skewed"
            else:
                shape = "not strongly skewed"

            observations.append(
                f"'{column}' is {shape} "
                f"(skewness: {skewness:.2f})."
            )

    if not numerical_df.empty:

        correlation = numerical_df.corr()

        pairs = []

        for i in range(len(correlation.columns)):
            for j in range(i + 1, len(correlation.columns)):

                value = correlation.iloc[i, j]

                if pd.notna(value):
                    pairs.append((
                        abs(value),
                        correlation.columns[i],
                        correlation.columns[j],
                        value
                    ))

        if pairs:
            strongest = max(pairs, key=lambda x: x[0])

            observations.append(
                f"The strongest absolute numerical correlation "
                f"is between '{strongest[1]}' and "
                f"'{strongest[2]}' "
                f"(correlation: {strongest[3]:.2f})."
            )

    return observations


def generate_dataset_health(df):
    """Calculate a basic dataset completeness score."""

    if df.empty or len(df.columns) == 0:
        return {
            "score": 0,
            "status": "No usable data",
            "missing_percentage": 100.0,
            "duplicate_rows": 0
        }

    total_cells = df.shape[0] * df.shape[1]

    missing_cells = int(df.isna().sum().sum())

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells else 100
    )

    duplicate_rows = int(df.duplicated().sum())

    # Completeness score only, not a full data-quality score
    score = round(100 - missing_percentage, 2)

    if score >= 95:
        status = "High completeness"
    elif score >= 80:
        status = "Moderate completeness"
    else:
        status = "Low completeness"

    return {
        "score": score,
        "status": status,
        "missing_percentage": round(
            missing_percentage, 2
        ),
        "duplicate_rows": duplicate_rows
    }
