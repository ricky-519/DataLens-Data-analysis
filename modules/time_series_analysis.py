
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing



def get_datetime_columns(df):
    """Detect datetime columns, including dates stored as strings."""
    datetime_columns = []

    for column in df.columns:
        series = df[column]

        # Already stored as a datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        # Ignore columns with no usable values
        non_null = series.dropna()

        if non_null.empty:
            continue

        # Try parsing date strings
        try:
            parsed = pd.to_datetime(
                non_null,
                errors="coerce",
                format="mixed"
            )
        except (TypeError, ValueError):
            parsed = pd.to_datetime(
                non_null,
                errors="coerce"
            )

        valid_ratio = parsed.notna().mean()

        if valid_ratio >= 0.80:
            datetime_columns.append(column)

    return datetime_columns


def get_numerical_columns(df):
    """Return numerical columns, excluding Boolean columns."""
    return [
        column for column in df.select_dtypes(include=np.number).columns
        if not pd.api.types.is_bool_dtype(df[column])
    ]


def prepare_time_series(df, date_column, value_column, frequency=None):
    """Prepare and aggregate a time series using a date and numerical column."""
    if date_column not in df.columns or value_column not in df.columns:
        raise ValueError("Select valid date and numerical columns.")

    data = df[[date_column, value_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column], errors="coerce", format="mixed"
    )
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values(date_column)

    if data.empty:
        raise ValueError("No valid date and numerical observations were found.")

    series = data.set_index(date_column)[value_column]

    if frequency:
        series = series.resample(frequency).mean()

    if series.empty:
        raise ValueError("No observations are available at the selected frequency.")

    return series


def get_time_series_summary(series):
    """Calculate basic time-series statistics."""
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        raise ValueError("The time series contains no valid numerical values.")

    return {
        "Observations": len(clean),
        "Start Date": clean.index.min(),
        "End Date": clean.index.max(),
        "Mean": clean.mean(),
        "Median": clean.median(),
        "Standard Deviation": clean.std(),
        "Minimum": clean.min(),
        "Maximum": clean.max(),
        "Total": clean.sum(),
        "Missing Values": int(series.isna().sum()),
        "Unique Values": int(clean.nunique())
    }


def get_resampled_series(series, frequency="MS", aggregation="mean"):
    """Resample a time series using a selected aggregation."""
    if aggregation == "sum":
        result = series.resample(frequency).sum()
    elif aggregation == "median":
        result = series.resample(frequency).median()
    elif aggregation == "min":
        result = series.resample(frequency).min()
    elif aggregation == "max":
        result = series.resample(frequency).max()
    else:
        result = series.resample(frequency).mean()

    return result


def get_rolling_statistics(series, window=3):
    """Calculate moving average and rolling standard deviation."""
    if window < 1:
        raise ValueError("Rolling window must be at least 1.")

    return pd.DataFrame({
        "Original": series,
        "Moving Average": series.rolling(window=window).mean(),
        "Rolling Standard Deviation": series.rolling(window=window).std()
    })


def get_growth_analysis(series):
    """Calculate period-over-period changes and percentage growth."""
    return pd.DataFrame({
        "Value": series,
        "Absolute Change": series.diff(),
        "Percentage Change": series.pct_change(fill_method=None) * 100
    })


def perform_seasonal_decomposition(series, period=12, model="additive"):
    """Separate a time series into trend, seasonal and residual components."""
    clean = series.dropna()

    if period < 2:
        raise ValueError("Seasonal period must be at least 2.")

    if len(clean) < period * 2:
        raise ValueError(
            f"At least {period * 2} observations are required "
            "for seasonal decomposition."
        )

    if clean.index.has_duplicates:
        raise ValueError("The time series must have a unique date index.")

    return seasonal_decompose(
        clean,
        model=model,
        period=period,
        extrapolate_trend="freq"
    )


def get_autocorrelation(series, lags=20):
    """Calculate ACF and PACF values."""
    clean = series.dropna()

    if len(clean) < 4:
        raise ValueError("At least 4 observations are required.")

    maximum_lags = min(lags, len(clean) // 2 - 1)

    if maximum_lags < 1:
        raise ValueError("Not enough observations for the selected lag count.")

    return {
        "ACF": acf(clean, nlags=maximum_lags),
        "PACF": pacf(clean, nlags=maximum_lags, method="ywm"),
        "Lags": list(range(maximum_lags + 1))
    }


def test_stationarity(series):
    """Run Augmented Dickey-Fuller and KPSS stationarity tests."""
    clean = series.dropna()

    if len(clean) < 10:
        raise ValueError("At least 10 observations are required.")

    if clean.nunique() < 2:
        raise ValueError("Stationarity tests require varying values.")

    adf_result = adfuller(clean, autolag="AIC")
    kpss_result = kpss(clean, regression="c", nlags="auto")

    return {
        "ADF Statistic": adf_result[0],
        "ADF p-value": adf_result[1],
        "ADF Stationary": adf_result[1] < 0.05,
        "KPSS Statistic": kpss_result[0],
        "KPSS p-value": kpss_result[1],
        "KPSS Stationary": kpss_result[1] >= 0.05
    }


def forecast_time_series(series, periods=6, seasonal=False, seasonal_periods=12):
    """Generate forecasts using Holt or Holt-Winters exponential smoothing."""
    clean = series.dropna().astype(float)

    if periods < 1:
        raise ValueError("Forecast periods must be at least 1.")

    if len(clean) < 4:
        raise ValueError("At least 4 observations are required for forecasting.")

    if seasonal:
        if seasonal_periods < 2 or len(clean) < seasonal_periods * 2:
            raise ValueError(
                "Seasonal forecasting requires at least two complete seasons."
            )

        model = ExponentialSmoothing(
            clean,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
    else:
        model = ExponentialSmoothing(
            clean,
            trend="add",
            initialization_method="estimated"
        )

    fitted_model = model.fit(optimized=True)
    forecast = fitted_model.forecast(periods)

    return {
        "Forecast": forecast,
        "Fitted Values": fitted_model.fittedvalues,
        "Model": fitted_model
    }


def generate_time_series_observations(series):
    """Generate basic observations from a time series."""
    clean = series.dropna()

    if len(clean) < 2:
        return ["Not enough observations to generate trend observations."]

    first_value = clean.iloc[0]
    last_value = clean.iloc[-1]

    observations = []

    if last_value > first_value:
        observations.append("The final observed value is higher than the initial value.")
    elif last_value < first_value:
        observations.append("The final observed value is lower than the initial value.")
    else:
        observations.append("The initial and final observed values are equal.")

    if clean.std() > 0:
        observations.append(
            f"The series has a standard deviation of {clean.std():.3f}."
        )

    if clean.is_monotonic_increasing:
        observations.append("The series is consistently non-decreasing.")
    elif clean.is_monotonic_decreasing:
        observations.append("The series is consistently non-increasing.")

    return observations
