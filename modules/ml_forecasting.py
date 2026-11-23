
"""
DataLens - ML-08: Time-Series Forecasting

Algorithms:
- Naive Forecast
- Seasonal Naive Forecast
- Moving Average
- Simple Exponential Smoothing
- Holt's Linear Trend
- ARIMA
- Seasonal ARIMA (SARIMA)

Uses chronological train/test splitting.
"""

from io import BytesIO
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


FORECAST_ALGORITHMS = {
    "Naive Forecast": {
        "description": "Uses the last observed training value for every forecast."
    },
    "Seasonal Naive Forecast": {
        "description": "Repeats the most recently observed seasonal pattern."
    },
    "Moving Average": {
        "description": "Forecasts using the average of recent observations."
    },
    "Simple Exponential Smoothing": {
        "description": "Weights recent observations more heavily than older observations."
    },
    "Holt's Linear Trend": {
        "description": "Models the level and trend of the time series."
    },
    "ARIMA": {
        "description": "Autoregressive Integrated Moving Average model."
    },
    "SARIMA": {
        "description": "Seasonal ARIMA model for series with recurring patterns."
    },
}


def get_forecasting_algorithms():
    """Return the available forecasting algorithms."""
    return FORECAST_ALGORITHMS.copy()


def prepare_time_series(
    df,
    date_column,
    value_column,
    frequency="D",
    aggregation="mean"
):
    """
    Prepare a regular time series from a DataFrame.

    Duplicate timestamps are aggregated.
    Missing timestamps are time-interpolated.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Please provide a non-empty DataFrame.")

    if date_column not in df.columns:
        raise ValueError("Selected date column does not exist.")

    if value_column not in df.columns:
        raise ValueError("Selected value column does not exist.")

    if date_column == value_column:
        raise ValueError("Date and value columns must be different.")

    data = df[[date_column, value_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna()
    data = data.sort_values(date_column)

    if len(data) < 10:
        raise ValueError(
            "At least 10 valid observations are required for forecasting."
        )

    if aggregation not in ["mean", "sum", "min", "max", "median"]:
        raise ValueError("Unsupported aggregation method.")

    grouped = data.groupby(date_column)[value_column]

    if aggregation == "mean":
        series = grouped.mean()
    elif aggregation == "sum":
        series = grouped.sum()
    elif aggregation == "min":
        series = grouped.min()
    elif aggregation == "max":
        series = grouped.max()
    else:
        series = grouped.median()

    series = series.sort_index()

    try:
        series = series.resample(frequency).mean()
    except Exception as error:
        raise ValueError(
            f"Invalid or unsupported frequency '{frequency}': {error}"
        )

    if series.empty:
        raise ValueError("No observations remain after resampling.")

    series = series.interpolate(method="time", limit_direction="both")

    series = series.dropna()

    if len(series) < 10:
        raise ValueError(
            "At least 10 observations are required after resampling."
        )

    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("The prepared time series contains invalid values.")

    series.name = value_column

    return series.astype(float)


def calculate_forecast_metrics(actual, predicted):
    """Calculate standard forecasting evaluation metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))

    nonzero_mask = actual != 0

    if nonzero_mask.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual[nonzero_mask] - predicted[nonzero_mask])
                    / actual[nonzero_mask]
                )
            ) * 100
        )
    else:
        mape = None

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": rmse,
        "MAPE": mape,
    }


def _naive_forecast(train, steps):
    return np.repeat(float(train.iloc[-1]), steps)


def _seasonal_naive_forecast(train, steps, seasonal_period):
    if seasonal_period < 1:
        raise ValueError("Seasonal period must be at least 1.")

    if len(train) < seasonal_period:
        raise ValueError(
            "Training observations must be greater than or equal "
            "to the seasonal period."
        )

    pattern = train.iloc[-seasonal_period:].to_numpy(dtype=float)

    repeats = int(np.ceil(steps / seasonal_period))

    return np.tile(pattern, repeats)[:steps]


def _moving_average_forecast(train, steps, window):
    if window < 1 or window > len(train):
        raise ValueError(
            "Moving average window must be between 1 "
            "and the number of training observations."
        )

    history = list(train.to_numpy(dtype=float))
    predictions = []

    for _ in range(steps):
        prediction = float(np.mean(history[-window:]))
        predictions.append(prediction)
        history.append(prediction)

    return np.asarray(predictions)


def _fit_model(
    algorithm,
    train,
    seasonal_period=7,
    moving_average_window=3,
    arima_order=(1, 1, 1),
    sarima_order=(1, 1, 1),
    sarima_seasonal_order=(1, 1, 1, 7)
):
    """Fit one forecasting algorithm and return its fitted model."""

    if algorithm in [
        "Naive Forecast",
        "Seasonal Naive Forecast",
        "Moving Average"
    ]:
        return {
            "algorithm": algorithm,
            "training_data": train.copy(),
            "seasonal_period": seasonal_period,
            "moving_average_window": moving_average_window,
        }

    if algorithm == "Simple Exponential Smoothing":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        model = SimpleExpSmoothing(
            train,
            initialization_method="estimated"
        )

        return model.fit(optimized=True)

    if algorithm == "Holt's Linear Trend":
        from statsmodels.tsa.holtwinters import Holt

        model = Holt(
            train,
            initialization_method="estimated"
        )

        return model.fit(optimized=True)

    if algorithm == "ARIMA":
        from statsmodels.tsa.arima.model import ARIMA

        return ARIMA(
            train,
            order=arima_order
        ).fit()

    if algorithm == "SARIMA":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if sarima_seasonal_order[3] < 2:
            raise ValueError(
                "SARIMA seasonal period must be at least 2."
            )

        return SARIMAX(
            train,
            order=arima_order,
            seasonal_order=sarima_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

    raise ValueError(f"Unknown forecasting algorithm: {algorithm}")


def _predict_model(
    algorithm,
    model,
    steps,
    seasonal_period=7,
    moving_average_window=3
):
    """Generate forecasts from a fitted model."""

    if algorithm == "Naive Forecast":
        return _naive_forecast(
            model["training_data"],
            steps
        )

    if algorithm == "Seasonal Naive Forecast":
        return _seasonal_naive_forecast(
            model["training_data"],
            steps,
            seasonal_period
        )

    if algorithm == "Moving Average":
        return _moving_average_forecast(
            model["training_data"],
            steps,
            moving_average_window
        )

    predictions = model.forecast(steps=steps)

    return np.asarray(predictions, dtype=float)


def _future_index(series, periods):
    """Create future timestamps using the series frequency."""

    frequency = series.index.freq

    if frequency is None:
        frequency = pd.infer_freq(series.index)

    if frequency is None:
        raise ValueError(
            "Could not determine the time frequency. "
            "Please select a valid frequency."
        )

    return pd.date_range(
        start=series.index[-1],
        periods=periods + 1,
        freq=frequency
    )[1:]


def run_time_series_forecasting(
    df,
    date_column,
    value_column,
    algorithm="Naive Forecast",
    frequency="D",
    aggregation="mean",
    test_size=0.2,
    forecast_horizon=12,
    seasonal_period=7,
    moving_average_window=3,
    arima_order=(1, 1, 1),
    sarima_order=(1, 1, 1),
    sarima_seasonal_order=(1, 1, 1, 7)
):
    """
    Run chronological model evaluation and future forecasting.

    Returns:
        Prepared time series
        Train and test data
        Test predictions
        Evaluation metrics
        Future forecast
        Fitted model
        Forecast comparison DataFrame
    """

    if algorithm not in FORECAST_ALGORITHMS:
        raise ValueError("Please select a supported forecasting algorithm.")

    if not 0.05 <= test_size <= 0.40:
        raise ValueError("Test size must be between 5% and 40%.")

    if forecast_horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    if seasonal_period < 1:
        raise ValueError("Seasonal period must be at least 1.")

    series = prepare_time_series(
        df=df,
        date_column=date_column,
        value_column=value_column,
        frequency=frequency,
        aggregation=aggregation
    )

    test_count = max(
        1,
        int(np.ceil(len(series) * test_size))
    )

    train = series.iloc[:-test_count]
    test = series.iloc[-test_count:]

    if len(train) < 5:
        raise ValueError(
            "Not enough training observations. "
            "Use more data or reduce the test percentage."
        )

    if algorithm == "Seasonal Naive Forecast":
        if len(train) < seasonal_period:
            raise ValueError(
                "The seasonal period is larger than the training data."
            )

    if algorithm == "Moving Average":
        if moving_average_window > len(train):
            raise ValueError(
                "Moving average window exceeds training data length."
            )

    # Evaluate on the chronological holdout.
    evaluation_model = _fit_model(
        algorithm=algorithm,
        train=train,
        seasonal_period=seasonal_period,
        moving_average_window=moving_average_window,
        arima_order=arima_order,
        sarima_order=sarima_order,
        sarima_seasonal_order=sarima_seasonal_order
    )

    test_predictions = _predict_model(
        algorithm=algorithm,
        model=evaluation_model,
        steps=len(test),
        seasonal_period=seasonal_period,
        moving_average_window=moving_average_window
    )

    test_predictions = pd.Series(
        test_predictions,
        index=test.index,
        name="Predicted"
    )

    metrics = calculate_forecast_metrics(
        test.to_numpy(),
        test_predictions.to_numpy()
    )

    comparison = pd.DataFrame({
        "Actual": test,
        "Predicted": test_predictions
    })

    comparison["Residual"] = (
        comparison["Actual"] - comparison["Predicted"]
    )

    # Refit using all observations before generating future forecasts.
    final_model = _fit_model(
        algorithm=algorithm,
        train=series,
        seasonal_period=seasonal_period,
        moving_average_window=moving_average_window,
        arima_order=arima_order,
        sarima_order=sarima_order,
        sarima_seasonal_order=sarima_seasonal_order
    )

    future_predictions = _predict_model(
        algorithm=algorithm,
        model=final_model,
        steps=forecast_horizon,
        seasonal_period=seasonal_period,
        moving_average_window=moving_average_window
    )

    future_dates = _future_index(
        series,
        forecast_horizon
    )

    future_forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_predictions
    })

    future_forecast["Date"] = pd.to_datetime(
        future_forecast["Date"]
    )

    return {
        "algorithm": algorithm,
        "series": series,
        "train": train,
        "test": test,
        "test_predictions": test_predictions,
        "comparison": comparison,
        "metrics": metrics,
        "model": final_model,
        "future_forecast": future_forecast,
        "forecast_horizon": forecast_horizon,
        "frequency": frequency,
        "date_column": date_column,
        "value_column": value_column,
        "summary": {
            "total_observations": len(series),
            "training_observations": len(train),
            "testing_observations": len(test),
            "forecast_horizon": forecast_horizon,
            "algorithm": algorithm,
        }
    }


def export_forecast_results(results):
    """Export future forecasts as CSV."""
    return results["future_forecast"].to_csv(
        index=False
    ).encode("utf-8")


def export_forecast_comparison(results):
    """Export actual vs predicted test observations."""
    output = results["comparison"].copy()
    output.index.name = "Date"

    return output.to_csv().encode("utf-8")


def export_forecasting_model(results):
    """Export the fitted forecasting model and metadata."""
    buffer = BytesIO()

    package = {
        "algorithm": results["algorithm"],
        "model": results["model"],
        "frequency": results["frequency"],
        "date_column": results["date_column"],
        "value_column": results["value_column"],
        "forecast_horizon": results["forecast_horizon"],
    }

    joblib.dump(package, buffer)
    buffer.seek(0)

    return buffer.getvalue()
