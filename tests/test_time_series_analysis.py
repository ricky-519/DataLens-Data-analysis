
import numpy as np
import pandas as pd

from modules.time_series_analysis import (
    get_datetime_columns,
    get_numerical_columns,
    prepare_time_series,
    get_time_series_summary,
    get_resampled_series,
    get_rolling_statistics,
    get_growth_analysis,
    perform_seasonal_decomposition,
    get_autocorrelation,
    test_stationarity,
    forecast_time_series,
    generate_time_series_observations
)

# Generate 48 months of sample data
np.random.seed(42)

dates = pd.date_range(
    start="2022-01-01",
    periods=48,
    freq="MS"
)

values = (
    100
    + np.arange(48) * 2
    + 15 * np.sin(2 * np.pi * np.arange(48) / 12)
    + np.random.normal(0, 4, 48)
)

df = pd.DataFrame({
    "date": dates,
    "sales": values,
    "customers": np.random.randint(50, 150, 48)
})

print("\n1. DATE COLUMN DETECTION")
print(get_datetime_columns(df))

print("\n2. NUMERICAL COLUMN DETECTION")
print(get_numerical_columns(df))

print("\n3. TIME SERIES PREPARATION")
series = prepare_time_series(df, "date", "sales")
print(series.head())

print("\n4. TIME SERIES SUMMARY")
print(get_time_series_summary(series))

print("\n5. MONTHLY RESAMPLING")
print(get_resampled_series(series, "MS", "mean").head())

print("\n6. ROLLING STATISTICS")
print(get_rolling_statistics(series, window=3).tail())

print("\n7. GROWTH ANALYSIS")
print(get_growth_analysis(series).tail())

print("\n8. SEASONAL DECOMPOSITION")
decomposition = perform_seasonal_decomposition(series, period=12)
print(decomposition.trend.dropna().head())

print("\n9. AUTOCORRELATION")
autocorrelation = get_autocorrelation(series, lags=12)
print("ACF:", autocorrelation["ACF"])
print("PACF:", autocorrelation["PACF"])

print("\n10. STATIONARITY TESTS")
print(test_stationarity(series))

print("\n11. FORECASTING")
forecast = forecast_time_series(
    series,
    periods=6,
    seasonal=True,
    seasonal_periods=12
)
print(forecast["Forecast"])

print("\n12. AUTOMATED OBSERVATIONS")
print(generate_time_series_observations(series))

print("\nALL TIME-SERIES TESTS COMPLETED SUCCESSFULLY")
