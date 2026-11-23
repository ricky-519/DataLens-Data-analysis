
import pandas as pd

from modules.statistical_analysis import (
    check_normality,
    check_equal_variance,
    one_sample_t_test,
    independent_t_test,
    paired_t_test,
    mann_whitney_test,
    wilcoxon_test,
    confidence_interval_mean,
    confidence_interval_difference,
    linear_regression
)

# Load Iris dataset
df = pd.read_csv("test_datasets/iris.csv")

print("\n========== 1. NORMALITY TEST ==========")
print(check_normality(df, ["sepal_length", "petal_length"]))

print("\n========== 2. EQUAL VARIANCE TEST ==========")
print(
    check_equal_variance(
        df, "sepal_length", "species"
    )
)

print("\n========== 3. ONE-SAMPLE T-TEST ==========")
print(
    one_sample_t_test(
        df, "sepal_length", 5.0
    )
)

print("\n========== 4. INDEPENDENT T-TEST ==========")
print(
    independent_t_test(
        df,
        "sepal_length",
        "species",
        "setosa",
        "versicolor"
    )
)

print("\n========== 5. PAIRED T-TEST ==========")
paired_df = pd.DataFrame({
    "before": [10, 12, 14, 16, 18, 20],
    "after": [12, 13, 15, 18, 19, 22]
})
print(paired_t_test(paired_df, "before", "after"))

print("\n========== 6. MANN-WHITNEY TEST ==========")
print(
    mann_whitney_test(
        df,
        "sepal_length",
        "species",
        "setosa",
        "versicolor"
    )
)

print("\n========== 7. WILCOXON TEST ==========")
print(paired_df)
print(wilcoxon_test(paired_df, "before", "after"))

print("\n========== 8. CONFIDENCE INTERVAL ==========")
print(confidence_interval_mean(df, "sepal_length"))

print("\n========== 9. GROUP DIFFERENCE INTERVAL ==========")
print(
    confidence_interval_difference(
        df,
        "sepal_length",
        "species",
        "setosa",
        "versicolor"
    )
)

print("\n========== 10. LINEAR REGRESSION ==========")
regression = linear_regression(
    df,
    ["sepal_length", "sepal_width"],
    "petal_length"
)

print("R-squared:", regression["R-squared"])
print("Adjusted R-squared:", regression["Adjusted R-squared"])
print("\nCoefficients:")
print(regression["Coefficients"])
print("\nRegression diagnostics:")
print("Residual normality p-value:",
      regression["Residual Normality P-value"])
print("Breusch-Pagan p-value:",
      regression["Breusch-Pagan P-value"])
print("Durbin-Watson:",
      regression["Durbin-Watson Statistic"])

print("\nALL TESTS EXECUTED.")
