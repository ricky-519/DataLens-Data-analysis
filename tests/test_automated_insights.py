
import pandas as pd

from modules.automated_insights import (
    get_dataset_overview,
    analyze_missing_values,
    analyze_numerical_insights,
    analyze_categorical_insights,
    analyze_correlation_insights,
    analyze_datetime_insights,
    analyze_data_quality_insights,
    generate_all_insights,
    get_insights_summary
)


# Load the Iris dataset
df = pd.read_csv("test_datasets/iris.csv")

print("\n1. DATASET OVERVIEW")
print(get_dataset_overview(df))

print("\n2. NUMERICAL INSIGHTS")
print(analyze_numerical_insights(df))

print("\n3. CATEGORICAL INSIGHTS")
print(analyze_categorical_insights(df))

print("\n4. CORRELATION INSIGHTS")
print(analyze_correlation_insights(df))

print("\n5. DATETIME INSIGHTS")
print(analyze_datetime_insights(df))

print("\n6. DATA QUALITY INSIGHTS")
print(analyze_data_quality_insights(df))

print("\n7. COMPLETE AUTOMATED INSIGHTS")
report = generate_all_insights(df)

print("Total insights:", report["total_insights"])
print("Summary:", get_insights_summary(report["insights"]))

print("\n8. TEST WITH SALES DATA")
try:
    sales_df = pd.read_csv(
        "test_datasets/datalens_time_series_sales_60_months.csv"
    )

    sales_report = generate_all_insights(sales_df)

    print("Date columns:", sales_report["datetime_columns"])
    print("Total insights:", sales_report["total_insights"])

except FileNotFoundError:
    print("Sales dataset not found in test_datasets; skipping this test.")

print("\nAll automated insight tests executed.")
