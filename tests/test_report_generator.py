import pandas as pd

from modules.report_generator import (
    get_report_overview,
    get_report_numerical_summary,
    get_report_categorical_summary,
    get_report_missing_values,
    generate_pdf_report,
)


def main():
    df = pd.read_csv("test_datasets/iris.csv")

    print("\n1. Testing dataset overview")
    print(get_report_overview(df))

    print("\n2. Testing numerical summary")
    print(get_report_numerical_summary(df).head())

    print("\n3. Testing categorical summary")
    print(get_report_categorical_summary(df).head())

    print("\n4. Testing missing-value report")
    print(get_report_missing_values(df))

    print("\n5. Testing PDF generation")

    pdf_bytes = generate_pdf_report(
        df,
        title="DataLens Test Report",
        author="DataLens",
        include_numerical=True,
        include_categorical=True,
        include_missing=True,
        include_insights=True,
        include_charts=True,
    )

    output_path = "test_report.pdf"

    with open(output_path, "wb") as file:
        file.write(pdf_bytes)

    print(f"\nPDF generated successfully: {output_path}")
    print(f"PDF size: {len(pdf_bytes):,} bytes")

    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 1000

    print("\nAll report generator tests passed!")


if __name__ == "__main__":
    main()