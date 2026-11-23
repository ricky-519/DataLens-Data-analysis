import io
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def _safe_text(value):
    """Convert values into safe, readable PDF text."""
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "N/A"
        return f"{value:,.3f}"

    return str(value)


def _make_table(data, col_widths=None):
    """Create a consistently styled ReportLab table."""
    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2EC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F4F7FA")
        ]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return table


def _section_heading(text, styles):
    return Paragraph(text, styles["SectionHeading"])


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

def get_report_overview(df):
    """Generate general dataset information for the report."""
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column for column in df.columns
        if column not in numerical_columns
    ]

    return {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Total cells": total_cells,
        "Missing cells": missing_cells,
        "Complete cells": total_cells - missing_cells,
        "Duplicate rows": int(df.duplicated().sum()),
        "Numerical columns": len(numerical_columns),
        "Categorical / other columns": len(categorical_columns),
        "Memory usage (KB)": round(
            df.memory_usage(deep=True).sum() / 1024, 2
        ),
    }


# ---------------------------------------------------------
# ANALYSIS TABLES
# ---------------------------------------------------------

def get_report_numerical_summary(df):
    """Create numerical descriptive statistics."""
    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.loc[
        :, ~numeric_df.columns.isin([])
    ]

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T
    summary["median"] = numeric_df.median()
    summary["missing"] = numeric_df.isna().sum()

    columns = [
        "count", "mean", "median", "std",
        "min", "25%", "50%", "75%", "max", "missing"
    ]

    return summary[columns].reset_index(names="Column")


def get_report_categorical_summary(df, top_n=5):
    """Summarize categorical columns and their most common values."""
    categorical_columns = [
        column for column in df.columns
        if (
            not pd.api.types.is_numeric_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]

    rows = []

    for column in categorical_columns:
        series = df[column].dropna()
        counts = series.value_counts()

        rows.append({
            "Column": column,
            "Unique values": int(series.nunique()),
            "Most common value": (
                _safe_text(counts.index[0]) if not counts.empty else "N/A"
            ),
            "Most common count": (
                int(counts.iloc[0]) if not counts.empty else 0
            ),
            f"Top {top_n} values": ", ".join(
                f"{_safe_text(value)} ({count})"
                for value, count in counts.head(top_n).items()
            ) if not counts.empty else "No values",
        })

    return pd.DataFrame(rows)


def get_report_missing_values(df):
    """Return columns containing missing values."""
    report = pd.DataFrame({
        "Column": df.columns,
        "Missing count": df.isna().sum().values,
        "Missing percentage": (
            df.isna().mean().values * 100
        )
    })

    return report[report["Missing count"] > 0].sort_values(
        "Missing count", ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------
# CHART GENERATION
# ---------------------------------------------------------

def _create_report_charts(df, output_directory, max_charts=4):
    """Create simple charts to embed in the PDF report."""
    chart_paths = []

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for column in numerical_columns[:max_charts]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()

        if len(values) < 2 or values.nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(values, bins="auto", edgecolor="black", alpha=0.75)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()

        safe_name = f"numerical_{len(chart_paths)}.png"
        path = os.path.join(output_directory, safe_name)
        fig.savefig(path, dpi=130)
        plt.close(fig)

        chart_paths.append(path)

    return chart_paths


# ---------------------------------------------------------
# AUTOMATED INSIGHTS
# ---------------------------------------------------------

def _get_insight_rows(insights_report):
    """Extract insight records from the Automated Insights result."""
    if not isinstance(insights_report, dict):
        return []

    insights = insights_report.get("insights", [])

    if isinstance(insights, pd.DataFrame):
        return insights.to_dict("records")

    if isinstance(insights, list):
        return insights

    return []


# ---------------------------------------------------------
# PDF GENERATION
# ---------------------------------------------------------

def generate_pdf_report(
    df,
    title="DataLens Analysis Report",
    author="DataLens User",
    insights_report=None,
    include_numerical=True,
    include_categorical=True,
    include_missing=True,
    include_insights=True,
    include_charts=True,
):
    """
    Generate a PDF report and return its contents as bytes.

    The original DataFrame is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The report input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot generate a report from an empty dataset.")

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author=author,
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#243B53"),
        spaceAfter=10,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#627D98"),
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#243B53"),
        spaceBefore=13,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="SmallReportText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        alignment=TA_LEFT,
    ))

    story = []

    # Cover / title
    story.append(Paragraph(title, styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated by DataLens | {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        styles["ReportSubtitle"]
    ))
    story.append(Paragraph(
        f"Prepared by: {_safe_text(author)}",
        styles["ReportSubtitle"]
    ))
    story.append(Spacer(1, 12))

    # Dataset overview
    story.append(_section_heading("1. Dataset Overview", styles))

    overview = get_report_overview(df)
    overview_table = [["Metric", "Value"]]

    for key, value in overview.items():
        overview_table.append([key, _safe_text(value)])

    story.append(_make_table(
        overview_table,
        col_widths=[85 * mm, 75 * mm]
    ))

    # Numerical summary
    if include_numerical:
        numerical_summary = get_report_numerical_summary(df)

        story.append(_section_heading(
            "2. Numerical Analysis", styles
        ))

        if numerical_summary.empty:
            story.append(Paragraph(
                "No numerical columns were available for analysis.",
                styles["SmallReportText"]
            ))
        else:
            table_data = [[
                "Column", "Count", "Mean", "Median",
                "Std Dev", "Min", "Max", "Missing"
            ]]

            for _, row in numerical_summary.iterrows():
                table_data.append([
                    _safe_text(row["Column"]),
                    _safe_text(row["count"]),
                    _safe_text(row["mean"]),
                    _safe_text(row["median"]),
                    _safe_text(row["std"]),
                    _safe_text(row["min"]),
                    _safe_text(row["max"]),
                    _safe_text(row["missing"]),
                ])

            story.append(_make_table(
                table_data,
                col_widths=[
                    34 * mm, 17 * mm, 19 * mm, 19 * mm,
                    19 * mm, 17 * mm, 17 * mm, 17 * mm
                ]
            ))

    # Categorical summary
    if include_categorical:
        categorical_summary = get_report_categorical_summary(df)

        story.append(_section_heading(
            "3. Categorical Analysis", styles
        ))

        if categorical_summary.empty:
            story.append(Paragraph(
                "No categorical columns were available for analysis.",
                styles["SmallReportText"]
            ))
        else:
            table_data = [[
                "Column", "Unique", "Most common",
                "Count", "Top values"
            ]]

            for _, row in categorical_summary.iterrows():
                table_data.append([
                    Paragraph(_safe_text(row["Column"]), styles["SmallReportText"]),
                    _safe_text(row["Unique values"]),
                    Paragraph(_safe_text(row["Most common value"]), styles["SmallReportText"]),
                    _safe_text(row["Most common count"]),
                    Paragraph(_safe_text(row["Top 5 values"]), styles["SmallReportText"]),
                ])

            story.append(_make_table(
                table_data,
                col_widths=[
                    30 * mm, 18 * mm, 30 * mm, 17 * mm, 64 * mm
                ]
            ))

    # Missing values
    if include_missing:
        missing_report = get_report_missing_values(df)

        story.append(_section_heading(
            "4. Data Quality – Missing Values", styles
        ))

        if missing_report.empty:
            story.append(Paragraph(
                "No missing values were detected.",
                styles["SmallReportText"]
            ))
        else:
            table_data = [[
                "Column", "Missing count", "Missing percentage"
            ]]

            for _, row in missing_report.iterrows():
                table_data.append([
                    _safe_text(row["Column"]),
                    _safe_text(row["Missing count"]),
                    f"{row['Missing percentage']:.2f}%",
                ])

            story.append(_make_table(
                table_data,
                col_widths=[65 * mm, 40 * mm, 45 * mm]
            ))

        story.append(Paragraph(
            f"Duplicate rows detected: {int(df.duplicated().sum())}",
            styles["SmallReportText"]
        ))

    # Automated insights
    if include_insights:
        insight_rows = _get_insight_rows(insights_report)

        story.append(_section_heading(
            "5. Automated Insights", styles
        ))

        if not insight_rows:
            story.append(Paragraph(
                "No automated insights were supplied for this report. "
                "Generate insights on the Automated Insights page and "
                "then create the report again to include them.",
                styles["SmallReportText"]
            ))
        else:
            for index, insight in enumerate(insight_rows, start=1):
                if not isinstance(insight, dict):
                    continue

                heading = (
                    insight.get("title")
                    or insight.get("category")
                    or f"Insight {index}"
                )

                description = (
                    insight.get("description")
                    or insight.get("message")
                    or ""
                )

                severity = insight.get("severity", "")

                story.append(Paragraph(
                    f"<b>{index}. {_safe_text(heading)}</b>",
                    styles["SmallReportText"]
                ))

                if severity:
                    story.append(Paragraph(
                        f"Severity: {_safe_text(severity)}",
                        styles["SmallReportText"]
                    ))

                if description:
                    story.append(Paragraph(
                        _safe_text(description),
                        styles["SmallReportText"]
                    ))

                story.append(Spacer(1, 5))

    # Charts
    if include_charts:
        story.append(_section_heading(
            "6. Selected Visualizations", styles
        ))

        with tempfile.TemporaryDirectory() as temp_directory:
            chart_paths = _create_report_charts(df, temp_directory)

            if not chart_paths:
                story.append(Paragraph(
                    "No suitable numerical columns were available "
                    "for chart generation.",
                    styles["SmallReportText"]
                ))
            else:
                for chart_path in chart_paths:
                    story.append(Image(
                        chart_path,
                        width=155 * mm,
                        height=77 * mm
                    ))
                    story.append(Spacer(1, 8))

            document.build(story)

    else:
        document.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes