
import pandas as pd
import numpy as np
import plotly.express as px


CHART_TYPES = [
    "Scatter Plot",
    "Line Chart",
    "Bar Chart",
    "Area Chart",
    "Histogram",
    "Box Plot",
    "Violin Plot",
    "Pie Chart",
    "Correlation Heatmap"
]


def get_visualization_columns(df):
    """Return columns grouped by their data types."""

    numerical = df.select_dtypes(include=np.number).columns.tolist()

    categorical = [
        col for col in df.columns
        if col not in numerical
        and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    datetime = [
        col for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    # Detect dates stored as text
    for col in categorical.copy():
        values = df[col].dropna()

        if values.empty:
            continue

        try:
            parsed = pd.to_datetime(
                values,
                errors="coerce",
                format="mixed"
            )
        except (TypeError, ValueError):
            parsed = pd.to_datetime(values, errors="coerce")

        if parsed.notna().mean() >= 0.80:
            datetime.append(col)
            categorical.remove(col)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime,
        "all": df.columns.tolist()
    }


def get_chart_recommendations(df):
    """Suggest suitable chart types based on available columns."""

    columns = get_visualization_columns(df)

    recommendations = []

    if len(columns["numerical"]) >= 2:
        recommendations.append("Scatter Plot")
        recommendations.append("Correlation Heatmap")

    if columns["numerical"]:
        recommendations.extend([
            "Histogram",
            "Box Plot",
            "Violin Plot"
        ])

    if columns["categorical"]:
        recommendations.extend([
            "Bar Chart",
            "Pie Chart"
        ])

    if columns["datetime"] and columns["numerical"]:
        recommendations.extend([
            "Line Chart",
            "Area Chart"
        ])

    return list(dict.fromkeys(recommendations))


def create_visualization(
    df,
    chart_type,
    x_column=None,
    y_column=None,
    color_column=None,
    aggregation="Mean",
    title=None,
    labels=None
):
    """Create an interactive Plotly visualization."""

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    columns = get_visualization_columns(df)

    valid_columns = columns["all"]

    if x_column is not None and x_column not in valid_columns:
        raise ValueError("Invalid X-axis column.")

    if y_column is not None and y_column not in valid_columns:
        raise ValueError("Invalid Y-axis column.")

    if color_column is not None and color_column not in valid_columns:
        raise ValueError("Invalid grouping column.")

    chart_title = title or chart_type

    # Scatter Plot
    if chart_type == "Scatter Plot":

        if not x_column or not y_column:
            raise ValueError("Scatter plot requires X and Y columns.")

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=chart_title,
            labels=labels,
            hover_data=[col for col in df.columns[:5]]
        )

    # Line Chart
    elif chart_type == "Line Chart":

        if not x_column or not y_column:
            raise ValueError("Line chart requires X and Y columns.")

        plot_df = df.copy()

        if x_column in columns["datetime"]:
            plot_df[x_column] = pd.to_datetime(
                plot_df[x_column],
                errors="coerce"
            )
            plot_df = plot_df.sort_values(x_column)

        fig = px.line(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=chart_title,
            labels=labels,
            markers=True
        )

    # Bar Chart
    elif chart_type == "Bar Chart":

        if not x_column:
            raise ValueError("Bar chart requires an X-axis column.")

        if y_column:
            grouped = df.groupby(
                x_column,
                dropna=False
            )[y_column]

            agg = aggregation.lower()

            if agg == "sum":
                plot_df = grouped.sum().reset_index()
            elif agg == "median":
                plot_df = grouped.median().reset_index()
            elif agg == "count":
                plot_df = grouped.count().reset_index()
            elif agg == "max":
                plot_df = grouped.max().reset_index()
            elif agg == "min":
                plot_df = grouped.min().reset_index()
            else:
                plot_df = grouped.mean().reset_index()

            y_axis = y_column

        else:
            plot_df = (
                df.groupby(x_column, dropna=False)
                .size()
                .reset_index(name="Count")
            )
            y_axis = "Count"

        fig = px.bar(
            plot_df,
            x=x_column,
            y=y_axis,
            title=chart_title,
            labels=labels
        )

    # Area Chart
    elif chart_type == "Area Chart":

        if not x_column or not y_column:
            raise ValueError("Area chart requires X and Y columns.")

        plot_df = df.copy()

        if x_column in columns["datetime"]:
            plot_df[x_column] = pd.to_datetime(
                plot_df[x_column],
                errors="coerce"
            )
            plot_df = plot_df.sort_values(x_column)

        fig = px.area(
            plot_df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=chart_title,
            labels=labels
        )

    # Histogram
    elif chart_type == "Histogram":

        if not x_column:
            raise ValueError("Histogram requires an X-axis column.")

        fig = px.histogram(
            df,
            x=x_column,
            color=color_column,
            title=chart_title,
            labels=labels,
            marginal="box"
        )

    # Box Plot
    elif chart_type == "Box Plot":

        if not y_column:
            raise ValueError("Box plot requires a numerical Y-axis column.")

        fig = px.box(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=chart_title,
            labels=labels
        )

    # Violin Plot
    elif chart_type == "Violin Plot":

        if not y_column:
            raise ValueError("Violin plot requires a numerical Y-axis column.")

        fig = px.violin(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            box=True,
            points="outliers",
            title=chart_title,
            labels=labels
        )

    # Pie Chart
    elif chart_type == "Pie Chart":

        if not x_column:
            raise ValueError("Pie chart requires a category column.")

        if y_column:
            plot_df = (
                df.groupby(x_column, dropna=False)[y_column]
                .sum()
                .reset_index()
            )
            values = y_column
        else:
            plot_df = (
                df.groupby(x_column, dropna=False)
                .size()
                .reset_index(name="Count")
            )
            values = "Count"

        fig = px.pie(
            plot_df,
            names=x_column,
            values=values,
            title=chart_title,
            hole=0.35
        )

    # Correlation Heatmap
    elif chart_type == "Correlation Heatmap":

        numerical_columns = columns["numerical"]

        if len(numerical_columns) < 2:
            raise ValueError(
                "Correlation heatmap requires at least two numerical columns."
            )

        correlation = df[numerical_columns].corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=chart_title,
            aspect="auto"
        )

    else:
        raise ValueError("Unsupported chart type.")

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        height=550,
        margin=dict(l=40, r=40, t=70, b=40)
    )

    return fig


def get_chart_data(df, x_column=None, y_column=None):
    """Return selected columns for chart data export."""

    selected = []

    if x_column and x_column in df.columns:
        selected.append(x_column)

    if y_column and y_column in df.columns and y_column not in selected:
        selected.append(y_column)

    if not selected:
        return df.copy()

    return df[selected].copy()
