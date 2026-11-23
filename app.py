import html
import numpy as np
from scipy import stats
import streamlit as st
import pandas as pd
from ui.machine_learning import render_machine_learning

from modules.data_loader import load_dataset
from modules.data_profiler import (
    get_column_profile,
    get_dataset_summary
)

from modules.data_description import (
    generate_dataset_description,
    generate_variable_profile,
    generate_initial_observations,
    generate_dataset_health
)

from modules.numerical_analysis import (
    get_numerical_columns,
    get_numerical_summary,
    get_variable_statistics,
    get_numerical_distribution
)

from modules.categorical_analysis import (
    get_categorical_columns,
    get_frequency_distribution,
    get_categorical_summary,
    get_category_statistics,
    get_category_observations
)

from modules.data_quality import (
    get_missing_value_report,
    get_duplicate_report,
    get_infinite_value_report,
    get_constant_columns,
    get_consistency_report,
    get_data_quality_summary
)

from modules.data_cleaning import (
    remove_missing_rows,
    fill_missing_values,
    remove_duplicate_rows,
    trim_text_values,
    standardise_text_case,
    convert_column_type,
    rename_column,
    get_cleaning_summary
)

from modules.data_cleaning import (
    scan_dataset_for_cleaning,
    apply_automatic_cleaning
)

from modules.outlier_analysis import (
    get_numerical_columns,
    get_outlier_summary,
    get_outlier_details,
    treat_outliers
)

from modules.distribution_analysis import (
    detect_variable_type,
    get_variable_type_report,
    get_columns_by_type,
    get_distribution_statistics,
    test_normality,
    compare_with_normal_distribution,
    get_categorical_distribution,
    get_categorical_statistics,
    get_datetime_distribution,
    get_datetime_statistics,
    get_dataset_distribution_summary,
    get_distribution_report,
    get_datetime_values
)

from modules.univariate_analysis import (
    analyze_numerical_variable,
    analyze_categorical_variable,
    analyze_datetime_variable,
    get_numerical_observations,
    get_categorical_observations,
    get_categorical_frequency,
    get_datetime_observations,
    generate_univariate_report
)

from modules.bivariate_analysis import (
    analyze_numerical_relationship,
    get_correlation_matrix,
    analyze_categorical_relationship,
    analyze_numerical_by_category,
    analyze_datetime_numerical,
    analyze_datetime_categorical,
    get_relationship_observations
)

from modules.group_analysis import (
    get_groupable_columns,
    get_numerical_columns,
    get_categorical_columns,
    get_group_summary,
    get_group_frequency,
    get_multi_group_summary,
    rank_groups,
    get_group_sizes,
    get_group_observations,
    generate_group_report
)

from modules.multivariate_analysis import (
    get_multivariate_columns,
    prepare_multivariate_data,
    get_multivariate_correlation,
    get_vif_analysis,
    perform_pca,
    get_pairwise_data,
    get_multivariate_observations,
    get_pca_observations
)

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

# Reuse existing Phase 12 implementations
from modules.bivariate_analysis import (
    analyze_numerical_by_category,
    analyze_categorical_relationship
)

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

from modules.visualization_studio import (
    CHART_TYPES,
    get_visualization_columns,
    get_chart_recommendations,
    create_visualization,
    get_chart_data
)

from modules.automated_insights import (
    generate_all_insights,
    get_insights_dataframe,
    get_insights_summary
)

from modules.report_generator import generate_pdf_report

# ==========================================
# DATALENS PREMIUM UI
# ==========================================

from ui.theme import inject_theme
from ui.sidebar import render_sidebar
from ui.dashboard import render_dashboard
from ui.components import page_header
import textwrap

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DataLens",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. DATALENS PREMIUM UI
# ==========================================

# Apply the selected theme before rendering the interface.
inject_theme()

# ------------------------------------------------------------------
# DATALENS UI REDESIGN: UI-05 TO UI-16
# Shared visual system for every analysis page. All existing analysis
# functions, calculations, session-state keys and exports are retained.
# ------------------------------------------------------------------

def section_heading(title, caption=None):
    """Render a consistent section heading across all DataLens pages."""
    safe_title = html.escape(str(title))
    safe_caption = html.escape(str(caption)) if caption else ""
    caption_html = (
        f'<p class="dl-section-caption">{safe_caption}</p>'
        if safe_caption else ""
    )
    st.markdown(
        f'<div class="dl-section-block"><h3>{safe_title}</h3>{caption_html}</div>',
        unsafe_allow_html=True,
    )


def page_intro(text):
    """Consistent supporting copy beneath page headers."""
    st.markdown(f'<div class="dl-page-intro">{html.escape(str(text))}</div>', unsafe_allow_html=True)


st.markdown("""
<style>
/* Global canvas and content rhythm */
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1500px;
    padding-top: 2.0rem;
    padding-bottom: 3rem;
}
[data-testid="stVerticalBlock"] { gap: 1rem; }
[data-testid="stMarkdownContainer"] p { line-height: 1.62; }
.dl-page-intro {
    color: var(--dl-muted); font-size: .98rem; line-height: 1.65;
    margin: -8px 0 22px; max-width: 900px;
}
.dl-section-block {
    margin: 1.55rem 0 .85rem; padding: 0 0 .65rem;
    border-bottom: 1px solid var(--dl-border);
}
.dl-section-block h3 {
    color: var(--dl-text); font-size: 1.12rem; font-weight: 680;
    letter-spacing: -.018em; margin: 0;
}
.dl-section-caption { color: var(--dl-muted); font-size: .84rem; margin: .35rem 0 0; }
/* Metrics and cards */
[data-testid="stMetric"] {
    background: var(--dl-surface); border: 1px solid var(--dl-border);
    border-radius: 14px; padding: 16px 18px; box-shadow: var(--dl-shadow);
}
[data-testid="stMetricLabel"] p { color: var(--dl-muted) !important; font-size: .78rem !important; }
[data-testid="stMetricValue"] { color: var(--dl-text) !important; font-weight: 720; }
[data-testid="stMetricDelta"] { font-size: .78rem; }
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; border-bottom: 1px solid var(--dl-border);
}
.stTabs [data-baseweb="tab"] {
    height: 43px; border-radius: 9px 9px 0 0; padding: 0 15px;
    color: var(--dl-muted); font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: var(--dl-accent) !important;
    border-bottom: 2px solid var(--dl-accent) !important;
}
/* Inputs and upload surfaces */
[data-testid="stFileUploader"] section {
    background: var(--dl-surface); border: 1.5px dashed var(--dl-border);
    border-radius: 16px; padding: 22px; transition: border-color .18s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--dl-accent); }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"] {
    border-radius: 10px !important;
}
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: var(--dl-text); }
/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; min-height: 42px; font-weight: 620;
    border: 1px solid var(--dl-border); transition: transform .16s ease, box-shadow .16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: var(--dl-shadow); border-color: var(--dl-accent);
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    border: 0; background: linear-gradient(110deg, var(--dl-accent), var(--dl-accent2));
    color: white;
}
/* Dataframes and charts */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--dl-border); border-radius: 12px; overflow: hidden;
}
[data-testid="stPlotlyChart"] { border-radius: 14px; overflow: hidden; }
/* Expanders and messages */
[data-testid="stExpander"] {
    border: 1px solid var(--dl-border); border-radius: 12px;
    background: var(--dl-surface);
}
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: var(--dl-border) !important; }
/* Page-specific layout refinements */
[data-testid="stFileUploader"] { margin: 14px 0 20px; }
[data-testid="stForm"] {
    background: var(--dl-surface); border: 1px solid var(--dl-border);
    border-radius: 14px; padding: 18px;
}
@media (max-width: 768px) {
    [data-testid="stAppViewContainer"] .main .block-container { padding: 1rem .85rem 2rem; }
    [data-testid="stMetric"] { padding: 12px; }
    .dl-section-block { margin-top: 1.15rem; }
}
</style>
""", unsafe_allow_html=True)

# Render the redesigned sidebar and retrieve the selected page.
page = render_sidebar()
st.markdown(f'<div class="dl-current-page" data-page="{html.escape(str(page))}"></div>', unsafe_allow_html=True)


# ==========================================
# 3. HOME DASHBOARD
# ==========================================

if page == "Home":
    render_dashboard()


# ==========================================
# 4. DATASET UPLOAD MODULE
# ==========================================

# ==========================================
# 4. DATASET UPLOAD MODULE
# ==========================================

elif page == "Dataset Upload":

    page_header("📁 Dataset Upload", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Upload a CSV, Excel or JSON dataset "
        "to begin exploring and performing analysis."
    )

    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, XLSX and JSON"
    )

    if uploaded_file is not None:

        try:

            df = load_dataset(uploaded_file)

            # Store dataset in session state
            st.session_state["dataset"] = df
            st.session_state["dataset_name"] = uploaded_file.name

            st.success("Dataset uploaded successfully!")

            # Dataset information
            section_heading("Dataset Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Rows",
                    f"{df.shape[0]:,}"
                )

            with col2:
                st.metric(
                    "Total Columns",
                    f"{df.shape[1]:,}"
                )

            with col3:
                st.metric(
                    "File Size",
                    f"{uploaded_file.size / 1024:.2f} KB"
                )

            st.write(
                "**File Name:**",
                uploaded_file.name
            )

            st.divider()

            # Dataset preview
            section_heading("Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.divider()

            # Column information
            section_heading("Column Information")

            column_info = pd.DataFrame({
                "Column Name": df.columns.astype(str),
                "Data Type": df.dtypes.astype(str).values,
                "Non-Null Values": df.notna().sum().values,
                "Missing Values": df.isna().sum().values
            })

            st.dataframe(
                column_info,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "Open Dataset Explorer from the sidebar "
                "to explore your dataset in detail."
            )

        except Exception as e:

            st.error(f"Dataset upload failed: {e}")


# ==========================================
# 5. DATASET EXPLORER MODULE
# ==========================================

elif page == "Dataset Explorer":

    page_header("🔍 Dataset Explorer", eyebrow="DATALENS WORKSPACE")
    # Check whether a dataset exists
    if "dataset" not in st.session_state:

        st.warning(
            "No dataset found. Please upload a dataset "
            "from the Dataset Upload module first."
        )

    else:

        df = st.session_state["dataset"]

        dataset_name = st.session_state.get(
            "dataset_name",
            "Uploaded Dataset"
        )

        st.write(f"**Current Dataset:** {dataset_name}")

        # Generate dataset summary
        summary = get_dataset_summary(df)

        # ======================================
        # 5.1 DATASET OVERVIEW
        # ======================================

        section_heading("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Rows",
                f"{summary['Rows']:,}"
            )

        with col2:
            st.metric(
                "Total Columns",
                f"{summary['Columns']:,}"
            )

        with col3:
            st.metric(
                "Missing Cells",
                f"{summary['Missing Cells']:,}"
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                f"{summary['Duplicate Rows']:,}"
            )

        col5, col6, col7 = st.columns(3)

        with col5:
            st.metric(
                "Numerical Columns",
                summary["Numerical Columns"]
            )

        with col6:
            st.metric(
                "Categorical Columns",
                summary["Categorical Columns"]
            )

        with col7:
            st.metric(
                "Memory Usage",
                f"{summary['Memory Usage (KB)']} KB"
            )

        st.divider()

        # ======================================
        # 5.2 EXPLORATION TABS
        # ======================================

        tab1, tab2, tab3, tab4 = st.tabs([
            "Column Profile",
            "Dataset Preview",
            "Statistical Summary",
            "Column Explorer"
        ])

        # ======================================
        # TAB 1: COLUMN PROFILE
        # ======================================

        with tab1:

            section_heading("Column Information")

            profile = get_column_profile(df)

            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "View data types, variable types, "
                "missing values and unique values."
            )

        # ======================================
        # TAB 2: DATASET PREVIEW
        # ======================================

        with tab2:

            section_heading("Dataset Preview")

            total_rows = len(df)

            if total_rows == 0:

                st.warning("The dataset contains no records.")

            else:

                # Corrected slider logic
                max_rows = min(100, total_rows)

                if max_rows > 5:

                    rows_to_display = st.slider(
                        "Number of rows to display",
                        min_value=5,
                        max_value=max_rows,
                        value=min(10, max_rows)
                    )

                else:

                    rows_to_display = max_rows

                    st.caption(
                        f"Displaying all {max_rows} available rows."
                    )

                st.dataframe(
                    df.head(rows_to_display),
                    use_container_width=True
                )

                st.caption(
                    f"Showing {rows_to_display} of "
                    f"{total_rows} total rows."
                )

        # ======================================
        # TAB 3: STATISTICAL SUMMARY
        # ======================================

        with tab3:

            section_heading("Numerical Summary")

            numerical_df = df.select_dtypes(
                include="number"
            )

            if not numerical_df.empty:

                numerical_summary = numerical_df.describe().T

                st.dataframe(
                    numerical_summary,
                    use_container_width=True
                )

            else:

                st.info(
                    "No numerical columns found in this dataset."
                )

            st.divider()

            section_heading("Categorical Summary")

            categorical_df = df.select_dtypes(
                include=["object", "category", "bool"]
            )

            if not categorical_df.empty:

                categorical_summary = categorical_df.describe().T

                st.dataframe(
                    categorical_summary,
                    use_container_width=True
                )

            else:

                st.info(
                    "No categorical columns found in this dataset."
                )

        # ======================================
        # TAB 4: INDIVIDUAL COLUMN EXPLORER
        # ======================================

        with tab4:

            section_heading("Explore Individual Columns")

            if len(df.columns) == 0:

                st.warning("No columns available to explore.")

            else:

                selected_column = st.selectbox(
                    "Select a column",
                    df.columns.tolist(),
                    format_func=str
                )

                series = df[selected_column]

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Data Type",
                        str(series.dtype)
                    )

                with col2:

                    st.metric(
                        "Unique Values",
                        series.nunique()
                    )

                with col3:

                    st.metric(
                        "Missing Values",
                        int(series.isna().sum())
                    )

                st.divider()

                # Distinct values
                st.write("### Distinct Values")

                unique_values = series.dropna().unique()

                if len(unique_values) > 0:

                    st.write([
                        str(value)
                        for value in unique_values[:50]
                    ])

                    if len(unique_values) > 50:

                        st.caption(
                            f"Showing first 50 of "
                            f"{len(unique_values)} unique values."
                        )

                else:

                    st.info("No non-missing values available.")

                st.divider()

                # Frequency analysis for non-numerical columns
                if (
                    pd.api.types.is_object_dtype(series)
                    or pd.api.types.is_categorical_dtype(series)
                    or pd.api.types.is_bool_dtype(series)
                ):

                    st.write("### Value Frequency")

                    frequency = (
                        series.value_counts(dropna=False)
                        .head(20)
                        .rename_axis("Value")
                        .reset_index(name="Count")
                    )

                    st.dataframe(
                        frequency,
                        use_container_width=True,
                        hide_index=True
                    )

                # Numerical statistics
                elif pd.api.types.is_numeric_dtype(series):

                    st.write("### Numerical Statistics")

                    st.dataframe(
                        series.describe().to_frame(
                            name="Value"
                        ),
                        use_container_width=True
                    )

                # Date/time columns
                elif pd.api.types.is_datetime64_any_dtype(series):

                    st.write("### Date/Time Statistics")

                    st.write(series.describe())


        # ======================================
        # 6. COMPLETION MESSAGE
        # ======================================

        st.divider()

        st.success(
            "Dataset Explorer loaded successfully!"
        )

        st.caption(
            "DataLens | Phase 2: Dataset Upload | "
            "Phase 3: Dataset Explorer"
        )
        
# ==========================================
# 6. DATASET DESCRIPTION & PROFILING
# ==========================================

elif page == "Dataset Description & Profiling":

    page_header("📋 Dataset Description & Profiling", eyebrow="DATALENS WORKSPACE")
    if "dataset" not in st.session_state:

        st.warning(
            "Please upload a dataset first "
            "using the Dataset Upload module."
        )

    else:

        df = st.session_state["dataset"]

        dataset_name = st.session_state.get(
            "dataset_name",
            "Uploaded Dataset"
        )

        st.write(f"**Dataset:** {dataset_name}")

        st.divider()

        # ----------------------------------
        # 6.1 AUTOMATIC DATASET DESCRIPTION
        # ----------------------------------

        section_heading("📝 Automatic Dataset Description")

        description = generate_dataset_description(df)

        st.info(description)

        st.divider()

        # ----------------------------------
        # 6.2 DATASET HEALTH
        # ----------------------------------

        section_heading("❤️ Dataset Completeness")

        health = generate_dataset_health(df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Completeness Score",
                f"{health['score']:.2f}%"
            )

        with col2:
            st.metric(
                "Missing Data",
                f"{health['missing_percentage']:.2f}%"
            )

        with col3:
            st.metric(
                "Duplicate Rows",
                health["duplicate_rows"]
            )

        st.write(
            f"**Completeness Status:** {health['status']}"
        )

        st.progress(
            health["score"] / 100
        )

        st.caption(
            "This score measures non-missing cells only. "
            "It is not a complete data-quality assessment."
        )

        st.divider()

        # ----------------------------------
        # 6.3 VARIABLE CLASSIFICATION
        # ----------------------------------

        section_heading("🔍 Variable Classification")

        variable_profile = generate_variable_profile(df)

        st.dataframe(
            variable_profile,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------
        # 6.4 VARIABLE TYPE DISTRIBUTION
        # ----------------------------------

        section_heading("📊 Variable Type Distribution")

        type_counts = (
            variable_profile["Variable Type"]
            .value_counts()
        )

        if not type_counts.empty:

            chart_data = type_counts.rename_axis(
                "Variable Type"
            ).reset_index(name="Number of Columns")

            st.bar_chart(
                chart_data,
                x="Variable Type",
                y="Number of Columns"
            )

        else:

            st.info("No variables available for classification.")

        st.divider()

        # ----------------------------------
        # 6.5 INITIAL DATASET OBSERVATIONS
        # ----------------------------------

        section_heading("💡 Initial Dataset Observations")

        observations = generate_initial_observations(df)

        for index, observation in enumerate(observations, start=1):

            st.write(f"**{index}.** {observation}")

        st.divider()

        # ----------------------------------
        # 6.6 DETAILED PROFILE
        # ----------------------------------

        section_heading("📑 Detailed Variable Profile")

        selected_variable = st.selectbox(
            "Select a variable for detailed profiling",
            df.columns.tolist(),
            key="phase4_selected_variable"
        )

        selected_series = df[selected_variable]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Values",
            len(selected_series)
        )

        col2.metric(
            "Unique Values",
            selected_series.nunique()
        )

        col3.metric(
            "Missing Values",
            int(selected_series.isna().sum())
        )

        st.write("**Variable Statistics**")

        if pd.api.types.is_numeric_dtype(selected_series):

            st.dataframe(
                selected_series.describe().to_frame(
                    name="Statistics"
                ),
                use_container_width=True
            )

        else:

            st.write(
                selected_series.describe().to_frame(
                    name="Statistics"
                )
            )

        st.success(
            "Dataset Description & Profiling completed successfully!"
        )

elif page == "Numerical Variable Analysis":

    page_header("📊 Numerical Variable Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Explore descriptive statistics and distributions "
        "of numerical variables in your dataset."
    )

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["dataset"]

        numerical_columns = get_numerical_columns(df)

        if not numerical_columns:
            st.warning(
                "No numerical variables were found in this dataset."
            )
        else:
            st.success(
                f"Found {len(numerical_columns)} numerical variables."
            )

            # Overall numerical summary
            section_heading("1. Numerical Summary")

            summary = get_numerical_summary(df)

            st.dataframe(
                summary.round(3),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Numerical Summary (CSV)",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="datalens_numerical_summary.csv",
                mime="text/csv"
            )

            st.divider()

            # Individual variable analysis
            section_heading("2. Individual Variable Analysis")

            selected_column = st.selectbox(
                "Select a numerical variable",
                numerical_columns,
                key="phase5_selected_variable"
            )

            series = get_numerical_distribution(df, selected_column)
            statistics = get_variable_statistics(df, selected_column)

            if series.empty:
                st.warning(
                    "This variable does not contain usable numerical values."
                )
            else:
                st.write(f"**Selected Variable:** {selected_column}")

                # Key metrics
                metric1, metric2, metric3, metric4 = st.columns(4)

                metric1.metric(
                    "Mean",
                    f"{statistics['Mean']:.3f}"
                )
                metric2.metric(
                    "Median",
                    f"{statistics['Median']:.3f}"
                )
                metric3.metric(
                    "Minimum",
                    f"{statistics['Minimum']:.3f}"
                )
                metric4.metric(
                    "Maximum",
                    f"{statistics['Maximum']:.3f}"
                )

                # Detailed statistics
                section_heading("3. Detailed Statistical Measures")

                statistics_df = pd.DataFrame(
                    list(statistics.items()),
                    columns=["Statistical Measure", "Value"]
                )

                st.dataframe(
                    statistics_df.round(3),
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()

                # Distribution charts
                section_heading("4. Distribution Analysis")

                try:
                    import plotly.express as px

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        st.write("**Histogram**")

                        fig_hist = px.histogram(
                            x=series,
                            nbins=20,
                            labels={"x": selected_column},
                            title=f"Distribution of {selected_column}"
                        )

                        st.plotly_chart(
                            fig_hist,
                            use_container_width=True
                        )

                    with chart_col2:
                        st.write("**Box Plot**")

                        fig_box = px.box(
                            y=series,
                            labels={"y": selected_column},
                            title=f"Box Plot of {selected_column}"
                        )

                        st.plotly_chart(
                            fig_box,
                            use_container_width=True
                        )

                except ImportError:
                    st.info(
                        "Install Plotly to display interactive charts."
                    )

                st.divider()

                # Statistical interpretation
                section_heading("5. Initial Statistical Interpretation")

                skewness = statistics["Skewness"]
                kurtosis = statistics["Kurtosis"]

                if pd.isna(skewness):
                    st.write(
                        "Skewness cannot be calculated for this variable."
                    )
                elif skewness > 1:
                    st.write(
                        "The distribution is strongly positively skewed."
                    )
                elif skewness < -1:
                    st.write(
                        "The distribution is strongly negatively skewed."
                    )
                else:
                    st.write(
                        "The distribution is not strongly skewed."
                    )

                if pd.isna(kurtosis):
                    st.write(
                        "Kurtosis cannot be calculated for this variable."
                    )
                elif kurtosis > 0:
                    st.write(
                        "The distribution has positive excess kurtosis."
                    )
                elif kurtosis < 0:
                    st.write(
                        "The distribution has negative excess kurtosis."
                    )
                else:
                    st.write(
                        "The distribution has approximately zero excess kurtosis."
                    )

                st.caption(
                    "These interpretations are descriptive. "
                    "They do not establish statistical significance "
                    "or prove that a variable follows a particular distribution."
                )

            st.success(
                "Numerical Variable Analysis completed successfully!"
            )
elif page == "Categorical Variable Analysis":

    page_header("🏷️ Categorical Variable Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Explore category frequencies, percentages, "
        "and distributions in your dataset."
    )

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["dataset"]
        categorical_columns = get_categorical_columns(df)

        if not categorical_columns:
            st.warning(
                "No categorical variables were detected in this dataset."
            )
        else:
            st.success(
                f"Found {len(categorical_columns)} categorical variables."
            )

            # Overall categorical summary
            section_heading("1. Categorical Variables Summary")

            summary = get_categorical_summary(df)

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Categorical Summary (CSV)",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="datalens_categorical_summary.csv",
                mime="text/csv"
            )

            st.divider()

            # Select a categorical variable
            section_heading("2. Individual Category Analysis")

            selected_column = st.selectbox(
                "Select a categorical variable",
                categorical_columns,
                key="phase6_selected_variable"
            )

            statistics = get_category_statistics(df, selected_column)
            frequency = get_frequency_distribution(df, selected_column)

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Records",
                statistics["Total Records"]
            )
            col2.metric(
                "Unique Categories",
                statistics["Unique Categories"]
            )
            col3.metric(
                "Missing Values",
                statistics["Missing Values"]
            )
            col4.metric(
                "Most Frequent Category",
                statistics["Most Frequent Category"]
            )

            st.divider()

            # Frequency distribution
            section_heading("3. Frequency Distribution")

            if frequency.empty:
                st.info("No non-missing category values are available.")
            else:
                st.dataframe(
                    frequency,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Frequency Distribution (CSV)",
                    data=frequency.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_column}_frequency.csv",
                    mime="text/csv",
                    key="phase6_frequency_download"
                )

                st.divider()

                # Charts
                section_heading("4. Category Visualization")

                try:
                    import plotly.express as px

                    chart_col1, chart_col2 = st.columns(2)

                    # Bar chart
                    with chart_col1:
                        st.write("**Category Frequency – Bar Chart**")

                        bar_data = frequency.head(20)

                        fig_bar = px.bar(
                            bar_data,
                            x="Category",
                            y="Frequency",
                            title=f"Frequency of {selected_column}",
                            labels={
                                "Category": selected_column,
                                "Frequency": "Number of Records"
                            }
                        )

                        fig_bar.update_layout(
                            xaxis_tickangle=-45
                        )

                        st.plotly_chart(
                            fig_bar,
                            use_container_width=True
                        )

                    # Pie chart
                    with chart_col2:
                        st.write("**Category Distribution – Pie Chart**")

                        pie_data = frequency.head(10).copy()

                        if len(frequency) > 10:
                            other_count = frequency.iloc[10:]["Frequency"].sum()

                            pie_data = pd.concat([
                                pie_data,
                                pd.DataFrame([{
                                    "Category": "Other",
                                    "Frequency": other_count,
                                    "Percentage": (
                                        other_count /
                                        frequency["Frequency"].sum() * 100
                                    )
                                }])
                            ], ignore_index=True)

                        fig_pie = px.pie(
                            pie_data,
                            names="Category",
                            values="Frequency",
                            title=f"Distribution of {selected_column}"
                        )

                        st.plotly_chart(
                            fig_pie,
                            use_container_width=True
                        )

                except ImportError:
                    st.info(
                        "Install Plotly to display interactive charts."
                    )

            st.divider()

            # Observations
            section_heading("5. Automatic Category Observations")

            observations = get_category_observations(
                df,
                selected_column
            )

            for index, observation in enumerate(
                observations, start=1
            ):
                st.write(f"**{index}.** {observation}")

            st.success(
                "Categorical Variable Analysis completed successfully!"
            )
elif page == "Data Quality Analysis":

    page_header("🔍 Data Quality Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Identify missing values, duplicate records, "
        "infinite values and potential data consistency issues."
    )

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset first.")

    else:
        df = st.session_state["dataset"]

        # Overall quality summary
        section_heading("1. Overall Data Quality")

        quality = get_data_quality_summary(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Quality Score", f"{quality['Quality Score']:.2f}%")
        col2.metric("Missing Cells", quality["Missing Cells"])
        col3.metric("Duplicate Rows", quality["Duplicate Rows"])
        col4.metric("Infinite Values", quality["Infinite Values"])

        st.write(f"**Quality Status:** {quality['Status']}")

        st.progress(quality["Quality Score"] / 100)

        st.caption(
            "The indicative score subtracts missing-cell percentage, "
            "duplicate-row percentage and infinite-value percentage. "
            "It does not assess every possible quality issue."
        )

        st.divider()

        # Missing values
        section_heading("2. Missing Value Analysis")

        missing_report = get_missing_value_report(df)

        st.dataframe(
            missing_report,
            use_container_width=True,
            hide_index=True
        )

        columns_with_missing = missing_report[
            missing_report["Missing Values"] > 0
        ]

        if not columns_with_missing.empty:
            st.write("**Missing Values by Column**")

            chart_data = columns_with_missing.set_index("Column")[
                "Missing Values"
            ]

            st.bar_chart(chart_data)
        else:
            st.success("No missing values detected.")

        st.divider()

        # Duplicate records
        section_heading("3. Duplicate Record Detection")

        duplicate_report = get_duplicate_report(df)

        col1, col2 = st.columns(2)

        col1.metric(
            "Duplicate Rows",
            duplicate_report["Duplicate Rows"]
        )

        col2.metric(
            "Duplicate Percentage",
            f"{duplicate_report['Duplicate Percentage']:.2f}%"
        )

        if duplicate_report["Duplicate Rows"] > 0:
            st.warning("Completely duplicated rows were detected.")

            duplicate_preview = df.loc[df.duplicated(keep=False)]

            st.write("**Duplicate Row Preview**")
            st.dataframe(
                duplicate_preview.head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No completely duplicated rows detected.")

        st.divider()

        # Infinite values
        section_heading("4. Infinite Value Detection")

        infinite_report = get_infinite_value_report(df)

        if not infinite_report.empty:
            st.dataframe(
                infinite_report,
                use_container_width=True,
                hide_index=True
            )

            if infinite_report["Total Infinite Values"].sum() > 0:
                st.warning("Infinite numerical values were detected.")
            else:
                st.success("No infinite numerical values detected.")
        else:
            st.info("No numerical columns are available for this check.")

        st.divider()

        # Constant columns
        section_heading("5. Constant Column Detection")

        constant_report = get_constant_columns(df)

        if not constant_report.empty:
            st.warning(
                f"Found {len(constant_report)} constant "
                "or entirely missing columns."
            )

            st.dataframe(
                constant_report,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No constant or entirely missing columns detected.")

        st.divider()

        # Consistency checks
        section_heading("6. Potential Data Consistency Issues")

        consistency_report = get_consistency_report(df)

        if not consistency_report.empty:
            st.warning(
                "Potential inconsistencies were found. "
                "Review these columns before cleaning."
            )

            st.dataframe(
                consistency_report,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success(
                "No issues were detected by the basic consistency checks."
            )

        st.divider()

        # Download report
        section_heading("7. Download Data Quality Report")

        report_sections = {
            "Overall Summary": pd.DataFrame([quality]),
            "Missing Values": missing_report,
            "Infinite Values": infinite_report,
            "Constant Columns": constant_report,
            "Consistency Issues": consistency_report
        }

        csv_parts = []

        for section_name, section_df in report_sections.items():
            csv_parts.append(f"\n{section_name}\n")
            csv_parts.append(section_df.to_csv(index=False))

        csv_report = "\n".join(csv_parts)

        st.download_button(
            "Download Complete Quality Report (CSV)",
            data=csv_report.encode("utf-8"),
            file_name="datalens_data_quality_report.csv",
            mime="text/csv"
        )

        st.success(
            "Data Quality Analysis completed successfully!"
        )
elif page == "Data Cleaning":

    page_header("🧹 Data Cleaning", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Clean, transform and prepare your dataset "
        "for further statistical analysis."
    )

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset first.")

    else:
        original_df = st.session_state["dataset"]

        # Initialise a separate working copy.
        if "cleaned_dataset" not in st.session_state:
            st.session_state["cleaned_dataset"] = original_df.copy()

        if "cleaning_source_name" not in st.session_state:
            st.session_state["cleaning_source_name"] = (
                st.session_state.get("dataset_name", "")
            )

        # Reset the working copy when a different dataset is uploaded.
        current_name = st.session_state.get("dataset_name", "")

        if st.session_state["cleaning_source_name"] != current_name:
            st.session_state["cleaned_dataset"] = original_df.copy()
            st.session_state["cleaning_source_name"] = current_name

        cleaned_df = st.session_state["cleaned_dataset"]

                # -----------------------------------------
        # AUTOMATIC DATA CLEANING
        # -----------------------------------------

        section_heading("🤖 Automatic Data Cleaning")

        st.write(
            "Scan your dataset, review recommended cleaning actions, "
            "preview the results and approve the changes."
        )

        if "phase8_auto_recommendations" not in st.session_state:
            st.session_state["phase8_auto_recommendations"] = None

        if "phase8_auto_preview" not in st.session_state:
            st.session_state["phase8_auto_preview"] = None

        if "phase8_auto_preview_selection" not in st.session_state:
            st.session_state["phase8_auto_preview_selection"] = None

        if "phase8_cleaning_history" not in st.session_state:
            st.session_state["phase8_cleaning_history"] = []

        # Scan dataset
        if st.button(
            "🔍 Scan Dataset for Cleaning Issues",
            key="phase8_scan_button"
        ):
            st.session_state["phase8_auto_recommendations"] = (
                scan_dataset_for_cleaning(cleaned_df)
            )

            st.session_state["phase8_auto_preview"] = None
            st.session_state["phase8_auto_preview_selection"] = None

            st.rerun()

        recommendations = st.session_state[
            "phase8_auto_recommendations"
        ]

        if recommendations is not None:

            if not recommendations:
                st.success(
                    "No automatically detectable cleaning issues found."
                )

            else:
                st.write(
                    f"**{len(recommendations)} cleaning recommendations found.**"
                )

                for index, action in enumerate(recommendations, start=1):
                    with st.container(border=True):
                        st.write(
                            f"**{index}. {action['title']}**"
                        )

                        st.write(action["description"])

                        st.caption(
                            f"Affected values/records: {action['affected']} "
                            f" | Risk: {action['risk']}"
                        )

                st.divider()

                st.write("**Select the actions you want to apply**")

                default_actions = [
                    action["id"]
                    for action in recommendations
                    if action["default_selected"]
                ]

                selected_actions = st.multiselect(
                    "Cleaning actions",
                    options=[
                        action["id"] for action in recommendations
                        if action["id"] != "mixed_types_review"
                    ],
                    default=default_actions,
                    format_func=lambda action_id: next(
                        action["title"]
                        for action in recommendations
                        if action["id"] == action_id
                    ),
                    key="phase8_auto_selected"
                )

                st.caption(
                    "Recommendations marked for manual review are not "
                    "automatically selectable. Review those columns yourself."
                )

                # Preview changes
                if st.button(
                    "👁️ Preview Cleaning Results",
                    key="phase8_preview_button"
                ):
                    try:
                        preview_df = apply_automatic_cleaning(
                            cleaned_df,
                            selected_actions,
                            recommendations
                        )

                        st.session_state["phase8_auto_preview"] = preview_df
                        st.session_state[
                            "phase8_auto_preview_selection"
                        ] = list(selected_actions)

                        st.rerun()

                    except Exception as error:
                        st.error(f"Preview failed: {error}")

                preview_df = st.session_state["phase8_auto_preview"]

                if preview_df is not None:

                    preview_selection = st.session_state[
                        "phase8_auto_preview_selection"
                    ]

                    if preview_selection == list(selected_actions):

                        st.divider()
                        section_heading("Preview of Proposed Changes")

                        preview_col1, preview_col2, preview_col3 = (
                            st.columns(3)
                        )

                        preview_col1.metric(
                            "Current Rows",
                            len(cleaned_df)
                        )

                        preview_col2.metric(
                            "Proposed Rows",
                            len(preview_df)
                        )

                        preview_col3.metric(
                            "Proposed Missing Cells",
                            int(preview_df.isna().sum().sum())
                        )

                        st.write("**Proposed cleaned dataset**")

                        st.dataframe(
                            preview_df.head(100),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.write("**Changes summary**")

                        changes = pd.DataFrame([
                            {
                                "Metric": "Rows removed",
                                "Before": len(cleaned_df),
                                "After": len(preview_df)
                            },
                            {
                                "Metric": "Missing cells",
                                "Before": int(cleaned_df.isna().sum().sum()),
                                "After": int(preview_df.isna().sum().sum())
                            },
                            {
                                "Metric": "Duplicate rows",
                                "Before": int(cleaned_df.duplicated().sum()),
                                "After": int(preview_df.duplicated().sum())
                            }
                        ])

                        st.dataframe(
                            changes,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.warning(
                            "Review the preview before approving. "
                            "Approval replaces the current cleaned working copy, "
                            "not the original uploaded dataset."
                        )

                        if st.button(
                            "✅ Approve and Apply Cleaning",
                            key="phase8_approve_button"
                        ):
                            history = st.session_state[
                                "phase8_cleaning_history"
                            ]

                            history.append(cleaned_df.copy(deep=True))

                            st.session_state[
                                "cleaned_dataset"
                            ] = preview_df.copy(deep=True)

                            st.session_state[
                                "phase8_cleaning_history"
                            ] = history

                            st.session_state[
                                "phase8_auto_preview"
                            ] = None

                            st.session_state[
                                "phase8_auto_preview_selection"
                            ] = None

                            st.session_state[
                                "phase8_auto_recommendations"
                            ] = None

                            st.success(
                                "Approved cleaning actions applied successfully."
                            )

                            st.rerun()

                    else:
                        st.info(
                            "Your selected actions have changed. "
                            "Generate a new preview before approval."
                        )

        st.divider()

        # Undo last approved operation
        section_heading("↩️ Undo and Reset")

        history = st.session_state["phase8_cleaning_history"]

        undo_col, reset_col = st.columns(2)

        with undo_col:
            if st.button(
                "Undo Last Cleaning",
                disabled=len(history) == 0,
                key="phase8_undo_button"
            ):
                previous_df = history.pop()

                st.session_state["cleaned_dataset"] = previous_df

                st.session_state[
                    "phase8_cleaning_history"
                ] = history

                st.session_state[
                    "phase8_auto_preview"
                ] = None

                st.session_state[
                    "phase8_auto_recommendations"
                ] = None

                st.rerun()

        with reset_col:
            if st.button(
                "Reset to Original Dataset",
                key="phase8_auto_reset_button"
            ):
                st.session_state["cleaned_dataset"] = (
                    original_df.copy(deep=True)
                )

                st.session_state["phase8_cleaning_history"] = []
                st.session_state["phase8_auto_preview"] = None
                st.session_state["phase8_auto_recommendations"] = None

                st.rerun()

        st.divider()

        section_heading("1. Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Original Rows", len(original_df))
        col2.metric("Cleaned Rows", len(cleaned_df))
        col3.metric(
            "Missing Cells",
            int(cleaned_df.isna().sum().sum())
        )
        col4.metric(
            "Duplicate Rows",
            int(cleaned_df.duplicated().sum())
        )

        st.divider()

        # Cleaning operations
        section_heading("2. Cleaning Operations")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Missing Values",
            "Duplicates & Text",
            "Data Types",
            "Rename Columns"
        ])

        # Missing value handling
        with tab1:
            st.write("**Handle Missing Values**")

            missing_columns = cleaned_df.columns[
                cleaned_df.isna().any()
            ].tolist()

            if missing_columns:
                selected_column = st.selectbox(
                    "Select a column",
                    missing_columns,
                    key="phase8_missing_column"
                )

                method = st.selectbox(
                    "Choose a filling method",
                    ["Mean", "Median", "Mode", "Custom Value"],
                    key="phase8_missing_method"
                )

                custom_value = None

                if method == "Custom Value":
                    custom_value = st.text_input(
                        "Enter the replacement value",
                        key="phase8_custom_value"
                    )

                if st.button(
                    "Apply Missing Value Treatment",
                    key="phase8_fill_button"
                ):
                    try:
                        value = custom_value

                        if method == "Custom Value":
                            if pd.api.types.is_numeric_dtype(
                                cleaned_df[selected_column]
                            ):
                                value = pd.to_numeric(
                                    custom_value, errors="raise"
                                )

                        updated_df = fill_missing_values(
                            cleaned_df,
                            selected_column,
                            method,
                            value
                        )

                        st.session_state["cleaned_dataset"] = updated_df
                        st.success("Missing values handled successfully.")
                        st.rerun()

                    except Exception as error:
                        st.error(str(error))

                st.divider()

                st.write("**Remove Rows With Missing Values**")

                row_removal_option = st.radio(
                    "Which rows should be removed?",
                    [
                        "Rows with missing values in any column",
                        "Rows with missing values in selected columns"
                    ],
                    key="phase8_missing_rows_option"
                )

                selected_columns = None

                if row_removal_option == (
                    "Rows with missing values in selected columns"
                ):
                    selected_columns = st.multiselect(
                        "Select columns",
                        cleaned_df.columns.tolist(),
                        key="phase8_missing_row_columns"
                    )

                if st.button(
                    "Remove Missing Rows",
                    key="phase8_remove_missing"
                ):
                    if (
                        row_removal_option ==
                        "Rows with missing values in selected columns"
                        and not selected_columns
                    ):
                        st.warning("Select at least one column.")
                    else:
                        updated_df = remove_missing_rows(
                            cleaned_df,
                            selected_columns
                        )

                        st.session_state["cleaned_dataset"] = updated_df
                        st.success("Rows with missing values removed.")
                        st.rerun()

            else:
                st.success("No missing values found in the cleaned dataset.")

        # Duplicate and text handling
        with tab2:
            st.write("**Remove Duplicate Rows**")

            duplicate_count = int(cleaned_df.duplicated().sum())

            st.metric("Duplicate Rows Found", duplicate_count)

            if st.button(
                "Remove Duplicate Rows",
                key="phase8_remove_duplicates"
            ):
                updated_df = remove_duplicate_rows(cleaned_df)

                st.session_state["cleaned_dataset"] = updated_df
                st.success("Duplicate rows removed.")
                st.rerun()

            st.divider()

            st.write("**Remove Extra Whitespace**")

            if st.button(
                "Trim All Text Columns",
                key="phase8_trim_text"
            ):
                updated_df = trim_text_values(cleaned_df)

                st.session_state["cleaned_dataset"] = updated_df
                st.success("Text whitespace cleaned.")
                st.rerun()

            st.divider()

            st.write("**Standardise Text Case**")

            text_columns = [
                column for column in cleaned_df.columns
                if (
                    pd.api.types.is_object_dtype(cleaned_df[column])
                    or pd.api.types.is_string_dtype(cleaned_df[column])
                )
            ]

            if text_columns:
                text_column = st.selectbox(
                    "Select a text column",
                    text_columns,
                    key="phase8_case_column"
                )

                text_case = st.selectbox(
                    "Choose text format",
                    ["Uppercase", "Lowercase", "Title Case"],
                    key="phase8_case_format"
                )

                if st.button(
                    "Apply Text Formatting",
                    key="phase8_case_button"
                ):
                    updated_df = standardise_text_case(
                        cleaned_df,
                        text_column,
                        text_case
                    )

                    st.session_state["cleaned_dataset"] = updated_df
                    st.success("Text formatting applied.")
                    st.rerun()
            else:
                st.info("No text columns available.")

        # Data type conversion
        with tab3:
            st.write("**Convert Column Data Type**")

            selected_column = st.selectbox(
                "Select a column to convert",
                cleaned_df.columns.tolist(),
                key="phase8_type_column"
            )

            target_type = st.selectbox(
                "Select target data type",
                [
                    "Integer",
                    "Float",
                    "String",
                    "Date/Time",
                    "Category"
                ],
                key="phase8_target_type"
            )

            if st.button(
                "Convert Data Type",
                key="phase8_convert_button"
            ):
                try:
                    updated_df = convert_column_type(
                        cleaned_df,
                        selected_column,
                        target_type
                    )

                    st.session_state["cleaned_dataset"] = updated_df
                    st.success("Column data type converted successfully.")
                    st.rerun()

                except Exception as error:
                    st.error(str(error))

        # Rename columns
        with tab4:
            st.write("**Rename a Column**")

            selected_column = st.selectbox(
                "Select a column",
                cleaned_df.columns.tolist(),
                key="phase8_rename_column"
            )

            new_name = st.text_input(
                "Enter the new column name",
                value=str(selected_column),
                key="phase8_new_name"
            )

            if st.button(
                "Rename Column",
                key="phase8_rename_button"
            ):
                try:
                    updated_df = rename_column(
                        cleaned_df,
                        selected_column,
                        new_name
                    )

                    st.session_state["cleaned_dataset"] = updated_df
                    st.success("Column renamed successfully.")
                    st.rerun()

                except Exception as error:
                    st.error(str(error))

        st.divider()

        # Compare original and cleaned data
        section_heading("3. Original vs Cleaned Dataset")

        cleaned_df = st.session_state["cleaned_dataset"]

        comparison = get_cleaning_summary(
            original_df,
            cleaned_df
        )

        comparison_df = pd.DataFrame(
            list(comparison.items()),
            columns=["Metric", "Value"]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Preview cleaned dataset
        section_heading("4. Cleaned Dataset Preview")

        st.dataframe(
            cleaned_df.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing up to 100 of {len(cleaned_df)} cleaned rows."
        )

        st.divider()

        # Download cleaned data
        section_heading("5. Export Cleaned Dataset")

        csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="datalens_cleaned_dataset.csv",
            mime="text/csv"
        )


        st.success("Data Cleaning module is ready.")

elif page == "Outlier Detection & Analysis":

    page_header("Outlier Detection & Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Identify unusual numerical observations using statistical "
        "and machine-learning techniques."
    )

    # Use the cleaned dataset when available
    analysis_df = st.session_state.get(
        "cleaned_dataset",
        st.session_state.get("dataset")
    )

    if analysis_df is None:
        st.warning("Please upload a dataset first.")

    else:
        numerical_columns = get_numerical_columns(analysis_df)

        if not numerical_columns:
            st.warning(
                "No numerical columns are available for outlier analysis."
            )

        else:
            section_heading("1. Detection Method")

            method = st.selectbox(
                "Choose an outlier detection method",
                [
                    "IQR",
                    "Z-Score",
                    "Modified Z-Score",
                    "Isolation Forest"
                ]
            )

            st.caption(
                "IQR uses the 1.5 × IQR rule. Z-Score uses a threshold "
                "of 3. Modified Z-Score uses 3.5. Isolation Forest "
                "uses a default contamination of 5%."
            )

            # Generate summary
            summary = get_outlier_summary(
                analysis_df,
                method=method
            )

            section_heading("2. Outlier Summary")

            total_outliers = int(summary["Outliers"].sum())

            col1, col2, col3 = st.columns(3)

            col1.metric("Numerical Columns", len(numerical_columns))
            col2.metric("Detected Outliers", total_outliers)
            col3.metric(
                "Columns with Outliers",
                int((summary["Outliers"] > 0).sum())
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Outlier Summary",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name="datalens_outlier_summary.csv",
                mime="text/csv"
            )

            st.divider()

            # Detailed analysis
            section_heading("3. Column-wise Outlier Analysis")

            selected_column = st.selectbox(
                "Select a numerical column",
                numerical_columns
            )

            outlier_rows = get_outlier_details(
                analysis_df,
                selected_column,
                method
            )

            st.write(
                f"Detected **{len(outlier_rows)}** outlier rows "
                f"in `{selected_column}`."
            )

            if not outlier_rows.empty:
                st.dataframe(
                    outlier_rows,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No outliers detected in this column.")

            # Charts
            section_heading("4. Outlier Visualization")

            import plotly.express as px

            chart_df = analysis_df[
                [selected_column]
            ].copy()

            chart_df["Detection"] = "Normal"

            chart_df.loc[
                outlier_rows.index,
                "Detection"
            ] = "Outlier"

            fig = px.box(
                chart_df,
                y=selected_column,
                points="all",
                title=f"Box Plot – {selected_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            fig2 = px.scatter(
                chart_df.reset_index(),
                x="index",
                y=selected_column,
                color="Detection",
                title=f"Outlier Distribution – {selected_column}"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.divider()

            # Treatment
            section_heading("5. Outlier Treatment")

            if method == "Isolation Forest":
                st.info(
                    "Automatic treatment is not available for Isolation "
                    "Forest in this version. Select IQR, Z-Score, or "
                    "Modified Z-Score to treat outliers."
                )

            else:
                treatment = st.selectbox(
                    "Choose treatment",
                    ["Keep", "Remove", "Cap"]
                )

                if treatment == "Cap":
                    st.caption(
                        "Capping replaces detected extreme values with "
                        "the selected method's lower or upper threshold."
                    )

                if st.button("Preview Outlier Treatment"):

                    preview_df = treat_outliers(
                        analysis_df,
                        selected_column,
                        method=method,
                        treatment=treatment
                    )

                    st.session_state[
                        "phase9_outlier_preview"
                    ] = preview_df

                    st.session_state[
                        "phase9_outlier_preview_method"
                    ] = method

                    st.session_state[
                        "phase9_outlier_preview_column"
                    ] = selected_column

                    st.session_state[
                        "phase9_outlier_preview_treatment"
                    ] = treatment

                preview_df = st.session_state.get(
                    "phase9_outlier_preview"
                )

                if preview_df is not None:
                    if (
                        st.session_state.get(
                            "phase9_outlier_preview_method"
                        ) == method
                        and st.session_state.get(
                            "phase9_outlier_preview_column"
                        ) == selected_column
                        and st.session_state.get(
                            "phase9_outlier_preview_treatment"
                        ) == treatment
                    ):

                        st.write("**Treatment Preview**")

                        before_col, after_col = st.columns(2)

                        with before_col:
                            st.metric(
                                "Original Rows",
                                len(analysis_df)
                            )

                        with after_col:
                            st.metric(
                                "Proposed Rows",
                                len(preview_df)
                            )

                        st.dataframe(
                            preview_df.head(100),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.warning(
                            "Review the result before applying. "
                            "This changes the cleaned working dataset, "
                            "not the original uploaded dataset."
                        )

                        if st.button("Approve and Apply Treatment"):

                            if "phase9_outlier_history" not in st.session_state:
                                st.session_state[
                                    "phase9_outlier_history"
                                ] = []

                            st.session_state[
                                "phase9_outlier_history"
                            ].append(
                                analysis_df.copy(deep=True)
                            )

                            st.session_state[
                                "cleaned_dataset"
                            ] = preview_df.copy(deep=True)

                            st.session_state[
                                "phase9_outlier_preview"
                            ] = None

                            st.success(
                                "Outlier treatment applied successfully."
                            )

                            st.rerun()
                    else:
                        st.info(
                            "Your selections have changed. Generate a "
                            "new preview before applying treatment."
                        )

            # Undo
            st.divider()
            section_heading("6. Undo Last Outlier Treatment")

            history = st.session_state.get(
                "phase9_outlier_history",
                []
            )

            if st.button(
                "Undo Last Outlier Treatment",
                disabled=len(history) == 0
            ):
                previous_df = history.pop()

                st.session_state["cleaned_dataset"] = previous_df
                st.session_state["phase9_outlier_history"] = history
                st.session_state["phase9_outlier_preview"] = None

                st.rerun()


elif page == "Distribution Analysis":

    page_header("Distribution Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Explore the distribution of numerical, categorical and "
        "date/time variables across your dataset."
    )

    analysis_df = st.session_state.get(
        "cleaned_dataset",
        st.session_state.get("dataset")
    )

    if analysis_df is None:
        st.warning("Please upload a dataset first.")

    else:
        import plotly.express as px
        import plotly.graph_objects as go

        # -----------------------------------------
        # 1. OVERALL DATASET DISTRIBUTION
        # -----------------------------------------

        section_heading("1. Overall Dataset Distribution")

        overview = get_dataset_distribution_summary(analysis_df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Rows", overview["Total Rows"])
        col2.metric("Numerical", overview["Numerical Columns"])
        col3.metric("Categorical", overview["Categorical Columns"])
        col4.metric("Date/Time", overview["Date/Time Columns"])

        st.metric(
            "Total Missing Cells",
            overview["Total Missing Cells"]
        )

        type_report = get_variable_type_report(analysis_df)

        st.write("**Variable Classification**")

        st.dataframe(
            type_report,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------
        # 2. OVERALL DISTRIBUTION CHART
        # -----------------------------------------

        section_heading("2. Variable Type Distribution")

        type_counts = (
            type_report["Variable Type"]
            .value_counts()
            .reindex(
                ["Numerical", "Categorical", "Date/Time"],
                fill_value=0
            )
            .reset_index()
        )

        type_counts.columns = ["Variable Type", "Columns"]

        fig_types = px.pie(
            type_counts,
            names="Variable Type",
            values="Columns",
            title="Dataset Variable Composition",
            hole=0.4
        )

        st.plotly_chart(
            fig_types,
            use_container_width=True
        )

        st.divider()

        # -----------------------------------------
        # 3. SELECT VARIABLE
        # -----------------------------------------

        section_heading("3. Individual Variable Distribution")

        selected_column = st.selectbox(
            "Select a dataset column",
            list(analysis_df.columns),
            key="phase10_selected_column"
        )

        detected_type = detect_variable_type(
            analysis_df[selected_column]
        )

        st.write(
            f"Automatically detected type: **{detected_type}**"
        )

        variable_type = st.selectbox(
            "Change variable type if necessary",
            ["Numerical", "Categorical", "Date/Time"],
            index=[
                "Numerical", "Categorical", "Date/Time"
            ].index(detected_type),
            key="phase10_variable_override"
        )

        st.caption(
            "You can override automatic classification when a numeric "
            "column represents category codes, or when dates are stored "
            "as text."
        )

        st.divider()

        # -----------------------------------------
        # 4. NUMERICAL DISTRIBUTION
        # -----------------------------------------

        if variable_type == "Numerical":

            section_heading("Numerical Distribution")

            values = pd.to_numeric(
                analysis_df[selected_column],
                errors="coerce"
            ).dropna()

            if values.empty:
                st.warning("No valid numerical values found.")

            else:
                numerical_stats = get_distribution_statistics(
                    analysis_df,
                    selected_column
                )

                stat_col1, stat_col2, stat_col3 = st.columns(3)

                stat_col1.metric(
                    "Mean",
                    f"{numerical_stats['Mean']:.3f}"
                )

                stat_col2.metric(
                    "Median",
                    f"{numerical_stats['Median']:.3f}"
                )

                stat_col3.metric(
                    "Standard Deviation",
                    f"{numerical_stats['Standard Deviation']:.3f}"
                )

                st.dataframe(
                    pd.DataFrame(
                        numerical_stats.items(),
                        columns=["Statistic", "Value"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    f"Distribution shape: "
                    f"{numerical_stats['Distribution Shape']}"
                )

                st.info(
                    f"Tail characteristics: "
                    f"{numerical_stats['Tail Characteristics']}"
                )

                # Histogram
                section_heading("Histogram")

                fig_hist = px.histogram(
                    x=values,
                    nbins=30,
                    title=f"Distribution of {selected_column}",
                    labels={"x": selected_column}
                )

                st.plotly_chart(
                    fig_hist,
                    use_container_width=True
                )

                # Density
                section_heading("Density Plot")

                if len(values) >= 2 and values.nunique() > 1:
                    kde = stats.gaussian_kde(values)

                    x_values = np.linspace(
                        values.min(),
                        values.max(),
                        200
                    )

                    fig_density = go.Figure()

                    fig_density.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=kde(x_values),
                            mode="lines",
                            name="Density"
                        )
                    )

                    fig_density.update_layout(
                        title=f"Density Distribution – {selected_column}",
                        xaxis_title=selected_column,
                        yaxis_title="Density"
                    )

                    st.plotly_chart(
                        fig_density,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "Density plot requires at least two distinct values."
                    )

                # Box plot
                section_heading("Box Plot")

                fig_box = px.box(
                    y=values,
                    title=f"Box Plot – {selected_column}"
                )

                st.plotly_chart(
                    fig_box,
                    use_container_width=True
                )

                # Normal distribution
                section_heading("Normal Distribution Comparison")

                normal_data = compare_with_normal_distribution(
                    analysis_df,
                    selected_column
                )

                if not normal_data.empty:

                    fig_normal = go.Figure()

                    fig_normal.add_trace(
                        go.Histogram(
                            x=values,
                            histnorm="probability density",
                            name="Observed Data",
                            opacity=0.5
                        )
                    )

                    fig_normal.add_trace(
                        go.Scatter(
                            x=normal_data["Value"],
                            y=normal_data["Normal Density"],
                            mode="lines",
                            name="Fitted Normal Distribution"
                        )
                    )

                    fig_normal.update_layout(
                        title="Observed Data vs Fitted Normal Curve",
                        xaxis_title=selected_column,
                        yaxis_title="Probability Density"
                    )

                    st.plotly_chart(
                        fig_normal,
                        use_container_width=True
                    )

                # Normality tests
                section_heading("Normality Tests")

                normality = test_normality(
                    analysis_df,
                    selected_column
                )

                st.dataframe(
                    normality,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "A p-value below 0.05 provides evidence against "
                    "normality. A higher p-value does not prove normality."
                )

        # -----------------------------------------
        # 5. CATEGORICAL DISTRIBUTION
        # -----------------------------------------

        elif variable_type == "Categorical":

            section_heading("Categorical Distribution")

            values = analysis_df[selected_column].dropna()

            if values.empty:
                st.warning("No valid categorical values found.")

            else:
                category_stats = get_categorical_statistics(
                    analysis_df,
                    selected_column
                )

                cat_col1, cat_col2, cat_col3 = st.columns(3)

                cat_col1.metric(
                    "Total Values",
                    category_stats["Total Values"]
                )

                cat_col2.metric(
                    "Unique Categories",
                    category_stats["Unique Categories"]
                )

                cat_col3.metric(
                    "Most Frequent Count",
                    category_stats["Most Frequent Count"]
                )

                st.write("**Category Statistics**")

                st.dataframe(
                    pd.DataFrame(
                        category_stats.items(),
                        columns=["Statistic", "Value"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                frequency = get_categorical_distribution(
                    analysis_df,
                    selected_column
                )

                section_heading("Frequency Distribution")

                st.dataframe(
                    frequency,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Category Distribution",
                    data=frequency.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="datalens_category_distribution.csv",
                    mime="text/csv"
                )

                # Limit charts to the top 20 categories for readability.
                chart_data = frequency.head(20)

                section_heading("Category Bar Chart")

                fig_bar = px.bar(
                    chart_data,
                    x="Category",
                    y="Count",
                    title=f"Category Frequency – {selected_column}"
                )

                st.plotly_chart(
                    fig_bar,
                    use_container_width=True
                )

                section_heading("Category Pie Chart")

                fig_pie = px.pie(
                    chart_data,
                    names="Category",
                    values="Count",
                    title=f"Category Proportions – {selected_column}"
                )

                st.plotly_chart(
                    fig_pie,
                    use_container_width=True
                )

                if len(frequency) > 20:
                    st.caption(
                        "Charts display the top 20 categories. "
                        "The complete frequency table is available above."
                    )

        # -----------------------------------------
        # 6. DATE/TIME DISTRIBUTION
        # -----------------------------------------

        elif variable_type == "Date/Time":

            section_heading("Date/Time Distribution")

            dates = get_datetime_values(
                analysis_df,
                selected_column
            ).dropna()

            if dates.empty:
                st.warning(
                    "No valid dates could be identified in this column."
                )

            else:
                date_stats = get_datetime_statistics(
                    analysis_df,
                    selected_column
                )

                date_col1, date_col2, date_col3 = st.columns(3)

                date_col1.metric(
                    "Valid Dates",
                    date_stats["Valid Dates"]
                )

                date_col2.metric(
                    "Unique Dates",
                    date_stats["Unique Dates"]
                )

                date_col3.metric(
                    "Date Range (Days)",
                    date_stats["Date Range (Days)"]
                )

                st.dataframe(
                    pd.DataFrame(
                        date_stats.items(),
                        columns=["Statistic", "Value"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                frequency = st.selectbox(
                    "Choose date grouping",
                    [
                        "Year",
                        "Month",
                        "Week",
                        "Day",
                        "Day of Week",
                        "Hour"
                    ],
                    index=1
                )

                date_distribution = get_datetime_distribution(
                    analysis_df,
                    selected_column,
                    frequency
                )

                section_heading("Date/Time Frequency Distribution")

                st.dataframe(
                    date_distribution,
                    use_container_width=True,
                    hide_index=True
                )

                fig_date = px.bar(
                    date_distribution,
                    x="Period",
                    y="Count",
                    title=(
                        f"{frequency} Distribution – "
                        f"{selected_column}"
                    )
                )

                st.plotly_chart(
                    fig_date,
                    use_container_width=True
                )

                st.download_button(
                    "Download Date Distribution",
                    data=date_distribution.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="datalens_date_distribution.csv",
                    mime="text/csv"
                )

        # -----------------------------------------
        # 7. COMPLETE DATASET REPORT
        # -----------------------------------------

        st.divider()

        section_heading("4. Complete Dataset Distribution Report")

        complete_report = get_distribution_report(
            analysis_df
        )

        st.dataframe(
            complete_report,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Complete Distribution Report",
            data=complete_report.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="datalens_complete_distribution_report.csv",
            mime="text/csv"
        )

elif page == "Univariate Analysis":

    import plotly.express as px

    page_header("Univariate Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Explore individual variables through statistical summaries, "
        "frequency distributions, visualizations and automatic insights."
    )

    # Select the active dataset
    analysis_df = st.session_state.get("cleaned_dataset")

    if analysis_df is None:
        analysis_df = st.session_state.get("dataset")

    if analysis_df is None:
        st.warning("Please upload a dataset before performing analysis.")

    else:
        # Import the variable classifier from Phase 10
        from modules.distribution_analysis import detect_variable_type

        section_heading("Select a Variable")

        column = st.selectbox(
            "Choose a column to analyse",
            analysis_df.columns.tolist()
        )

        detected_type = detect_variable_type(analysis_df[column])

        variable_type = st.selectbox(
            "Variable Type",
            ["Numerical", "Categorical", "Date/Time"],
            index=[
                "Numerical",
                "Categorical",
                "Date/Time"
            ].index(detected_type)
        )

        st.divider()

        # ------------------------------------------
        # NUMERICAL ANALYSIS
        # ------------------------------------------

        if variable_type == "Numerical":

            section_heading("Numerical Variable Analysis")

            result = analyze_numerical_variable(
                analysis_df, column
            )

            if result:

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Mean",
                    f"{result['Mean']:.2f}"
                )

                col2.metric(
                    "Median",
                    f"{result['Median']:.2f}"
                )

                col3.metric(
                    "Standard Deviation",
                    f"{result['Standard Deviation']:.2f}"
                )

                col4.metric(
                    "Unique Values",
                    result["Unique Values"]
                )

                section_heading("Statistical Summary")

                st.dataframe(
                    generate_univariate_report(
                        analysis_df, column, variable_type
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                values = pd.to_numeric(
                    analysis_df[column],
                    errors="coerce"
                ).dropna()

                section_heading("Distribution Visualization")

                if not values.empty:

                    fig = px.histogram(
                        x=values,
                        nbins=30,
                        title=f"Distribution of {column}",
                        labels={"x": column, "y": "Frequency"}
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    fig = px.box(
                        x=values,
                        points="outliers",
                        title=f"Box Plot of {column}",
                        labels={"x": column}
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                section_heading("Automatic Observations")

                for observation in get_numerical_observations(
                    analysis_df, column
                ):
                    st.write(f"• {observation}")

            else:
                st.warning("No valid numerical values found.")

        # ------------------------------------------
        # CATEGORICAL ANALYSIS
        # ------------------------------------------

        elif variable_type == "Categorical":

            section_heading("Categorical Variable Analysis")

            result = analyze_categorical_variable(
                analysis_df, column
            )

            if result:

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Unique Categories",
                    result["Unique Categories"]
                )

                col2.metric(
                    "Most Frequent Category",
                    result["Most Frequent Category"]
                )

                col3.metric(
                    "Most Frequent Count",
                    result["Most Frequent Count"]
                )

                section_heading("Statistical Summary")

                st.dataframe(
                    generate_univariate_report(
                        analysis_df, column, variable_type
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                frequency_df = get_categorical_frequency(
                    analysis_df, column
                )

                section_heading("Category Frequency Distribution")

                st.dataframe(
                    frequency_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Frequency Report",
                    frequency_df.to_csv(index=False),
                    file_name=f"{column}_frequency.csv",
                    mime="text/csv"
                )

                chart_df = frequency_df.head(20)

                fig = px.bar(
                    chart_df,
                    x="Category",
                    y="Count",
                    title=f"Category Distribution of {column}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                section_heading("Automatic Observations")

                for observation in get_categorical_observations(
                    analysis_df, column
                ):
                    st.write(f"• {observation}")

            else:
                st.warning("No valid categorical values found.")

        # ------------------------------------------
        # DATE/TIME ANALYSIS
        # ------------------------------------------

        elif variable_type == "Date/Time":

            section_heading("Date/Time Variable Analysis")

            result = analyze_datetime_variable(
                analysis_df, column
            )

            if result:

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Valid Dates",
                    result["Valid Dates"]
                )

                col2.metric(
                    "Unique Dates",
                    result["Unique Dates"]
                )

                col3.metric(
                    "Date Range (Days)",
                    result["Date Range (Days)"]
                )

                section_heading("Statistical Summary")

                st.dataframe(
                    generate_univariate_report(
                        analysis_df, column, variable_type
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                dates = pd.to_datetime(
                    analysis_df[column],
                    errors="coerce",
                    format="mixed"
                ).dropna()

                date_df = pd.DataFrame({"Date": dates})

                frequency = st.selectbox(
                    "Choose date grouping",
                    ["Year", "Month", "Day of Week"]
                )

                if frequency == "Year":
                    date_df["Period"] = date_df["Date"].dt.year.astype(str)

                elif frequency == "Month":
                    date_df["Period"] = (
                        date_df["Date"].dt.to_period("M").astype(str)
                    )

                else:
                    date_df["Period"] = (
                        date_df["Date"].dt.day_name()
                    )

                counts = (
                    date_df["Period"]
                    .value_counts()
                    .rename_axis("Period")
                    .reset_index(name="Count")
                )

                if frequency == "Day of Week":
                    weekday_order = [
                        "Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"
                    ]

                    counts["Period"] = pd.Categorical(
                        counts["Period"],
                        categories=weekday_order,
                        ordered=True
                    )

                    counts = counts.sort_values("Period")

                fig = px.bar(
                    counts,
                    x="Period",
                    y="Count",
                    title=f"{frequency} Distribution of {column}"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                section_heading("Automatic Observations")

                for observation in get_datetime_observations(
                    analysis_df, column
                ):
                    st.write(f"• {observation}")

            else:
                st.warning("No valid date/time values found.")

elif page == "Bivariate Analysis":

    import plotly.express as px

    from modules.distribution_analysis import detect_variable_type

    page_header("Bivariate Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Explore relationships between two variables using "
        "correlation, statistical tests, group comparisons and visualizations."
    )

    # ------------------------------------------
    # SELECT ACTIVE DATASET
    # ------------------------------------------

    analysis_df = st.session_state.get("cleaned_dataset")

    if analysis_df is None:
        analysis_df = st.session_state.get("dataset")

    if analysis_df is None:
        st.warning("Please upload a dataset before performing bivariate analysis.")

    else:

        # ------------------------------------------
        # IDENTIFY VARIABLE TYPES
        # ------------------------------------------

        numerical_columns = []
        categorical_columns = []
        datetime_columns = []

        for col in analysis_df.columns:

            variable_type = detect_variable_type(analysis_df[col])

            if variable_type == "Numerical":
                numerical_columns.append(col)

            elif variable_type == "Categorical":
                categorical_columns.append(col)

            elif variable_type == "Date/Time":
                datetime_columns.append(col)

        section_heading("Choose Relationship Type")

        relationship_type = st.selectbox(
            "Select the relationship you want to investigate",
            [
                "Numerical vs Numerical",
                "Categorical vs Categorical",
                "Numerical vs Categorical",
                "Date/Time vs Numerical",
                "Date/Time vs Categorical"
            ]
        )

        st.divider()

        # ------------------------------------------
        # NUMERICAL VS NUMERICAL
        # ------------------------------------------

        if relationship_type == "Numerical vs Numerical":

            section_heading("Numerical Relationship Analysis")

            if len(numerical_columns) < 2:
                st.warning(
                    "At least two numerical variables are required."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:
                    column_x = st.selectbox(
                        "First numerical variable",
                        numerical_columns,
                        key="biv_num_x"
                    )

                available_y = [
                    col for col in numerical_columns
                    if col != column_x
                ]

                with col2:
                    column_y = st.selectbox(
                        "Second numerical variable",
                        available_y,
                        key="biv_num_y"
                    )

                result = analyze_numerical_relationship(
                    analysis_df, column_x, column_y
                )

                if result:

                    section_heading("Correlation Results")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Pearson Correlation",
                        f"{result['Pearson Correlation']:.3f}"
                    )

                    c2.metric(
                        "Spearman Correlation",
                        f"{result['Spearman Correlation']:.3f}"
                    )

                    c3.metric(
                        "Covariance",
                        f"{result['Covariance']:.3f}"
                    )

                    c4.metric(
                        "Valid Observations",
                        result["Valid Observations"]
                    )

                    st.write(
                        "**Relationship Strength:**",
                        result["Relationship Strength"]
                    )

                    st.write(
                        "**Relationship Direction:**",
                        result["Relationship Direction"]
                    )

                    section_heading("Statistical Test Results")

                    result_df = pd.DataFrame(
                        result.items(),
                        columns=["Metric", "Value"]
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    fig = px.scatter(
                        analysis_df,
                        x=column_x,
                        y=column_y,
                        trendline="ols",
                        title=f"{column_x} vs {column_y}",
                        opacity=0.7
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    section_heading("Automatic Observations")

                    for observation in get_relationship_observations(
                        result
                    ):
                        st.write(f"• {observation}")

                    st.download_button(
                        "Download Relationship Report",
                        result_df.to_csv(index=False),
                        file_name="numerical_relationship_report.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning(
                        "Insufficient valid observations to calculate correlation."
                    )

        # ------------------------------------------
        # CATEGORICAL VS CATEGORICAL
        # ------------------------------------------

        elif relationship_type == "Categorical vs Categorical":

            section_heading("Categorical Association Analysis")

            if len(categorical_columns) < 2:
                st.warning(
                    "At least two categorical variables are required."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:
                    column_x = st.selectbox(
                        "First categorical variable",
                        categorical_columns,
                        key="biv_cat_x"
                    )

                available_y = [
                    col for col in categorical_columns
                    if col != column_x
                ]

                with col2:
                    column_y = st.selectbox(
                        "Second categorical variable",
                        available_y,
                        key="biv_cat_y"
                    )

                result, contingency_table = (
                    analyze_categorical_relationship(
                        analysis_df, column_x, column_y
                    )
                )

                section_heading("Cross-Tabulation")

                st.dataframe(
                    contingency_table,
                    use_container_width=True
                )

                section_heading("Chi-Square Test Results")

                if result.get("P-value") is not None:

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Chi-Square Statistic",
                        f"{result['Chi-square Statistic']:.3f}"
                    )

                    c2.metric(
                        "P-value",
                        f"{result['P-value']:.5f}"
                    )
                    cramers_v = result.get("Cramer's V")

                    if cramers_v is not None:
                        c3.metric(
                            "Cramer's V",
                            str(round(float(cramers_v), 3))
                        )
                    else:
                        c3.metric(
                            "Cramer's V",
                            "N/A"
                        )

                    st.write(
                        "**Association Strength:**",
                        result["Association Strength"]
                    )

                    st.info(result["Interpretation"])

                    expected = stats.chi2_contingency(
                        contingency_table
                    )[3]

                    if (expected < 5).any():
                        st.warning(
                            "Some expected cell counts are below 5. "
                            "The Chi-square approximation may be unreliable."
                        )

                    fig = px.imshow(
                        contingency_table,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title=f"{column_x} vs {column_y}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:
                    st.info(result["Interpretation"])

                report_df = pd.DataFrame(
                    result.items(),
                    columns=["Metric", "Value"]
                )

                st.download_button(
                    "Download Association Report",
                    report_df.to_csv(index=False),
                    file_name="categorical_relationship_report.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Chi-square tests association, not causation. "
                    "Expected cell counts and independent observations "
                    "should be considered when interpreting the result."
                )

        # ------------------------------------------
        # NUMERICAL VS CATEGORICAL
        # ------------------------------------------

        elif relationship_type == "Numerical vs Categorical":

            section_heading("Group Comparison Analysis")

            if not numerical_columns or not categorical_columns:
                st.warning(
                    "The dataset needs at least one numerical and "
                    "one categorical variable."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:
                    numerical_column = st.selectbox(
                        "Numerical variable",
                        numerical_columns,
                        key="biv_group_num"
                    )

                with col2:
                    categorical_column = st.selectbox(
                        "Categorical variable",
                        categorical_columns,
                        key="biv_group_cat"
                    )

                summary, test_results = (
                    analyze_numerical_by_category(
                        analysis_df,
                        numerical_column,
                        categorical_column
                    )
                )

                if not summary.empty:

                    section_heading("Group-wise Statistical Summary")

                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True
                    )

                    section_heading("Statistical Test Results")

                    st.dataframe(
                        pd.DataFrame(
                            test_results.items(),
                            columns=["Test", "Result"]
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    fig = px.box(
                        analysis_df,
                        x=categorical_column,
                        y=numerical_column,
                        color=categorical_column,
                        title=(
                            f"{numerical_column} "
                            f"across {categorical_column}"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    fig = px.bar(
                        summary,
                        x=categorical_column,
                        y="Mean",
                        title=f"Mean {numerical_column} by Category",
                        error_y=(
                            summary["Standard_Deviation"].fillna(0)
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Group Comparison Report",
                        summary.to_csv(index=False),
                        file_name="group_comparison_report.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "ANOVA compares group means. Kruskal-Wallis "
                        "is a rank-based alternative. A significant "
                        "result does not identify which specific "
                        "groups differ."
                    )

                else:
                    st.warning(
                        "No valid observations available for group comparison."
                    )

        # ------------------------------------------
        # DATETIME VS NUMERICAL
        # ------------------------------------------

        elif relationship_type == "Date/Time vs Numerical":

            section_heading("Time-Based Numerical Analysis")

            if not datetime_columns or not numerical_columns:
                st.warning(
                    "The dataset needs at least one date/time and "
                    "one numerical variable."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:
                    datetime_column = st.selectbox(
                        "Date/time variable",
                        datetime_columns,
                        key="biv_time_num_date"
                    )

                with col2:
                    numerical_column = st.selectbox(
                        "Numerical variable",
                        numerical_columns,
                        key="biv_time_num_value"
                    )

                frequency = st.selectbox(
                    "Time grouping",
                    ["Year", "Quarter", "Month", "Day"],
                    key="biv_time_num_frequency"
                )

                summary = analyze_datetime_numerical(
                    analysis_df,
                    datetime_column,
                    numerical_column,
                    frequency
                )

                if not summary.empty:

                    section_heading("Time-Based Summary")

                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True
                    )

                    fig = px.line(
                        summary,
                        x="Period",
                        y="Mean",
                        markers=True,
                        title=(
                            f"Mean {numerical_column} over time"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Time-Based Report",
                        summary.to_csv(index=False),
                        file_name="time_numerical_report.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning(
                        "No valid date and numerical observations found."
                    )

        # ------------------------------------------
        # DATETIME VS CATEGORICAL
        # ------------------------------------------

        elif relationship_type == "Date/Time vs Categorical":

            section_heading("Time-Based Category Analysis")

            if not datetime_columns or not categorical_columns:
                st.warning(
                    "The dataset needs at least one date/time and "
                    "one categorical variable."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:
                    datetime_column = st.selectbox(
                        "Date/time variable",
                        datetime_columns,
                        key="biv_time_cat_date"
                    )

                with col2:
                    categorical_column = st.selectbox(
                        "Categorical variable",
                        categorical_columns,
                        key="biv_time_cat_value"
                    )

                frequency = st.selectbox(
                    "Time grouping",
                    ["Year", "Month", "Day of Week"],
                    key="biv_time_cat_frequency"
                )

                result = analyze_datetime_categorical(
                    analysis_df,
                    datetime_column,
                    categorical_column,
                    frequency
                )

                if not result.empty:

                    section_heading("Category Frequency Over Time")

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    fig = px.bar(
                        result.reset_index().melt(
                            id_vars="Period",
                            var_name="Category",
                            value_name="Count"
                        ),
                        x="Period",
                        y="Count",
                        color="Category",
                        barmode="group",
                        title=(
                            f"{categorical_column} across "
                            f"{frequency}"
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Time-Based Category Report",
                        result.to_csv(),
                        file_name="time_categorical_report.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning(
                        "No valid date and categorical observations found."
                    )

        # ------------------------------------------
        # CORRELATION MATRIX
        # ------------------------------------------

        with st.expander("Explore Full Numerical Correlation Matrix"):

            if len(numerical_columns) >= 2:

                correlation_method = st.selectbox(
                    "Correlation method",
                    ["pearson", "spearman", "kendall"]
                )

                correlation_matrix = get_correlation_matrix(
                    analysis_df,
                    numerical_columns,
                    correlation_method
                )

                st.dataframe(
                    correlation_matrix,
                    use_container_width=True
                )

                fig = px.imshow(
                    correlation_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    title=f"{correlation_method.title()} Correlation Matrix"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.download_button(
                    "Download Correlation Matrix",
                    correlation_matrix.to_csv(),
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )

            else:
                st.info(
                    "At least two numerical variables are required "
                    "for a correlation matrix."
                )


elif page == "Group-wise Analysis":

    import plotly.express as px

    page_header("Group-wise Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Analyse, compare and rank groups within your dataset "
        "using descriptive statistics and interactive visualizations."
    )

    # ------------------------------------------
    # 1. GET ACTIVE DATASET
    # ------------------------------------------

    analysis_df = st.session_state.get("cleaned_dataset")

    if analysis_df is None:
        analysis_df = st.session_state.get("dataset")

    if analysis_df is None:
        st.warning("Please upload a dataset first.")

    elif analysis_df.empty:
        st.warning("The selected dataset is empty.")

    else:

        groupable_columns = get_groupable_columns(analysis_df)
        numerical_columns = get_numerical_columns(analysis_df)
        categorical_columns = get_categorical_columns(analysis_df)

        # Define once for use throughout this page
        include_missing = st.checkbox(
            "Include missing group values",
            value=False,
            key="group_include_missing"
        )

        if not groupable_columns:

            st.warning("No columns are available for group analysis.")

        else:

            # ------------------------------------------
            # 2. GROUP-WISE NUMERICAL ANALYSIS
            # ------------------------------------------

            section_heading("1. Group-wise Numerical Analysis")

            st.write(
                "Compare numerical values across one or more groups."
            )

            selected_groups = st.multiselect(
                "Select grouping column(s)",
                options=groupable_columns,
                key="group_numerical_columns"
            )

            available_metrics = [
                col for col in numerical_columns
                if col not in selected_groups
            ]

            if selected_groups and available_metrics:

                selected_metric = st.selectbox(
                    "Select numerical variable",
                    available_metrics,
                    key="group_numerical_metric"
                )

                try:

                    summary_df = get_group_summary(
                        analysis_df,
                        selected_groups,
                        selected_metric,
                        include_missing_groups=include_missing
                    )

                    if not summary_df.empty:

                        st.markdown("#### Group-wise Summary")

                        st.dataframe(
                            summary_df,
                            use_container_width=True
                        )

                        st.download_button(
                            "Download Group Summary CSV",
                            data=summary_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="group_wise_summary.csv",
                            mime="text/csv"
                        )

                        # Create a readable group label
                        chart_df = summary_df.copy()

                        chart_df["Group"] = (
                            chart_df[selected_groups]
                            .astype("string")
                            .fillna("Missing")
                            .agg(" | ".join, axis=1)
                        )

                        st.markdown("#### Group Comparison Chart")

                        chart_metric = st.selectbox(
                            "Choose metric for visualization",
                            [
                                "Mean",
                                "Median",
                                "Sum",
                                "Count",
                                "Standard_Deviation",
                                "Minimum",
                                "Maximum"
                            ],
                            key="group_chart_metric"
                        )

                        fig = px.bar(
                            chart_df,
                            x="Group",
                            y=chart_metric,
                            title=(
                                f"{chart_metric} of {selected_metric} "
                                "Across Groups"
                            ),
                            text_auto=".2f"
                        )

                        fig.update_layout(
                            xaxis_title="Groups",
                            yaxis_title=chart_metric,
                            xaxis_tickangle=-30
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # Automatic observations
                        if len(selected_groups) == 1:

                            st.markdown(
                                "#### Automatic Observations"
                            )

                            observations = get_group_observations(
                                summary_df,
                                selected_groups[0],
                                chart_metric
                            )

                            for observation in observations:
                                st.info(observation)

                            # Group ranking
                            st.markdown("#### Group Ranking")

                            ranking_metric = st.selectbox(
                                "Select ranking metric",
                                [
                                    "Mean",
                                    "Median",
                                    "Sum",
                                    "Count",
                                    "Standard_Deviation",
                                    "Minimum",
                                    "Maximum"
                                ],
                                key="group_ranking_metric"
                            )

                            ranking_order = st.radio(
                                "Ranking order",
                                [
                                    "Highest to Lowest",
                                    "Lowest to Highest"
                                ],
                                horizontal=True,
                                key="group_ranking_order"
                            )

                            ranking_df = rank_groups(
                                summary_df,
                                selected_groups[0],
                                ranking_metric,
                                ascending=(
                                    ranking_order == "Lowest to Highest"
                                )
                            )

                            st.dataframe(
                                ranking_df,
                                use_container_width=True
                            )

                            st.download_button(
                                "Download Group Ranking CSV",
                                data=ranking_df.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="group_ranking.csv",
                                mime="text/csv"
                            )

                    else:

                        st.warning(
                            "No valid numerical data is available "
                            "for the selected groups."
                        )

                except (ValueError, KeyError) as e:

                    st.error(f"Group-wise numerical analysis error: {e}")

            elif selected_groups and not available_metrics:

                st.info(
                    "Select a grouping column and make sure at least "
                    "one different numerical column is available."
                )

            else:

                st.info(
                    "Select at least one grouping column to begin "
                    "numerical group analysis."
                )

            # ------------------------------------------
            # 3. GROUP-WISE CATEGORICAL ANALYSIS
            # ------------------------------------------

            st.divider()

            section_heading("2. Group-wise Categorical Analysis")

            st.write(
                "Explore how categories are distributed within groups."
            )

            if categorical_columns:

                frequency_group = st.selectbox(
                    "Select grouping variable",
                    groupable_columns,
                    key="frequency_group"
                )

                category_options = [
                    col for col in categorical_columns
                    if col != frequency_group
                ]

                if category_options:

                    frequency_category = st.selectbox(
                        "Select categorical variable",
                        category_options,
                        key="frequency_category"
                    )

                    try:

                        frequency_df = get_group_frequency(
                            analysis_df,
                            frequency_group,
                            frequency_category,
                            include_missing_groups=include_missing
                        )

                        if not frequency_df.empty:

                            st.markdown(
                                "#### Category Frequency Distribution"
                            )

                            st.dataframe(
                                frequency_df,
                                use_container_width=True
                            )

                            st.download_button(
                                "Download Category Frequency CSV",
                                data=frequency_df.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="group_category_frequency.csv",
                                mime="text/csv"
                            )

                            fig = px.bar(
                                frequency_df,
                                x=frequency_group,
                                y="Count",
                                color=frequency_category,
                                barmode="group",
                                title=(
                                    "Category Distribution Across Groups"
                                )
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                "No valid categorical data is available."
                            )

                    except (ValueError, KeyError) as e:

                        st.error(
                            f"Categorical analysis error: {e}"
                        )

                else:

                    st.info(
                        "No other categorical column is available "
                        "for comparison with the selected group."
                    )

            else:

                st.info(
                    "This dataset does not contain categorical columns."
                )

            # ------------------------------------------
            # 4. GROUP SIZE DISTRIBUTION
            # ------------------------------------------

            st.divider()

            section_heading("3. Group Size Distribution")

            size_groups = st.multiselect(
                "Select column(s) to calculate group sizes",
                groupable_columns,
                key="size_group_columns"
            )

            if size_groups:

                try:

                    size_df = get_group_sizes(
                        analysis_df,
                        size_groups,
                        include_missing_groups=include_missing
                    )

                    if not size_df.empty:

                        st.markdown("#### Group Size Summary")

                        st.dataframe(
                            size_df,
                            use_container_width=True
                        )

                        size_chart_df = size_df.copy()

                        size_chart_df["Group"] = (
                            size_chart_df[size_groups]
                            .astype("string")
                            .fillna("Missing")
                            .agg(" | ".join, axis=1)
                        )

                        fig = px.bar(
                            size_chart_df,
                            x="Group",
                            y="Record Count",
                            title="Number of Records in Each Group",
                            text="Record Count"
                        )

                        fig.update_layout(
                            xaxis_title="Groups",
                            yaxis_title="Number of Records",
                            xaxis_tickangle=-30
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.download_button(
                            "Download Group Size CSV",
                            data=size_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="group_size_distribution.csv",
                            mime="text/csv"
                        )

                    else:

                        st.warning(
                            "No group size information is available."
                        )

                except (ValueError, KeyError) as e:

                    st.error(f"Group size analysis error: {e}")

            else:

                st.info(
                    "Select one or more columns to analyse group sizes."
                )

            # ------------------------------------------
            # 5. MULTIPLE VARIABLE GROUP COMPARISON
            # ------------------------------------------

            st.divider()

            section_heading("4. Multiple Variable Group Comparison")

            st.write(
                "Compare multiple numerical variables across "
                "one or more grouping columns."
            )

            multi_groups = st.multiselect(
                "Select grouping column(s)",
                groupable_columns,
                key="multi_group_columns"
            )

            available_multi_metrics = [
                col for col in numerical_columns
                if col not in multi_groups
            ]

            multi_metrics = st.multiselect(
                "Select numerical variables",
                available_multi_metrics,
                key="multi_group_metrics"
            )

            if multi_groups and multi_metrics:

                try:

                    multi_summary = get_multi_group_summary(
                        analysis_df,
                        multi_groups,
                        multi_metrics
                    )

                    if not multi_summary.empty:

                        st.markdown(
                            "#### Multiple Variable Comparison Results"
                        )

                        st.dataframe(
                            multi_summary,
                            use_container_width=True
                        )

                        st.download_button(
                            "Download Multiple Group Comparison CSV",
                            data=multi_summary.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="multiple_group_comparison.csv",
                            mime="text/csv"
                        )

                    else:

                        st.warning(
                            "No valid data is available for this comparison."
                        )

                except (ValueError, KeyError, TypeError) as e:

                    st.error(
                        f"Multiple variable comparison error: {e}"
                    )

            else:

                st.info(
                    "Select at least one grouping column and "
                    "one numerical variable."
                )

            # ------------------------------------------
            # 6. COMPLETION MESSAGE
            # ------------------------------------------

            st.divider()

            st.success(
                "Group-wise Analysis is ready. "
                "Explore the different sections to compare your data."
            )

elif page == "Multivariate Analysis":

    import plotly.express as px
    import plotly.graph_objects as go

    page_header("Multivariate Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Explore relationships among multiple numerical variables "
        "using correlation, pairwise analysis, multicollinearity "
        "and Principal Component Analysis."
    )

    # ------------------------------------------
    # 1. GET ACTIVE DATASET
    # ------------------------------------------

    analysis_df = st.session_state.get("cleaned_dataset")

    if analysis_df is None:
        analysis_df = st.session_state.get("dataset")

    if analysis_df is None:

        st.warning("Please upload a dataset first.")

    elif analysis_df.empty:

        st.warning("The selected dataset is empty.")

    else:

        numerical_columns = get_multivariate_columns(analysis_df)

        if len(numerical_columns) < 2:

            st.warning(
                "Multivariate Analysis requires at least "
                "two numerical columns."
            )

        else:

            section_heading("Dataset Overview")

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Records", len(analysis_df))
            c2.metric("Numerical Variables", len(numerical_columns))
            c3.metric(
                "Missing Numerical Cells",
                int(
                    analysis_df[numerical_columns]
                    .isna()
                    .sum()
                    .sum()
                )
            )

            # ------------------------------------------
            # 2. SELECT VARIABLES
            # ------------------------------------------

            st.divider()

            section_heading("Select Variables")

            selected_columns = st.multiselect(
                "Choose numerical variables for analysis",
                options=numerical_columns,
                default=numerical_columns,
                key="multivariate_selected_columns"
            )

            if len(selected_columns) < 2:

                st.info(
                    "Select at least two numerical variables "
                    "to perform multivariate analysis."
                )

            else:

                try:

                    prepared_data = prepare_multivariate_data(
                        analysis_df,
                        selected_columns
                    )

                    if prepared_data.shape[1] < 2:

                        st.warning(
                            "At least two valid, non-constant "
                            "variables are required."
                        )

                    else:

                        # ------------------------------------------
                        # 3. CORRELATION MATRIX
                        # ------------------------------------------

                        st.divider()

                        section_heading("1. Multivariate Correlation Analysis")

                        correlation_method = st.selectbox(
                            "Select correlation method",
                            ["pearson", "spearman", "kendall"],
                            key="multivariate_correlation_method"
                        )

                        correlation_matrix = get_multivariate_correlation(
                            analysis_df,
                            selected_columns,
                            method=correlation_method
                        )

                        st.markdown("#### Correlation Matrix")

                        st.dataframe(
                            correlation_matrix.style.format("{:.3f}"),
                            use_container_width=True
                        )

                        fig = px.imshow(
                            correlation_matrix,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            zmin=-1,
                            zmax=1,
                            aspect="auto",
                            title=(
                                f"{correlation_method.title()} "
                                "Correlation Heatmap"
                            )
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.markdown("#### Automatic Correlation Observations")

                        observations = get_multivariate_observations(
                            correlation_matrix
                        )

                        for observation in observations:
                            st.info(observation)

                        st.download_button(
                            "Download Correlation Matrix CSV",
                            data=correlation_matrix.to_csv().encode("utf-8"),
                            file_name="multivariate_correlation.csv",
                            mime="text/csv"
                        )

                        # ------------------------------------------
                        # 4. PAIRWISE ANALYSIS
                        # ------------------------------------------

                        st.divider()

                        section_heading("2. Pairwise Relationship Analysis")

                        pair_data = get_pairwise_data(
                            analysis_df,
                            selected_columns
                        )

                        color_options = [
                            col for col in analysis_df.columns
                            if col not in selected_columns
                            and (
                                analysis_df[col].nunique(dropna=True) <= 20
                                or analysis_df[col].dtype == "object"
                                or str(analysis_df[col].dtype) == "category"
                            )
                        ]

                        selected_color = st.selectbox(
                            "Colour points by (optional)",
                            ["None"] + color_options,
                            key="multivariate_pair_color"
                        )

                        pair_plot_df = pair_data.copy()

                        if selected_color != "None":
                            pair_plot_df[selected_color] = (
                                analysis_df.loc[
                                    pair_plot_df.index,
                                    selected_color
                                ].astype("string").fillna("Missing")
                            )

                        fig = px.scatter_matrix(
                            pair_plot_df,
                            dimensions=pair_data.columns.tolist(),
                            color=(
                                selected_color
                                if selected_color != "None"
                                else None
                            ),
                            title="Pairwise Variable Relationships",
                            height=850
                        )

                        fig.update_traces(
                            diagonal_visible=False,
                            showupperhalf=False
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # ------------------------------------------
                        # 5. MULTICOLLINEARITY ANALYSIS
                        # ------------------------------------------

                        st.divider()

                        section_heading("3. Multicollinearity Analysis")

                        st.write(
                            "Variance Inflation Factor (VIF) helps "
                            "identify overlapping information among "
                            "numerical predictor variables."
                        )

                        if st.button(
                            "Calculate VIF",
                            key="calculate_multivariate_vif"
                        ):

                            try:

                                vif_df = get_vif_analysis(
                                    analysis_df,
                                    selected_columns
                                )

                                st.session_state[
                                    "multivariate_vif_result"
                                ] = vif_df

                            except ValueError as e:

                                st.error(str(e))

                        vif_df = st.session_state.get(
                            "multivariate_vif_result"
                        )

                        if (
                            vif_df is not None
                            and all(
                                col in vif_df.columns
                                for col in ["Variable", "VIF", "Interpretation"]
                            )
                        ):

                            st.dataframe(
                                vif_df,
                                use_container_width=True
                            )

                            st.download_button(
                                "Download VIF Report CSV",
                                data=vif_df.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="multicollinearity_vif.csv",
                                mime="text/csv"
                            )

                            st.caption(
                                "General interpretation: VIF below 5 "
                                "is often considered low, 5–10 moderate, "
                                "and above 10 high. These are guidelines, "
                                "not universal cutoffs."
                            )

                        # ------------------------------------------
                        # 6. PRINCIPAL COMPONENT ANALYSIS
                        # ------------------------------------------

                        st.divider()

                        section_heading("4. Principal Component Analysis (PCA)")

                        st.write(
                            "PCA transforms the selected numerical "
                            "variables into fewer components while "
                            "preserving as much variance as possible."
                        )

                        max_components = min(
                            len(selected_columns),
                            len(prepared_data)
                        )

                        number_of_components = st.slider(
                            "Number of principal components",
                            min_value=1,
                            max_value=max_components,
                            value=min(2, max_components),
                            key="multivariate_pca_components"
                        )

                        if st.button(
                            "Run Principal Component Analysis",
                            key="run_multivariate_pca"
                        ):

                            try:

                                pca_result = perform_pca(
                                    analysis_df,
                                    selected_columns,
                                    number_of_components
                                )

                                st.session_state[
                                    "multivariate_pca_result"
                                ] = pca_result

                            except ValueError as e:

                                st.error(str(e))

                        pca_result = st.session_state.get(
                            "multivariate_pca_result"
                        )

                        if (
                            pca_result is not None
                            and pca_result["scores"].shape[1]
                            == number_of_components
                        ):

                            variance_df = pca_result[
                                "explained_variance"
                            ]

                            st.markdown("#### Explained Variance")

                            st.dataframe(
                                variance_df,
                                use_container_width=True
                            )

                            total_variance = float(
                                variance_df[
                                    "Explained Variance Ratio"
                                ].sum() * 100
                            )

                            st.metric(
                                "Total Variance Explained",
                                f"{total_variance:.2f}%"
                            )

                            fig = px.bar(
                                variance_df,
                                x="Component",
                                y="Explained Variance (%)",
                                title="Explained Variance by Component",
                                text_auto=".2f"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            # Cumulative variance chart

                            fig = px.line(
                                variance_df,
                                x="Component",
                                y="Cumulative Variance (%)",
                                markers=True,
                                title="Cumulative Explained Variance"
                            )

                            fig.update_yaxes(
                                range=[0, 105]
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            # PCA loadings

                            st.markdown("#### PCA Component Loadings")

                            st.dataframe(
                                pca_result["loadings"].style.format("{:.3f}"),
                                use_container_width=True
                            )

                            fig = px.imshow(
                                pca_result["loadings"],
                                text_auto=".2f",
                                color_continuous_scale="RdBu_r",
                                aspect="auto",
                                title="PCA Loading Heatmap"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            # PCA score visualization

                            st.markdown("#### PCA Score Visualization")

                            scores = pca_result["scores"].copy()

                            if scores.shape[1] >= 2:

                                fig = px.scatter(
                                    scores,
                                    x="PC1",
                                    y="PC2",
                                    title="PCA: First Two Principal Components"
                                )

                                st.plotly_chart(
                                    fig,
                                    use_container_width=True
                                )

                            # PCA observations

                            st.markdown("#### PCA Observations")

                            for observation in get_pca_observations(
                                pca_result
                            ):
                                st.info(observation)

                            # Download PCA results

                            st.download_button(
                                "Download PCA Explained Variance CSV",
                                data=variance_df.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="pca_explained_variance.csv",
                                mime="text/csv"
                            )

                            st.download_button(
                                "Download PCA Loadings CSV",
                                data=pca_result["loadings"].to_csv().encode(
                                    "utf-8"
                                ),
                                file_name="pca_loadings.csv",
                                mime="text/csv"
                            )

                            st.download_button(
                                "Download PCA Scores CSV",
                                data=scores.to_csv(index=False).encode(
                                    "utf-8"
                                ),
                                file_name="pca_scores.csv",
                                mime="text/csv"
                            )

                except (ValueError, KeyError, TypeError) as e:

                    st.error(
                        f"Multivariate analysis error: {e}"
                    )

            st.divider()

            st.success(
                "Multivariate Analysis is ready. "
                "Explore correlations, pairwise relationships, "
                "multicollinearity and PCA."
            )

elif page == "Statistical Analysis":

    import plotly.express as px

    page_header("Statistical Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Perform hypothesis tests, evaluate statistical assumptions, "
        "calculate confidence intervals and build linear regression models."
    )

    analysis_df = st.session_state.get("cleaned_dataset")
    if analysis_df is None:
        analysis_df = st.session_state.get("dataset")

    if analysis_df is None:
        st.warning("Please upload a dataset first.")

    elif analysis_df.empty:
        st.warning("The selected dataset is empty.")

    else:
        numerical_columns = [
            column for column in analysis_df.select_dtypes(include="number").columns
            if not pd.api.types.is_bool_dtype(analysis_df[column])
        ]

        categorical_columns = [
            column for column in analysis_df.columns
            if (
                not pd.api.types.is_numeric_dtype(analysis_df[column])
                or pd.api.types.is_bool_dtype(analysis_df[column])
            )
        ]

        if not numerical_columns:
            st.warning("This page requires at least one numerical variable.")

        else:
            section_heading("Dataset Overview")

            c1, c2, c3 = st.columns(3)
            c1.metric("Records", len(analysis_df))
            c2.metric("Numerical Variables", len(numerical_columns))
            c3.metric("Categorical Variables", len(categorical_columns))

            st.divider()

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Hypothesis Tests",
                "Existing Tests",
                "Assumptions",
                "Confidence Intervals",
                "Regression"
            ])

            # ==========================================
            # TAB 1: HYPOTHESIS TESTS
            # ==========================================

            with tab1:
                section_heading("Hypothesis Testing")

                test_type = st.selectbox(
                    "Select a statistical test",
                    [
                        "One-Sample T-Test",
                        "Independent T-Test (Welch)",
                        "Paired T-Test",
                        "Mann-Whitney U Test",
                        "Wilcoxon Signed-Rank Test"
                    ],
                    key="stat_test_type"
                )

                alpha = st.selectbox(
                    "Significance Level (alpha)",
                    [0.01, 0.05, 0.10],
                    index=1,
                    key="stat_test_alpha"
                )

                try:
                    if test_type == "One-Sample T-Test":

                        column = st.selectbox(
                            "Numerical variable",
                            numerical_columns,
                            key="stat_one_sample_column"
                        )

                        population_mean = st.number_input(
                            "Hypothesized population mean",
                            value=0.0,
                            key="stat_population_mean"
                        )

                        if st.button("Run One-Sample T-Test"):
                            result = one_sample_t_test(
                                analysis_df,
                                column,
                                population_mean,
                                alpha
                            )
                            st.session_state["stat_test_result"] = result

                    elif test_type in [
                        "Independent T-Test (Welch)",
                        "Mann-Whitney U Test"
                    ]:

                        if not categorical_columns:
                            st.info(
                                "This test requires a categorical grouping variable."
                            )
                        else:
                            numerical_column = st.selectbox(
                                "Numerical variable",
                                numerical_columns,
                                key="stat_ind_num"
                            )

                            group_column = st.selectbox(
                                "Grouping variable",
                                categorical_columns,
                                key="stat_ind_group"
                            )

                            group_values = (
                                analysis_df[group_column]
                                .dropna()
                                .unique()
                                .tolist()
                            )

                            if len(group_values) < 2:
                                st.warning("The grouping variable needs at least two groups.")
                            else:
                                g1, g2 = st.columns(2)

                                group_a = g1.selectbox(
                                    "First group",
                                    group_values,
                                    key="stat_group_a"
                                )

                                remaining_groups = [
                                    value for value in group_values
                                    if value != group_a
                                ]

                                group_b = g2.selectbox(
                                    "Second group",
                                    remaining_groups,
                                    key="stat_group_b"
                                )

                                if st.button("Run Selected Test"):
                                    if test_type == "Independent T-Test (Welch)":
                                        result = independent_t_test(
                                            analysis_df,
                                            numerical_column,
                                            group_column,
                                            group_a,
                                            group_b,
                                            alpha
                                        )
                                    else:
                                        result = mann_whitney_test(
                                            analysis_df,
                                            numerical_column,
                                            group_column,
                                            group_a,
                                            group_b,
                                            alpha
                                        )

                                    st.session_state["stat_test_result"] = result

                    else:
                        col1, col2 = st.columns(2)

                        column_a = col1.selectbox(
                            "First measurement",
                            numerical_columns,
                            key="stat_paired_a"
                        )

                        column_b = col2.selectbox(
                            "Second measurement",
                            numerical_columns,
                            index=min(1, len(numerical_columns) - 1),
                            key="stat_paired_b"
                        )

                        if column_a == column_b:
                            st.warning("Select two different measurements.")
                        elif st.button("Run Paired Test"):
                            if test_type == "Paired T-Test":
                                result = paired_t_test(
                                    analysis_df, column_a, column_b, alpha
                                )
                            else:
                                result = wilcoxon_test(
                                    analysis_df, column_a, column_b, alpha
                                )

                            st.session_state["stat_test_result"] = result

                    if "stat_test_result" in st.session_state:
                        result = st.session_state["stat_test_result"]

                        st.divider()
                        section_heading("Test Results")

                        m1, m2 = st.columns(2)
                        m1.metric("Test Statistic", f"{result['Test Statistic']:.4f}")
                        m2.metric("P-value", f"{result['P-value']:.6g}")

                        st.write("**Test:**", result["Test"])
                        st.write("**Interpretation:**", result["Interpretation"])

                        with st.expander("View all test details"):
                            st.json({
                                key: value
                                for key, value in result.items()
                                if not isinstance(value, pd.DataFrame)
                            })

                except (ValueError, KeyError, TypeError) as e:
                    st.error(f"Unable to perform the test: {e}")

            # ==========================================
            # TAB 2: EXISTING PHASE 12 TESTS
            # ==========================================

            with tab2:
                section_heading("Reuse Existing Statistical Tests")

                st.caption(
                    "These tests reuse the implementations already developed "
                    "in Phase 12."
                )

                existing_test = st.selectbox(
                    "Select an existing test",
                    [
                        "One-Way ANOVA",
                        "Kruskal-Wallis Test",
                        "Chi-Square Test"
                    ],
                    key="stat_existing_test"
                )

                try:
                    if existing_test in [
                        "One-Way ANOVA",
                        "Kruskal-Wallis Test"
                    ]:

                        if not categorical_columns:
                            st.info("A categorical grouping variable is required.")
                        else:
                            num_col = st.selectbox(
                                "Numerical variable",
                                numerical_columns,
                                key="stat_existing_num"
                            )

                            group_col = st.selectbox(
                                "Grouping variable",
                                categorical_columns,
                                key="stat_existing_group"
                            )

                            if st.button("Run Existing Group Test"):
                                result = analyze_numerical_by_category(
                                    analysis_df,
                                    num_col,
                                    group_col
                                )

                                st.write("**Phase 12 analysis result**")
                                st.write(result)

                    else:
                        if len(categorical_columns) < 2:
                            st.info(
                                "At least two categorical variables are required."
                            )
                        else:
                            col_a, col_b = st.columns(2)

                            category_a = col_a.selectbox(
                                "First categorical variable",
                                categorical_columns,
                                key="stat_chi_a"
                            )

                            category_b_options = [
                                col for col in categorical_columns
                                if col != category_a
                            ]

                            category_b = col_b.selectbox(
                                "Second categorical variable",
                                category_b_options,
                                key="stat_chi_b"
                            )

                            if st.button("Run Chi-Square Test"):
                                result = analyze_categorical_relationship(
                                    analysis_df,
                                    category_a,
                                    category_b
                                )

                                st.write("**Phase 12 analysis result**")
                                st.write(result)

                except (ValueError, KeyError, TypeError) as e:
                    st.error(f"Unable to perform the existing test: {e}")

            # ==========================================
            # TAB 3: ASSUMPTION CHECKS
            # ==========================================

            with tab3:
                section_heading("Statistical Assumption Checks")

                st.write("**1. Normality testing**")

                selected_normality_columns = st.multiselect(
                    "Select numerical variables",
                    numerical_columns,
                    default=numerical_columns[:min(3, len(numerical_columns))],
                    key="stat_normality_columns"
                )

                if st.button("Check Normality"):
                    if not selected_normality_columns:
                        st.warning("Select at least one variable.")
                    else:
                        normality_result = check_normality(
                            analysis_df,
                            selected_normality_columns
                        )
                        st.dataframe(
                            normality_result,
                            use_container_width=True
                        )

                st.divider()
                st.write("**2. Equal variance testing (Levene's test)**")

                if categorical_columns:
                    variance_num = st.selectbox(
                        "Numerical variable",
                        numerical_columns,
                        key="stat_variance_num"
                    )

                    variance_group = st.selectbox(
                        "Grouping variable",
                        categorical_columns,
                        key="stat_variance_group"
                    )

                    if st.button("Check Equal Variance"):
                        try:
                            variance_result = check_equal_variance(
                                analysis_df,
                                variance_num,
                                variance_group
                            )

                            st.json(variance_result)

                        except (ValueError, KeyError, TypeError) as e:
                            st.error(str(e))
                else:
                    st.info("No categorical grouping variable is available.")

                st.caption(
                    "A p-value below 0.05 indicates evidence against the "
                    "corresponding assumption. These tests should be interpreted "
                    "alongside plots, sample sizes and study design."
                )

            # ==========================================
            # TAB 4: CONFIDENCE INTERVALS
            # ==========================================

            with tab4:
                section_heading("Confidence Interval Analysis")

                interval_type = st.radio(
                    "Select interval type",
                    [
                        "Single Mean",
                        "Difference Between Two Group Means"
                    ],
                    key="stat_interval_type"
                )

                confidence_level = st.selectbox(
                    "Confidence level",
                    [0.90, 0.95, 0.99],
                    index=1,
                    format_func=lambda x: f"{int(x * 100)}%",
                    key="stat_confidence_level"
                )

                try:
                    if interval_type == "Single Mean":

                        ci_column = st.selectbox(
                            "Numerical variable",
                            numerical_columns,
                            key="stat_ci_column"
                        )

                        if st.button("Calculate Confidence Interval"):
                            result = confidence_interval_mean(
                                analysis_df,
                                ci_column,
                                confidence_level
                            )

                            st.session_state["stat_ci_result"] = result

                    else:
                        if not categorical_columns:
                            st.info("A categorical grouping variable is required.")
                        else:
                            ci_num = st.selectbox(
                                "Numerical variable",
                                numerical_columns,
                                key="stat_ci_num"
                            )

                            ci_group = st.selectbox(
                                "Grouping variable",
                                categorical_columns,
                                key="stat_ci_group"
                            )

                            ci_values = (
                                analysis_df[ci_group]
                                .dropna()
                                .unique()
                                .tolist()
                            )

                            if len(ci_values) >= 2:
                                ci_a, ci_b = st.columns(2)

                                ci_group_a = ci_a.selectbox(
                                    "First group",
                                    ci_values,
                                    key="stat_ci_a"
                                )

                                ci_remaining = [
                                    value for value in ci_values
                                    if value != ci_group_a
                                ]

                                ci_group_b = ci_b.selectbox(
                                    "Second group",
                                    ci_remaining,
                                    key="stat_ci_b"
                                )

                                if st.button("Calculate Group Difference Interval"):
                                    result = confidence_interval_difference(
                                        analysis_df,
                                        ci_num,
                                        ci_group,
                                        ci_group_a,
                                        ci_group_b,
                                        confidence_level
                                    )

                                    st.session_state["stat_ci_result"] = result
                            else:
                                st.warning("At least two groups are required.")

                    if "stat_ci_result" in st.session_state:
                        result = st.session_state["stat_ci_result"]

                        st.divider()
                        section_heading("Confidence Interval Result")
                        st.json(result)

                except (ValueError, KeyError, TypeError) as e:
                    st.error(f"Unable to calculate interval: {e}")

            # ==========================================
            # TAB 5: REGRESSION
            # ==========================================

            with tab5:
                section_heading("Linear Regression")

                st.write(
                    "Build a simple or multiple linear regression model "
                    "using numerical predictors."
                )

                target = st.selectbox(
                    "Target variable (dependent variable)",
                    numerical_columns,
                    key="stat_reg_target"
                )

                predictor_options = [
                    column for column in numerical_columns
                    if column != target
                ]

                predictors = st.multiselect(
                    "Predictor variable(s) (independent variables)",
                    predictor_options,
                    default=predictor_options[:min(2, len(predictor_options))],
                    key="stat_reg_predictors"
                )

                if st.button("Run Linear Regression"):
                    try:
                        if not predictors:
                            st.warning("Select at least one predictor.")
                        else:
                            regression_result = linear_regression(
                                analysis_df,
                                predictors,
                                target
                            )

                            st.session_state["stat_regression_result"] = (
                                regression_result
                            )

                    except (ValueError, KeyError, TypeError) as e:
                        st.error(f"Regression failed: {e}")

                if "stat_regression_result" in st.session_state:
                    result = st.session_state["stat_regression_result"]

                    st.divider()
                    section_heading("Model Performance")

                    r1, r2, r3 = st.columns(3)
                    r1.metric("R-squared", f"{result['R-squared']:.4f}")
                    r2.metric(
                        "Adjusted R-squared",
                        f"{result['Adjusted R-squared']:.4f}"
                    )
                    r3.metric("Observations", result["Observations"])

                    section_heading("Regression Coefficients")
                    st.dataframe(
                        result["Coefficients"],
                        use_container_width=True
                    )

                    section_heading("Actual vs Predicted")

                    prediction_data = result["Predictions"].reset_index(
                        drop=True
                    )

                    fig = px.scatter(
                        prediction_data,
                        x="Actual",
                        y="Predicted",
                        title="Actual Values vs Predicted Values"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    section_heading("Regression Diagnostics")

                    d1, d2, d3 = st.columns(3)

                    normality_p = result["Residual Normality P-value"]
                    bp_p = result["Breusch-Pagan P-value"]
                    dw = result["Durbin-Watson Statistic"]

                    d1.metric(
                        "Residual Normality p-value",
                        f"{normality_p:.4f}"
                        if pd.notna(normality_p) else "N/A"
                    )

                    d2.metric(
                        "Breusch-Pagan p-value",
                        f"{bp_p:.4f}"
                        if pd.notna(bp_p) else "N/A"
                    )

                    d3.metric("Durbin-Watson", f"{dw:.4f}")

                    with st.expander("View full regression model summary"):
                        st.code(result["Model Summary"])

                    st.download_button(
                        "Download Regression Coefficients",
                        data=result["Coefficients"].to_csv(index=False),
                        file_name="regression_coefficients.csv",
                        mime="text/csv"
                    )

                    st.download_button(
                        "Download Predictions and Residuals",
                        data=result["Predictions"].to_csv(index=False),
                        file_name="regression_predictions.csv",
                        mime="text/csv"
                    )

# ==========================================
# PHASE 16: TIME-SERIES ANALYSIS
# ==========================================

elif page == "Time-Series Analysis":

    page_header("📈 Time-Series Analysis", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Analyze how numerical variables change over time using "
        "trend analysis, resampling, rolling statistics, growth "
        "analysis, seasonality, stationarity and forecasting."
    )

    # Use cleaned dataset when available
    analysis_df = st.session_state.get("cleaned_dataset")

    if analysis_df is None:
        analysis_df = st.session_state.get("dataset")

    if analysis_df is None:
        st.warning("Please upload a dataset first.")

    else:
        date_columns = get_datetime_columns(analysis_df)
        numerical_columns = get_numerical_columns(analysis_df)

        if not date_columns:
            st.warning(
                "No date/time columns were detected. "
                "Please convert a suitable column to Date/Time "
                "using the Data Cleaning module."
            )

        elif not numerical_columns:
            st.warning(
                "No numerical columns are available for time-series analysis."
            )

        else:
            section_heading("1. Configure Time Series")

            col1, col2 = st.columns(2)

            with col1:
                date_column = st.selectbox(
                    "Select date/time column",
                    date_columns,
                    key="phase16_date_column"
                )

            with col2:
                value_column = st.selectbox(
                    "Select numerical variable",
                    numerical_columns,
                    key="phase16_value_column"
                )

            frequency_options = {
                "Daily": "D",
                "Weekly": "W",
                "Monthly": "MS",
                "Quarterly": "QS",
                "Yearly": "YS"
            }

            selected_frequency = st.selectbox(
                "Select time frequency",
                list(frequency_options.keys()),
                index=2,
                key="phase16_frequency"
            )

            aggregation = st.selectbox(
                "Aggregation method",
                ["mean", "sum", "median", "min", "max"],
                key="phase16_aggregation"
            )

            try:
                # Prepare date-indexed series
                raw_series = prepare_time_series(
                    analysis_df,
                    date_column,
                    value_column
                )

                series = get_resampled_series(
                    raw_series,
                    frequency_options[selected_frequency],
                    aggregation
                )

                if series.dropna().empty:
                    st.warning(
                        "No valid observations are available "
                        "for the selected time frequency."
                    )
                    st.stop()

                st.divider()

                # ======================================
                # OVERVIEW
                # ======================================

                section_heading("2. Time-Series Overview")

                summary = get_time_series_summary(series)

                metric1, metric2, metric3, metric4 = st.columns(4)

                metric1.metric(
                    "Observations",
                    summary["Observations"]
                )

                metric2.metric(
                    "Mean",
                    f"{summary['Mean']:.3f}"
                )

                metric3.metric(
                    "Minimum",
                    f"{summary['Minimum']:.3f}"
                )

                metric4.metric(
                    "Maximum",
                    f"{summary['Maximum']:.3f}"
                )

                st.write("**Detailed Summary**")

                summary_df = pd.DataFrame(
                    list(summary.items()),
                    columns=["Metric", "Value"]
                )

                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Time-Series Summary",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="datalens_time_series_summary.csv",
                    mime="text/csv"
                )

                import plotly.express as px
                import plotly.graph_objects as go

                st.divider()

                # ======================================
                # TABS
                # ======================================

                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Trend Analysis",
                    "Rolling Statistics",
                    "Growth Analysis",
                    "Seasonality",
                    "Stationarity & ACF",
                    "Forecasting"
                ])

                # ======================================
                # TAB 1: TREND ANALYSIS
                # ======================================

                with tab1:

                    section_heading("Time-Series Trend")

                    trend_df = series.dropna().rename(
                        "Value"
                    ).reset_index()

                    trend_df.columns = ["Date", "Value"]

                    fig = px.line(
                        trend_df,
                        x="Date",
                        y="Value",
                        markers=True,
                        title=f"{value_column} Over Time"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    observations = generate_time_series_observations(
                        series
                    )

                    st.write("**Automatic Observations**")

                    for index, observation in enumerate(
                        observations,
                        start=1
                    ):
                        st.write(f"**{index}.** {observation}")

                # ======================================
                # TAB 2: ROLLING STATISTICS
                # ======================================

                with tab2:

                    section_heading("Moving Average and Rolling Statistics")

                    window = st.slider(
                        "Rolling window size",
                        min_value=2,
                        max_value=max(2, min(24, len(series.dropna()))),
                        value=min(3, max(2, len(series.dropna()))),
                        key="phase16_rolling_window"
                    )

                    if len(series.dropna()) < 2:
                        st.info(
                            "At least two observations are required."
                        )
                    else:
                        rolling_df = get_rolling_statistics(
                            series,
                            window
                        )

                        st.dataframe(
                            rolling_df,
                            use_container_width=True
                        )

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(
                                x=rolling_df.index,
                                y=rolling_df["Original"],
                                mode="lines",
                                name="Original"
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=rolling_df.index,
                                y=rolling_df["Moving Average"],
                                mode="lines",
                                name="Moving Average"
                            )
                        )

                        fig.update_layout(
                            title="Original Series vs Moving Average",
                            xaxis_title="Date",
                            yaxis_title=value_column
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.download_button(
                            "Download Rolling Statistics",
                            data=rolling_df.to_csv().encode("utf-8"),
                            file_name="datalens_rolling_statistics.csv",
                            mime="text/csv"
                        )

                # ======================================
                # TAB 3: GROWTH ANALYSIS
                # ======================================

                with tab3:

                    section_heading("Period-over-Period Growth")

                    growth_df = get_growth_analysis(series)

                    st.dataframe(
                        growth_df,
                        use_container_width=True
                    )

                    fig = px.line(
                        growth_df.reset_index(),
                        x=growth_df.index.name or "index",
                        y="Percentage Change",
                        title="Percentage Change Over Time",
                        markers=True
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Growth Analysis",
                        data=growth_df.to_csv().encode("utf-8"),
                        file_name="datalens_growth_analysis.csv",
                        mime="text/csv"
                    )

                # ======================================
                # TAB 4: SEASONALITY
                # ======================================

                with tab4:

                    section_heading("Seasonal Decomposition")

                    st.write(
                        "Separate the series into trend, seasonal "
                        "and residual components."
                    )

                    period = st.number_input(
                        "Seasonal period",
                        min_value=2,
                        max_value=max(2, len(series.dropna()) // 2),
                        value=min(12, max(2, len(series.dropna()) // 2)),
                        key="phase16_seasonal_period"
                    )

                    decomposition_model = st.selectbox(
                        "Decomposition model",
                        ["Additive", "Multiplicative"],
                        key="phase16_decomposition_model"
                    )

                    if st.button(
                        "Perform Seasonal Decomposition",
                        key="phase16_decompose_button"
                    ):
                        try:
                            decomposition = perform_seasonal_decomposition(
                                series,
                                period=int(period),
                                model=decomposition_model.lower()
                            )

                            decomposition_df = pd.DataFrame({
                                "Observed": decomposition.observed,
                                "Trend": decomposition.trend,
                                "Seasonal": decomposition.seasonal,
                                "Residual": decomposition.resid
                            })

                            st.session_state[
                                "phase16_decomposition_result"
                            ] = decomposition_df

                        except Exception as error:
                            st.error(
                                f"Seasonal decomposition failed: {error}"
                            )

                    decomposition_df = st.session_state.get(
                        "phase16_decomposition_result"
                    )

                    if decomposition_df is not None:

                        st.dataframe(
                            decomposition_df,
                            use_container_width=True
                        )

                        st.line_chart(decomposition_df)

                        st.download_button(
                            "Download Decomposition",
                            data=decomposition_df.to_csv().encode("utf-8"),
                            file_name="datalens_decomposition.csv",
                            mime="text/csv"
                        )

                # ======================================
                # TAB 5: STATIONARITY AND AUTOCORRELATION
                # ======================================

                with tab5:

                    section_heading("Stationarity Testing")

                    st.caption(
                        "ADF: a low p-value provides evidence of stationarity. "
                        "KPSS: a high p-value provides evidence consistent "
                        "with stationarity."
                    )

                    if st.button(
                        "Run Stationarity Tests",
                        key="phase16_stationarity_button"
                    ):
                        try:
                            stationarity_result = test_stationarity(
                                series
                            )

                            st.session_state[
                                "phase16_stationarity_result"
                            ] = stationarity_result

                        except Exception as error:
                            st.error(
                                f"Stationarity testing failed: {error}"
                            )

                    stationarity_result = st.session_state.get(
                        "phase16_stationarity_result"
                    )

                    if stationarity_result is not None:
                        st.json(stationarity_result)

                    st.divider()

                    section_heading("Autocorrelation Analysis")

                    lag_count = st.slider(
                        "Number of lags",
                        min_value=1,
                        max_value=max(
                            1,
                            min(30, len(series.dropna()) // 2 - 1)
                        ),
                        value=min(
                            12,
                            max(1, len(series.dropna()) // 2 - 1)
                        ),
                        key="phase16_lags"
                    )

                    if st.button(
                        "Calculate ACF and PACF",
                        key="phase16_acf_button"
                    ):
                        try:
                            acf_result = get_autocorrelation(
                                series,
                                lags=lag_count
                            )

                            acf_df = pd.DataFrame({
                                "Lag": acf_result["Lags"],
                                "ACF": acf_result["ACF"],
                                "PACF": acf_result["PACF"]
                            })

                            st.session_state[
                                "phase16_acf_result"
                            ] = acf_df

                        except Exception as error:
                            st.error(
                                f"Autocorrelation analysis failed: {error}"
                            )

                    acf_df = st.session_state.get(
                        "phase16_acf_result"
                    )

                    if acf_df is not None:

                        st.dataframe(
                            acf_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        fig = go.Figure()

                        fig.add_trace(
                            go.Bar(
                                x=acf_df["Lag"],
                                y=acf_df["ACF"],
                                name="ACF"
                            )
                        )

                        fig.update_layout(
                            title="Autocorrelation Function",
                            xaxis_title="Lag",
                            yaxis_title="ACF"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        fig2 = go.Figure()

                        fig2.add_trace(
                            go.Bar(
                                x=acf_df["Lag"],
                                y=acf_df["PACF"],
                                name="PACF"
                            )
                        )

                        fig2.update_layout(
                            title="Partial Autocorrelation Function",
                            xaxis_title="Lag",
                            yaxis_title="PACF"
                        )

                        st.plotly_chart(
                            fig2,
                            use_container_width=True
                        )

                # ======================================
                # TAB 6: FORECASTING
                # ======================================

                with tab6:

                    section_heading("Time-Series Forecasting")

                    forecast_periods = st.number_input(
                        "Number of future periods",
                        min_value=1,
                        max_value=100,
                        value=6,
                        key="phase16_forecast_periods"
                    )

                    use_seasonality = st.checkbox(
                        "Use seasonal forecasting",
                        value=False,
                        key="phase16_forecast_seasonal"
                    )

                    seasonal_periods = 12

                    if use_seasonality:
                        seasonal_periods = st.number_input(
                            "Forecast seasonal period",
                            min_value=2,
                            max_value=max(
                                2,
                                len(series.dropna()) // 2
                            ),
                            value=min(
                                12,
                                max(2, len(series.dropna()) // 2)
                            ),
                            key="phase16_forecast_seasonal_period"
                        )

                    st.warning(
                        "Forecasts are model-based estimates, not "
                        "guaranteed future outcomes. Review the data "
                        "and model suitability before relying on them."
                    )

                    if st.button(
                        "Generate Forecast",
                        key="phase16_forecast_button"
                    ):
                        try:
                            forecast_result = forecast_time_series(
                                series,
                                periods=int(forecast_periods),
                                seasonal=use_seasonality,
                                seasonal_periods=int(seasonal_periods)
                            )

                            st.session_state[
                                "phase16_forecast_result"
                            ] = forecast_result["Forecast"]

                        except Exception as error:
                            st.error(
                                f"Forecasting failed: {error}"
                            )

                    forecast = st.session_state.get(
                        "phase16_forecast_result"
                    )

                    if forecast is not None:

                        st.write("**Forecast Values**")

                        forecast_df = forecast.rename(
                            "Forecast"
                        ).to_frame()

                        st.dataframe(
                            forecast_df,
                            use_container_width=True
                        )

                        historical_df = series.dropna().rename(
                            "Historical"
                        ).to_frame()

                        combined_df = pd.concat([
                            historical_df,
                            forecast_df
                        ])

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(
                                x=historical_df.index,
                                y=historical_df["Historical"],
                                mode="lines",
                                name="Historical"
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df["Forecast"],
                                mode="lines+markers",
                                name="Forecast"
                            )
                        )

                        fig.update_layout(
                            title="Historical Data and Forecast",
                            xaxis_title="Date",
                            yaxis_title=value_column
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        st.download_button(
                            "Download Forecast",
                            data=forecast_df.to_csv().encode("utf-8"),
                            file_name="datalens_forecast.csv",
                            mime="text/csv"
                        )

                st.divider()

                st.success(
                    "Time-Series Analysis module loaded successfully!"
                )

                st.caption(
                    "DataLens | Phase 16: Time-Series Analysis"
                )

            except Exception as error:
                st.error(
                    f"Unable to perform time-series analysis: {error}"
                )

elif page == "Visualization Studio":

    page_header("Visualization Studio", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Create interactive visualizations from your dataset "
        "using customizable charts and filters."
    )

    import plotly.io as pio

    # Select original or cleaned dataset
    original_df = st.session_state.get("dataset")
    cleaned_df = st.session_state.get("cleaned_dataset")

    if original_df is None:
        st.warning("Please upload a dataset first.")
    else:
        dataset_choice = st.radio(
            "Choose dataset",
            ["Original Dataset", "Cleaned Dataset"],
            horizontal=True
        )

        if dataset_choice == "Cleaned Dataset":
            analysis_df = (
                cleaned_df
                if cleaned_df is not None
                else original_df
            )
        else:
            analysis_df = original_df

        if dataset_choice == "Cleaned Dataset" and cleaned_df is None:
            st.info("No cleaned dataset is available. Using the original dataset.")

        if analysis_df is None or analysis_df.empty:
            st.warning("The selected dataset is empty.")
        else:
            column_info = get_visualization_columns(analysis_df)

            numerical_columns = column_info["numerical"]
            categorical_columns = column_info["categorical"]
            datetime_columns = column_info["datetime"]
            all_columns = column_info["all"]

            # Dataset overview
            section_heading("Dataset Overview")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Rows", analysis_df.shape[0])
            c2.metric("Columns", analysis_df.shape[1])
            c3.metric("Numerical", len(numerical_columns))
            c4.metric("Categorical", len(categorical_columns))

            st.divider()

            # Chart recommendations
            section_heading("Recommended Visualizations")

            recommendations = get_chart_recommendations(analysis_df)

            if recommendations:
                st.write(", ".join(recommendations))
            else:
                st.info("No chart recommendations available for this dataset.")

            st.divider()

            # Chart configuration
            section_heading("Create Your Visualization")

            chart_type = st.selectbox(
                "Select Chart Type",
                CHART_TYPES
            )

            st.caption(
                "Choose the columns and settings required for your selected chart."
            )

            x_column = None
            y_column = None
            color_column = None

            # Column selection based on chart type
            if chart_type == "Correlation Heatmap":

                st.info(
                    "A correlation heatmap uses all numerical columns automatically."
                )

            elif chart_type == "Histogram":

                if not numerical_columns:
                    st.warning("Histogram requires a numerical column.")
                else:
                    x_column = st.selectbox(
                        "Numerical Column",
                        numerical_columns
                    )

                    color_options = ["None"] + categorical_columns

                    selected_color = st.selectbox(
                        "Group By (Optional)",
                        color_options
                    )

                    color_column = (
                        None if selected_color == "None"
                        else selected_color
                    )

            elif chart_type == "Pie Chart":

                if not categorical_columns:
                    st.warning("Pie chart requires a categorical column.")
                else:
                    x_column = st.selectbox(
                        "Category Column",
                        categorical_columns
                    )

                    value_options = ["Count"] + numerical_columns

                    selected_value = st.selectbox(
                        "Values (Optional)",
                        value_options
                    )

                    y_column = (
                        None if selected_value == "Count"
                        else selected_value
                    )

            elif chart_type == "Bar Chart":

                x_options = categorical_columns + datetime_columns

                if not x_options:
                    st.warning("Bar chart requires a categorical or date column.")
                else:
                    x_column = st.selectbox(
                        "X-axis",
                        x_options
                    )

                    y_options = ["Count"] + numerical_columns

                    selected_y = st.selectbox(
                        "Y-axis",
                        y_options
                    )

                    y_column = (
                        None if selected_y == "Count"
                        else selected_y
                    )

                    aggregation = st.selectbox(
                        "Aggregation",
                        ["Mean", "Sum", "Median", "Count", "Max", "Min"],
                        disabled=(y_column is None)
                    )

            elif chart_type in ["Box Plot", "Violin Plot"]:

                if not numerical_columns:
                    st.warning("This chart requires a numerical column.")
                else:
                    y_column = st.selectbox(
                        "Numerical Column",
                        numerical_columns
                    )

                    x_options = ["None"] + categorical_columns

                    selected_x = st.selectbox(
                        "Group By (Optional)",
                        x_options
                    )

                    x_column = (
                        None if selected_x == "None"
                        else selected_x
                    )

                    color_options = ["None"] + categorical_columns

                    selected_color = st.selectbox(
                        "Color By (Optional)",
                        color_options
                    )

                    color_column = (
                        None if selected_color == "None"
                        else selected_color
                    )

            elif chart_type in [
                "Scatter Plot",
                "Line Chart",
                "Area Chart"
            ]:

                if len(all_columns) < 2:
                    st.warning("This chart requires at least two columns.")
                else:

                    if chart_type == "Scatter Plot":
                        x_options = numerical_columns + categorical_columns + datetime_columns
                    else:
                        x_options = datetime_columns + categorical_columns + numerical_columns

                    if not x_options:
                        x_options = all_columns

                    x_column = st.selectbox(
                        "X-axis",
                        x_options
                    )

                    y_options = [
                        col for col in numerical_columns
                        if col != x_column
                    ]

                    if not y_options:
                        y_options = [
                            col for col in all_columns
                            if col != x_column
                        ]

                    y_column = st.selectbox(
                        "Y-axis",
                        y_options
                    )

                    color_options = ["None"] + categorical_columns

                    selected_color = st.selectbox(
                        "Color By (Optional)",
                        color_options
                    )

                    color_column = (
                        None if selected_color == "None"
                        else selected_color
                    )

            # Chart title
            chart_title = st.text_input(
                "Chart Title",
                value=chart_type
            )

            # Generate chart
            if st.button("Generate Visualization", type="primary"):

                try:
                    if chart_type != "Correlation Heatmap":
                        if x_column is None and chart_type not in [
                            "Box Plot",
                            "Violin Plot"
                        ]:
                            raise ValueError("Please select an X-axis column.")

                        if y_column is None and chart_type in [
                            "Scatter Plot",
                            "Line Chart",
                            "Area Chart"
                        ]:
                            raise ValueError("Please select a Y-axis column.")

                    if chart_type == "Bar Chart":
                        selected_aggregation = aggregation
                    else:
                        selected_aggregation = "Mean"

                    fig = create_visualization(
                        analysis_df,
                        chart_type=chart_type,
                        x_column=x_column,
                        y_column=y_column,
                        color_column=color_column,
                        aggregation=selected_aggregation,
                        title=chart_title
                    )

                    st.session_state["phase17_chart"] = fig
                    st.session_state["phase17_chart_type"] = chart_type
                    st.session_state["phase17_x_column"] = x_column
                    st.session_state["phase17_y_column"] = y_column

                except Exception as e:
                    st.error(f"Unable to create visualization: {e}")

            # Display generated chart
            if st.session_state.get("phase17_chart") is not None:

                st.divider()
                section_heading("Generated Visualization")

                fig = st.session_state["phase17_chart"]

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Chart downloads
                section_heading("Export Visualization")

                html_data = fig.to_html(
                    include_plotlyjs="cdn"
                )

                st.download_button(
                    "Download Interactive Chart (HTML)",
                    data=html_data,
                    file_name="datalens_visualization.html",
                    mime="text/html"
                )

                export_df = get_chart_data(
                    analysis_df,
                    st.session_state.get("phase17_x_column"),
                    st.session_state.get("phase17_y_column")
                )

                csv_data = export_df.to_csv(index=False)

                st.download_button(
                    "Download Chart Data (CSV)",
                    data=csv_data,
                    file_name="datalens_chart_data.csv",
                    mime="text/csv"
                )

                if st.button("Clear Visualization"):
                    st.session_state["phase17_chart"] = None
                    st.rerun()


elif page == "Automated Insights":

    page_header("Automated Insights", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Automatically discover patterns, data quality issues, "
        "statistical observations and relationships in your dataset."
    )

    original_df = st.session_state.get("dataset")
    cleaned_df = st.session_state.get("cleaned_dataset")

    if original_df is None:
        st.warning("Please upload a dataset first.")

    else:
        dataset_choice = st.radio(
            "Select dataset",
            ["Original Dataset", "Cleaned Dataset"],
            horizontal=True,
            key="phase18_dataset_choice"
        )

        if dataset_choice == "Cleaned Dataset" and cleaned_df is not None:
            analysis_df = cleaned_df
        else:
            analysis_df = original_df

            if dataset_choice == "Cleaned Dataset":
                st.info(
                    "No cleaned dataset is available. "
                    "Using the original dataset."
                )

        if analysis_df.empty:
            st.warning("The selected dataset is empty.")

        else:
            st.divider()

            section_heading("Insight Configuration")

            correlation_threshold = st.slider(
                "Minimum absolute correlation strength",
                min_value=0.50,
                max_value=0.95,
                value=0.70,
                step=0.05,
                help="Controls which numerical correlations are reported."
            )

            if st.button(
                "Generate Automated Insights",
                type="primary"
            ):

                try:
                    report = generate_all_insights(
                        analysis_df,
                        correlation_threshold=correlation_threshold
                    )

                    st.session_state["phase18_report"] = report
                    st.session_state["phase18_report_dataset"] = dataset_choice
                    st.session_state["phase18_dataset_shape"] = analysis_df.shape

                except Exception as e:
                    st.error(f"Unable to generate insights: {e}")

            report = st.session_state.get("phase18_report")

            if report is not None:

                st.divider()
                section_heading("Dataset Overview")

                overview = report["overview"]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Rows", overview["Rows"])
                c2.metric("Columns", overview["Columns"])
                c3.metric("Missing Cells", overview["Missing Cells"])
                c4.metric("Duplicate Rows", overview["Duplicate Rows"])

                st.divider()

                # Insight summary
                section_heading("Insights Summary")

                summary = get_insights_summary(
                    report["insights"]
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Total Insights", summary["Total"])
                c2.metric("Warnings", summary["Warnings"])
                c3.metric("Information", summary["Info"])
                c4.metric("Categories", summary["Categories"])

                st.divider()

                # Missing values
                section_heading("Missing Value Analysis")

                missing_df = report["missing_values"]

                if missing_df.empty:
                    st.success("No missing values detected.")
                else:
                    st.dataframe(
                        missing_df,
                        use_container_width=True,
                        hide_index=True
                    )

                st.divider()

                # Detected date columns
                section_heading("Detected Date/Time Columns")

                datetime_columns = report["datetime_columns"]

                if datetime_columns:
                    st.success(
                        "Detected date/time columns: "
                        + ", ".join(datetime_columns)
                    )
                else:
                    st.info("No date/time columns were detected.")

                st.divider()

                # Insight filters
                section_heading("Detailed Automated Insights")

                insights_df = get_insights_dataframe(
                    report["insights"]
                )

                if insights_df.empty:
                    st.info(
                        "No specific insights were detected "
                        "using the current rules and thresholds."
                    )

                else:
                    category_options = [
                        "All"
                    ] + sorted(
                        insights_df["Category"].unique().tolist()
                    )

                    severity_options = [
                        "All",
                        "Warning",
                        "Info"
                    ]

                    f1, f2 = st.columns(2)

                    selected_category = f1.selectbox(
                        "Filter by category",
                        category_options,
                        key="phase18_category_filter"
                    )

                    selected_severity = f2.selectbox(
                        "Filter by severity",
                        severity_options,
                        key="phase18_severity_filter"
                    )

                    filtered_df = insights_df.copy()

                    if selected_category != "All":
                        filtered_df = filtered_df[
                            filtered_df["Category"] == selected_category
                        ]

                    if selected_severity != "All":
                        filtered_df = filtered_df[
                            filtered_df["Severity"] == selected_severity
                        ]

                    st.caption(
                        f"Showing {len(filtered_df)} of "
                        f"{len(insights_df)} insights."
                    )

                    for _, insight in filtered_df.iterrows():

                        severity = insight["Severity"]

                        with st.expander(
                            f"{severity}: {insight['Title']}"
                        ):

                            st.write(
                                "**Category:**",
                                insight["Category"]
                            )

                            st.write(
                                "**Finding:**",
                                insight["Description"]
                            )

                            st.write(
                                "**Recommendation:**",
                                insight["Recommendation"]
                            )

                    st.divider()

                    # Export insights
                    section_heading("Export Insights")

                    csv_data = insights_df.to_csv(index=False)

                    st.download_button(
                        "Download Insights Report (CSV)",
                        data=csv_data,
                        file_name="datalens_automated_insights.csv",
                        mime="text/csv"
                    )

                    st.download_button(
                        "Download Missing Values Report (CSV)",
                        data=missing_df.to_csv(index=False),
                        file_name="datalens_missing_values_report.csv",
                        mime="text/csv"
                    )

                    st.success(
                        "Automated insights report generated successfully."
                    )
elif page == "Report Generator":

    page_header("Report Generator", eyebrow="DATALENS WORKSPACE")
    st.write(
        "Generate a downloadable PDF report containing dataset summaries, "
        "statistical information, data quality findings, automated insights "
        "and visualizations."
    )

    # -------------------------------------------------
    # SELECT DATASET
    # -------------------------------------------------

    original_df = st.session_state.get("dataset")
    cleaned_df = st.session_state.get("cleaned_dataset")

    if original_df is None:
        st.warning("Please upload a dataset before generating a report.")
        st.stop()

    dataset_options = ["Original Dataset"]

    if cleaned_df is not None:
        dataset_options.append("Cleaned Dataset")

    dataset_choice = st.radio(
        "Select dataset for the report",
        dataset_options,
        horizontal=True,
        key="phase19_dataset_choice"
    )

    if dataset_choice == "Cleaned Dataset" and cleaned_df is not None:
        report_df = cleaned_df.copy()
    else:
        report_df = original_df.copy()

    st.divider()

    # -------------------------------------------------
    # REPORT DETAILS
    # -------------------------------------------------

    section_heading("Report Details")

    col1, col2 = st.columns(2)

    with col1:
        report_title = st.text_input(
            "Report title",
            value="DataLens Analysis Report",
            key="phase19_report_title"
        )

    with col2:
        report_author = st.text_input(
            "Prepared by",
            value="DataLens User",
            key="phase19_report_author"
        )

    st.divider()

    # -------------------------------------------------
    # REPORT SECTIONS
    # -------------------------------------------------

    section_heading("Choose Report Sections")

    col1, col2 = st.columns(2)

    with col1:
        include_numerical = st.checkbox(
            "Numerical Analysis",
            value=True,
            key="phase19_numerical"
        )

        include_categorical = st.checkbox(
            "Categorical Analysis",
            value=True,
            key="phase19_categorical"
        )

        include_missing = st.checkbox(
            "Data Quality and Missing Values",
            value=True,
            key="phase19_missing"
        )

    with col2:
        include_insights = st.checkbox(
            "Automated Insights",
            value=True,
            key="phase19_insights"
        )

        include_charts = st.checkbox(
            "Visualizations",
            value=True,
            key="phase19_charts"
        )

    st.divider()

    # -------------------------------------------------
    # DATASET PREVIEW
    # -------------------------------------------------

    section_heading("Report Preview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{report_df.shape[0]:,}")
    c2.metric("Columns", f"{report_df.shape[1]:,}")
    c3.metric(
        "Missing Values",
        f"{int(report_df.isna().sum().sum()):,}"
    )

    with st.expander("Preview selected dataset"):
        st.dataframe(report_df.head(10), use_container_width=True)

    # -------------------------------------------------
    # AUTOMATED INSIGHTS
    # -------------------------------------------------

    insights_report = None

    saved_insights = st.session_state.get("phase18_report")
    saved_insights_dataset = st.session_state.get(
        "phase18_report_dataset"
    )

    if (
        saved_insights is not None
        and saved_insights_dataset == dataset_choice
    ):
        insights_report = saved_insights

        st.success(
            "Previously generated Automated Insights will be included "
            "in the PDF."
        )
    elif include_insights:
        st.info(
            "No matching Automated Insights report was found. "
            "The PDF can still be generated, but its insights section "
            "will explain that no insights were supplied. "
            "Generate insights on the Automated Insights page first "
            "to include them."
        )

    st.divider()

    # -------------------------------------------------
    # GENERATE PDF
    # -------------------------------------------------

    if st.button(
        "Generate PDF Report",
        type="primary",
        use_container_width=True,
        key="phase19_generate_button"
    ):

        if not report_title.strip():
            st.error("Please enter a report title.")

        elif not report_author.strip():
            st.error("Please enter the author's name.")

        else:
            try:
                with st.spinner("Generating your PDF report..."):

                    pdf_bytes = generate_pdf_report(
                        df=report_df,
                        title=report_title.strip(),
                        author=report_author.strip(),
                        insights_report=insights_report,
                        include_numerical=include_numerical,
                        include_categorical=include_categorical,
                        include_missing=include_missing,
                        include_insights=include_insights,
                        include_charts=include_charts
                    )

                    st.session_state["phase19_pdf_bytes"] = pdf_bytes
                    st.session_state["phase19_pdf_title"] = report_title.strip()

                st.success("Your PDF report has been generated successfully!")

            except Exception as e:
                st.error(f"Unable to generate PDF report: {e}")

    # -------------------------------------------------
    # DOWNLOAD PDF
    # -------------------------------------------------

    pdf_bytes = st.session_state.get("phase19_pdf_bytes")

    if pdf_bytes:
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="DataLens_Analysis_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key="phase19_download_button"
        )
        
elif page == "Machine Learning Studio":
    render_machine_learning()
