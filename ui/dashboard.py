
import streamlit as st
import textwrap

from ui.components import page_header, info_card


# ==========================================
# HTML RENDERING
# ==========================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# ==========================================
# HOME DASHBOARD
# ==========================================

def render_dashboard():

    # PAGE HEADER
    page_header(
        title="Welcome to DataLens",
        subtitle=(
            "Your intelligent workspace for exploratory "
            "data analysis and statistical insights."
        ),
        eyebrow="ANALYTICS OVERVIEW"
    )

    # GET ACTIVE DATASET
    df = st.session_state.get("dataset")

    # ======================================
    # EMPTY DATASET STATE
    # ======================================

    if df is None:

        render_html("""
            <div class="dl-hero">

                <div class="dl-eyebrow">
                    YOUR DATA JOURNEY STARTS HERE
                </div>

                <h1>
                    Turn raw data into meaningful insights.
                </h1>

                <p>
                    Upload your dataset, explore its structure,
                    discover patterns and generate professional reports.
                </p>

            </div>
        """)

        st.markdown("### Get started")

        col1, col2, col3 = st.columns(3)

        with col1:
            info_card(
                "STEP 01",
                "Upload Data",
                "Import CSV, Excel or JSON datasets into your workspace.",
                "Dataset Upload"
            )

        with col2:
            info_card(
                "STEP 02",
                "Explore Data",
                "Understand columns, data types, distributions and quality.",
                "Dataset Explorer"
            )

        with col3:
            info_card(
                "STEP 03",
                "Generate Insights",
                "Perform statistical analysis and create detailed reports.",
                "Analysis & Reports"
            )

        st.info(
            "Select **Dataset Upload** from the sidebar to begin."
        )

        return

    # ======================================
    # DATASET METRICS
    # ======================================

    total_rows = df.shape[0]
    total_columns = df.shape[1]

    numerical_columns = len(
        df.select_dtypes(include="number").columns
    )

    categorical_columns = len(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    missing_values = int(
        df.isnull().sum().sum()
    )

    # ======================================
    # ACTIVE DATASET HEADER
    # ======================================

    dataset_name = st.session_state.get(
        "dataset_name",
        "Active Dataset"
    )

    render_html(f"""
        <div class="dl-hero">

            <div class="dl-eyebrow">
                ACTIVE DATASET
            </div>

            <h1>{dataset_name}</h1>

            <p>
                Your dataset is ready for exploration and analysis.
            </p>

        </div>
    """)

    # ======================================
    # OVERVIEW METRICS
    # ======================================

    st.markdown("### Dataset at a glance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("Total Columns", f"{total_columns:,}")
    col3.metric("Numerical Columns", f"{numerical_columns:,}")
    col4.metric("Missing Values", f"{missing_values:,}")

    # ======================================
    # DATASET DETAILS
    # ======================================

    st.markdown("---")

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown("### Column overview")

        column_summary = (
            df.dtypes.astype(str).to_frame(
                name="Data Type"
            )
        )

        column_summary.insert(
            0,
            "Column",
            column_summary.index
        )

        column_summary.reset_index(
            drop=True,
            inplace=True
        )

        st.dataframe(
            column_summary,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown("### Quick actions")

        st.caption(
            "Continue your analysis using these modules."
        )

        if st.button(
            "Explore Dataset",
            use_container_width=True
        ):
            st.session_state["datalens_page"] = "Dataset Explorer"
            st.rerun()

        if st.button(
            "Check Data Quality",
            use_container_width=True
        ):
            st.session_state["datalens_page"] = "Data Quality Analysis"
            st.rerun()

        if st.button(
            "Open Visualization Studio",
            use_container_width=True
        ):
            st.session_state["datalens_page"] = "Visualization Studio"
            st.rerun()

        cleaned = st.session_state.get("cleaned_dataset")

        if cleaned is not None:
            st.caption(
                f"Cleaned dataset available: "
                f"{cleaned.shape[0]:,} rows × "
                f"{cleaned.shape[1]:,} columns"
            )

    # ======================================
    # DATASET PREVIEW
    # ======================================

    st.markdown("---")

    st.markdown("### Dataset preview")

    st.caption(
        "First 10 records from the active dataset."
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )
