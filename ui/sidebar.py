
import streamlit as st
import textwrap

from ui.theme import get_theme


# ==========================================
# HTML RENDERING
# ==========================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# ==========================================
# NAVIGATION GROUPS
# ==========================================

PAGE_GROUPS = {
    "HOME": [
        "Home",
    ],

    "DATA MANAGEMENT": [
        "Dataset Upload",
        "Dataset Explorer",
        "Dataset Description & Profiling",
    ],

    "DATA QUALITY": [
        "Data Quality Analysis",
        "Data Cleaning",
        "Outlier Detection & Analysis",
    ],

    "EXPLORATORY ANALYSIS": [
        "Numerical Variable Analysis",
        "Categorical Variable Analysis",
        "Distribution Analysis",
        "Univariate Analysis",
        "Bivariate Analysis",
        "Group-wise Analysis",
        "Multivariate Analysis",
    ],

    "ADVANCED ANALYSIS": [
        "Statistical Analysis",
        "Time-Series Analysis",
    ],

    "MACHINE LEARNING": [
        "Machine Learning Studio",
    ],

    "INSIGHTS & REPORTS": [
        "Visualization Studio",
        "Automated Insights",
        "Report Generator",
    ],
}


# ==========================================
# NAVIGATION ICONS
# ==========================================

PAGE_ICONS = {
    "Home": "🏠",

    "Dataset Upload": "📤",
    "Dataset Explorer": "🗂️",
    "Dataset Description & Profiling": "📋",

    "Data Quality Analysis": "🔍",
    "Data Cleaning": "🧹",
    "Outlier Detection & Analysis": "🎯",

    "Numerical Variable Analysis": "🔢",
    "Categorical Variable Analysis": "🏷️",
    "Distribution Analysis": "📊",
    "Univariate Analysis": "📈",
    "Bivariate Analysis": "🔗",
    "Group-wise Analysis": "👥",
    "Multivariate Analysis": "🧩",

    "Statistical Analysis": "🧮",
    "Time-Series Analysis": "⏱️",

    "Machine Learning Studio": "🤖",

    "Visualization Studio": "🎨",
    "Automated Insights": "✨",
    "Report Generator": "📑",
}


# ==========================================
# SIDEBAR RENDER FUNCTION
# ==========================================

def render_sidebar():

    with st.sidebar:

        # BRANDING
        render_html("""
            <div class="dl-brand">
                <div class="dl-brand-mark">D</div>

                <div class="dl-brand-content">
                    <div class="dl-brand-name">DataLens</div>
                    <div class="dl-brand-sub">
                        ANALYTICS WORKSPACE
                    </div>
                </div>
            </div>
        """)

        st.markdown("---")

        # THEME SELECTION
        render_html("""
            <div class="dl-sidebar-label">
                APPEARANCE
            </div>
        """)

        current_theme = get_theme()

        st.radio(
            "Appearance",
            ["Dark", "Light"],
            index=["Dark", "Light"].index(current_theme),
            horizontal=True,
            key="datalens_theme",
            label_visibility="collapsed"
        )

        st.markdown("---")

        # NAVIGATION
        render_html("""
            <div class="dl-sidebar-label">
                WORKSPACE
            </div>
        """)

        selected_page = st.session_state.get(
            "datalens_page",
            "Home"
        )

        if selected_page not in PAGE_ICONS:
            selected_page = "Home"

        for group_name, pages in PAGE_GROUPS.items():

            render_html(f"""
                <div class="dl-nav-group">
                    {group_name}
                </div>
            """)

            for page_name in pages:

                icon = PAGE_ICONS.get(page_name, "📌")

                is_active = selected_page == page_name

                if st.button(
                    f"{icon}   {page_name}",
                    key=f"nav_{page_name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):

                    st.session_state["datalens_page"] = page_name
                    st.rerun()

        st.markdown("---")

        # ACTIVE DATASET
        render_html("""
            <div class="dl-sidebar-label">
                ACTIVE DATASET
            </div>
        """)

        df = st.session_state.get("dataset")

        if df is not None:

            dataset_name = st.session_state.get(
                "dataset_name",
                "Uploaded Dataset"
            )

            render_html(f"""
                <div class="dl-dataset-card">

                    <div class="dl-dataset-status">
                        <span class="dl-status-dot"></span>
                        DATASET LOADED
                    </div>

                    <div class="dl-dataset-name">
                        {dataset_name}
                    </div>

                    <div class="dl-dataset-stats">
                        {df.shape[0]:,} rows · {df.shape[1]:,} columns
                    </div>

                </div>
            """)

        else:

            render_html("""
                <div class="dl-dataset-empty">
                    <div style="font-size:1.3rem;">📂</div>
                    <div>No dataset loaded</div>
                    <div class="dl-dataset-hint">
                        Upload data to get started
                    </div>
                </div>
            """)

        # FOOTER
        render_html("""
            <div class="dl-sidebar-footer">
                <span>DataLens</span>
                <span>EDA & Statistical Analysis</span>
            </div>
        """)

    return st.session_state.get(
        "datalens_page",
        "Home"
    )
