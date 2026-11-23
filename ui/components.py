
import streamlit as st
import textwrap


# ==========================================
# HTML RENDERING
# ==========================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


# ==========================================
# PAGE HEADER
# ==========================================

def page_header(
    title,
    subtitle="",
    eyebrow="DATALENS WORKSPACE"
):

    render_html(f"""
        <div class="dl-page-header">

            <div class="dl-eyebrow">
                {eyebrow}
            </div>

            <h1 class="dl-page-title">
                {title}
            </h1>

            <p class="dl-page-subtitle">
                {subtitle}
            </p>

        </div>
    """)


# ==========================================
# SECTION HEADING
# ==========================================

def section_heading(title, caption=None):

    render_html(f"""
        <div class="dl-section-heading">
            <h3>{title}</h3>
        </div>
    """)

    if caption:
        st.caption(caption)


# ==========================================
# INFORMATION CARD
# ==========================================

def info_card(
    kicker,
    title,
    description,
    pill=None
):

    pill_html = ""

    if pill:
        pill_html = f"""
            <div class="dl-card-pill-wrapper">
                <span class="dl-pill">{pill}</span>
            </div>
        """

    render_html(f"""
        <div class="dl-card">

            <div class="dl-card-kicker">
                {kicker}
            </div>

            <div class="dl-card-title">
                {title}
            </div>

            <div class="dl-card-copy">
                {description}
            </div>

            {pill_html}

        </div>
    """)


# ==========================================
# METRIC CARD
# ==========================================

def metric_card(label, value, help_text=None):

    st.metric(
        label=label,
        value=value,
        help=help_text
    )


# ==========================================
# DIVIDER
# ==========================================

def page_divider():

    render_html("""
        <hr class="dl-divider">
    """)
