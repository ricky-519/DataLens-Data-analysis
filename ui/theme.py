
import streamlit as st


# ==========================================
# DATALENS THEME CONFIGURATION
# ==========================================

THEMES = {

    "Dark": {
        "bg": "#0B1020",
        "surface": "#11182B",
        "surface2": "#151F35",
        "sidebar": "#0D1426",
        "text": "#E8ECF5",
        "muted": "#929DB3",
        "border": "rgba(148, 163, 184, 0.16)",
        "accent": "#8B7CFF",
        "accent2": "#55B8FF",
        "input": "#10182A",
        "shadow": "0 12px 34px rgba(0, 0, 0, 0.22)",
        "hover": "#1A2540",
        "success": "#22C55E",
    },

    "Light": {
        "bg": "#F5F7FB",
        "surface": "#FFFFFF",
        "surface2": "#F8FAFD",
        "sidebar": "#FFFFFF",
        "text": "#182238",
        "muted": "#69758B",
        "border": "rgba(71, 85, 105, 0.16)",
        "accent": "#6257E8",
        "accent2": "#2389D7",
        "input": "#FFFFFF",
        "shadow": "0 10px 28px rgba(28, 43, 73, 0.07)",
        "hover": "#F0F2FA",
        "success": "#16A34A",
    },

}


# ==========================================
# GET SELECTED THEME
# ==========================================

def get_theme():

    if "datalens_theme" not in st.session_state:
        st.session_state["datalens_theme"] = "Dark"

    return st.session_state["datalens_theme"]


# ==========================================
# INJECT GLOBAL THEME
# ==========================================

def inject_theme():

    theme_name = get_theme()
    c = THEMES[theme_name]

    css = f"""

    <style>

    /* ======================================
       GLOBAL VARIABLES
    ====================================== */

    :root {{

        --dl-bg: {c['bg']};
        --dl-surface: {c['surface']};
        --dl-surface2: {c['surface2']};

        --dl-sidebar: {c['sidebar']};

        --dl-text: {c['text']};
        --dl-muted: {c['muted']};

        --dl-border: {c['border']};

        --dl-accent: {c['accent']};
        --dl-accent2: {c['accent2']};

        --dl-input: {c['input']};

        --dl-shadow: {c['shadow']};

        --dl-hover: {c['hover']};

        --dl-success: {c['success']};

    }}


    /* ======================================
       GLOBAL APPLICATION
    ====================================== */

    html,
    body,
    [data-testid="stAppViewContainer"] {{

        background: var(--dl-bg);
        color: var(--dl-text);

        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;

    }}

    [data-testid="stHeader"] {{

        background: transparent;

    }}

    [data-testid="stAppViewContainer"] > .main {{

        background: var(--dl-bg);

    }}

    .block-container {{

        max-width: 1480px;

        padding-top: 2.2rem;
        padding-bottom: 3.5rem;

    }}


    /* ======================================
       SIDEBAR
    ====================================== */

    [data-testid="stSidebar"] {{

        background: var(--dl-sidebar);

        border-right: 1px solid var(--dl-border);

    }}

    [data-testid="stSidebar"] > div:first-child {{

        padding-top: 1.3rem;
        padding-bottom: 2rem;

    }}

    [data-testid="stSidebar"] * {{

        color: var(--dl-text);

    }}

    [data-testid="stSidebar"] hr {{

        border-color: var(--dl-border);

        margin-top: 18px;
        margin-bottom: 18px;

    }}


    /* ======================================
       BRANDING
    ====================================== */

    .dl-brand {{

        display: flex;
        align-items: center;

        gap: 12px;

        padding: 8px 4px 13px 4px;

    }}

    .dl-brand-mark {{

        width: 42px;
        height: 42px;

        min-width: 42px;

        border-radius: 13px;

        display: flex;
        align-items: center;
        justify-content: center;

        color: white !important;

        font-size: 21px;
        font-weight: 850;

        background: linear-gradient(
            135deg,
            var(--dl-accent),
            var(--dl-accent2)
        );

        box-shadow: 0 7px 20px
            color-mix(in srgb, var(--dl-accent) 28%, transparent);

    }}

    .dl-brand-content {{

        display: flex;
        flex-direction: column;

    }}

    .dl-brand-name {{

        font-size: 1.35rem;
        font-weight: 850;

        letter-spacing: -0.045em;

        color: var(--dl-text);

        line-height: 1.2;

    }}

    .dl-brand-sub {{

        color: var(--dl-muted);

        font-size: 0.66rem;

        font-weight: 700;

        letter-spacing: 0.11em;

        margin-top: 3px;

    }}


    /* ======================================
       SIDEBAR LABELS
    ====================================== */

    .dl-sidebar-label {{

        color: var(--dl-muted);

        font-size: 0.70rem;

        font-weight: 800;

        letter-spacing: 0.12em;

        margin: 14px 0 12px 0;

    }}

    .dl-nav-group {{

        color: var(--dl-muted);

        font-size: 0.65rem;

        font-weight: 800;

        letter-spacing: 0.11em;

        margin: 19px 0 7px 2px;

    }}


    /* ======================================
       SIDEBAR NAVIGATION BUTTONS
    ====================================== */

    [data-testid="stSidebar"] .stButton > button {{

        width: 100%;

        min-height: 39px;

        text-align: left;

        justify-content: flex-start;

        border-radius: 9px;

        border: 1px solid transparent;

        background: transparent;

        color: var(--dl-muted);

        font-size: 0.82rem;

        font-weight: 550;

        padding: 7px 10px;

        transition: all 0.18s ease;

    }}

    [data-testid="stSidebar"] .stButton > button:hover {{

        background: var(--dl-hover);

        border-color: var(--dl-border);

        color: var(--dl-text);

        transform: translateX(2px);

    }}

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{

        background: color-mix(
            in srgb,
            var(--dl-accent) 15%,
            transparent
        );

        color: var(--dl-accent) !important;

        border: 1px solid color-mix(
            in srgb,
            var(--dl-accent) 25%,
            transparent
        );

        font-weight: 750;

    }}


    /* ======================================
       DATASET STATUS CARD
    ====================================== */

    .dl-dataset-card,
    .dl-dataset-empty {{

        background: var(--dl-surface);

        border: 1px solid var(--dl-border);

        border-radius: 12px;

        padding: 14px;

        margin-top: 10px;

    }}

    .dl-dataset-status {{

        font-size: 0.65rem;

        font-weight: 800;

        color: var(--dl-muted);

        letter-spacing: 0.08em;

    }}

    .dl-status-dot {{

        display: inline-block;

        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: var(--dl-success);

        margin-right: 5px;

    }}

    .dl-dataset-name {{

        font-weight: 750;

        font-size: 0.88rem;

        margin-top: 9px;

        overflow-wrap: anywhere;

        color: var(--dl-text);

    }}

    .dl-dataset-stats,
    .dl-dataset-hint {{

        font-size: 0.76rem;

        color: var(--dl-muted);

        margin-top: 6px;

    }}

    .dl-dataset-empty {{

        color: var(--dl-text);

        text-align: center;

        font-size: 0.85rem;

    }}

    .dl-sidebar-footer {{

        display: flex;

        flex-direction: column;

        gap: 4px;

        color: var(--dl-muted);

        font-size: 0.7rem;

        margin-top: 22px;

    }}


    /* ======================================
       HEADINGS AND TYPOGRAPHY
    ====================================== */

    h1, h2, h3, h4, h5 {{

        color: var(--dl-text) !important;

        letter-spacing: -0.035em;

        font-weight: 750;

    }}

    p,
    label,
    .stMarkdown {{

        color: var(--dl-text);

    }}

    [data-testid="stCaptionContainer"],
    .stCaption {{

        color: var(--dl-muted) !important;

    }}


    /* ======================================
       METRIC CARDS
    ====================================== */

    [data-testid="stMetric"] {{

        background: var(--dl-surface);

        border: 1px solid var(--dl-border);

        border-radius: 16px;

        padding: 18px;

        box-shadow: var(--dl-shadow);

        transition: transform 0.2s ease;

    }}

    [data-testid="stMetric"]:hover {{

        transform: translateY(-2px);

    }}

    [data-testid="stMetricLabel"] p {{

        color: var(--dl-muted) !important;

        font-size: 0.82rem;

    }}

    [data-testid="stMetricValue"] {{

        color: var(--dl-text) !important;

        font-size: 1.6rem;

        font-weight: 800;

    }}


    /* ======================================
       BUTTONS
    ====================================== */

    .stButton > button,
    .stDownloadButton > button {{

        border-radius: 10px;

        border: 1px solid var(--dl-border);

        min-height: 2.55rem;

        font-weight: 650;

        transition: all 0.18s ease;

    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{

        border-color: var(--dl-accent);

        transform: translateY(-1px);

    }}

    .stButton > button[kind="primary"] {{

        background: linear-gradient(
            115deg,
            var(--dl-accent),
            var(--dl-accent2)
        );

        color: white;

        border: none;

    }}


    /* ======================================
       INPUTS
    ====================================== */

    [data-baseweb="input"],
    [data-baseweb="select"],
    [data-baseweb="textarea"] {{

        background: var(--dl-input);

        border-radius: 10px;

    }}

    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {{

        color: var(--dl-text);

    }}


    /* ======================================
       TABS
    ====================================== */

    [data-baseweb="tab-list"] {{

        gap: 7px;

        border-bottom: 1px solid var(--dl-border);

    }}

    [data-baseweb="tab"] {{

        background: transparent;

        border-radius: 9px 9px 0 0;

        padding: 10px 15px;

    }}

    [aria-selected="true"][data-baseweb="tab"] {{

        color: var(--dl-accent) !important;

        border-bottom: 2px solid var(--dl-accent);

    }}


    /* ======================================
       DATA TABLES
    ====================================== */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{

        border: 1px solid var(--dl-border);

        border-radius: 12px;

        overflow: hidden;

    }}


    /* ======================================
       EXPANDERS
    ====================================== */

    [data-testid="stExpander"] {{

        background: var(--dl-surface);

        border: 1px solid var(--dl-border);

        border-radius: 13px;

    }}


    /* ======================================
       HERO SECTION
    ====================================== */

    .dl-hero {{

        padding: 29px;

        border: 1px solid var(--dl-border);

        border-radius: 20px;

        background: linear-gradient(
            115deg,
            var(--dl-surface),
            var(--dl-surface2)
        );

        box-shadow: var(--dl-shadow);

        margin: 5px 0 24px 0;

    }}

    .dl-hero h1 {{

        margin: 0 0 8px 0;

        font-size: 2.05rem;

        font-weight: 850;

    }}

    .dl-hero p {{

        color: var(--dl-muted);

        margin: 0;

    }}


    /* ======================================
       REUSABLE INFORMATION CARDS
    ====================================== */

    .dl-card {{

        background: var(--dl-surface);

        border: 1px solid var(--dl-border);

        border-radius: 16px;

        padding: 20px;

        height: 100%;

        box-shadow: var(--dl-shadow);

        transition: transform 0.2s ease;

    }}

    .dl-card:hover {{

        transform: translateY(-3px);

    }}

    .dl-card-title {{

        font-size: 1.08rem;

        font-weight: 750;

        margin-bottom: 7px;

        color: var(--dl-text);

    }}

    .dl-card-copy {{

        color: var(--dl-muted);

        font-size: 0.88rem;

        line-height: 1.55;

    }}


    /* ======================================
       ALERTS AND DIVIDERS
    ====================================== */

    [data-testid="stAlert"] {{

        border-radius: 12px;

    }}

    hr {{

        border-color: var(--dl-border) !important;

    }}


    /* ======================================
       RESPONSIVE DESIGN
    ====================================== */

    @media (max-width: 800px) {{

        .block-container {{

            padding: 1.1rem 1rem 2rem;

        }}

        .dl-hero {{

            padding: 21px;

        }}

        .dl-hero h1 {{

            font-size: 1.65rem;

        }}

        [data-testid="stMetric"] {{

            padding: 12px;

        }}

    }}
    

/* ======================================
   PAGE HEADER
====================================== */

.dl-page-header {{
    margin-bottom: 24px;
}}

.dl-eyebrow {{
    color: var(--dl-accent);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

.dl-page-title {{
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 7px 0;
    letter-spacing: -0.04em;
}}

.dl-page-subtitle {{
    color: var(--dl-muted);
    font-size: 0.95rem;
    margin: 0;
}}

/* ======================================
   SECTION HEADINGS
====================================== */

.dl-section-heading h3 {{
    font-size: 1.2rem;
    font-weight: 750;
    margin: 22px 0 8px 0;
}}

/* ======================================
   CARD LABELS
====================================== */

.dl-card-kicker {{
    color: var(--dl-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}

.dl-card-pill-wrapper {{
    margin-top: 14px;
}}

.dl-pill {{
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: color-mix(in srgb, var(--dl-accent) 13%, transparent);
    color: var(--dl-accent);
    font-size: 0.72rem;
    font-weight: 700;
}}

.dl-divider {{
    border: none;
    border-top: 1px solid var(--dl-border);
    margin: 25px 0;
}}


    </style>

    """

    st.markdown(
        css,
        unsafe_allow_html=True
    )
