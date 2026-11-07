
import streamlit as st


def load_theme():

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #080808;
        --surface: #111111;
        --surface2: #191919;
        --surface3: #202020;
        --border: #303030;
        --white: #F5F5F5;
        --muted: #999999;
        --cyan: #00D4D8;
        --amber: #F59E0B;
        --green: #36C98F;
        --mono: 'IBM Plex Mono', monospace;
    }

    * { box-sizing: border-box; }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--white);
    }

    .block-container {
        max-width: 1700px;
        padding: 1.3rem 2rem 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    h1, h2, h3, h4 {
        color: var(--white) !important;
        letter-spacing: -0.7px;
    }

    p {
        color: var(--muted);
        line-height: 1.6;
    }

    hr {
        border-color: var(--border) !important;
    }

    /* HEADER */

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        flex-wrap: wrap;
        padding: 17px 20px;
        background: #101010;
        border: 1px solid var(--border);
        border-radius: 9px;
        margin-bottom: 25px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .brand-mark {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid var(--cyan);
        border-radius: 10px;
        color: var(--cyan);
        font-size: 27px;
        font-weight: 800;
    }

    .brand-name {
        font-size: 28px;
        font-weight: 800;
        color: white;
    }

    .brand-name span { color: var(--cyan); }

    .brand-caption {
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 1.4px;
        color: var(--muted);
        margin-top: 4px;
    }

    /* STATUS */

    .status-chip {
        display: inline-block;
        padding: 8px 12px;
        border: 1px solid var(--border);
        border-radius: 5px;
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: .5px;
    }

    .status-warning {
        color: var(--amber);
        border-color: #634b24;
        background: #1d170d;
    }

    .status-good {
        color: var(--green);
        border-color: #285d48;
        background: #0d1b14;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid var(--border);
    }

    .nav-title {
        font-family: var(--mono);
        color: var(--cyan);
        font-size: 16px;
        letter-spacing: 1.5px;
        margin: 20px 0 9px;
    }

    /* BUTTONS */

    div.stButton > button {
        width: 100%;
        background: #151515;
        color: #eeeeee;
        border: 1px solid #303030;
        border-radius: 6px;
        min-height: 43px;
        font-weight: 600;
        transition: all .2s ease;
    }

    div.stButton > button:hover {
        background: #252525;
        color: var(--cyan);
        border-color: var(--cyan);
    }

    div.stButton > button[kind="primary"] {
        background: #102a2b;
        border-color: var(--cyan);
        color: var(--cyan);
    }

    /* HERO */

    .hero {
        padding: 25px 0 27px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 25px;
    }

    .eyebrow {
        color: var(--cyan);
        font-family: var(--mono);
        font-size: 14px;
        letter-spacing: 1.4px;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: clamp(30px, 3vw, 38px);
        line-height: 1.1;
        font-weight: 800;
        margin: 0 0 13px;
    }

    .hero p {
        max-width: 700px;
        font-size: 14px;
    }

    /* UNIVERSAL EV-SENSE CARD STYLE */

    .panel {
        background: #111315 !important;
        border: 1px solid #34383C !important;
        border-top: 2px solid #16C7C7 !important;
        border-radius: 10px !important;
        padding: 18px !important;
        min-height: 120px;
        box-shadow: none !important;
        transition: border-color 0.2s ease,
                    background 0.2s ease;
    }

    .panel:hover {
        border-color: #16C7C7 !important;
        background: #151819 !important;
    }

    .panel-label {
        color: #999999;
        font-family: var(--mono);
        font-size: 11px;
        letter-spacing: 1px;
    }

    .panel-value {
        color: #F5F5F5;
        font-size: 23px;
        font-weight: 700;
        margin-top: 13px;
    }

    .panel-note {
        color: #A1A6AA;
        font-size: 12px;
        margin-top: 10px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        margin: 25px 0 15px;
    }

    /* FAULT GUIDE, ABOUT & SYSTEM INFORMATION CARDS */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #111315 !important;
        border: 1px solid #34383C !important;
        border-top: 2px solid #16C7C7 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] p {
        color: #A1A6AA !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] h3,
    div[data-testid="stVerticalBlockBorderWrapper"] h4 {
        color: #F5F5F5 !important;
    }

    /* METRICS */

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 7px;
        padding: 17px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--cyan) !important;
    }

    /* INTERACTIVE FAULT CARDS */

    .fault-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 7px;
        padding: 15px;
        min-height: 105px;
        transition: .2s ease;
    }

    .fault-card:hover {
        border-color: var(--cyan);
        background: var(--surface2);
    }

    .fault-code {
        color: var(--cyan);
        font-family: var(--mono);
        font-size: 12px;
        font-weight: 600;
    }

    .fault-name {
        color: var(--white);
        font-size: 13px;
        font-weight: 600;
        margin-top: 10px;
    }

    /* SIGNAL AREA */

    .signal-panel {
        background: #0e0e0e;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px;
        min-height: 190px;
    }

    .signal-empty {
        border: 1px dashed #444;
        padding: 30px 12px;
        text-align: center;
        margin-top: 15px;
    }

    .signal-empty strong {
        color: var(--amber);
        font-family: var(--mono);
        font-size: 12px;
    }

    .signal-empty p {
        font-size: 11px;
        margin-bottom: 0;
    }

    /* FOOTER */

    .footer {
        border-top: 1px solid var(--border);
        padding-top: 15px;
        color: #777;
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: .7px;
    }

    /* RESPONSIVE */

    @media(max-width: 800px) {
        .block-container {
            padding: 1rem;
        }

        .top-header {
            align-items: flex-start;
            flex-direction: column;
        }

        .hero h1 {
            font-size: 34px;
        }
    }

    @media(prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
        }
    }

    
    /* HIDE STREAMLIT DEFAULT PAGE NAVIGATION */
    [data-testid="stSidebarNav"] {
      display: none !important;
    }

    [data-testid="stSidebarNavItems"] {
     display: none !important;
    }

    
    /* SIDEBAR TOP SPACING */
    [data-testid="stSidebar"] > div:first-child {
     padding-top: 0 !important;
    }

    [data-testid="stSidebarContent"] {
     padding-top: 12px !important;
    }

    /* SIDEBAR BRAND POSITION */
    [data-testid="stSidebar"] .brand {
     margin-top: 0 !important;
    }

    
    /* =================================
    EV-SENSE AI — COMPACT TYPOGRAPHY
    ================================= */

    /* Main body text */
    .stApp p,
    .stApp li {
        font-size: 15px !important;
        line-height: 1.6 !important;
    }

    /* Section headings */
    .stApp h2 {
        font-size: 24px !important;
        font-weight: 700 !important;
    }

    /* Main page heading */
    .stApp h1 {
        font-size: 36px !important;
        line-height: 1.2 !important;
    }

    /* Dashboard card titles */
    .panel h3,
    .fault-name {
        font-size: 17px !important;
        font-weight: 650 !important;
    }

    /* Technical labels */
    .panel-label,
    .fault-code,
    .nav-heading {
        font-size: 12px !important;
        letter-spacing: 1px !important;
    }

    /* Small descriptions */
    .fault-category,
    .stCaption,
    small {
        font-size: 12px !important;
    }

    /* Sidebar navigation */
    [data-testid="stSidebar"] a,
    [data-testid="stPageLink"] a {
        font-size: 15px !important;
    }

    /* Buttons */
    .stButton button {
        font-size: 14px !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
    }

    html {
      scroll-behavior: smooth !important;
    }

    
    /* FAULT GUIDE — ACTUAL CARD ALIGNMENT */

    div[class*="st-key-fault-details-"] 
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: 280px !important;
        min-height: 280px !important;
        max-height: 280px !important;
        box-sizing: border-box !important;

        background: #111315 !important;
        border: 1px solid #34383C !important;
        border-radius: 10px !important;
    }

    /* Keep inner content spacing consistent */
    div[class*="st-key-fault-details-"] 
    [data-testid="stVerticalBlock"] {
        padding: 18px !important;
    }

    /* Existing dark theme */
    div[class*="st-key-fault-details-"] p {
        color: #A1A6AA !important;
    }

    div[class*="st-key-fault-details-"] h3 {
        color: #FFFFFF !important;
    }

    
    /* =========================================
    EV-SENSE AI — GLOBAL INDUSTRIAL CARD STYLE
    ========================================= */

    /* All bordered Streamlit containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #111315 !important;
        border: 1px solid #34383C !important;
        border-top: 2px solid #16C7C7 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        padding: 4px !important;
        transition: border-color 0.2s ease;
    }

    /* Subtle hover effect */
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #16C7C7 !important;
    }

    /* Card headings */
    div[data-testid="stVerticalBlockBorderWrapper"] h3,
    div[data-testid="stVerticalBlockBorderWrapper"] h4 {
        color: #FFFFFF !important;
    }

    /* Card descriptions */
    div[data-testid="stVerticalBlockBorderWrapper"] p {
        color: #A1A6AA !important;
    }

    /* Card labels */
    div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {
        color: #A1A6AA !important;
    }


    </style>
    """, unsafe_allow_html=True)


def ev_logo(size=48):
    return f"""
    <div class="brand-mark"
         style="width:{size}px;height:{size}px;flex:0 0 {size}px;">
        ϟ
    </div>
    """
