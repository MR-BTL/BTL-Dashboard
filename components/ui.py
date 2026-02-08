"""UI components and styling."""

import os
import base64
import streamlit as st
from config import BRAND, LOGO_WIDTH, LOGO_PATHS


def _read_logo_as_base64(path: str) -> str:
    """Read logo file and convert to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def apply_theme():
    """Apply custom CSS theme to Streamlit."""
    bg = BRAND["white"]
    fg = BRAND["black"]
    card = BRAND["light_gray"]
    subtle = BRAND["mid_gray"]
    border = "#E2E6EA"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background: {bg} !important;
        }}

        body, p, div, span, label {{
            color: {fg} !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {fg} !important;
            font-weight: 700 !important;
        }}

        section[data-testid="stSidebar"] {{
            background: {bg} !important;
            border-right: 1px solid {border} !important;
        }}

        .block-container {{
            padding-top: 1.2rem;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 12px;
            overflow: hidden;
        }}

        .stButton button, .stDownloadButton button {{
            background: {BRAND["red"]} !important;
            color: {BRAND["white"]} !important;
            border: 0 !important;
            border-radius: 10px !important;
            padding: 0.6rem 1rem !important;
            font-weight: 700 !important;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
            opacity: 0.9;
        }}

        input, textarea {{
            border-radius: 10px !important;
            background-color: {card} !important;
            color: {fg} !important;
            border: 1px solid {border} !important;
        }}
        
        [data-baseweb="select"] {{
            background-color: {card} !important;
        }}
        
        [data-baseweb="select"] > div {{
            background-color: {card} !important;
            color: {fg} !important;
            border-color: {border} !important;
        }}
        
        [data-baseweb="popover"] {{
            background-color: {card} !important;
        }}
        
        [data-baseweb="menu"] {{
            background-color: {card} !important;
        }}
        
        [data-baseweb="menu"] li {{
            background-color: {card} !important;
            color: {fg} !important;
        }}
        
        [data-baseweb="menu"] li:hover {{
            background-color: {border} !important;
        }}
        
        label {{
            color: {fg} !important;
        }}
        
        .stSelectbox > div > div {{
            background-color: {card} !important;
        }}
        
        .stMultiSelect > div > div {{
            background-color: {card} !important;
        }}
        
        .stDateInput > div > div {{
            background-color: {card} !important;
        }}
        
        .stTextInput > div > div > input {{
            background-color: {card} !important;
            color: {fg} !important;
            border-color: {border} !important;
        }}
        
        [data-baseweb="input"] {{
            background-color: {card} !important;
            color: {fg} !important;
        }}
        
        [data-baseweb="input"] input {{
            background-color: {card} !important;
            color: {fg} !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {fg} !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {subtle} !important;
        }}

        hr {{
            border: none;
            height: 1px;
            background: {border};
            margin: 1.2rem 0;
        }}

        .mini-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 14px 16px;
        }}
        .mini-card .k {{
            color: {subtle};
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .mini-card .v {{
            font-size: 22px;
            font-weight: 800;
            margin-top: 6px;
        }}
        .mini-card .s {{
            color: {subtle};
            font-size: 12px;
            margin-top: 6px;
        }}
        
        .js-plotly-plot {{
            background-color: {bg} !important;
        }}
        
        .plotly {{
            background-color: {bg} !important;
        }}
        
        .plot-container {{
            background-color: {bg} !important;
        }}
        
        .main .block-container {{
            background-color: {bg} !important;
        }}
        
        header[data-testid="stHeader"] {{
            background-color: {bg} !important;
        }}
        
        [data-baseweb="switch"] {{
            background-color: {card} !important;
        }}
        
        [data-baseweb="base"] {{
            background-color: {card} !important;
            color: {fg} !important;
        }}
        
        .stCaption {{
            color: {subtle} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def display_logo(width=LOGO_WIDTH):
    """Display company logo if available."""
    logo_path = None
    for p in LOGO_PATHS:
        if os.path.exists(p):
            logo_path = p
            break

    if logo_path:
        ext = os.path.splitext(logo_path)[1].lower()
        mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime = mime_types.get(ext, "image/png")
        b64 = _read_logo_as_base64(logo_path)
        st.markdown(
            f"""
            <div style="display:flex;justify-content:center;align-items:center;width:100%;padding:8px 0 2px 0;">
                <img src="data:{mime};base64,{b64}" width="{width}" style="display:block;margin:0 auto;" />
            </div>
            """,
            unsafe_allow_html=True
        )


def mini_card(label: str, value: str, sub: str = ""):
    """Display a mini metric card."""
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="k">{label}</div>
            <div class="v">{value}</div>
            <div class="s">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
