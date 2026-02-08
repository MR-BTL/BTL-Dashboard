import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import os
import re
import requests
from datetime import timedelta
from functools import reduce

# ============================================================
# Branding
# ============================================================
BRAND = {
    "white": "#FFFFFF",
    "black": "#0B0B0B",
    "red": "#DC143C",
    "light_gray": "#F5F6F8",
    "mid_gray": "#A0A4AA",
    "dark_gray": "#2A2F36",
}

LOGO_WIDTH = 200
LOGO_PATHS = ["logo.png", "logo.jpg", "logo.jpeg", "assets/logo.png", "assets/logo.jpg", "assets/logo.jpeg"]

APP_TITLE = "Activation Agents Dashboard"
APP_SUBTITLE = "Operations monitoring and performance insights"

# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Helpers: UI
# ============================================================
def _read_logo_as_base64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def apply_theme():
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

        /* Main text */
        body, p, div, span, label {{
            color: {fg} !important;
        }}

        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            color: {fg} !important;
            font-weight: 700 !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {bg} !important;
            border-right: 1px solid {border} !important;
        }}

        /* Cards-ish feel */
        .block-container {{
            padding-top: 1.2rem;
        }}

        /* Dataframe border */
        [data-testid="stDataFrame"] {{
            border: 1px solid {border};
            border-radius: 12px;
            overflow: hidden;
        }}

        /* Buttons */
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

        /* Input fields */
        input, textarea {{
            border-radius: 10px !important;
            background-color: {card} !important;
            color: {fg} !important;
            border: 1px solid {border} !important;
        }}
        
        /* Select boxes and multiselect */
        [data-baseweb="select"] {{
            background-color: {card} !important;
        }}
        
        [data-baseweb="select"] > div {{
            background-color: {card} !important;
            color: {fg} !important;
            border-color: {border} !important;
        }}
        
        /* Multiselect dropdown */
        [data-baseweb="popover"] {{
            background-color: {card} !important;
        }}
        
        /* Select dropdown items */
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
        
        /* Input labels */
        label {{
            color: {fg} !important;
        }}
        
        /* Styled select containers */
        .stSelectbox > div > div {{
            background-color: {card} !important;
        }}
        
        .stMultiSelect > div > div {{
            background-color: {card} !important;
        }}
        
        /* Date input wrapper */
        .stDateInput > div > div {{
            background-color: {card} !important;
        }}
        
        /* Text input containers */
        .stTextInput > div > div > input {{
            background-color: {card} !important;
            color: {fg} !important;
            border-color: {border} !important;
        }}
        
        /* Date input */
        [data-baseweb="input"] {{
            background-color: {card} !important;
            color: {fg} !important;
        }}
        
        [data-baseweb="input"] input {{
            background-color: {card} !important;
            color: {fg} !important;
        }}

        /* Metric styling */
        [data-testid="stMetricValue"] {{
            color: {fg} !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {subtle} !important;
        }}

        /* Soft section divider */
        hr {{
            border: none;
            height: 1px;
            background: {border};
            margin: 1.2rem 0;
        }}

        /* Custom mini-cards */
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
        
        /* Plotly charts background */
        .js-plotly-plot {{
            background-color: {bg} !important;
        }}
        
        .plotly {{
            background-color: {bg} !important;
        }}
        
        .plot-container {{
            background-color: {bg} !important;
        }}
        
        /* Main content area */
        .main .block-container {{
            background-color: {bg} !important;
        }}
        
        /* Header area */
        header[data-testid="stHeader"] {{
            background-color: {bg} !important;
        }}
        
        /* Toggle switch styling */
        [data-baseweb="switch"] {{
            background-color: {card} !important;
        }}
        
        /* Info/Error/Success boxes */
        [data-baseweb="base"] {{
            background-color: {card} !important;
            color: {fg} !important;
        }}
        
        /* Caption text */
        .stCaption {{
            color: {subtle} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def display_logo(width=LOGO_WIDTH):
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

def style_chart(fig, title: str, show_legend: bool = True, height: int = 400):
    """Apply consistent styling to Plotly charts"""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=BRAND["black"])),
        font=dict(family="Inter", size=12, color=BRAND["black"]),
        plot_bgcolor=BRAND["white"],
        paper_bgcolor=BRAND["white"],
        showlegend=show_legend,
        height=height,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(gridcolor="#E2E6EA", linecolor="#E2E6EA"),
        yaxis=dict(gridcolor="#E2E6EA", linecolor="#E2E6EA"),
    )
    return fig

def safe_chart(chart_func, *args, empty_message: str = "No data available for the selected filters.", **kwargs):
    """Safely create a chart with error handling"""
    try:
        fig = chart_func(*args, **kwargs)
        if fig is not None:
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

# Apply theme
apply_theme()

# ============================================================
# Helpers: Robust data ops
# ============================================================
def normalize_colname(c: str) -> str:
    c = str(c).strip()
    c = c.replace("\n", " ")
    c = re.sub(r"\s+", " ", c)
    c = c.lower().strip()
    c = c.replace("-", "_")
    c = c.replace(" ", "_")
    return c

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df

def safe_read_sheet(xls: pd.ExcelFile, name: str) -> pd.DataFrame:
    """Read Excel sheet with memory optimization."""
    try:
        # Use openpyxl engine for better memory handling with large files
        # Read in chunks if file is very large (optional optimization)
        df = pd.read_excel(xls, name, engine='openpyxl')
        return df
    except Exception:
        return pd.DataFrame()

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize data types to reduce memory usage."""
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert object columns to category if they have limited unique values
    for col in df.select_dtypes(include=['object']).columns:
        if len(df) > 0 and df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            try:
                # Ensure we're working with the full column
                df[col] = df[col].astype('category')
            except (TypeError, ValueError, KeyError):
                # Skip if conversion fails
                pass
    
    # Downcast numeric columns - ensure we're working with the full column
    for col in df.select_dtypes(include=['int64']).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        except (TypeError, ValueError, KeyError):
            pass
    
    for col in df.select_dtypes(include=['float64']).columns:
        try:
            df[col] = pd.to_numeric(df[col], downcast='float')
        except (TypeError, ValueError, KeyError):
            pass
    
    return df

def limit_display_rows(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Limit dataframe rows for display to improve UI performance."""
    if df.empty or len(df) <= max_rows:
        return df
    return df.head(max_rows)

def optimize_chart_data(df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    """Limit data points for charts to improve rendering performance."""
    if df.empty or len(df) <= max_points:
        return df
    # Sample data if too large, maintaining distribution
    if len(df) > max_points:
        # Use systematic sampling for better distribution
        step = len(df) // max_points
        return df.iloc[::step].head(max_points)
    return df

def safe_fillna(df: pd.DataFrame, value=0) -> pd.DataFrame:
    """Safely fill NaN values, handling categorical columns properly.
    Only fills numeric columns with the value. Categorical/string columns are left as-is.
    """
    if df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.name == 'category':
            # For categorical columns, only fillna if it's a numeric categorical
            try:
                # Check if categories are numeric
                if pd.api.types.is_numeric_dtype(df[col].cat.categories.dtype):
                    # Numeric categorical - convert to numeric and fillna
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(value)
                else:
                    # String categorical (like zone, agent_name) - don't fill with numeric value
                    # Convert to object to preserve NaN values
                    df[col] = df[col].astype('object')
            except (AttributeError, TypeError):
                # If checking categories fails, convert to object
                df[col] = df[col].astype('object')
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numeric columns - fillna normally
            df[col] = df[col].fillna(value)
        # For non-numeric, non-categorical columns, skip fillna with numeric values
    return df

def clean_user_id(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": "", "None": ""})
    return s

def to_datetime_any(s: pd.Series) -> pd.Series:
    # handles: 2-Feb, 2/2/2026, 2/2/2026 15:46:58 etc.
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def ensure_numeric(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)

def safe_value_counts(df: pd.DataFrame, col: str, top=20) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    vc = df[col].astype(str).replace("nan", "").replace("None", "")
    vc = vc[vc.str.strip() != ""].value_counts().head(top).reset_index()
    vc.columns = [col, "count"]
    return vc

# ============================================================
# Google Sheets download
# ============================================================
def convert_google_sheets_url(url: str) -> str | None:
    pattern = r"/spreadsheets/d/([a-zA-Z0-9-_]+)"
    match = re.search(pattern, url)
    if not match:
        return None
    sheet_id = match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

def load_file_from_url(url: str) -> BytesIO:
    export_url = convert_google_sheets_url(url)
    if not export_url:
        raise ValueError("Invalid Google Sheets URL format")
    resp = requests.get(export_url, timeout=60)
    resp.raise_for_status()
    return BytesIO(resp.content)

# ============================================================
# Core: Load + Process (robust, scalable)
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def load_and_process(google_sheets_url: str) -> dict:
    file = load_file_from_url(google_sheets_url)
    xls = pd.ExcelFile(file)

    # ---- Load sheets safely ----
    users = safe_read_sheet(xls, "Users")
    tasks = safe_read_sheet(xls, "Tasks")
    inter = safe_read_sheet(xls, "interactions")
    stocklog = safe_read_sheet(xls, "Stocklog")
    sv_tasks = safe_read_sheet(xls, "sv-tasks")

    # Optional legacy sheets
    login = safe_read_sheet(xls, "Login")
    main_brand = safe_read_sheet(xls, "main")
    purchase = safe_read_sheet(xls, "purchase")

    # ---- Standardize columns ----
    users = standardize_columns(users)
    tasks = standardize_columns(tasks)
    inter = standardize_columns(inter)
    stocklog = standardize_columns(stocklog)
    sv_tasks = standardize_columns(sv_tasks)
    login = standardize_columns(login)
    main_brand = standardize_columns(main_brand)
    purchase = standardize_columns(purchase)
    
    # Note: optimize_dtypes will be called AFTER all column assignments are complete
    # to avoid length mismatch issues

    # ---- Normalize key columns (users) ----
    # expected users schema (new): username, user-id, email, password, user-photo, user-mobile, zone, area, channel, sv, role
    # after standardize: username, user_id, email, ...
    if not users.empty:
        # alias mapping
        alias = {
            "user-id": "user_id",
            "userid": "user_id",
            "user_id": "user_id",
            "user_name": "username",
        }
        users = users.rename(columns={k: v for k, v in alias.items() if k in users.columns})

        if "user_id" in users.columns:
            users["user_id"] = clean_user_id(users["user_id"])
        if "username" in users.columns:
            users["username"] = users["username"].astype(str).str.strip()

    # ---- Users lookup ----
    wanted_user_cols = ["user_id", "username", "zone", "area", "channel", "sv", "role"]
    for c in wanted_user_cols:
        if c not in users.columns:
            users[c] = np.nan
    users_lookup = users[wanted_user_cols].drop_duplicates()

    # ---- Tasks normalize ----
    # expected tasks schema (new): task-date, place-name, place-code, user-id, user-name, channel, ..., status, check-in/out, shift
    if not tasks.empty:
        alias = {
            "user-id": "user_id",
            "user_id": "user_id",
            "user-name": "user_name",
            "username": "user_name",
            "task-date": "task_date",
            "task_date": "task_date",
            "place-name": "place_name",
            "place_name": "place_name",
            "place-code": "place_code",
            "place_code": "place_code",
            "check-in-time": "check_in_time",
            "check_out_time": "check_out_time",
            "check-out-time": "check_out_time",
            "check-in-date": "check_in_date",
            "check-out-date": "check_out_date",
        }
        tasks = tasks.rename(columns={k: v for k, v in alias.items() if k in tasks.columns})

        if "user_id" in tasks.columns:
            tasks["user_id"] = clean_user_id(tasks["user_id"])

        # Dates
        if "task_date" in tasks.columns:
            tasks["task_dt"] = to_datetime_any(tasks["task_date"])
            tasks["task_day"] = tasks["task_dt"].dt.date
        else:
            tasks["task_dt"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index, dtype='datetime64[ns]')
            tasks["task_day"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index)

        # Status normalize
        if "status" in tasks.columns:
            tasks["status_norm"] = (
                tasks["status"].astype(str).str.strip().str.lower()
                .replace({"nan": "", "none": ""})
            )
        else:
            tasks["status_norm"] = pd.Series([""] * len(tasks), index=tasks.index, dtype='string')

        # Check-in/out time detection (robust)
        checkin_col = pick_first_existing_col(tasks, [
            "check_in_time", "checkin_time", "checkin", "check_in", "check_in_datetime",
            "check_in_time_", "check_in_time__"
        ])
        checkout_col = pick_first_existing_col(tasks, [
            "check_out_time", "checkout_time", "checkout", "check_out", "check_out_datetime",
            "check_out_time_", "check_out_time__"
        ])

        # Some schemas store separately date + time
        checkin_date_col = pick_first_existing_col(tasks, ["check_in_date", "checkin_date", "check_in"])
        checkout_date_col = pick_first_existing_col(tasks, ["check_out_date", "checkout_date", "check_out"])

        # Build check-in/out datetimes
        tasks["checkin_dt"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index, dtype='datetime64[ns]')
        tasks["checkout_dt"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index, dtype='datetime64[ns]')

        # If explicit datetime columns exist, use them
        if checkin_col:
            tasks["checkin_dt"] = to_datetime_any(tasks[checkin_col])
        if checkout_col:
            tasks["checkout_dt"] = to_datetime_any(tasks[checkout_col])

        # If those are empty but separate date/time exist, try combining
        # (best-effort; won’t crash)
        if tasks["checkin_dt"].isna().all() and checkin_date_col and "check_in_time" in tasks.columns:
            tasks["checkin_dt"] = to_datetime_any(
                tasks[checkin_date_col].astype(str) + " " + tasks["check_in_time"].astype(str)
            )
        if tasks["checkout_dt"].isna().all() and checkout_date_col and "check_out_time" in tasks.columns:
            tasks["checkout_dt"] = to_datetime_any(
                tasks[checkout_date_col].astype(str) + " " + tasks["check_out_time"].astype(str)
            )

        # Shift duration hours (0..24)
        tasks["shift_hours"] = (tasks["checkout_dt"] - tasks["checkin_dt"]).dt.total_seconds() / 3600
        tasks["shift_hours"] = tasks["shift_hours"].where((tasks["shift_hours"] >= 0) & (tasks["shift_hours"] <= 24))

        def shift_bucket(x):
            if pd.isna(x):
                return "Unknown"
            if x < 1:
                return "<1h"
            if x <= 8:
                return "1–8h"
            if x <= 12:
                return "8–12h"
            return ">12h"

        tasks["shift_bucket"] = tasks["shift_hours"].apply(shift_bucket)

        # Metrics numeric
        for m in ["dcc", "ecc", "qr", "bbos"]:
            col = m
            if col not in tasks.columns:
                # maybe uppercase in original
                if m.upper() in tasks.columns:
                    tasks[m] = tasks[m.upper()]
                else:
                    tasks[m] = pd.Series([0] * len(tasks), index=tasks.index, dtype='float64')
            tasks[m] = pd.to_numeric(tasks[m], errors="coerce").fillna(0)
        
        # Distance fields normalization
        distance_aliases = {
            "in-distance": "in_distance",
            "in_distance": "in_distance",
            "in-distance_": "in_distance",
            "out-distance": "out_distance",
            "out_distance": "out_distance",
            "out-distance_": "out_distance",
        }
        for old_col, new_col in distance_aliases.items():
            if old_col in tasks.columns and new_col not in tasks.columns:
                tasks[new_col] = pd.to_numeric(tasks[old_col], errors="coerce")
        
        # Ensure distance columns exist
        if "in_distance" not in tasks.columns:
            tasks["in_distance"] = np.nan
        if "out_distance" not in tasks.columns:
            tasks["out_distance"] = np.nan
        
        # Calculate total distance (in + out) if both exist
        if "in_distance" in tasks.columns and "out_distance" in tasks.columns:
            tasks["total_distance"] = tasks["in_distance"].fillna(0) + tasks["out_distance"].fillna(0)
            # Replace 0 with NaN if both were NaN
            tasks["total_distance"] = tasks["total_distance"].replace(0, np.nan)
        elif "in_distance" in tasks.columns:
            tasks["total_distance"] = tasks["in_distance"]
        elif "out_distance" in tasks.columns:
            tasks["total_distance"] = tasks["out_distance"]
        else:
            tasks["total_distance"] = np.nan

        # Place display - ensure proper length matching
        if "place_code" not in tasks.columns:
            tasks["place_code"] = pd.Series([""] * len(tasks), index=tasks.index)
        if "place_name" not in tasks.columns:
            tasks["place_name"] = pd.Series([""] * len(tasks), index=tasks.index)
        
        # Ensure all columns have the same index before using np.where
        place_code_series = tasks["place_code"].astype(str).str.strip().replace("nan", "")
        place_name_series = tasks["place_name"].astype(str).fillna("")
        
        tasks["place_display"] = pd.Series(
            np.where(
                place_code_series != "",
                place_name_series + " (" + tasks["place_code"].astype(str).fillna("") + ")",
                place_name_series,
            ),
            index=tasks.index
        )

        # Merge agent attributes from users_lookup (non-destructive)
        tasks = tasks.merge(
            users_lookup[["user_id", "username", "zone", "area", "channel", "sv", "role"]],
            on="user_id",
            how="left",
            suffixes=("", "_u")
        )
        # choose best agent name
        if "user_name" in tasks.columns:
            tasks["agent_name"] = tasks["user_name"].fillna(tasks["username"])
        else:
            tasks["agent_name"] = tasks["username"]

    # ---- Interactions normalize ----
    if not inter.empty:
        alias = {
            "interaction-id": "interaction_id",
            "interaction_id": "interaction_id",
            "user-id": "user_id",
            "user_id": "user_id",
            "user-name": "user_name",
            "u-email": "u_email",
            "main_brand": "main_brand",
            "consumer_interactions": "consumer_interactions",
            "consumer interactions": "consumer_interactions",
            "pack_purchase": "pack_purchase",
            "pack_purchase_": "pack_purchase",
            "place-name": "place_name",
            "place-code": "place_code",
            "day": "day",
            "date": "date",
            "url": "url",
            "location": "location",
        }
        # Apply only keys that exist (after standardize, keys are snake_case)
        inter = inter.rename(columns={k: v for k, v in alias.items() if k in inter.columns})

        if "user_id" in inter.columns:
            inter["user_id"] = clean_user_id(inter["user_id"])

        # Interaction day (prefer day column)
        if "day" in inter.columns:
            inter["day_dt"] = to_datetime_any(inter["day"])
            inter["interaction_day"] = inter["day_dt"].dt.date
        elif "date" in inter.columns:
            inter["day_dt"] = to_datetime_any(inter["date"])
            inter["interaction_day"] = inter["day_dt"].dt.date
        else:
            inter["day_dt"] = pd.Series([pd.NaT] * len(inter), index=inter.index, dtype='datetime64[ns]')
            inter["interaction_day"] = pd.Series([pd.NaT] * len(inter), index=inter.index)

        # Consumer interactions col
        ci_col = pick_first_existing_col(inter, ["consumer_interactions", "consumerinteractions", "consumer_interaction"])
        if not ci_col:
            # some sheets might have "consumer_interactions" already created
            ci_col = "consumer_interactions" if "consumer_interactions" in inter.columns else None

        if ci_col:
            inter["interaction_type"] = inter[ci_col].astype(str).str.strip().str.upper()
        else:
            inter["interaction_type"] = pd.Series([""] * len(inter), index=inter.index, dtype='string')

        # Place cols - ensure proper length matching
        if "place_name" not in inter.columns:
            inter["place_name"] = pd.Series([""] * len(inter), index=inter.index)
        if "place_code" not in inter.columns:
            inter["place_code"] = pd.Series([""] * len(inter), index=inter.index)
        
        # Ensure all columns have the same index before using np.where
        place_code_series = inter["place_code"].astype(str).str.strip().replace("nan", "")
        place_name_series = inter["place_name"].astype(str).fillna("")
        
        inter["place_display"] = pd.Series(
            np.where(
                place_code_series != "",
                place_name_series + " (" + inter["place_code"].astype(str).fillna("") + ")",
                place_name_series,
            ),
            index=inter.index
        )

        # Pack purchase parsing: BBOS if contains "+"
        if "pack_purchase" not in inter.columns:
            # sometimes it is "pack_purchase" or "pack_purchase" already
            if "pack_purchase" not in inter.columns:
                inter["pack_purchase"] = ""
        pp = inter["pack_purchase"].astype(str).fillna("").str.strip()
        inter["has_bbos"] = pp.str.contains(r"\+", regex=True, na=False)
        inter["ecc_item"] = np.where(inter["has_bbos"], pp.str.split("+", n=1).str[1].fillna("").str.strip(), "")

        # Merge agent attributes from users_lookup
        inter = inter.merge(
            users_lookup[["user_id", "username", "zone", "area", "channel", "sv", "role"]],
            on="user_id",
            how="left",
            suffixes=("", "_u")
        )
        if "user_name" in inter.columns:
            inter["agent_name"] = inter["user_name"].fillna(inter["username"])
        else:
            inter["agent_name"] = inter["username"]

        # Flags per type
        inter["is_dcc"] = (inter["interaction_type"] == "DCC").astype(int)
        inter["is_ecc"] = (inter["interaction_type"] == "ECC").astype(int)
        inter["is_qr"] = (inter["interaction_type"] == "QR").astype(int)
        inter["is_bbos"] = (inter["interaction_type"] == "BBOS").astype(int)
        
        # Normalize brand columns
        if "main_brand" in inter.columns:
            inter["main_brand"] = inter["main_brand"].astype(str).str.strip().str.title()
            inter["main_brand"] = inter["main_brand"].replace({"Nan": "", "None": "", "": np.nan})
        
        # Check for occasional_brand column (may be named differently)
        occasional_col = pick_first_existing_col(inter, ["occasional_brand", "occasional brand", "occasionalbrand"])
        if occasional_col:
            inter["occasional_brand"] = inter[occasional_col].astype(str).str.strip().str.title()
            inter["occasional_brand"] = inter["occasional_brand"].replace({"Nan": "", "None": "", "": np.nan})
        elif "occasional_brand" not in inter.columns:
            inter["occasional_brand"] = np.nan

    # ---- Stocklog normalize ----
    if not stocklog.empty:
        alias = {
            "agent-name": "agent_name",
            "agent_name": "agent_name",
            "supervisior-name": "supervisor_name",
            "sv-email": "sv_email",
            "transaction-type": "transaction_type",
            "qtys": "qtys",
            "date": "date",
            "location": "location",
            "url": "url",
        }
        stocklog = stocklog.rename(columns={k: v for k, v in alias.items() if k in stocklog.columns})

        if "date" in stocklog.columns:
            stocklog["date_dt"] = to_datetime_any(stocklog["date"])
            stocklog["stock_day"] = stocklog["date_dt"].dt.date
        else:
            stocklog["date_dt"] = pd.NaT
            stocklog["stock_day"] = pd.NaT

        if "qtys" in stocklog.columns:
            stocklog["qtys"] = pd.to_numeric(stocklog["qtys"], errors="coerce").fillna(0)
        else:
            stocklog["qtys"] = 0

        if "transaction_type" in stocklog.columns:
            stocklog["transaction_type"] = stocklog["transaction_type"].astype(str).str.strip().str.title()
        else:
            stocklog["transaction_type"] = ""

        if "agent_name" not in stocklog.columns:
            stocklog["agent_name"] = ""

        # Pivot transaction types per agent/day
        stock_pivot = (
            stocklog.pivot_table(
                index=["agent_name", "stock_day"],
                columns="transaction_type",
                values="qtys",
                aggfunc="sum",
                fill_value=0
            )
            .reset_index()
        )

        # Normalize known tx types
        lower_cols = {c.lower(): c for c in stock_pivot.columns}
        def get_tx(name: str, aliases: list = None) -> pd.Series:
            """Get transaction type, checking name and optional aliases."""
            names_to_check = [name]
            if aliases:
                names_to_check.extend(aliases)
            for n in names_to_check:
                if n in lower_cols:
                    return stock_pivot[lower_cols[n]].astype(float)
            return pd.Series([0.0] * len(stock_pivot), index=stock_pivot.index)

        stock_pivot["qty_release"] = get_tx("release")
        # Check for both "back" and "return" as the sheet uses "Back"
        stock_pivot["qty_return"] = get_tx("back", aliases=["return"])
        # qty_used is the difference between Release and Back (Return)
        # If "used" transaction type exists, use it; otherwise calculate as release - return
        qty_used_from_tx = get_tx("used")
        stock_pivot["qty_used"] = np.where(
            qty_used_from_tx > 0,
            qty_used_from_tx,
            stock_pivot["qty_release"] - stock_pivot["qty_return"]
        )
        stock_agg = stock_pivot.copy()
    else:
        stock_agg = pd.DataFrame(columns=["agent_name", "stock_day", "qty_release", "qty_return", "qty_used"])

    # ---- Derived views (unfiltered; filtering happens in UI) ----
    # Tasks summary per agent
    if not tasks.empty:
        tasks_summary = (
            tasks.groupby("agent_name", dropna=False)
            .agg(
                tasks_count=("task_day", "count"),
                completed=("status_norm", lambda s: (s == "completed").sum()),
                pending=("status_norm", lambda s: (s == "pending").sum()),
                in_progress=("status_norm", lambda s: s.isin(["in progress", "in_progress", "inprogress"]).sum()),
                avg_shift_hours=("shift_hours", "mean"),
                working_days=("task_day", lambda s: pd.Series(s).nunique()),
            )
            .reset_index()
        )
    else:
        tasks_summary = pd.DataFrame(columns=["agent_name","tasks_count","completed","pending","in_progress","avg_shift_hours","working_days"])

    # Interactions summary per agent
    if not inter.empty:
        inter_summary = (
            inter.groupby("agent_name", dropna=False)
            .agg(
                interactions=("interaction_day", "count"),
                dcc=("is_dcc", "sum"),
                ecc=("is_ecc", "sum"),
                qr=("is_qr", "sum"),
                bbos=("is_bbos", "sum"),
                bbos_from_pack=("has_bbos", "sum"),
            )
            .reset_index()
        )
    else:
        inter_summary = pd.DataFrame(columns=["agent_name","interactions","dcc","ecc","qr","bbos","bbos_from_pack"])

    # Metric validations (per agent/day): Tasks vs Interactions
    def build_metric_validation(metric: str, tasks_df: pd.DataFrame, inter_df: pd.DataFrame) -> pd.DataFrame:
        # metric in tasks: dcc/ecc/qr/bbos numeric total
        # metric in inter: is_dcc/is_ecc/is_qr/is_bbos sum
        if tasks_df.empty and inter_df.empty:
            return pd.DataFrame()

        t = pd.DataFrame(columns=["agent_name","day","tasks_total"])
        i = pd.DataFrame(columns=["agent_name","day","inter_total"])

        if not tasks_df.empty and "task_day" in tasks_df.columns:
            t = (
                tasks_df.groupby(["agent_name","task_day"], as_index=False)[metric]
                .sum()
                .rename(columns={"task_day":"day", metric:"tasks_total"})
            )

        inter_flag = f"is_{metric}"
        if not inter_df.empty and "interaction_day" in inter_df.columns and inter_flag in inter_df.columns:
            i = (
                inter_df.groupby(["agent_name","interaction_day"], as_index=False)[inter_flag]
                .sum()
                .rename(columns={"interaction_day":"day", inter_flag:"inter_total"})
            )

        v = pd.merge(t, i, on=["agent_name","day"], how="outer")
        v["tasks_total"] = v["tasks_total"].fillna(0)
        v["inter_total"] = v["inter_total"].fillna(0)
        v["difference"] = v["tasks_total"] - v["inter_total"]
        v["match_flag"] = np.where(v["difference"] == 0, "Match", "Mismatch")
        return v

    dcc_check = build_metric_validation("dcc", tasks, inter)
    ecc_check = build_metric_validation("ecc", tasks, inter)
    qr_check  = build_metric_validation("qr",  tasks, inter)
    bbos_check= build_metric_validation("bbos",tasks, inter)

    # Stock validation (Used stock = Release - Back):
    # interactions_total should NOT exceed qty_used (the difference between Release and Back)
    if not stock_agg.empty and not inter.empty:
        # Ensure qty_used is calculated (Release - Back)
        if "qty_used" not in stock_agg.columns:
            stock_agg["qty_used"] = stock_agg["qty_release"] - stock_agg["qty_return"]
        else:
            # Fill any missing qty_used with calculated value
            stock_agg["qty_used"] = stock_agg["qty_used"].fillna(stock_agg["qty_release"] - stock_agg["qty_return"])
        
        inter_agent_day = (
            inter.groupby(["agent_name","interaction_day"], as_index=False)
            .agg(
                interactions_total=("interaction_type", "count"),
                bbos_pack=("has_bbos", "sum"),
                bbos_type=("is_bbos", "sum"),
            )
            .rename(columns={"interaction_day":"stock_day"})
        )
        stock_check = stock_agg.merge(inter_agent_day, on=["agent_name","stock_day"], how="left")
        stock_check[["interactions_total","bbos_pack","bbos_type","qty_used"]] = stock_check[["interactions_total","bbos_pack","bbos_type","qty_used"]].fillna(0)

        # Validate: interactions should NOT exceed qty_used (Release - Back)
        stock_check["diff_used_vs_interactions"] = stock_check["qty_used"] - stock_check["interactions_total"]
        stock_check["stock_flag_interactions"] = np.where(stock_check["diff_used_vs_interactions"] >= 0, "OK", "Interactions Exceed Used Stock")

        # BBOS pack validation also uses qty_used
        stock_check["diff_used_vs_bbos_pack"] = stock_check["qty_used"] - stock_check["bbos_pack"]
        stock_check["stock_flag_bbos_pack"] = np.where(stock_check["diff_used_vs_bbos_pack"] >= 0, "OK", "BBOS Pack Exceed Used Stock")
    else:
        stock_check = pd.DataFrame(columns=[
            "agent_name","stock_day","qty_release","qty_return","qty_used",
            "interactions_total","bbos_pack","bbos_type",
            "diff_used_vs_interactions","stock_flag_interactions",
            "diff_used_vs_bbos_pack","stock_flag_bbos_pack"
        ])

    # Attendance-like validation (worked but no interactions):
    # Agents with tasks (any status) but zero interactions on same day
    if not tasks.empty:
        task_day_any = (
            tasks.groupby(["agent_name","task_day"], as_index=False)
            .size()
            .rename(columns={"size":"tasks_any", "task_day":"day"})
        )
    else:
        task_day_any = pd.DataFrame(columns=["agent_name","day","tasks_any"])

    if not inter.empty:
        inter_day_any = (
            inter.groupby(["agent_name","interaction_day"], as_index=False)
            .size()
            .rename(columns={"size":"interactions_any", "interaction_day":"day"})
        )
    else:
        inter_day_any = pd.DataFrame(columns=["agent_name","day","interactions_any"])

    attendance_view = task_day_any.merge(inter_day_any, on=["agent_name","day"], how="left")
    attendance_view["interactions_any"] = attendance_view["interactions_any"].fillna(0)
    attendance_view["activity_flag"] = np.where(
        (attendance_view["tasks_any"] > 0) & (attendance_view["interactions_any"] == 0),
        "Worked_no_interactions",
        "OK"
    )

    # Date bounds (global)
    date_candidates = []
    for df, col in [
        (tasks, "task_day"),
        (inter, "interaction_day"),
        (stock_agg, "stock_day"),
        (sv_tasks, "task_date" if "task_date" in sv_tasks.columns else ""),
    ]:
        if not df.empty and col and col in df.columns:
            date_candidates.append(pd.to_datetime(df[col], errors="coerce"))

    if date_candidates:
        all_dates = pd.concat(date_candidates).dropna()
        min_date = all_dates.min().date() if not all_dates.empty else None
        max_date = all_dates.max().date() if not all_dates.empty else None
    else:
        min_date, max_date = None, None

    # Supervisor aggregations (for performance analysis) - optimized
    supervisor_perf = pd.DataFrame()
    if not tasks.empty and "sv" in tasks.columns:
        # Use vectorized operations for better performance
        supervisor_perf = tasks.groupby("sv", dropna=False, sort=False).agg(
            team_size=("agent_name", "nunique"),
            total_tasks=("task_day", "count"),
            completed_tasks=("status_norm", lambda s: (s == "completed").sum()),
            avg_completion_rate=("status_norm", lambda s: (s == "completed").sum() / len(s) * 100 if len(s) > 0 else 0),
            total_shift_hours=("shift_hours", "sum"),
            avg_shift_hours=("shift_hours", "mean"),
            working_days=("task_day", "nunique"),  # Use nunique directly instead of lambda
        ).reset_index()
        supervisor_perf["avg_completion_rate"] = supervisor_perf["avg_completion_rate"].round(1)
    
    # Add interaction metrics to supervisor performance - optimized
    if not inter.empty and "sv" in inter.columns and not supervisor_perf.empty:
        inter_by_sv = inter.groupby("sv", dropna=False, sort=False).agg(
            total_interactions=("interaction_day", "count"),
            dcc_count=("is_dcc", "sum"),
            ecc_count=("is_ecc", "sum"),
            qr_count=("is_qr", "sum"),
            bbos_count=("is_bbos", "sum"),
        ).reset_index()
        supervisor_perf = supervisor_perf.merge(inter_by_sv, on="sv", how="left", sort=False)
        supervisor_perf[["total_interactions", "dcc_count", "ecc_count", "qr_count", "bbos_count"]] = \
            supervisor_perf[["total_interactions", "dcc_count", "ecc_count", "qr_count", "bbos_count"]].fillna(0)
    elif not inter.empty and "sv" in inter.columns:
        supervisor_perf = inter.groupby("sv", dropna=False, sort=False).agg(
            team_size=("agent_name", "nunique"),
            total_interactions=("interaction_day", "count"),
            dcc_count=("is_dcc", "sum"),
            ecc_count=("is_ecc", "sum"),
            qr_count=("is_qr", "sum"),
            bbos_count=("is_bbos", "sum"),
        ).reset_index()

    # Optimize data types AFTER all processing is complete to avoid length mismatch issues
    if not users.empty:
        users = optimize_dtypes(users)
    if not tasks.empty:
        tasks = optimize_dtypes(tasks)
    if not inter.empty:
        inter = optimize_dtypes(inter)
    if not stocklog.empty:
        stocklog = optimize_dtypes(stocklog)
    if not stock_agg.empty:
        stock_agg = optimize_dtypes(stock_agg)
    if not stock_check.empty:
        stock_check = optimize_dtypes(stock_check)
    if not supervisor_perf.empty:
        supervisor_perf = optimize_dtypes(supervisor_perf)
    
    return {
        "users": users,
        "users_lookup": users_lookup,
        "tasks": tasks,
        "inter": inter,
        "stocklog": stocklog,
        "stock_agg": stock_agg,
        "stock_check": stock_check,
        "sv_tasks": sv_tasks,
        "tasks_summary": tasks_summary,
        "inter_summary": inter_summary,
        "dcc_check": dcc_check,
        "ecc_check": ecc_check,
        "qr_check": qr_check,
        "bbos_check": bbos_check,
        "attendance_view": attendance_view,
        "main_brand": main_brand,
        "supervisor_perf": supervisor_perf,
        "min_date": min_date,
        "max_date": max_date,
    }

# ============================================================
# Filtering helpers (apply to EVERYTHING)
# ============================================================
def filter_date(df: pd.DataFrame, col: str, dr: tuple | None) -> pd.DataFrame:
    """Filter by date range - optimized for large datasets."""
    if df.empty or dr is None or col not in df.columns:
        return df
    start, end = dr
    # Use vectorized operations - avoid creating intermediate Series if possible
    if df[col].dtype == 'datetime64[ns]':
        mask = (df[col].dt.date >= start) & (df[col].dt.date <= end)
    else:
        s = pd.to_datetime(df[col], errors="coerce").dt.date
        mask = (s >= start) & (s <= end)
    return df.loc[mask]  # Return view when possible, copy only if needed

def filter_in(df: pd.DataFrame, col: str, values: list) -> pd.DataFrame:
    """Filter dataframe by column values. Empty list means no filter (show all).
    Optimized for large datasets."""
    if df.empty or col not in df.columns:
        return df
    if not values:  # Empty list means no filter applied
        return df
    
    # Optimize: use category dtype if available, otherwise convert efficiently
    if df[col].dtype.name == 'category':
        # Categories are faster for isin operations
        values_set = set(str(v) for v in values)
        mask = df[col].astype(str).isin(values_set)
    else:
        # Convert to string for comparison to handle type mismatches
        values_set = set(str(v) for v in values)  # Use set for faster lookup
        mask = df[col].astype(str).isin(values_set)
    return df.loc[mask]  # Return view when possible

def filter_text_contains(df: pd.DataFrame, col: str, q: str) -> pd.DataFrame:
    """Filter by text contains - optimized for large datasets."""
    if df.empty or col not in df.columns or not q:
        return df
    # Use vectorized string operations
    if df[col].dtype.name == 'category':
        mask = df[col].astype(str).str.contains(q, case=False, na=False)
    else:
        mask = df[col].astype(str).str.contains(q, case=False, na=False)
    return df.loc[mask]  # Return view when possible

def to_excel_download(sheets: dict[str, pd.DataFrame], filename="export.xlsx") -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    output.seek(0)
    return output

# ============================================================
# Header
# ============================================================
display_logo()

st.title("📊 " + APP_TITLE)
st.caption(APP_SUBTITLE)
st.markdown("---")

# ============================================================
# Sidebar: Inputs
# ============================================================
st.sidebar.markdown("### 📌 Data Source")
google_sheets_url = st.sidebar.text_input(
    "Google Sheets URL",
    placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0"
)

if not google_sheets_url:
    st.info("Paste your Google Sheets URL in the sidebar to load data.")
    st.stop()

if "docs.google.com/spreadsheets" not in google_sheets_url:
    st.error("Invalid Google Sheets URL.")
    st.stop()

# Load with progress indicator
try:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading data from Google Sheets...")
    progress_bar.progress(10)
    
    with st.spinner("Loading & processing data..."):
        data = load_and_process(google_sheets_url)
    
    progress_bar.progress(100)
    status_text.text("✅ Data loaded successfully")
    progress_bar.empty()
    status_text.empty()
    st.success("✅ Data loaded")
    
    # Show data size info for large datasets
    total_rows = 0
    if not data.get("tasks", pd.DataFrame()).empty:
        total_rows += len(data["tasks"])
    if not data.get("inter", pd.DataFrame()).empty:
        total_rows += len(data["inter"])
    if total_rows > 10000:
        st.info(f"📊 Large dataset detected: {total_rows:,} total rows. Filters and displays are optimized for performance.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

users_lookup = data["users_lookup"]
tasks = data["tasks"]
inter = data["inter"]
stock_agg = data["stock_agg"]
stock_check = data["stock_check"]
dcc_check = data["dcc_check"]
ecc_check = data["ecc_check"]
qr_check = data["qr_check"]
bbos_check = data["bbos_check"]
attendance_view = data["attendance_view"]
supervisor_perf = data["supervisor_perf"]
min_date = data["min_date"]
max_date = data["max_date"]

# ============================================================
# Sidebar: Global Filters (meaningful and MANY)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🔎 Global Filters")

# Date range
date_range = None
if min_date and max_date:
    dr = st.sidebar.date_input("📅 Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    date_range = dr if isinstance(dr, tuple) else (dr, dr)

# Build filter options from users_lookup + tasks + inter
def unique_sorted(df, col):
    if df.empty or col not in df.columns:
        return []
    vals = df[col].dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    return sorted(vals.unique().tolist())

agents = unique_sorted(users_lookup, "username")
zones = unique_sorted(users_lookup, "zone")
areas = unique_sorted(users_lookup, "area")
channels = unique_sorted(users_lookup, "channel")
svs = unique_sorted(users_lookup, "sv")

# Filters default to empty (no filter applied = show all)
agent_filter = st.sidebar.multiselect("👤 Agents", agents, default=[])
zone_filter = st.sidebar.multiselect("🌍 Zones", zones, default=[])
area_filter = st.sidebar.multiselect("📍 Areas", areas, default=[])
channel_filter = st.sidebar.multiselect("📡 Channels", channels, default=[])
sv_filter = st.sidebar.multiselect("🧑‍💼 Supervisors (SV)", svs, default=[])

# Tasks-specific
task_status_options = []
if not tasks.empty and "status_norm" in tasks.columns:
    task_status_options = sorted(tasks["status_norm"].dropna().unique().tolist())
task_status_filter = st.sidebar.multiselect("✅ Task Status", task_status_options, default=[])

shift_bucket_options = []
if not tasks.empty and "shift_bucket" in tasks.columns:
    shift_bucket_options = sorted(tasks["shift_bucket"].dropna().unique().tolist())
shift_bucket_filter = st.sidebar.multiselect("⏱️ Shift Bucket", shift_bucket_options, default=[])

# Place filters
place_options = []
if not tasks.empty and "place_display" in tasks.columns:
    place_options = sorted([p for p in tasks["place_display"].dropna().astype(str).unique() if p.strip() != "" and p != "nan"])
place_filter = st.sidebar.multiselect("🏢 Places", place_options, default=[])

# Interactions-specific
interaction_types = []
if not inter.empty and "interaction_type" in inter.columns:
    interaction_types = sorted([x for x in inter["interaction_type"].dropna().astype(str).unique() if x.strip() != "" and x != "nan"])
interaction_type_filter = st.sidebar.multiselect("👥 Interaction Type", interaction_types, default=[])

age_options = unique_sorted(inter, "age_range")
age_filter = st.sidebar.multiselect("🎂 Age Range", age_options, default=[])

interaction_id_search = st.sidebar.text_input("🔍 Interaction ID contains", "")

st.sidebar.markdown("---")

# Filter status indicator
st.sidebar.markdown("### 📊 Active Filters")
active_filters = []
if date_range:
    active_filters.append(f"📅 Date: {date_range[0]} to {date_range[1]}")
if agent_filter:
    active_filters.append(f"👤 {len(agent_filter)} agent(s)")
if zone_filter:
    active_filters.append(f"🌍 {len(zone_filter)} zone(s)")
if area_filter:
    active_filters.append(f"📍 {len(area_filter)} area(s)")
if channel_filter:
    active_filters.append(f"📡 {len(channel_filter)} channel(s)")
if sv_filter:
    active_filters.append(f"🧑‍💼 {len(sv_filter)} supervisor(s)")
if task_status_filter:
    active_filters.append(f"✅ {len(task_status_filter)} status(es)")
if shift_bucket_filter:
    active_filters.append(f"⏱️ {len(shift_bucket_filter)} shift bucket(s)")
if place_filter:
    active_filters.append(f"🏢 {len(place_filter)} place(s)")
if interaction_type_filter:
    active_filters.append(f"👥 {len(interaction_type_filter)} interaction type(s)")
if age_filter:
    active_filters.append(f"🎂 {len(age_filter)} age range(s)")
if interaction_id_search:
    active_filters.append(f"🔍 Search: '{interaction_id_search}'")

if active_filters:
    for f in active_filters[:5]:  # Show first 5
        st.sidebar.caption(f"• {f}")
    if len(active_filters) > 5:
        st.sidebar.caption(f"... and {len(active_filters) - 5} more")
else:
    st.sidebar.caption("No filters applied (showing all data)")

st.sidebar.markdown("---")
st.sidebar.caption("All filters apply across KPIs, tables and charts.")

# ============================================================
# Apply filters consistently
# ============================================================
# Filter users_lookup first (base population)
# Empty filter list = show all, non-empty = filter
u = users_lookup.copy()
if agent_filter:
    u = filter_in(u, "username", agent_filter)
if zone_filter:
    u = filter_in(u, "zone", zone_filter)
if area_filter:
    u = filter_in(u, "area", area_filter)
if channel_filter:
    u = filter_in(u, "channel", channel_filter)
if sv_filter:
    u = filter_in(u, "sv", sv_filter)

valid_agents = set(u["username"].dropna().astype(str).tolist())

# Tasks filtered - optimized: chain filters without intermediate copies
tasks_f = tasks if not tasks.empty else pd.DataFrame()
if not tasks_f.empty:
    # Apply filters sequentially, but only copy at the end
    if date_range:
        tasks_f = filter_date(tasks_f, "task_day", date_range)
    if valid_agents:
        tasks_f = filter_in(tasks_f, "agent_name", list(valid_agents))
    if task_status_filter:
        tasks_f = filter_in(tasks_f, "status_norm", task_status_filter)
    if shift_bucket_filter:
        tasks_f = filter_in(tasks_f, "shift_bucket", shift_bucket_filter)
    if place_filter:
        tasks_f = filter_in(tasks_f, "place_display", place_filter)
    # Final copy only if we filtered (to ensure we have a proper DataFrame)
    if date_range or valid_agents or task_status_filter or shift_bucket_filter or place_filter:
        tasks_f = tasks_f.copy()

# Interactions filtered - optimized: chain filters without intermediate copies
inter_f = inter if not inter.empty else pd.DataFrame()
if not inter_f.empty:
    # Apply filters sequentially, but only copy at the end
    if date_range:
        inter_f = filter_date(inter_f, "interaction_day", date_range)
    if valid_agents:
        inter_f = filter_in(inter_f, "agent_name", list(valid_agents))
    if interaction_type_filter:
        inter_f = filter_in(inter_f, "interaction_type", interaction_type_filter)
    if age_filter:
        inter_f = filter_in(inter_f, "age_range", age_filter)
    if place_filter:
        inter_f = filter_in(inter_f, "place_display", place_filter)
    if interaction_id_search:
        inter_f = filter_text_contains(inter_f, "interaction_id", interaction_id_search)
    # Final copy only if we filtered
    if date_range or valid_agents or interaction_type_filter or age_filter or place_filter or interaction_id_search:
        inter_f = inter_f.copy()

# Stock filtered (pivot already; filter by date + agents)
stock_f = stock_agg.copy() if not stock_agg.empty else stock_agg
if not stock_f.empty:
    if date_range:
        stock_f = filter_date(stock_f, "stock_day", date_range)
    # Always filter by valid_agents (derived from user filters)
    if valid_agents:
        stock_f = filter_in(stock_f, "agent_name", list(valid_agents))

stock_check_f = stock_check.copy() if not stock_check.empty else stock_check
if not stock_check_f.empty:
    if date_range:
        stock_check_f = filter_date(stock_check_f, "stock_day", date_range)
    # Always filter by valid_agents (derived from user filters)
    if valid_agents:
        stock_check_f = filter_in(stock_check_f, "agent_name", list(valid_agents))

# Validations filtered (by date + agent)
def filter_validation(v: pd.DataFrame) -> pd.DataFrame:
    if v.empty:
        return v
    if date_range and "day" in v.columns:
        v = filter_date(v, "day", date_range)
    if "agent_name" in v.columns and valid_agents:
        v = filter_in(v, "agent_name", list(valid_agents))
    return v

dcc_f = filter_validation(dcc_check)
ecc_f = filter_validation(ecc_check)
qr_f  = filter_validation(qr_check)
bbos_f= filter_validation(bbos_check)

att_f = attendance_view.copy()
if not att_f.empty:
    if date_range and "day" in att_f.columns:
        att_f = filter_date(att_f, "day", date_range)
    if valid_agents:
        att_f = filter_in(att_f, "agent_name", list(valid_agents))

# ============================================================
# KPIs (MUST update by filters)
# ============================================================
st.subheader("📌 Overview (fully filter-driven)")

# Total agents: if filters applied, use filtered count; otherwise use all
if agent_filter or zone_filter or area_filter or channel_filter or sv_filter:
    total_agents = len(valid_agents)
else:
    total_agents = len(users_lookup["username"].dropna().unique()) if not users_lookup.empty else 0

active_agents_set = set()
if not tasks_f.empty:
    active_agents_set |= set(tasks_f["agent_name"].dropna().astype(str).tolist())
if not inter_f.empty:
    active_agents_set |= set(inter_f["agent_name"].dropna().astype(str).tolist())

active_agents = len(active_agents_set)
inactive_agents = max(total_agents - active_agents, 0)

total_tasks = len(tasks_f) if not tasks_f.empty else 0
total_interactions = len(inter_f) if not inter_f.empty else 0

# Task status counts
completed_tasks = int((tasks_f["status_norm"] == "completed").sum()) if (not tasks_f.empty and "status_norm" in tasks_f.columns) else 0
pending_tasks = int((tasks_f["status_norm"] == "pending").sum()) if (not tasks_f.empty and "status_norm" in tasks_f.columns) else 0
in_progress_tasks = int(tasks_f["status_norm"].isin(["in progress","in_progress","inprogress"]).sum()) if (not tasks_f.empty and "status_norm" in tasks_f.columns) else 0

# Totals per metric from tasks
dcc_total = float(tasks_f["dcc"].sum()) if (not tasks_f.empty and "dcc" in tasks_f.columns) else 0
ecc_total = float(tasks_f["ecc"].sum()) if (not tasks_f.empty and "ecc" in tasks_f.columns) else 0
qr_total  = float(tasks_f["qr"].sum())  if (not tasks_f.empty and "qr"  in tasks_f.columns) else 0
bbos_total= float(tasks_f["bbos"].sum())if (not tasks_f.empty and "bbos"in tasks_f.columns) else 0

# Working days + shifts
working_days = int(tasks_f["task_day"].nunique()) if (not tasks_f.empty and "task_day" in tasks_f.columns) else 0
total_shift_hours = float(tasks_f["shift_hours"].sum()) if (not tasks_f.empty and "shift_hours" in tasks_f.columns) else 0
avg_shift_hours = float(tasks_f["shift_hours"].mean()) if (not tasks_f.empty and "shift_hours" in tasks_f.columns) else 0

# Zones / Areas count (filtered population)
zones_count = int(u["zone"].dropna().nunique()) if not u.empty else 0
areas_count = int(u["area"].dropna().nunique()) if not u.empty else 0

# Additional KPIs
avg_interactions_per_agent = (total_interactions / active_agents) if active_agents > 0 else 0
shift_efficiency = (total_interactions / total_shift_hours) if total_shift_hours > 0 else 0

# Brand diversity (unique brands)
unique_brands = 0
if not inter_f.empty and "main_brand" in inter_f.columns:
    unique_brands = int(inter_f["main_brand"].dropna().nunique())

# KPI layout
c1, c2, c3, c4 = st.columns(4)
with c1: mini_card("Total Agents", str(total_agents), f"Zones: {zones_count} • Areas: {areas_count}")
with c2: mini_card("Active Agents", str(active_agents), f"Avg {avg_interactions_per_agent:.1f} interactions/agent")
with c3: mini_card("Total Interactions", f"{total_interactions:,}", f"Across {unique_brands} brands")
with c4: mini_card("Working Days", str(working_days), f"Avg {avg_shift_hours:.1f}h/shift")

c5, c6, c7, c8 = st.columns(4)
with c5: mini_card("Total Tasks", f"{total_tasks:,}", f"Completed {completed_tasks:,} • Pending {pending_tasks:,}")
with c6: mini_card("Shift Efficiency", f"{shift_efficiency:.1f}", f"Interactions per hour • {total_shift_hours:,.1f}h total")
with c7: mini_card("Metrics (Tasks)", f"DCC {int(dcc_total)} • ECC {int(ecc_total)}", f"QR {int(qr_total)} • BBOS {int(bbos_total)}")
with c8: mini_card("Task Status", f"{completed_tasks:,} completed", f"{pending_tasks:,} pending • {in_progress_tasks:,} in-progress")

st.markdown("---")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview Dashboard",
    "⚡ Operations & Performance",
    "🎯 Interactions & Brands",
    "📦 Stock & Inventory",
])

# ============================================================
# TAB 1: Overview Dashboard
# ============================================================
with tab1:
    st.subheader("📊 Overview Dashboard")
    
    # Top performing agents (combined view)
    st.markdown("### Top Performing Agents")
    colA, colB = st.columns(2)
    
    if not inter_f.empty or not tasks_f.empty:
        # Combine agent performance metrics
        agent_perf = []
        
        if not inter_f.empty:
            inter_agents = inter_f.groupby("agent_name").size().reset_index(name="interactions")
            agent_perf.append(inter_agents)
        
        if not tasks_f.empty:
            task_agents = tasks_f.groupby("agent_name").size().reset_index(name="tasks")
            agent_perf.append(task_agents)
        
        if agent_perf:
            agent_combined = reduce(lambda x, y: pd.merge(x, y, on="agent_name", how="outer"), agent_perf)
            # Safely fillna - handle categorical columns properly
            agent_combined = safe_fillna(agent_combined, value=0)
            
            if "interactions" in agent_combined.columns and "tasks" in agent_combined.columns:
                agent_combined["total_activity"] = agent_combined["interactions"] + agent_combined["tasks"]
                top_agents = agent_combined.sort_values("total_activity", ascending=False).head(15)
                
                fig = px.bar(top_agents, x="agent_name", y=["interactions", "tasks"],
                            title="Top 15 Agents by Activity",
                            barmode="group",
                            labels={"value": "Count", "agent_name": "Agent", "variable": "Type"})
                fig = style_chart(fig, "Top 15 Agents by Activity")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            elif "interactions" in agent_combined.columns:
                top_agents = agent_combined.sort_values("interactions", ascending=False).head(15)
                fig = px.bar(top_agents, x="agent_name", y="interactions", title="Top 15 Agents by Interactions")
                fig = style_chart(fig, "Top 15 Agents by Interactions")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            elif "tasks" in agent_combined.columns:
                top_agents = agent_combined.sort_values("tasks", ascending=False).head(15)
                fig = px.bar(top_agents, x="agent_name", y="tasks", title="Top 15 Agents by Tasks")
                fig = style_chart(fig, "Top 15 Agents by Tasks")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No agent performance data available for the selected filters.")
    
    # Quick insights (original 3 metrics)
    st.markdown("### Quick Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if active_agents > 0:
            st.metric("Avg Interactions/Agent", f"{avg_interactions_per_agent:.1f}")
        else:
            st.metric("Avg Interactions/Agent", "N/A")
    
    with col2:
        if total_shift_hours > 0:
            st.metric("Interactions/Hour", f"{shift_efficiency:.1f}")
        else:
            st.metric("Interactions/Hour", "N/A")
    
    with col3:
        if total_tasks > 0:
            completion_pct = (completed_tasks / total_tasks) * 100
            st.metric("Task Completion Rate", f"{completion_pct:.1f}%")
        else:
            st.metric("Task Completion Rate", "N/A")
    
    st.markdown("---")
    
    # Additional Performance Metrics - Charts and Tables
    st.markdown("### Performance Metrics")
    
    # Calculate metrics for charts/tables
    # Average Task Duration
    avg_task_duration = 0
    task_durations_data = pd.DataFrame()
    if not tasks_f.empty and "checkin_dt" in tasks_f.columns and "checkout_dt" in tasks_f.columns:
        completed_with_times = tasks_f[
            (tasks_f["status_norm"] == "completed") & 
            tasks_f["checkin_dt"].notna() & 
            tasks_f["checkout_dt"].notna()
        ]
        if not completed_with_times.empty:
            durations = (completed_with_times["checkout_dt"] - completed_with_times["checkin_dt"]).dt.total_seconds() / 3600
            durations = durations[(durations >= 0) & (durations <= 24)]
            if not durations.empty:
                avg_task_duration = durations.mean()
                task_durations_data = completed_with_times.copy()
                task_durations_data["duration_hours"] = durations
    
    # Zone Performance
    zone_perf_data = pd.DataFrame()
    if not inter_f.empty and "zone" in inter_f.columns:
        zone_inter = inter_f.groupby("zone").size().reset_index(name="interactions")
        zone_perf_data = zone_inter
    if not tasks_f.empty and "zone" in tasks_f.columns:
        zone_tasks = tasks_f.groupby("zone").size().reset_index(name="tasks")
        if not zone_perf_data.empty:
            zone_perf_data = zone_perf_data.merge(zone_tasks, on="zone", how="outer")
            # Safely fillna - handle categorical columns properly
            zone_perf_data = safe_fillna(zone_perf_data, value=0)
        else:
            zone_perf_data = zone_tasks
    
    # Agent Activity
    agent_activity = {}
    agent_activity_data = pd.DataFrame()
    if not inter_f.empty and "agent_name" in inter_f.columns:
        inter_by_agent = inter_f.groupby("agent_name").size().reset_index(name="interactions")
        agent_activity_data = inter_by_agent
        for _, row in inter_by_agent.iterrows():
            agent_activity[row["agent_name"]] = row["interactions"]
    if not tasks_f.empty and "agent_name" in tasks_f.columns:
        tasks_by_agent = tasks_f.groupby("agent_name").size().reset_index(name="tasks")
        if not agent_activity_data.empty:
            agent_activity_data = agent_activity_data.merge(tasks_by_agent, on="agent_name", how="outer")
            # Safely fillna - handle categorical columns properly
            agent_activity_data = safe_fillna(agent_activity_data, value=0)
            agent_activity_data["total_activity"] = agent_activity_data["interactions"].fillna(0) + agent_activity_data["tasks"].fillna(0)
        else:
            agent_activity_data = tasks_by_agent
            agent_activity_data["total_activity"] = agent_activity_data["tasks"]
        for _, row in tasks_by_agent.iterrows():
            agent_activity[row["agent_name"]] = agent_activity.get(row["agent_name"], 0) + row["tasks"]
    
    # Stock Utilization
    stock_util_data = pd.DataFrame()
    stock_utilization_rate = 0
    if not stock_f.empty and "qty_release" in stock_f.columns and "qty_used" in stock_f.columns:
        stock_util_data = stock_f.groupby("agent_name").agg({
            "qty_release": "sum",
            "qty_used": "sum"
        }).reset_index()
        stock_util_data["utilization_rate"] = (stock_util_data["qty_used"] / stock_util_data["qty_release"] * 100).fillna(0)
        total_release = stock_f["qty_release"].sum()
        total_used = stock_f["qty_used"].sum()
        if total_release > 0:
            stock_utilization_rate = (total_used / total_release) * 100
    
    # Agent Engagement
    agent_engagement_data = pd.DataFrame()
    agent_engagement_rate = 0
    if total_agents > 0 and not tasks_f.empty and "agent_name" in tasks_f.columns:
        agents_with_completed = tasks_f[tasks_f["status_norm"] == "completed"]["agent_name"].nunique()
        agent_engagement_rate = (agents_with_completed / total_agents) * 100
        # Create engagement data
        all_agents = set(tasks_f["agent_name"].dropna().unique())
        agents_completed = set(tasks_f[tasks_f["status_norm"] == "completed"]["agent_name"].dropna().unique())
        agent_engagement_data = pd.DataFrame({
            "agent_name": list(all_agents),
            "has_completed": [1 if agent in agents_completed else 0 for agent in all_agents]
        })
    
    # Low Performers Table
    low_performers_data = pd.DataFrame()
    if agent_activity and active_agents > 0:
        avg_activity = sum(agent_activity.values()) / len(agent_activity)
        low_performers_list = [agent for agent, count in agent_activity.items() if count < avg_activity]
        if low_performers_list and not agent_activity_data.empty:
            low_performers_data = agent_activity_data[agent_activity_data["agent_name"].isin(low_performers_list)].copy()
            low_performers_data = low_performers_data.sort_values("total_activity" if "total_activity" in low_performers_data.columns else "interactions" if "interactions" in low_performers_data.columns else "tasks", ascending=True)
    
    # Best vs Worst Zone comparison
    best_zone = "N/A"
    worst_zone = "N/A"
    if not zone_perf_data.empty:
        if "interactions" in zone_perf_data.columns:
            best_zone = zone_perf_data.loc[zone_perf_data["interactions"].idxmax(), "zone"]
            worst_zone = zone_perf_data.loc[zone_perf_data["interactions"].idxmin(), "zone"]
        elif "tasks" in zone_perf_data.columns:
            best_zone = zone_perf_data.loc[zone_perf_data["tasks"].idxmax(), "zone"]
            worst_zone = zone_perf_data.loc[zone_perf_data["tasks"].idxmin(), "zone"]
    
    # Create charts and tables
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Average Task Duration Distribution
        if not task_durations_data.empty and "duration_hours" in task_durations_data.columns:
            st.markdown("#### Task Duration Distribution")
            # Optimize chart data for large datasets
            chart_data = optimize_chart_data(task_durations_data, max_points=5000)
            fig = px.histogram(chart_data, x="duration_hours", nbins=20,
                             title=f"Task Duration Distribution (Avg: {avg_task_duration:.1f}h)",
                             labels={"duration_hours": "Duration (hours)", "count": "Number of Tasks"})
            fig = style_chart(fig, f"Task Duration Distribution (Avg: {avg_task_duration:.1f}h)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No task duration data available")
    
    with col_chart2:
        # Zone Performance Comparison
        if not zone_perf_data.empty:
            st.markdown("#### Zone Performance")
            if "interactions" in zone_perf_data.columns and "tasks" in zone_perf_data.columns:
                zone_perf_sorted = zone_perf_data.sort_values("interactions", ascending=False).head(10)
                fig = px.bar(zone_perf_sorted, x="zone", y=["interactions", "tasks"],
                           title="Top Zones by Activity",
                           barmode="group",
                           labels={"value": "Count", "zone": "Zone", "variable": "Type"})
            elif "interactions" in zone_perf_data.columns:
                zone_perf_sorted = zone_perf_data.sort_values("interactions", ascending=False).head(10)
                fig = px.bar(zone_perf_sorted, x="zone", y="interactions",
                           title="Top Zones by Interactions")
            else:
                zone_perf_sorted = zone_perf_data.sort_values("tasks", ascending=False).head(10)
                fig = px.bar(zone_perf_sorted, x="zone", y="tasks",
                           title="Top Zones by Tasks")
            fig = style_chart(fig, "Zone Performance")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No zone performance data available")
    
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        # Stock Utilization by Agent
        if not stock_util_data.empty:
            st.markdown("#### Stock Utilization by Agent")
            stock_util_sorted = stock_util_data.sort_values("utilization_rate", ascending=True).head(15)
            fig = px.bar(stock_util_sorted, x="agent_name", y="utilization_rate",
                        title=f"Stock Utilization Rate (Overall: {stock_utilization_rate:.1f}%)",
                        labels={"utilization_rate": "Utilization Rate (%)", "agent_name": "Agent"})
            fig = style_chart(fig, f"Stock Utilization Rate (Overall: {stock_utilization_rate:.1f}%)")
            fig.update_layout(xaxis_tickangle=45, yaxis_title="Utilization Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stock utilization data available")
    
    with col_chart4:
        # Average Shift Hours Trend
        if not tasks_f.empty and "task_day" in tasks_f.columns and "shift_hours" in tasks_f.columns:
            st.markdown("#### Average Shift Hours Trend")
            shift_trend = tasks_f.groupby("task_day")["shift_hours"].mean().reset_index()
            shift_trend.columns = ["date", "avg_shift_hours"]
            fig = px.line(shift_trend, x="date", y="avg_shift_hours",
                         title=f"Average Shift Hours Over Time (Overall: {avg_shift_hours:.1f}h)",
                         labels={"avg_shift_hours": "Avg Shift Hours", "date": "Date"})
            fig = style_chart(fig, f"Average Shift Hours Over Time (Overall: {avg_shift_hours:.1f}h)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shift hours data available")
    
    # Tables section
    st.markdown("#### Low Performers (Below Average)")
    if not low_performers_data.empty:
        display_cols = [c for c in ["agent_name", "interactions", "tasks", "total_activity"] if c in low_performers_data.columns]
        if display_cols:
            st.dataframe(low_performers_data[display_cols],
                       use_container_width=True, height=300)
            # Download button
            csv = low_performers_data[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Low Performers Table",
                data=csv,
                file_name="low_performers.csv",
                mime="text/csv"
            )
        st.metric("Total Low Performers", f"{len(low_performers_data)}")
    else:
        st.info("No low performers identified")
        st.metric("Total Low Performers", "0")
    
    # Additional metrics row
    interactions_per_task = (total_interactions / total_tasks) if total_tasks > 0 else 0
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.markdown("#### Efficiency Metrics")
        if interactions_per_task > 0:
            st.metric("Interactions per Task", f"{interactions_per_task:.2f}")
        else:
            st.metric("Interactions per Task", "N/A")
    
    with col_metric2:
        st.markdown("#### Engagement")
        if agent_engagement_rate > 0:
            st.metric("Agent Engagement Rate", f"{agent_engagement_rate:.1f}%")
        else:
            st.metric("Agent Engagement Rate", "N/A")
    
    with col_metric3:
        st.markdown("#### Zone Comparison")
        if best_zone != "N/A" and worst_zone != "N/A":
            st.metric("Best Zone", str(best_zone))
            st.metric("Worst Zone", str(worst_zone))
        else:
            st.info("Zone comparison data not available")

# ============================================================
# TAB 2: Operations & Performance
# ============================================================
with tab2:
    st.subheader("⚡ Operations & Performance")
    
    # Supervisor Performance Section
    st.markdown("### Supervisor Performance")
    if not supervisor_perf.empty and "sv" in supervisor_perf.columns:
        # Filter supervisor performance by valid supervisors and zones/areas
        sv_filtered = supervisor_perf.copy()
        if sv_filter:
            sv_filtered = sv_filtered[sv_filtered["sv"].isin(sv_filter)]
        
        # Also filter by valid agents' supervisors
        if not u.empty and "sv" in u.columns:
            valid_svs = set(u["sv"].dropna().astype(str).tolist())
            sv_filtered = sv_filtered[sv_filtered["sv"].isin(valid_svs)]
        
        if not sv_filtered.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                if "total_interactions" in sv_filtered.columns:
                    sv_sorted = sv_filtered.sort_values("total_interactions", ascending=False).head(15)
                    fig = px.bar(sv_sorted, x="sv", y="total_interactions",
                                title="Supervisors by Total Interactions")
                    fig = style_chart(fig, "Supervisors by Total Interactions")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                elif "total_tasks" in sv_filtered.columns:
                    sv_sorted = sv_filtered.sort_values("total_tasks", ascending=False).head(15)
                    fig = px.bar(sv_sorted, x="sv", y="total_tasks",
                                title="Supervisors by Total Tasks")
                    fig = style_chart(fig, "Supervisors by Total Tasks")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if "team_size" in sv_filtered.columns and "avg_completion_rate" in sv_filtered.columns:
                    sv_sorted = sv_filtered.sort_values("avg_completion_rate", ascending=False).head(15)
                    fig = px.bar(sv_sorted, x="sv", y="avg_completion_rate",
                                title="Supervisors by Avg Completion Rate (%)")
                    fig = style_chart(fig, "Supervisors by Avg Completion Rate (%)")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Supervisor performance table
            with st.expander("📋 View Supervisor Performance Details"):
                display_cols = [c for c in ["sv", "team_size", "total_tasks", "completed_tasks", 
                                           "avg_completion_rate", "total_interactions", 
                                           "total_shift_hours", "avg_shift_hours", "working_days"]
                               if c in sv_filtered.columns]
                if display_cols:
                    sv_display = sv_filtered[display_cols].sort_values("total_interactions" if "total_interactions" in display_cols else "total_tasks", 
                                                                      ascending=False)
                    st.dataframe(sv_display, use_container_width=True, height=400)
                    # Download button
                    csv = sv_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Supervisor Performance Table",
                        data=csv,
                        file_name="supervisor_performance.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No supervisor data available for the selected filters.")
    else:
        st.info("No supervisor performance data available.")
    
    st.markdown("---")
    
    # Agent Performance Section
    st.markdown("### Agent Performance")
    if not tasks_f.empty or not inter_f.empty:
        # Build comprehensive agent performance
        agent_perf_list = []
        
        if not tasks_f.empty:
            perf_tasks = tasks_f.groupby("agent_name").agg(
                tasks=("task_day","count"),
                completed=("status_norm", lambda s: (s=="completed").sum()),
                pending=("status_norm", lambda s: (s=="pending").sum()),
                in_progress=("status_norm", lambda s: s.isin(["in progress","in_progress","inprogress"]).sum()),
                working_days=("task_day", lambda s: pd.Series(s).nunique()),
                avg_shift=("shift_hours", "mean"),
                total_shift=("shift_hours", "sum"),
                dcc=("dcc","sum"), ecc=("ecc","sum"), qr=("qr","sum"), bbos=("bbos","sum"),
            ).reset_index()
            agent_perf_list.append(perf_tasks)
        
        if not inter_f.empty:
            perf_inter = inter_f.groupby("agent_name").agg(
                interactions=("interaction_day", "count"),
                dcc_inter=("is_dcc", "sum"),
                ecc_inter=("is_ecc", "sum"),
                qr_inter=("is_qr", "sum"),
                bbos_inter=("is_bbos", "sum"),
            ).reset_index()
            agent_perf_list.append(perf_inter)
        
        if agent_perf_list:
            perf = reduce(lambda x, y: pd.merge(x, y, on="agent_name", how="outer"), agent_perf_list)
            # Safely fillna - handle categorical columns properly
            perf = safe_fillna(perf, value=0)
            
            if "tasks" in perf.columns and perf["tasks"].sum() > 0:
                perf["completion_rate"] = np.where(perf["tasks"]>0, (perf["completed"]/perf["tasks"])*100, 0).round(1)
            
            if "working_days" in perf.columns:
                top_working = perf.sort_values("working_days", ascending=False).head(15)
                fig = px.bar(top_working, x="agent_name", y="working_days",
                            title="Top 15 Agents by Working Days")
                fig = style_chart(fig, "Top 15 Agents by Working Days")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            elif "interactions" in perf.columns:
                top_inter = perf.sort_values("interactions", ascending=False).head(15)
                fig = px.bar(top_inter, x="agent_name", y="interactions",
                            title="Top 15 Agents by Interactions")
                fig = style_chart(fig, "Top 15 Agents by Interactions")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Shift Analysis
            st.markdown("#### Shift Analysis")
            
            if not tasks_f.empty and "status_norm" in tasks_f.columns:
                status_counts = tasks_f["status_norm"].value_counts().reset_index()
                status_counts.columns = ["status", "count"]
                fig = px.pie(status_counts, names="status", values="count",
                            title="Tasks by Status")
                fig = style_chart(fig, "Tasks by Status")
                st.plotly_chart(fig, use_container_width=True)
            
            # Distance Analysis Section
            st.markdown("---")
            st.markdown("#### Distance Analysis")
            
            if not tasks_f.empty:
                has_distance = False
                distance_conditions = []
                
                if "in_distance" in tasks_f.columns:
                    has_distance = True
                    distance_conditions.append(tasks_f["in_distance"].notna())
                if "out_distance" in tasks_f.columns:
                    has_distance = True
                    distance_conditions.append(tasks_f["out_distance"].notna())
                if "total_distance" in tasks_f.columns:
                    has_distance = True
                    distance_conditions.append(tasks_f["total_distance"].notna())
                
                if has_distance and distance_conditions:
                    # Filter tasks with distance data
                    import operator
                    combined_condition = reduce(lambda x, y: x | y, distance_conditions)
                    distance_tasks = tasks_f[combined_condition].copy()
                else:
                    distance_tasks = pd.DataFrame()
                
                if not distance_tasks.empty:
                    # Distance metrics summary
                    col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
                    
                    with col_dist1:
                        avg_in_dist = distance_tasks["in_distance"].mean() if "in_distance" in distance_tasks.columns else 0
                        st.metric("Avg In-Distance", f"{avg_in_dist:.2f}" if avg_in_dist > 0 else "N/A")
                    
                    with col_dist2:
                        avg_out_dist = distance_tasks["out_distance"].mean() if "out_distance" in distance_tasks.columns else 0
                        st.metric("Avg Out-Distance", f"{avg_out_dist:.2f}" if avg_out_dist > 0 else "N/A")
                    
                    with col_dist3:
                        avg_total_dist = distance_tasks["total_distance"].mean() if "total_distance" in distance_tasks.columns else 0
                        st.metric("Avg Total Distance", f"{avg_total_dist:.2f}" if avg_total_dist > 0 else "N/A")
                    
                    with col_dist4:
                        total_dist = distance_tasks["total_distance"].sum() if "total_distance" in distance_tasks.columns else 0
                        st.metric("Total Distance", f"{total_dist:.2f}" if total_dist > 0 else "N/A")
                    
                    # Distance charts
                    col_chart_dist1, col_chart_dist2 = st.columns(2)
                    
                    with col_chart_dist1:
                        # Distance by Agent
                        if "agent_name" in distance_tasks.columns and "total_distance" in distance_tasks.columns:
                            agent_dist = distance_tasks.groupby("agent_name").agg(
                                avg_distance=("total_distance", "mean"),
                                total_distance=("total_distance", "sum"),
                                task_count=("total_distance", "count")
                            ).reset_index()
                            agent_dist = agent_dist.sort_values("total_distance", ascending=False).head(15)
                            
                            if not agent_dist.empty:
                                fig = px.bar(
                                    agent_dist,
                                    x="agent_name",
                                    y="total_distance",
                                    title="Total Distance by Agent (Top 15)",
                                    labels={"total_distance": "Total Distance", "agent_name": "Agent"}
                                )
                                fig = style_chart(fig, "Total Distance by Agent (Top 15)")
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col_chart_dist2:
                        # Distance by Zone
                        if "zone" in distance_tasks.columns and "total_distance" in distance_tasks.columns:
                            zone_dist = distance_tasks.groupby("zone").agg(
                                avg_distance=("total_distance", "mean"),
                                total_distance=("total_distance", "sum"),
                                task_count=("total_distance", "count")
                            ).reset_index()
                            zone_dist = zone_dist.sort_values("total_distance", ascending=False)
                            
                            if not zone_dist.empty:
                                fig = px.bar(
                                    zone_dist,
                                    x="zone",
                                    y="total_distance",
                                    title="Total Distance by Zone",
                                    labels={"total_distance": "Total Distance", "zone": "Zone"}
                                )
                                fig = style_chart(fig, "Total Distance by Zone")
                                fig.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Distance trends over time
                    if "task_day" in distance_tasks.columns and "total_distance" in distance_tasks.columns:
                        daily_dist = distance_tasks.groupby("task_day").agg(
                            avg_distance=("total_distance", "mean"),
                            total_distance=("total_distance", "sum"),
                            task_count=("total_distance", "count")
                        ).reset_index()
                        daily_dist = daily_dist.sort_values("task_day")
                        
                        if not daily_dist.empty:
                            fig = px.line(
                                daily_dist,
                                x="task_day",
                                y=["avg_distance", "total_distance"],
                                title="Distance Trends Over Time",
                                labels={"value": "Distance", "task_day": "Date", "variable": "Metric"}
                            )
                            fig = style_chart(fig, "Distance Trends Over Time")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Distance summary table
                    st.markdown("##### Distance Summary by Agent")
                    if "agent_name" in distance_tasks.columns:
                        dist_summary_cols = []
                        if "in_distance" in distance_tasks.columns:
                            dist_summary_cols.append("in_distance")
                        if "out_distance" in distance_tasks.columns:
                            dist_summary_cols.append("out_distance")
                        if "total_distance" in distance_tasks.columns:
                            dist_summary_cols.append("total_distance")
                        
                        if dist_summary_cols:
                            dist_summary = distance_tasks.groupby("agent_name").agg({
                                "in_distance": ["mean", "sum", "count"] if "in_distance" in distance_tasks.columns else None,
                                "out_distance": ["mean", "sum", "count"] if "out_distance" in distance_tasks.columns else None,
                                "total_distance": ["mean", "sum", "count"] if "total_distance" in distance_tasks.columns else None,
                            }).reset_index()
                            
                            # Flatten column names
                            dist_summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in dist_summary.columns.values]
                            
                            # Create a cleaner summary
                            clean_summary = []
                            for agent in distance_tasks["agent_name"].unique():
                                agent_data = distance_tasks[distance_tasks["agent_name"] == agent]
                                row = {"agent_name": agent}
                                if "in_distance" in agent_data.columns:
                                    row["avg_in_distance"] = agent_data["in_distance"].mean()
                                    row["total_in_distance"] = agent_data["in_distance"].sum()
                                    row["tasks_with_in_dist"] = agent_data["in_distance"].notna().sum()
                                if "out_distance" in agent_data.columns:
                                    row["avg_out_distance"] = agent_data["out_distance"].mean()
                                    row["total_out_distance"] = agent_data["out_distance"].sum()
                                    row["tasks_with_out_dist"] = agent_data["out_distance"].notna().sum()
                                if "total_distance" in agent_data.columns:
                                    row["avg_total_distance"] = agent_data["total_distance"].mean()
                                    row["total_total_distance"] = agent_data["total_distance"].sum()
                                    row["tasks_with_total_dist"] = agent_data["total_distance"].notna().sum()
                                clean_summary.append(row)
                            
                            dist_summary_df = pd.DataFrame(clean_summary)
                            dist_summary_df = dist_summary_df.fillna(0)
                            
                            # Sort by total distance
                            if "total_total_distance" in dist_summary_df.columns:
                                dist_summary_df = dist_summary_df.sort_values("total_total_distance", ascending=False)
                            elif "total_in_distance" in dist_summary_df.columns:
                                dist_summary_df = dist_summary_df.sort_values("total_in_distance", ascending=False)
                            
                            st.dataframe(dist_summary_df, use_container_width=True, height=400)
                            
                            # Download button
                            csv = dist_summary_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "⬇️ Download Distance Summary Table",
                                data=csv,
                                file_name="distance_summary_by_agent.csv",
                                mime="text/csv"
                            )
                    
                    # Detailed distance table (in expander)
                    with st.expander("📋 View Detailed Distance Data"):
                        detail_cols = ["agent_name", "task_day", "place_display"]
                        if "in_distance" in distance_tasks.columns:
                            detail_cols.append("in_distance")
                        if "out_distance" in distance_tasks.columns:
                            detail_cols.append("out_distance")
                        if "total_distance" in distance_tasks.columns:
                            detail_cols.append("total_distance")
                        if "zone" in distance_tasks.columns:
                            detail_cols.append("zone")
                        if "area" in distance_tasks.columns:
                            detail_cols.append("area")
                        
                        available_detail_cols = [c for c in detail_cols if c in distance_tasks.columns]
                        if available_detail_cols:
                            dist_detail = distance_tasks[available_detail_cols].sort_values(
                                "total_distance" if "total_distance" in available_detail_cols else "task_day",
                                ascending=False
                            )
                            dist_detail_display = limit_display_rows(dist_detail, max_rows=1000)
                            st.dataframe(dist_detail_display, use_container_width=True, height=500)
                            if len(dist_detail) > 1000:
                                st.caption(f"⚠️ Showing first 1,000 of {len(dist_detail):,} rows. Use download button for full data.")
                            
                            # Download button
                            csv = dist_detail.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "⬇️ Download Detailed Distance Data",
                                data=csv,
                                file_name="distance_detailed.csv",
                                mime="text/csv"
                            )
                else:
                    st.info("No distance data available for the selected filters.")
            else:
                st.info("Distance fields (in-distance, out-distance) not found in the data.")
            
            # Agent performance table (in expander)
            with st.expander("📋 View Detailed Agent Performance Table"):
                display_cols = [c for c in ["agent_name", "tasks", "completed", "pending", "in_progress",
                                           "completion_rate", "working_days", "avg_shift", "total_shift",
                                           "interactions", "dcc", "ecc", "qr", "bbos"]
                               if c in perf.columns]
                if display_cols:
                    sort_col = "tasks" if "tasks" in display_cols else "interactions" if "interactions" in display_cols else display_cols[0]
                    perf_display = perf[display_cols].sort_values(sort_col, ascending=False)
                    perf_display_limited = limit_display_rows(perf_display, max_rows=1000)
                    st.dataframe(perf_display_limited, use_container_width=True, height=500)
                    if len(perf_display) > 1000:
                        st.caption(f"⚠️ Showing first 1,000 of {len(perf_display):,} rows. Use download button for full data.")
                    # Download button
                    csv = perf_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Agent Performance Table",
                        data=csv,
                        file_name="agent_performance.csv",
                        mime="text/csv"
                    )
    else:
        st.info("No performance data available for the selected filters.")

# ============================================================
# TAB 3: Interactions & Brands
# ============================================================
with tab3:
    st.subheader("🎯 Interactions & Brands")
    
    if inter_f.empty:
        st.info("No interactions found for selected filters.")
    else:
        # Brand Performance Section
        st.markdown("### Brand Performance")
        
        if "main_brand" in inter_f.columns:
            brand_data = inter_f[inter_f["main_brand"].notna() & (inter_f["main_brand"] != "")]
            
            if not brand_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top brands by interaction count
                    top_brands = brand_data.groupby("main_brand").size().reset_index(name="count")
                    top_brands = top_brands.sort_values("count", ascending=False).head(15)
                    fig = px.bar(top_brands, x="main_brand", y="count",
                                title="Top 15 Brands by Interactions")
                    fig = style_chart(fig, "Top 15 Brands by Interactions")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Brand distribution pie chart
                    brand_dist = brand_data["main_brand"].value_counts().head(10).reset_index()
                    brand_dist.columns = ["brand", "count"]
                    fig = px.pie(brand_dist, names="brand", values="count",
                                title="Top 10 Brands Distribution")
                    fig = style_chart(fig, "Top 10 Brands Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Brand trends over time
                if "interaction_day" in brand_data.columns:
                    brand_trends = brand_data.groupby(["interaction_day", "main_brand"]).size().reset_index(name="count")
                    top_5_brands = brand_data["main_brand"].value_counts().head(5).index.tolist()
                    brand_trends_filtered = brand_trends[brand_trends["main_brand"].isin(top_5_brands)]
                    
                    if not brand_trends_filtered.empty:
                        fig = px.line(brand_trends_filtered, x="interaction_day", y="count", color="main_brand",
                                    title="Top 5 Brands Trends Over Time")
                        fig = style_chart(fig, "Top 5 Brands Trends Over Time")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No brand data available for the selected filters.")
        else:
            st.info("Brand data not available in interactions.")
        
        st.markdown("---")
        
        # Demographic Insights Section
        st.markdown("### Demographic Insights")
        
        colA, colB = st.columns(2)
        
        with colA:
            # Gender distribution
            if "gender" in inter_f.columns:
                gender_data = inter_f[inter_f["gender"].notna() & (inter_f["gender"] != "")]
                if not gender_data.empty:
                    gender_dist = gender_data["gender"].value_counts().reset_index()
                    gender_dist.columns = ["gender", "count"]
                    fig = px.pie(gender_dist, names="gender", values="count",
                                title="Interactions by Gender")
                    fig = style_chart(fig, "Interactions by Gender")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No gender data available.")
            else:
                st.info("Gender data not available.")
        
        with colB:
            # Age range distribution
            if "age_range" in inter_f.columns:
                age_data = inter_f[inter_f["age_range"].notna() & (inter_f["age_range"] != "")]
                if not age_data.empty:
                    age_dist = age_data["age_range"].value_counts().reset_index()
                    age_dist.columns = ["age_range", "count"]
                    fig = px.bar(age_dist, x="age_range", y="count",
                                title="Interactions by Age Range")
                    fig = style_chart(fig, "Interactions by Age Range")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No age data available.")
            else:
                st.info("Age range data not available.")
        
        # Brand preferences by demographics
        if "main_brand" in inter_f.columns and ("gender" in inter_f.columns or "age_range" in inter_f.columns):
            st.markdown("#### Brand Preferences by Demographics")
            
            if "gender" in inter_f.columns and "main_brand" in inter_f.columns:
                brand_gender = inter_f[inter_f["main_brand"].notna() & inter_f["gender"].notna() & 
                                      (inter_f["main_brand"] != "") & (inter_f["gender"] != "")]
                if not brand_gender.empty:
                    top_brands_list = brand_gender["main_brand"].value_counts().head(8).index.tolist()
                    brand_gender_filtered = brand_gender[brand_gender["main_brand"].isin(top_brands_list)]
                    
                    if not brand_gender_filtered.empty:
                        brand_gender_cross = brand_gender_filtered.groupby(["main_brand", "gender"]).size().reset_index(name="count")
                        fig = px.bar(brand_gender_cross, x="main_brand", y="count", color="gender",
                                    title="Top Brands by Gender",
                                    barmode="group")
                        fig = style_chart(fig, "Top Brands by Gender")
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
            
            if "age_range" in inter_f.columns and "main_brand" in inter_f.columns:
                brand_age = inter_f[inter_f["main_brand"].notna() & inter_f["age_range"].notna() & 
                                   (inter_f["main_brand"] != "") & (inter_f["age_range"] != "")]
                if not brand_age.empty:
                    top_brands_list = brand_age["main_brand"].value_counts().head(8).index.tolist()
                    brand_age_filtered = brand_age[brand_age["main_brand"].isin(top_brands_list)]
                    
                    if not brand_age_filtered.empty:
                        brand_age_cross = brand_age_filtered.groupby(["main_brand", "age_range"]).size().reset_index(name="count")
                        fig = px.bar(brand_age_cross, x="main_brand", y="count", color="age_range",
                                    title="Top Brands by Age Range",
                                    barmode="group")
                        fig = style_chart(fig, "Top Brands by Age Range")
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Interaction Type Analysis
        st.markdown("### Interaction Type Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "interaction_type" in inter_f.columns:
                it = inter_f.groupby("interaction_type").size().reset_index(name="count")
                it = it.sort_values("count", ascending=False)
                fig = px.bar(it, x="interaction_type", y="count",
                            title="Interactions by Type")
                fig = style_chart(fig, "Interactions by Type")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Interaction type by demographics
            if "interaction_type" in inter_f.columns and "gender" in inter_f.columns:
                it_gender = inter_f[inter_f["interaction_type"].notna() & inter_f["gender"].notna() &
                                   (inter_f["interaction_type"] != "") & (inter_f["gender"] != "")]
                if not it_gender.empty:
                    it_gender_cross = it_gender.groupby(["interaction_type", "gender"]).size().reset_index(name="count")
                    fig = px.bar(it_gender_cross, x="interaction_type", y="count", color="gender",
                                title="Interaction Types by Gender",
                                barmode="group")
                    fig = style_chart(fig, "Interaction Types by Gender")
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Pack Purchase Insights
        st.markdown("### Pack Purchase Insights")
        
        colC, colD = st.columns(2)
        
        with colC:
            if "has_bbos" in inter_f.columns:
                bb = inter_f[inter_f["has_bbos"] == True]
                if not bb.empty:
                    bb_by_agent = bb.groupby("agent_name").size().reset_index(name="bbos_pack")
                    bb_by_agent = bb_by_agent.sort_values("bbos_pack", ascending=False).head(15)
                    fig = px.bar(bb_by_agent, x="agent_name", y="bbos_pack",
                                title="BBOS (Pack contains '+') by Agent")
                    fig = style_chart(fig, "BBOS (Pack contains '+') by Agent")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with colD:
            if "ecc_item" in inter_f.columns:
                ecc_items = inter_f[inter_f["ecc_item"].astype(str).str.strip() != ""]
                if not ecc_items.empty:
                    ecc_top = ecc_items.groupby("ecc_item").size().reset_index(name="count")
                    ecc_top = ecc_top.sort_values("count", ascending=False).head(15)
                    fig = px.bar(ecc_top, x="ecc_item", y="count",
                                title="Top ECC Items (text after '+')")
                    fig = style_chart(fig, "Top ECC Items (text after '+')")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Detailed interactions table (in expander)
        with st.expander("📋 View Detailed Interactions"):
            show_cols = [c for c in [
                "interaction_day","interaction_id","agent_name","place_display","zone","area","channel","sv","role",
                "interaction_type","gender","age_range","main_brand","pack_purchase","ecc_item","url","location"
            ] if c in inter_f.columns]
            if show_cols:
                inter_display = inter_f[show_cols].sort_values("interaction_day", ascending=False)
                inter_display_limited = limit_display_rows(inter_display, max_rows=1000)
                st.dataframe(inter_display_limited, use_container_width=True, height=500)
                if len(inter_display) > 1000:
                    st.caption(f"⚠️ Showing first 1,000 of {len(inter_display):,} rows. Use download button for full data.")
                # Download button
                csv = inter_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Interactions Table",
                    data=csv,
                    file_name="interactions_detailed.csv",
                    mime="text/csv"
                )

# ============================================================
# TAB 4: Stock & Inventory
# ============================================================
with tab4:
    st.subheader("📦 Stock & Inventory")
    
    if stock_f.empty and stock_check_f.empty:
        st.info("No stock data available for the selected filters.")
    else:
        # Stock Overview
        st.markdown("### Stock Overview")
        
        if not stock_f.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Stock trends over time
                if "stock_day" in stock_f.columns:
                    stock_trends = stock_f.groupby("stock_day").agg(
                        release=("qty_release", "sum"),
                        returned=("qty_return", "sum"),
                        used=("qty_used", "sum") if "qty_used" in stock_f.columns else pd.Series([0] * len(stock_f)),
                    ).reset_index()
                    
                    if not stock_trends.empty:
                        fig = px.line(stock_trends, x="stock_day", y=["release", "returned", "used"],
                                    title="Stock Trends Over Time",
                                    labels={"value": "Quantity", "stock_day": "Date", "variable": "Type"})
                        fig = style_chart(fig, "Stock Trends Over Time")
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Stock summary metrics
                total_release = float(stock_f["qty_release"].sum()) if "qty_release" in stock_f.columns else 0
                total_return = float(stock_f["qty_return"].sum()) if "qty_return" in stock_f.columns else 0
                total_used = float(stock_f["qty_used"].sum()) if "qty_used" in stock_f.columns else 0
                net_stock = total_release - total_return - total_used
                
                st.metric("Total Released", f"{total_release:,.0f}")
                st.metric("Total Returned", f"{total_return:,.0f}")
                st.metric("Total Used", f"{total_used:,.0f}")
                st.metric("Net Stock", f"{net_stock:,.0f}")
        
        st.markdown("---")
        
        # Used Stock vs Interactions Analysis
        st.markdown("### Used Stock (Release - Back) vs Interactions Analysis")
        
        if not stock_check_f.empty:
            colA, colB = st.columns(2)
            
            with colA:
                # Used stock (Release - Back) vs interactions comparison
                by_agent = stock_check_f.groupby("agent_name").agg(
                    used=("qty_used","sum"),
                    interactions=("interactions_total","sum"),
                    diff=("diff_used_vs_interactions","sum")
                ).reset_index().sort_values("diff", ascending=True).head(20)
                
                if not by_agent.empty:
                    fig = px.bar(by_agent, x="agent_name", y=["used", "interactions"],
                                title="Used Stock (Release - Back) vs Interactions by Agent",
                                barmode="group",
                                labels={"value": "Quantity", "agent_name": "Agent", "variable": "Type"})
                    fig = style_chart(fig, "Used Stock (Release - Back) vs Interactions by Agent")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with colB:
                # Stock efficiency (utilization rate based on used stock)
                if "qty_used" in stock_check_f.columns and "interactions_total" in stock_check_f.columns:
                    stock_check_f["utilization_rate"] = np.where(
                        stock_check_f["qty_used"] > 0,
                        (stock_check_f["interactions_total"] / stock_check_f["qty_used"]) * 100,
                        0
                    )
                    efficiency_by_agent = stock_check_f.groupby("agent_name")["utilization_rate"].mean().reset_index()
                    efficiency_by_agent = efficiency_by_agent.sort_values("utilization_rate", ascending=False).head(20)
                    
                    if not efficiency_by_agent.empty:
                        fig = px.bar(efficiency_by_agent, x="agent_name", y="utilization_rate",
                                    title="Stock Utilization Rate by Agent (%) (Based on Used Stock)")
                        fig = style_chart(fig, "Stock Utilization Rate by Agent (%) (Based on Used Stock)")
                        fig.update_layout(xaxis_tickangle=45, yaxis_title="Utilization Rate (%)")
                        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Stock Alerts
        st.markdown("### Stock Alerts")
        
        if not stock_check_f.empty:
            # Find issues
            mism = stock_check_f[stock_check_f["stock_flag_interactions"] != "OK"]
            
            if not mism.empty:
                st.warning(f"⚠️ Found {len(mism)} stock issue(s): Interactions exceed used stock (Release - Back)")
                
                alert_cols = [c for c in ["agent_name", "stock_day", "qty_release", "qty_return", "qty_used", "interactions_total",
                                         "diff_used_vs_interactions", "stock_flag_interactions"]
                             if c in mism.columns]
                if alert_cols:
                    mism_display = mism[alert_cols].sort_values("diff_used_vs_interactions")
                    st.dataframe(mism_display, use_container_width=True, height=300)
                    # Download button
                    csv = mism_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Stock Alerts Table",
                        data=csv,
                        file_name="stock_alerts.csv",
                        mime="text/csv"
                    )
            else:
                st.success("✅ No stock issues: All interactions are within used stock limits (Release - Back).")
        
        # Stock details table (in expander)
        with st.expander("📋 View Detailed Stock Data"):
            if not stock_check_f.empty:
                cols = [c for c in [
                    "agent_name","stock_day","qty_release","qty_return","qty_used",
                    "interactions_total","diff_used_vs_interactions","stock_flag_interactions",
                    "bbos_pack","diff_used_vs_bbos_pack","stock_flag_bbos_pack",
                ] if c in stock_check_f.columns]
                if cols:
                    stock_display = stock_check_f[cols].sort_values(["stock_day", "agent_name"], ascending=[False, True])
                    st.dataframe(stock_display, use_container_width=True, height=500)
                    # Download button
                    csv = stock_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Detailed Stock Data Table",
                        data=csv,
                        file_name="stock_detailed.csv",
                        mime="text/csv"
                    )

# ============================================================
# Optional: Validations Section (Collapsible)
# ============================================================
st.markdown("---")
with st.expander("🔍 Operational Validations (Click to expand)", expanded=False):
    st.subheader("🎯 Operational Validations")
    st.caption("These validations are for operational quality checks. Expand to view.")
    
    col1, col2 = st.columns(2)

    def show_validation_block(title: str, vdf: pd.DataFrame):
        st.markdown(f"### {title}")
        if vdf.empty:
            st.info("No data for this validation under current filters.")
            return

        # Summary
        summary = vdf["match_flag"].value_counts().reset_index()
        summary.columns = ["match_flag", "rows"]
        fig = px.bar(summary, x="match_flag", y="rows", title=f"{title} – Match vs Mismatch")
        fig = style_chart(fig, f"{title} – Match vs Mismatch")
        st.plotly_chart(fig, use_container_width=True)

        # Top mismatches
        mismatches = vdf[vdf["match_flag"] == "Mismatch"].sort_values("difference", ascending=False)
        if not mismatches.empty:
            st.markdown("#### Top Mismatches")
            st.dataframe(mismatches.head(50), use_container_width=True, height=300)

        xlsx = to_excel_download({title[:25]: vdf})
        st.download_button(
            f"⬇️ Download {title} (filtered)",
            data=xlsx,
            file_name=f"{title.replace(' ','_').lower()}_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col1:
        show_validation_block("DCC Validation (Tasks vs Interactions)", dcc_f)
        show_validation_block("QR Validation (Tasks vs Interactions)", qr_f)

    with col2:
        show_validation_block("ECC Validation (Tasks vs Interactions)", ecc_f)
        show_validation_block("BBOS Validation (Tasks vs Interactions)", bbos_f)

    st.markdown("---")
    st.markdown("### Attendance Check")
    st.caption("Shows days where an agent had tasks but zero interactions on the same day.")
    if att_f.empty:
        st.info("No attendance issues found under current filters.")
    else:
        issues = att_f[att_f["activity_flag"] == "Worked_no_interactions"]
        if issues.empty:
            st.success("✅ No 'worked but no interactions' issues under current filters.")
        else:
            st.warning(f"⚠️ Found {len(issues)} issue row(s)")
            issues_display = issues.sort_values("tasks_any", ascending=False)
            st.dataframe(issues_display, use_container_width=True, height=300)
            # Download button
            csv = issues_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Attendance Issues Table",
                data=csv,
                file_name="attendance_issues.csv",
                mime="text/csv"
            )

# ============================================================
# Consolidated Summary Table
# ============================================================
st.markdown("---")
st.subheader("📊 Consolidated Agent Summary")
st.caption("Key metrics from performance, stock, validations, and attendance data")

# Build consolidated table
consolidated_list = []

# Start with agent performance data
if not tasks_f.empty or not inter_f.empty:
    agent_perf_consolidated = []
    
    if not tasks_f.empty:
        perf_tasks_cons = tasks_f.groupby("agent_name").agg(
            tasks=("task_day", "count"),
            completed=("status_norm", lambda s: (s=="completed").sum()),
            pending=("status_norm", lambda s: (s=="pending").sum()),
            working_days=("task_day", lambda s: pd.Series(s).nunique()),
            avg_shift_hours=("shift_hours", "mean"),
        ).reset_index()
        agent_perf_consolidated.append(perf_tasks_cons)
    
    if not inter_f.empty:
        perf_inter_cons = inter_f.groupby("agent_name").agg(
            interactions=("interaction_day", "count"),
        ).reset_index()
        agent_perf_consolidated.append(perf_inter_cons)
    
    if agent_perf_consolidated:
        consolidated_base = reduce(lambda x, y: pd.merge(x, y, on="agent_name", how="outer"), agent_perf_consolidated)
        # Safely fillna - handle categorical columns properly
        consolidated_base = safe_fillna(consolidated_base, value=0)
        
        if "tasks" in consolidated_base.columns and consolidated_base["tasks"].sum() > 0:
            consolidated_base["completion_rate"] = np.where(
                consolidated_base["tasks"] > 0,
                (consolidated_base["completed"] / consolidated_base["tasks"]) * 100,
                0
            ).round(1)
        
        consolidated_list.append(consolidated_base)

# Add stock utilization data
if not stock_check_f.empty and "agent_name" in stock_check_f.columns:
    stock_summary = stock_check_f.groupby("agent_name").agg(
        total_stock_released=("qty_release", "sum"),
        total_stock_returned=("qty_return", "sum"),
        total_stock_used=("qty_used", "sum"),
        total_interactions_stock=("interactions_total", "sum"),
        stock_issues_count=("stock_flag_interactions", lambda s: (s != "OK").sum()),
    ).reset_index()
    
    stock_summary["stock_utilization_rate"] = np.where(
        stock_summary["total_stock_used"] > 0,
        (stock_summary["total_interactions_stock"] / stock_summary["total_stock_used"]) * 100,
        0
    ).round(1)
    
    if consolidated_list:
        consolidated_list[0] = consolidated_list[0].merge(stock_summary, on="agent_name", how="outer")
    else:
        consolidated_list.append(stock_summary)

# Add validation mismatch counts
validation_cols = {}
if not dcc_f.empty and "agent_name" in dcc_f.columns:
    dcc_mismatches = dcc_f[dcc_f["match_flag"] == "Mismatch"].groupby("agent_name").size().reset_index(name="dcc_mismatches")
    validation_cols["dcc_mismatches"] = dcc_mismatches

if not ecc_f.empty and "agent_name" in ecc_f.columns:
    ecc_mismatches = ecc_f[ecc_f["match_flag"] == "Mismatch"].groupby("agent_name").size().reset_index(name="ecc_mismatches")
    validation_cols["ecc_mismatches"] = ecc_mismatches

if not qr_f.empty and "agent_name" in qr_f.columns:
    qr_mismatches = qr_f[qr_f["match_flag"] == "Mismatch"].groupby("agent_name").size().reset_index(name="qr_mismatches")
    validation_cols["qr_mismatches"] = qr_mismatches

if not bbos_f.empty and "agent_name" in bbos_f.columns:
    bbos_mismatches = bbos_f[bbos_f["match_flag"] == "Mismatch"].groupby("agent_name").size().reset_index(name="bbos_mismatches")
    validation_cols["bbos_mismatches"] = bbos_mismatches

for col_name, val_df in validation_cols.items():
    if consolidated_list:
        consolidated_list[0] = consolidated_list[0].merge(val_df, on="agent_name", how="left")
    else:
        consolidated_list.append(val_df)

# Add attendance issues flag
if not att_f.empty and "agent_name" in att_f.columns:
    attendance_issues = att_f[att_f["activity_flag"] == "Worked_no_interactions"].groupby("agent_name").size().reset_index(name="attendance_issues_count")
    if consolidated_list:
        consolidated_list[0] = consolidated_list[0].merge(attendance_issues, on="agent_name", how="left")
    else:
        consolidated_list.append(attendance_issues)

# Add low performer flag
if not low_performers_data.empty and "agent_name" in low_performers_data.columns:
    low_performers_flag = pd.DataFrame({
        "agent_name": low_performers_data["agent_name"].unique(),
        "is_low_performer": True
    })
    if consolidated_list:
        consolidated_list[0] = consolidated_list[0].merge(low_performers_flag, on="agent_name", how="left")
    else:
        consolidated_list.append(low_performers_flag)

# Fill NaN values and create final consolidated table
if consolidated_list:
    consolidated_final = reduce(lambda x, y: pd.merge(x, y, on="agent_name", how="outer"), consolidated_list) if len(consolidated_list) > 1 else consolidated_list[0]
    # Safely fillna - handle categorical columns properly
    consolidated_final = safe_fillna(consolidated_final, value=0)
    
    # Calculate total validation issues
    val_cols = [c for c in consolidated_final.columns if "mismatches" in c.lower()]
    if val_cols:
        consolidated_final["total_validation_issues"] = consolidated_final[val_cols].sum(axis=1)
    
    # Select and order columns for display
    display_order = [
        "agent_name", "tasks", "completed", "pending", "completion_rate", 
        "interactions", "working_days", "avg_shift_hours",
        "total_stock_used", "total_interactions_stock", "stock_utilization_rate", "stock_issues_count",
        "dcc_mismatches", "ecc_mismatches", "qr_mismatches", "bbos_mismatches", "total_validation_issues",
        "attendance_issues_count", "is_low_performer"
    ]
    
    available_cols = [c for c in display_order if c in consolidated_final.columns]
    consolidated_display = consolidated_final[available_cols].copy()
    
    # Sort by total activity (tasks + interactions) or just available metrics
    if "tasks" in consolidated_display.columns and "interactions" in consolidated_display.columns:
        consolidated_display["total_activity"] = consolidated_display["tasks"] + consolidated_display["interactions"]
        consolidated_display = consolidated_display.sort_values("total_activity", ascending=False)
    elif "interactions" in consolidated_display.columns:
        consolidated_display = consolidated_display.sort_values("interactions", ascending=False)
    elif "tasks" in consolidated_display.columns:
        consolidated_display = consolidated_display.sort_values("tasks", ascending=False)
    
    if not consolidated_display.empty:
        consolidated_display_limited = limit_display_rows(consolidated_display, max_rows=1000)
        st.dataframe(consolidated_display_limited, use_container_width=True, height=500)
        if len(consolidated_display) > 1000:
            st.caption(f"⚠️ Showing first 1,000 of {len(consolidated_display):,} rows. Use download button for full data.")
        # Download button
        csv = consolidated_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Consolidated Summary Table",
            data=csv,
            file_name="consolidated_agent_summary.csv",
            mime="text/csv"
        )
    else:
        st.info("No consolidated data available for the selected filters.")
else:
    st.info("No data available to create consolidated summary.")

# ============================================================
# Global export
# ============================================================
st.markdown("---")
st.subheader("⬇️ Export (filtered)")
# Filter supervisor_perf for export
supervisor_perf_f = supervisor_perf.copy()
if not supervisor_perf_f.empty and "sv" in supervisor_perf_f.columns:
    if sv_filter:
        supervisor_perf_f = supervisor_perf_f[supervisor_perf_f["sv"].isin(sv_filter)]
    if not u.empty and "sv" in u.columns:
        valid_svs = set(u["sv"].dropna().astype(str).tolist())
        supervisor_perf_f = supervisor_perf_f[supervisor_perf_f["sv"].isin(valid_svs)]

export = to_excel_download({
    "agents_filtered": u,
    "tasks_filtered": tasks_f,
    "interactions_filtered": inter_f,
    "supervisor_performance": supervisor_perf_f,
    "dcc_validation": dcc_f,
    "ecc_validation": ecc_f,
    "qr_validation": qr_f,
    "bbos_validation": bbos_f,
    "stock_validation": stock_check_f,
    "attendance_check": att_f,
})
st.download_button(
    "Download Full Filtered Export (Excel)",
    data=export,
    file_name="dashboard_filtered_export.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
