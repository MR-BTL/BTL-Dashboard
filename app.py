import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from PIL import Image
import os
import requests
import re
from urllib.parse import urlparse, parse_qs
import base64

# Brand colors: White, Black, Red (majority white, rare red)
BRAND_COLORS = {
    "white": "#FFFFFF",
    "black": "#000000",
    "red": "#DC143C",  # Crimson red
    "light_gray": "#F5F5F5",
    "dark_gray": "#333333"
}

# Color palettes for consistent visuals (colorful theme)
PALETTE_MAIN = px.colors.qualitative.Set2
PALETTE_ALT = px.colors.qualitative.Set3

# ----------------------------------------
# 1. App config & Branding
# ----------------------------------------
st.set_page_config(
    page_title="Activation Agents Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize dark mode in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Apply Century Gothic font and brand styling with dynamic dark/light mode support
# Get dark mode state
dark_mode = st.session_state.dark_mode

# Determine text colors based on mode
text_color = BRAND_COLORS["white"] if dark_mode else BRAND_COLORS["black"]
metric_label_color = "#CCCCCC" if dark_mode else BRAND_COLORS["dark_gray"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Century+Gothic:wght@400;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {{
        font-family: 'Century Gothic', sans-serif !important;
    }}
    
    /* Text colors based on dark/light mode */
    body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] p, div, span {{
        color: {text_color} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {text_color} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {metric_label_color} !important;
    }}
    
    /* Font styling */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Century Gothic', sans-serif !important;
        font-weight: 700 !important;
    }}
    
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        font-family: 'Century Gothic', sans-serif !important;
    }}
    
    /* Buttons - rare red accent */
    .stDownloadButton>button {{
        background-color: {BRAND_COLORS["red"]} !important;
        color: {BRAND_COLORS["white"]} !important;
        border: none !important;
        font-family: 'Century Gothic', sans-serif !important;
    }}
    
    .stDownloadButton>button:hover {{
        background-color: {BRAND_COLORS["dark_gray"]} !important;
    }}
    
    /* Cards for photos */
    .photo-card {{
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: var(--background-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }}
    
    .photo-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}
    
    .photo-card img {{
        width: 100%;
        max-height: 300px;
        object-fit: cover;
        border-radius: 4px;
        margin-bottom: 10px;
    }}
    
    .photo-info {{
        font-size: 13px;
        line-height: 1.6;
        color: var(--text-color);
        font-family: 'Century Gothic', sans-serif;
    }}
    
    .photo-info strong {{
        color: {BRAND_COLORS["red"]};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------
# 1.1. Company Logo
# ----------------------------------------
LOGO_WIDTH = 200  # Adjust logo width here (in pixels)
LOGO_PATHS = ["logo.png", "logo.jpg", "logo.jpeg", "assets/logo.png", "assets/logo.jpg"]

def display_logo(width=LOGO_WIDTH):
    """Display company logo at the top of the dashboard"""
    logo_path = None
    for path in LOGO_PATHS:
        if os.path.exists(path):
            logo_path = path
            break
    
    if logo_path:
        try:
            # Determine image MIME type
            ext = os.path.splitext(logo_path)[1].lower()
            mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
            mime_type = mime_types.get(ext, 'image/png')
            
            # Use HTML/CSS for perfect centering
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 10px 0;">
                    <img src="data:{mime_type};base64,{image_to_base64(logo_path)}" width="{width}" style="margin: 0 auto; display: block;">
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")
        except Exception as e:
            st.warning(f"Could not load logo: {e}")

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Display logo if it exists
display_logo()

st.title("📊 Activation Agents Performance & Data Quality Dashboard")
st.caption("From raw Excel → validation → interactive analytics dashboard")


# ----------------------------------------
# 2. Load & process data
# ----------------------------------------
def convert_google_sheets_url(url):
    """Convert Google Sheets share URL to export format"""
    # Pattern to extract sheet ID and gid
    # Example: https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid={GID}
    pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url)
    
    if not match:
        return None
    
    sheet_id = match.group(1)
    
    # Return export URL (export entire workbook as xlsx - all sheets included)
    # Removing gid parameter exports the entire workbook with all sheets
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    return export_url

def load_file_from_url(url):
    """Download Excel file from Google Sheets URL"""
    export_url = convert_google_sheets_url(url)
    if not export_url:
        raise ValueError("Invalid Google Sheets URL format")
    
    response = requests.get(export_url)
    response.raise_for_status()
    return BytesIO(response.content)

@st.cache_data
def load_and_process(file_or_url) -> dict:
    # Check if it's a URL (string) or file (BytesIO/UploadedFile)
    if isinstance(file_or_url, str):
        # It's a Google Sheets URL
        file = load_file_from_url(file_or_url)
    else:
        # It's an uploaded file
        file = file_or_url
    
    xls = pd.ExcelFile(file)

    # Load sheets
    users      = pd.read_excel(xls, "Users")
    login      = pd.read_excel(xls, "Login")
    tasks      = pd.read_excel(xls, "Tasks")
    stocklog   = pd.read_excel(xls, "Stocklog")
    inter      = pd.read_excel(xls, "interactions")
    main_brand = pd.read_excel(xls, "main")
    purchase   = pd.read_excel(xls, "purchase")

    # Clean column names
    for df in [users, login, tasks, stocklog, inter]:
        df.columns = df.columns.str.strip()

    # Helper to normalize user-id everywhere
    def clean_user_id(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace("nan", "", regex=False)
        )

    # Normalize IDs
    for df in [users, login, tasks, inter]:
        if "user-id" in df.columns:
            df["user-id"] = clean_user_id(df["user-id"])

    if "username" in users.columns:
        users["username"] = users["username"].astype(str).str.strip()

    # Lookup table
    users_lookup = users[["user-id", "username", "zone", "area", "role"]].drop_duplicates()

    # -------- Login --------
    login["timestamp"] = pd.to_datetime(login["timestamp"])
    login["login_date"] = login["timestamp"].dt.date

    login_daily = (
        login.groupby(["user-id", "login_date"])
        .size()
        .reset_index(name="login_count")
        .merge(users_lookup, on="user-id", how="left")
    )

    # -------- Tasks --------
    tasks["task-date"] = pd.to_datetime(tasks["task-date"])
    tasks["task_day"] = tasks["task-date"].dt.date

    # Numeric fields for each metric
    for col, newcol in [("DCC", "DCC_fixed"), ("ECC", "ECC_fixed"), ("QR", "QR_fixed"), ("BBOS", "BBOS_fixed")]:
        if col in tasks.columns:
            tasks[newcol] = pd.to_numeric(tasks[col], errors="coerce").fillna(0)
        else:
            tasks[newcol] = 0

    # Aggregates per metric per agent/day (NO merge with users yet)
    tasks_dcc = (
        tasks.groupby(["user-id", "task_day"], as_index=False)["DCC_fixed"]
        .sum()
        .rename(columns={"DCC_fixed": "dcc_total_tasks"})
    )

    tasks_ecc = (
        tasks.groupby(["user-id", "task_day"], as_index=False)["ECC_fixed"]
        .sum()
        .rename(columns={"ECC_fixed": "ecc_total_tasks"})
    )

    tasks_qr = (
        tasks.groupby(["user-id", "task_day"], as_index=False)["QR_fixed"]
        .sum()
        .rename(columns={"QR_fixed": "qr_total_tasks"})
    )

    tasks_bbos = (
        tasks.groupby(["user-id", "task_day"], as_index=False)["BBOS_fixed"]
        .sum()
        .rename(columns={"BBOS_fixed": "bbos_total_tasks"})
    )

    # Tasks summary (for performance view)
    tasks_summary = (
        tasks.groupby(["user-id"], as_index=False)
        .agg(
            tasks_count=("task-date", "count"),
            avg_in_distance=("in-distance", "mean"),
            avg_out_distance=("out-distance", "mean"),
        )
        .merge(users_lookup, on="user-id", how="left")
    )
    
    # Detailed tasks table with place information
    tasks_detailed = tasks.copy()
    # Extract place name (not coordinates)
    place_name_cols_tasks = [c for c in tasks_detailed.columns if 'place' in c.lower() and 'name' in c.lower() and 'status' not in c.lower()]
    place_cols_tasks = [c for c in tasks_detailed.columns if 'place' in c.lower() and 'code' not in c.lower() and 'name' not in c.lower() and 'status' not in c.lower()]
    
    if place_name_cols_tasks:
        tasks_detailed["place_name"] = tasks_detailed[place_name_cols_tasks[0]].astype(str).fillna("")
    elif place_cols_tasks:
        # Check if it's coordinates
        place_data_tasks = tasks_detailed[place_cols_tasks[0]].astype(str)
        is_coords_tasks = place_data_tasks.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
        if not is_coords_tasks:
            tasks_detailed["place_name"] = place_data_tasks.fillna("")
        else:
            tasks_detailed["place_name"] = ""
    else:
        tasks_detailed["place_name"] = ""
    
    # Merge with users to get agent name
    tasks_detailed = tasks_detailed.merge(users_lookup[["user-id", "username", "zone", "area"]], on="user-id", how="left")
    
    # Convert task-date to date
    if "task-date" in tasks_detailed.columns:
        tasks_detailed["date"] = pd.to_datetime(tasks_detailed["task-date"], errors='coerce').dt.date
    
    # Select relevant columns for display
    tasks_display_cols = ["date", "username", "place_name", "zone", "area", "status", "shift"]
    if "in-distance" in tasks_detailed.columns:
        tasks_display_cols.insert(3, "in-distance")
    if "out-distance" in tasks_detailed.columns:
        tasks_display_cols.insert(4, "out-distance")
    
    existing_display_cols = [c for c in tasks_display_cols if c in tasks_detailed.columns]
    tasks_detailed = tasks_detailed[existing_display_cols] if existing_display_cols else tasks_detailed

    # Tasks per day
    tasks_per_day = (
        tasks.groupby("task_day")
        .size()
        .reset_index(name="tasks_count")
        .rename(columns={"task_day": "date"})
    )

    # Task distributions
    tasks_status_counts = (
        tasks.groupby("status")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    tasks_shift_counts = (
        tasks.groupby("shift")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # -------- Interactions --------
    inter["date"] = pd.to_datetime(inter["date"])
    inter["day"] = pd.to_datetime(inter["day"])
    inter["interaction_day"] = inter["day"].dt.date

    inter_daily = (
        inter.groupby(["user-id", "interaction_day"])
        .size()
        .reset_index(name="interactions_count")
    )
    inter_daily["user-id"] = clean_user_id(inter_daily["user-id"])

    # Flags per interaction type
    inter["is_DCC"]  = inter["Consumer interactions"].eq("DCC").astype(int)
    inter["is_ECC"]  = inter["Consumer interactions"].eq("ECC").astype(int)
    inter["is_QR"]   = inter["Consumer interactions"].eq("QR").astype(int)
    inter["is_BBOS"] = inter["Consumer interactions"].eq("BBOS").astype(int)

    def _sum_flag(colname, newname):
        g = (
            inter.groupby(["user-id", "interaction_day"], as_index=False)[colname]
            .sum()
            .rename(columns={colname: newname})
        )
        g["user-id"] = clean_user_id(g["user-id"])
        return g

    dcc_inter_daily  = _sum_flag("is_DCC", "dcc_from_interactions")
    ecc_inter_daily  = _sum_flag("is_ECC", "ecc_from_interactions")
    qr_inter_daily   = _sum_flag("is_QR", "qr_from_interactions")
    bbos_inter_daily = _sum_flag("is_BBOS", "bbos_from_interactions")

    inter_daily = (
        inter_daily
        .merge(dcc_inter_daily,  on=["user-id", "interaction_day"], how="left")
        .merge(ecc_inter_daily,  on=["user-id", "interaction_day"], how="left")
        .merge(qr_inter_daily,   on=["user-id", "interaction_day"], how="left")
        .merge(bbos_inter_daily, on=["user-id", "interaction_day"], how="left")
        .merge(users_lookup,     on="user-id", how="left")
    )

    for c in ["dcc_from_interactions", "ecc_from_interactions", "qr_from_interactions", "bbos_from_interactions"]:
        inter_daily[c] = inter_daily[c].fillna(0)

    # Interactions per day
    inter_per_day = (
        inter.groupby("interaction_day")
        .size()
        .reset_index(name="interactions_count")
        .rename(columns={"interaction_day": "date"})
    )

    # Interactions distributions
    inter_type_counts = (
        inter.groupby("Consumer interactions")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    inter_agent_counts = (
        inter.groupby("user-name")
        .size()
        .reset_index(name="interactions_count")
        .sort_values("interactions_count", ascending=False)
    )

    inter_gender_counts = (
        inter.groupby("Gender")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    inter_age_counts = (
        inter.groupby("Age Range")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    inter_main_brand_counts = (
        inter.groupby("Main Brand")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    inter_pack_purchase_counts = (
        inter.groupby("Pack Purchase")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # -------- Stocklog --------
    stocklog["date"] = pd.to_datetime(stocklog["date"])
    stocklog["stock_day"] = stocklog["date"].dt.date

    stock_agg = (
        stocklog.groupby(["agent-name", "stock_day"], as_index=False)
        .agg(
            issued=("issued", "sum"),
            used=("used", "sum"),
            returned=("returned", "sum"),
            balance=("balance", "sum"),
        )
    )

    stock_agg = stock_agg.merge(
        users_lookup.rename(columns={"username": "agent-name"}),
        on="agent-name",
        how="left",
    )

    inter_agent_daily = (
        inter.groupby(["user-name", "interaction_day"], as_index=False)
        .size()
        .rename(columns={"size": "interactions_count"})
    )
    inter_agent_daily = inter_agent_daily.rename(
        columns={"user-name": "agent-name", "interaction_day": "stock_day"}
    )

    # -------- Validation builders --------
    def build_validation(tasks_df, tasks_day_col, tasks_total_col,
                         inter_col_name, label_prefix):
        # Ensure user-id clean
        tasks_df = tasks_df.copy()
        tasks_df["user-id"] = clean_user_id(tasks_df["user-id"])

        v = pd.merge(
            tasks_df.rename(columns={tasks_day_col: "day_key"}),
            inter_daily[["user-id", "interaction_day", inter_col_name]]
                .rename(columns={"interaction_day": "day_key"}),
            on=["user-id", "day_key"],
            how="outer",
        )
        v[tasks_total_col] = v[tasks_total_col].fillna(0)
        v[inter_col_name]  = v[inter_col_name].fillna(0)
        diff_col = f"{label_prefix}_difference"
        flag_col = f"{label_prefix}_match_flag"
        v[diff_col] = v[tasks_total_col] - v[inter_col_name]
        v[flag_col] = np.where(v[diff_col] == 0, "Match", "Mismatch")
        v = v.merge(users_lookup, on="user-id", how="left")
        return v

    # DCC / ECC / QR / BBOS validations
    dcc_check = build_validation(
        tasks_dcc, "task_day", "dcc_total_tasks",
        "dcc_from_interactions", "dcc"
    )
    ecc_check = build_validation(
        tasks_ecc, "task_day", "ecc_total_tasks",
        "ecc_from_interactions", "ecc"
    )
    qr_check = build_validation(
        tasks_qr, "task_day", "qr_total_tasks",
        "qr_from_interactions", "qr"
    )
    bbos_check = build_validation(
        tasks_bbos, "task_day", "bbos_total_tasks",
        "bbos_from_interactions", "bbos"
    )

    dcc_check_view = dcc_check[
        [
            "user-id","username","zone","area","role",
            "day_key","dcc_total_tasks","dcc_from_interactions",
            "dcc_difference","dcc_match_flag"
        ]
    ].sort_values("dcc_difference", ascending=False)

    ecc_check_view = ecc_check[
        [
            "user-id","username","zone","area","role",
            "day_key","ecc_total_tasks","ecc_from_interactions",
            "ecc_difference","ecc_match_flag"
        ]
    ].sort_values("ecc_difference", ascending=False)

    qr_check_view = qr_check[
        [
            "user-id","username","zone","area","role",
            "day_key","qr_total_tasks","qr_from_interactions",
            "qr_difference","qr_match_flag"
        ]
    ].sort_values("qr_difference", ascending=False)

    bbos_check_view = bbos_check[
        [
            "user-id","username","zone","area","role",
            "day_key","bbos_total_tasks","bbos_from_interactions",
            "bbos_difference","bbos_match_flag"
        ]
    ].sort_values("bbos_difference", ascending=False)

    # -------- Stock validation --------
    stock_check = stock_agg.merge(
        inter_agent_daily,
        on=["agent-name", "stock_day"],
        how="left",
    )
    stock_check["interactions_count"] = stock_check["interactions_count"].fillna(0)

    stock_check["used_minus_interactions"] = (
        stock_check["used"] - stock_check["interactions_count"]
    )
    stock_check["stock_flag"] = np.where(
        stock_check["used_minus_interactions"] == 0, "Balanced", "Imbalanced"
    )

    stock_check_view = stock_check[
        [
            "agent-name",
            "user-id",
            "zone",
            "area",
            "role",
            "stock_day",
            "issued",
            "used",
            "returned",
            "balance",
            "interactions_count",
            "used_minus_interactions",
            "stock_flag",
        ]
    ].sort_values(["used_minus_interactions", "agent-name"], ascending=[False, True])

    # -------- Attendance validation --------
    attendance = login_daily.merge(
        inter_daily.rename(columns={"interaction_day": "login_date"}),
        on=["user-id", "login_date"],
        how="left",
    )
    attendance["interactions_count"] = attendance["interactions_count"].fillna(0)

    attendance["activity_flag"] = np.where(
        (attendance["login_count"] > 0) & (attendance["interactions_count"] == 0),
        "Attended_no_interactions",
        "OK",
    )

    attendance_view = attendance.merge(users_lookup, on="user-id", how="left")

    attendance_view = attendance_view[
        [
            "user-id",
            "username",
            "zone",
            "area",
            "role",
            "login_date",
            "login_count",
            "interactions_count",
            "activity_flag",
        ]
    ]

    # -------- NEW: Enhanced Interactions Processing --------
    # Extract place name/code (handle various column name variations)
    # Try to find place-name, place_name, place code, etc.
    place_name_cols = [col for col in inter.columns if ('place' in col.lower() and 'name' in col.lower()) or col.lower() == 'place-name']
    place_code_cols = [col for col in inter.columns if 'place' in col.lower() and 'code' in col.lower()]
    place_cols = [col for col in inter.columns if 'place' in col.lower() and 'code' not in col.lower() and 'name' not in col.lower()]
    
    # Extract place name (not coordinates)
    if place_name_cols:
        inter["place_name"] = inter[place_name_cols[0]].astype(str)
    elif place_cols:
        # If we have a place column, check if it's coordinates (has numbers/lat/long pattern)
        place_data = inter[place_cols[0]].astype(str)
        # Check if it looks like coordinates (contains numbers with decimal points or lat/long patterns)
        is_coords = place_data.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
        if not is_coords:
            inter["place_name"] = place_data
        else:
            # If it's coordinates, try to find place name in other columns
            inter["place_name"] = ""
    else:
        inter["place_name"] = ""
    
    # Extract place code
    if place_code_cols:
        inter["place_code"] = inter[place_code_cols[0]].astype(str)
    else:
        inter["place_code"] = ""
    
    # Combine place name and code for display
    inter["place_display"] = inter.apply(
        lambda row: f"{row['place_name']} ({row['place_code']})" if row['place_code'] and str(row['place_code']) != 'nan' and str(row['place_code']) != '' 
        else row['place_name'], axis=1
    )
    
    # Extract interaction ID (handle various column name variations)
    id_cols = [col for col in inter.columns if 'id' in col.lower() and 'interaction' in col.lower()]
    if id_cols:
        inter["interaction_id"] = inter[id_cols[0]].astype(str)
    else:
        inter["interaction_id"] = inter.index.astype(str)
    
    # Extract photo URLs (handle various column name variations)
    photo_cols = [col for col in inter.columns if 'photo' in col.lower() or 'image' in col.lower() or 'url' in col.lower()]
    if photo_cols:
        inter["photo_url"] = inter[photo_cols[0]].astype(str)
    else:
        inter["photo_url"] = ""
    
    # ========================================================================
    # SIMPLIFIED ECC/BBOS Parsing:
    # 
    # BBOS = ANY pack purchase that contains "+" symbol (count = 1 for each)
    # ECC = ANY TEXT that comes after the "+" sign
    # 
    # Examples:
    #   "Marlboro +1 cricket light" -> BBOS count=1, ECC="1 cricket light"
    #   "Camel +lighter" -> BBOS count=1, ECC="lighter"
    #   "Winston" -> BBOS count=0, ECC=""
    # ========================================================================
    def parse_pack_purchase_simple(pack_str):
        if pd.isna(pack_str) or str(pack_str).strip() == "":
            return {"ecc_item": "", "bbos_count": 0, "has_bbos": False}
        
        pack_str = str(pack_str).strip()
        
        # Simple check: does it have a "+" sign?
        if "+" in pack_str:
            # Split by + and take everything after it
            parts = pack_str.split("+", 1)  # Split only on first +
            ecc_text = parts[1].strip() if len(parts) > 1 else ""
            
            return {
                "ecc_item": ecc_text,  # Everything after + is ECC
                "bbos_count": 1,  # Count as 1 BBOS if it has +
                "has_bbos": True
            }
        
        return {"ecc_item": "", "bbos_count": 0, "has_bbos": False}
    
    # Apply simplified parsing
    inter["pack_purchase_parsed"] = inter.get("Pack Purchase", pd.Series()).apply(parse_pack_purchase_simple)
    inter["bbos_count"] = inter["pack_purchase_parsed"].apply(lambda x: x.get("bbos_count", 0) if isinstance(x, dict) else 0)
    inter["ecc_item"] = inter["pack_purchase_parsed"].apply(lambda x: x.get("ecc_item", "") if isinstance(x, dict) else "")
    inter["has_bbos"] = inter["pack_purchase_parsed"].apply(lambda x: x.get("has_bbos", False) if isinstance(x, dict) else False)
    
    # Main vs Occasional brands - Fixed to avoid duplicates
    # Get the main brands reference list from main_brand sheet
    main_brands_reference = set()
    if not main_brand.empty:
        # Try to find brand column (could be "brand", "Brand", first column, etc.)
        brand_col = None
        for col in main_brand.columns:
            if 'brand' in str(col).lower():
                brand_col = col
                break
        if brand_col is None and len(main_brand.columns) > 0:
            brand_col = main_brand.columns[0]
        
        if brand_col:
            main_brands_reference = set(main_brand[brand_col].dropna().astype(str).str.strip().str.lower().unique())
    
    # Mark each interaction
    if "Main Brand" in inter.columns:
        inter["is_main_brand"] = inter["Main Brand"].notna()
        # An interaction is occasional brand if it has a brand but it's NOT in the main brands list
        inter["brand_lower"] = inter["Main Brand"].astype(str).str.strip().str.lower()
        inter["is_occasional_brand"] = (inter["Main Brand"].notna()) & (~inter["brand_lower"].isin(main_brands_reference))
    else:
        inter["is_main_brand"] = False
        inter["is_occasional_brand"] = False
    
    # -------- NEW: Performance Tables --------
    # Find check-in and check-out columns first (VERY flexible search)
    checkin_col = None
    checkout_col = None
    
    # Priority 1: Exact matches (case-insensitive)
    exact_checkin_patterns = [
        'check-in', 'checkin', 'check_in', 'check-in time', 'checkin time',
        'check in', 'time check-in', 'time checkin'
    ]
    exact_checkout_patterns = [
        'check-out', 'checkout', 'check_out', 'check-out time', 'checkout time',
        'check out', 'time check-out', 'time checkout'
    ]
    
    # Try exact matches first
    for col in tasks.columns:
        col_lower = col.lower().strip()
        if col_lower in exact_checkin_patterns:
            checkin_col = col
            break
    
    for col in tasks.columns:
        col_lower = col.lower().strip()
        if col_lower in exact_checkout_patterns:
            checkout_col = col
            break
    
    # Priority 2: If not found, look for columns containing both keywords
    if not checkin_col:
        for col in tasks.columns:
            col_lower = col.lower().replace('-', '').replace('_', '').replace(' ', '')
            # Must contain "check" and "in" but NOT "out"
            if 'check' in col_lower and 'in' in col_lower and 'out' not in col_lower:
                checkin_col = col
                break
    
    if not checkout_col:
        for col in tasks.columns:
            col_lower = col.lower().replace('-', '').replace('_', '').replace(' ', '')
            # Must contain "check" and "out"
            if 'check' in col_lower and 'out' in col_lower:
                checkout_col = col
                break
    
    # Priority 3: Look for just "in" or "out" with "time" if check-related columns not found
    if not checkin_col:
        for col in tasks.columns:
            col_lower = col.lower().strip()
            # Look for time-in, in-time, etc.
            if ('time' in col_lower and 'in' in col_lower and 'out' not in col_lower):
                checkin_col = col
                break
    
    if not checkout_col:
        for col in tasks.columns:
            col_lower = col.lower().strip()
            # Look for time-out, out-time, etc.
            if ('time' in col_lower and 'out' in col_lower):
                checkout_col = col
                break
    
    # Performance table: date, agent name, shift (calculated), place name
    if not tasks.empty and "task-date" in tasks.columns and "user-id" in tasks.columns:
        # Start with all required columns
        required_cols = ["task-date", "user-id"]
        available_cols = [c for c in required_cols if c in tasks.columns]
        
        if len(available_cols) == len(required_cols):
            performance_table = tasks.copy()
            
            # Calculate shift from check-in/check-out times
            if checkin_col and checkout_col:
                # Parse times
                performance_table["checkin_parsed"] = pd.to_datetime(performance_table[checkin_col], errors='coerce')
                performance_table["checkout_parsed"] = pd.to_datetime(performance_table[checkout_col], errors='coerce')
                
                # Calculate shift duration in hours
                performance_table["shift_hours"] = (
                    performance_table["checkout_parsed"] - performance_table["checkin_parsed"]
                ).dt.total_seconds() / 3600
                
                # Clean up invalid shifts (negative or >24 hours)
                performance_table["shift_hours"] = performance_table["shift_hours"].apply(
                    lambda x: x if pd.notna(x) and 0 <= x <= 24 else None
                )
                
                # Format shift for display (e.g., "8.5" hours)
                performance_table["shift_display"] = performance_table["shift_hours"].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else ""
                )
                
                # Create category
                def categorize_shift(val):
                    if pd.isna(val):
                        return "Unknown"
                    elif val < 0 or val > 24:
                        return "Invalid"
                    elif val < 1:
                        return "Less than 1 hour"
                    elif val <= 8:
                        return "1-8 hours"
                    else:
                        return "More than 8 hours"
                
                performance_table["shift_category"] = performance_table["shift_hours"].apply(categorize_shift)
            elif "shift" in tasks.columns:
                # Fallback: use existing shift column if available
                shift_numeric = pd.to_numeric(performance_table["shift"], errors='coerce')
                performance_table["shift_hours"] = shift_numeric
                performance_table["shift_display"] = shift_numeric.apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else ""
                )
                
                def categorize_shift(val):
                    if pd.isna(val):
                        return "Unknown"
                    elif val < 0 or val > 24:
                        return "Invalid"
                    elif val < 1:
                        return "Less than 1 hour"
                    elif val <= 8:
                        return "1-8 hours"
                    else:
                        return "More than 8 hours"
                
                performance_table["shift_category"] = shift_numeric.apply(categorize_shift)
            else:
                performance_table["shift_display"] = ""
                performance_table["shift_category"] = "Unknown"
            
            # Add place name (not coordinates)
            place_name_cols = [c for c in tasks.columns if 'place' in c.lower() and 'name' in c.lower()]
            place_cols = [c for c in tasks.columns if 'place' in c.lower() and 'code' not in c.lower() and 'name' not in c.lower() and 'status' not in c.lower()]
            
            if place_name_cols:
                performance_table["place_name"] = tasks[place_name_cols[0]].astype(str).fillna("")
            elif place_cols:
                # Check if it's coordinates
                place_data = tasks[place_cols[0]].astype(str)
                is_coords = place_data.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
                if not is_coords:
                    performance_table["place_name"] = place_data.fillna("")
                else:
                    performance_table["place_name"] = ""
            else:
                performance_table["place_name"] = ""
            
            # Merge with users to get agent name (keep all rows even if no match)
            if not users_lookup.empty and "user-id" in users_lookup.columns and "username" in users_lookup.columns:
                performance_table = performance_table.merge(
                    users_lookup[["user-id", "username"]], 
                    on="user-id", 
                    how="left"
                )
                performance_table["agent_name"] = performance_table["username"].fillna("Unknown")
            else:
                performance_table["agent_name"] = performance_table["user-id"].astype(str)
            
            # Rename date column
            performance_table = performance_table.rename(columns={"task-date": "date"})
            
            # Convert date to date object (not datetime) - handle errors
            if "date" in performance_table.columns:
                try:
                    performance_table["date"] = pd.to_datetime(performance_table["date"], errors='coerce').dt.date
                except:
                    # If conversion fails, keep as is
                    pass
            
            # Select columns: date, agent, shift (display), category, place
            perf_cols = ["date", "agent_name", "shift_display", "shift_category", "place_name"]
            existing_perf_cols = [c for c in perf_cols if c in performance_table.columns]
            performance_table = performance_table[existing_perf_cols]
            
            # Rename for better display
            if "shift_display" in performance_table.columns:
                performance_table = performance_table.rename(columns={"shift_display": "shift"})
            if "shift_category" in performance_table.columns:
                performance_table = performance_table.rename(columns={"shift_category": "category"})
            
            # Remove rows where date is null (invalid dates)
            if "date" in performance_table.columns:
                performance_table = performance_table[performance_table["date"].notna()]
        else:
            performance_table = pd.DataFrame(columns=["date", "agent_name", "shift", "category", "place_name"])
    else:
        performance_table = pd.DataFrame(columns=["date", "agent_name", "shift", "category", "place_name"])
    
    # Check-in/Check-out table: date, place-name, agent, check-in time, check-out time
    # Use the checkin_col and checkout_col already found above
    
    if not tasks.empty and "task-date" in tasks.columns and "user-id" in tasks.columns:
        checkinout_table = tasks.copy()
        
        # Merge with users to get agent name
        if not users_lookup.empty and "user-id" in users_lookup.columns and "username" in users_lookup.columns:
            checkinout_table = checkinout_table.merge(users_lookup[["user-id", "username"]], on="user-id", how="left")
        
        # Convert date
        if "task-date" in checkinout_table.columns:
            try:
                checkinout_table["date"] = pd.to_datetime(checkinout_table["task-date"], errors='coerce').dt.date
            except:
                checkinout_table["date"] = None
        
        # Extract check-in and check-out times using improved detection
        if checkin_col:
            checkinout_table["check_in_time"] = pd.to_datetime(checkinout_table[checkin_col], errors='coerce')
        else:
            checkinout_table["check_in_time"] = pd.NaT
        
        if checkout_col:
            checkinout_table["check_out_time"] = pd.to_datetime(checkinout_table[checkout_col], errors='coerce')
        else:
            checkinout_table["check_out_time"] = pd.NaT
        
        # Calculate shift duration in hours and NORMALIZE (remove negative/unrealistic values)
        checkinout_table["shift_duration"] = (checkinout_table["check_out_time"] - checkinout_table["check_in_time"]).dt.total_seconds() / 3600
        
        # NORMALIZE: Set negative or >24 hour shifts to NaN (data quality issue)
        checkinout_table["shift_duration"] = checkinout_table["shift_duration"].apply(
            lambda x: x if pd.notna(x) and 0 <= x <= 24 else None
        )
        
        # Extract place name (not coordinates)
        place_name_cols = [col for col in checkinout_table.columns if 'place' in col.lower() and 'name' in col.lower() and 'status' not in col.lower()]
        place_cols_task = [col for col in checkinout_table.columns if 'place' in col.lower() and 'code' not in col.lower() and 'name' not in col.lower() and 'status' not in col.lower()]
        
        if place_name_cols:
            checkinout_table["place_name"] = checkinout_table[place_name_cols[0]].astype(str).fillna("")
        elif place_cols_task:
            # Check if it's coordinates
            place_data = checkinout_table[place_cols_task[0]].astype(str)
            is_coords = place_data.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
            if not is_coords:
                checkinout_table["place_name"] = place_data.fillna("")
            else:
                checkinout_table["place_name"] = ""
        else:
            checkinout_table["place_name"] = ""
        
        # Add status tracking
        checkinout_table["status"] = "pending"
        checkinout_table.loc[checkinout_table["check_in_time"].notna(), "status"] = "in-progress"
        checkinout_table.loc[checkinout_table["check_out_time"].notna(), "status"] = "completed"
        
        # Select only columns that exist
        available_cols = ["date", "place_name", "username", "check_in_time", "check_out_time", "shift_duration", "status"]
        existing_cols = [col for col in available_cols if col in checkinout_table.columns]
        
        if existing_cols and "date" in existing_cols:
            checkinout_table = checkinout_table[existing_cols]
            # Don't drop all rows, keep rows with at least a date
            checkinout_table = checkinout_table[checkinout_table["date"].notna()]
        else:
            checkinout_table = pd.DataFrame(columns=["date", "place_name", "username", "check_in_time", "check_out_time", "shift_duration", "status"])
    else:
        checkinout_table = pd.DataFrame(columns=["date", "place_name", "username", "check_in_time", "check_out_time", "shift_duration", "status"])
    
    # -------- NEW: Enhanced Stock Validation --------
    # Count actual interactions from pack purchase (BBOS = pack purchase with +)
    inter_with_pack = inter[inter["has_bbos"] == True].copy() if "has_bbos" in inter.columns else pd.DataFrame()
    
    if not inter_with_pack.empty and "user-name" in inter_with_pack.columns:
        # Group by agent and day, count BBOS (pack purchases with +)
        pack_interactions = (
            inter_with_pack.groupby(["user-name", "interaction_day"], as_index=False)
            .agg(pack_purchase_interactions=("bbos_count", "sum"))
        )
        pack_interactions = pack_interactions.rename(columns={"user-name": "agent-name", "interaction_day": "stock_day"})
        
        # Merge with stock data
        stock_check_enhanced = stock_agg.merge(
            pack_interactions,
            on=["agent-name", "stock_day"],
            how="left"
        )
        stock_check_enhanced["pack_purchase_interactions"] = stock_check_enhanced["pack_purchase_interactions"].fillna(0)
        stock_check_enhanced["used_minus_pack_interactions"] = (
            stock_check_enhanced["used"] - stock_check_enhanced["pack_purchase_interactions"]
        )
        stock_check_enhanced["pack_stock_flag"] = np.where(
            stock_check_enhanced["used_minus_pack_interactions"] == 0, "Balanced", "Mismatch"
        )
    else:
        stock_check_enhanced = stock_agg.copy()
        stock_check_enhanced["pack_purchase_interactions"] = 0
        stock_check_enhanced["used_minus_pack_interactions"] = stock_check_enhanced["used"]
        stock_check_enhanced["pack_stock_flag"] = "Unknown"
    
    # Add place-status (active/inactive/closed) - assume from tasks or users if available
    if "place-status" in tasks.columns:
        place_status_map = tasks.groupby("place-name")["place-status"].first().to_dict() if "place-name" in tasks.columns else {}
    elif "status" in tasks.columns:
        # Try to infer from status column
        place_status_map = {}
    else:
        place_status_map = {}
    
    # -------- NEW: Active/Inactive Agents --------
    # IMPORTANT: Active agents should NEVER exceed total agents
    # Total agents = unique user-id in users_lookup
    total_agents_count = users_lookup["user-id"].nunique() if not users_lookup.empty else 0
    
    if not login_daily.empty and "login_date" in login_daily.columns and "user-id" in login_daily.columns:
        recent_date = login_daily["login_date"].max()
        if pd.notna(recent_date):
            # Convert to date for comparison (both sides need to be date objects)
            threshold_date = (pd.to_datetime(recent_date) - pd.Timedelta(days=30)).date()
            
            # Get unique user-ids who logged in within last 30 days
            active_user_ids = login_daily[
                login_daily["login_date"] >= threshold_date
            ]["user-id"].unique()
            
            # Only count those that actually exist in users_lookup
            active_agents = len([uid for uid in active_user_ids if uid in users_lookup["user-id"].values])
        else:
            # If no valid dates, count unique users who have logged in at least once
            active_user_ids = login_daily["user-id"].unique()
            active_agents = len([uid for uid in active_user_ids if uid in users_lookup["user-id"].values])
    else:
        # No login data available - all agents are inactive
        active_agents = 0
    
    # Ensure active never exceeds total
    active_agents = min(active_agents, total_agents_count)
    inactive_agents = total_agents_count - active_agents
    
    # -------- NEW: Places vs Tasks --------
    # Extract place name (not coordinates)
    place_name_cols = [c for c in tasks.columns if 'place' in c.lower() and 'name' in c.lower()]
    place_cols = [c for c in tasks.columns if 'place' in c.lower() and 'code' not in c.lower() and 'name' not in c.lower()]
    
    if place_name_cols:
        places_tasks = tasks.groupby(place_name_cols[0]).size().reset_index(name="tasks_count").sort_values("tasks_count", ascending=False)
        places_tasks = places_tasks.rename(columns={place_name_cols[0]: "place_name"})
    elif "place-name" in tasks.columns:
        # Check if it's coordinates
        place_data = tasks["place-name"].astype(str)
        is_coords = place_data.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
        if not is_coords:
            places_tasks = tasks.groupby("place-name").size().reset_index(name="tasks_count").sort_values("tasks_count", ascending=False)
            places_tasks = places_tasks.rename(columns={"place-name": "place_name"})
        else:
            places_tasks = pd.DataFrame(columns=["place_name", "tasks_count"])
    elif place_cols:
        # Check if it's coordinates
        place_data = tasks[place_cols[0]].astype(str)
        is_coords = place_data.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
        if not is_coords:
            places_tasks = tasks.groupby(place_cols[0]).size().reset_index(name="tasks_count").sort_values("tasks_count", ascending=False)
            places_tasks = places_tasks.rename(columns={place_cols[0]: "place_name"})
        else:
            places_tasks = pd.DataFrame(columns=["place_name", "tasks_count"])
    else:
        places_tasks = pd.DataFrame(columns=["place_name", "tasks_count"])
    
    # -------- NEW: DCC/ECC/QR/BBOS Totals --------
    dcc_total = tasks["DCC_fixed"].sum() if "DCC_fixed" in tasks.columns else 0
    ecc_total = tasks["ECC_fixed"].sum() if "ECC_fixed" in tasks.columns else 0
    qr_total = tasks["QR_fixed"].sum() if "QR_fixed" in tasks.columns else 0
    bbos_total = tasks["BBOS_fixed"].sum() if "BBOS_fixed" in tasks.columns else 0
    
    # -------- NEW: Interactions by Place --------
    # Use place_display if available, otherwise place_name
    place_col_for_group = "place_display" if "place_display" in inter.columns else "place_name"
    if place_col_for_group in inter.columns:
        inter_by_place = inter.groupby(place_col_for_group).agg({
            "is_DCC": "sum",
            "is_ECC": "sum",
            "Gender": lambda x: x.value_counts().to_dict() if x.notna().any() else {},
            "Age Range": lambda x: x.value_counts().to_dict() if x.notna().any() else {}
        }).reset_index()
        inter_by_place = inter_by_place.rename(columns={place_col_for_group: "place_name", "is_DCC": "dcc_count", "is_ECC": "ecc_count"})
        inter_by_place = inter_by_place[inter_by_place["place_name"] != ""]
    else:
        inter_by_place = pd.DataFrame(columns=["place_name", "dcc_count", "ecc_count"])

    # -------- Global date bounds --------
    date_candidates = []
    for df, col in [
        (dcc_check_view,  "day_key"),
        (ecc_check_view,  "day_key"),
        (qr_check_view,   "day_key"),
        (bbos_check_view, "day_key"),
        (stock_check_view, "stock_day"),
        (attendance_view,  "login_date"),
        (tasks_per_day,    "date"),
        (inter_per_day,    "date"),
    ]:
        if not df.empty:
            date_candidates.append(pd.to_datetime(df[col]))

    if date_candidates:
        all_dates = pd.concat(date_candidates)
        min_date = all_dates.min().date()
        max_date = all_dates.max().date()
    else:
        min_date = None
        max_date = None

    return {
        "users": users,
        "users_lookup": users_lookup,
        "login_daily": login_daily,
        "tasks": tasks,
        "tasks_summary": tasks_summary,
        "tasks_detailed": tasks_detailed,
        "tasks_dcc": tasks_dcc,
        "tasks_ecc": tasks_ecc,
        "tasks_qr": tasks_qr,
        "tasks_bbos": tasks_bbos,
        "tasks_per_day": tasks_per_day,
        "tasks_status_counts": tasks_status_counts,
        "tasks_shift_counts": tasks_shift_counts,
        "inter": inter,
        "inter_daily": inter_daily,
        "inter_per_day": inter_per_day,
        "inter_type_counts": inter_type_counts,
        "inter_agent_counts": inter_agent_counts,
        "inter_gender_counts": inter_gender_counts,
        "inter_age_counts": inter_age_counts,
        "inter_main_brand_counts": inter_main_brand_counts,
        "inter_pack_purchase_counts": inter_pack_purchase_counts,
        "stocklog": stocklog,
        "stock_agg": stock_agg,
        "dcc_check_view": dcc_check_view,
        "ecc_check_view": ecc_check_view,
        "qr_check_view": qr_check_view,
        "bbos_check_view": bbos_check_view,
        "stock_check_view": stock_check_view,
        "stock_check_enhanced": stock_check_enhanced,
        "attendance_view": attendance_view,
        "min_date": min_date,
        "max_date": max_date,
        # New data
        "performance_table": performance_table,
        "checkinout_table": checkinout_table,
        "active_agents": active_agents,
        "inactive_agents": inactive_agents,
        "dcc_total": dcc_total,
        "ecc_total": ecc_total,
        "qr_total": qr_total,
        "bbos_total": bbos_total,
        "places_tasks": places_tasks,
        "inter_by_place": inter_by_place,
        "main_brand": main_brand,
        # Debug info
        "checkin_col": checkin_col,
        "checkout_col": checkout_col,
    }


# ----------------------------------------
# 3. Helper filters / export
# ----------------------------------------
def filter_by_zone_area(df, zones, areas, zone_col="zone", area_col="area"):
    if df is None or df.empty:
        return df
    filtered = df.copy()
    if zones and zone_col in filtered.columns:
        filtered = filtered[filtered[zone_col].isin(zones)]
    if areas and area_col in filtered.columns:
        filtered = filtered[filtered[area_col].isin(areas)]
    return filtered


def filter_by_date(df, date_col, date_range):
    if df is None or df.empty or date_range is None:
        return df
    start, end = date_range
    if date_col not in df.columns:
        return df
    mask = (pd.to_datetime(df[date_col]).dt.date >= start) & (
        pd.to_datetime(df[date_col]).dt.date <= end
    )
    return df[mask]


def filter_by_agents(df, agents, username_col="username", alt_agent_col=None):
    if df is None or df.empty or not agents:
        return df
    filtered = df.copy()
    if username_col in filtered.columns:
        filtered = filtered[filtered[username_col].isin(agents)]
    elif alt_agent_col and alt_agent_col in filtered.columns:
        filtered = filtered[filtered[alt_agent_col].isin(agents)]
    return filtered

def filter_by_place(df, places, place_col="place_name"):
    if df is None or df.empty or not places:
        return df
    filtered = df.copy()
    place_cols = [col for col in filtered.columns if 'place' in col.lower()]
    if place_cols:
        filtered = filtered[filtered[place_cols[0]].isin(places)]
    return filtered

def filter_by_interaction_id(df, interaction_id_search):
    if df is None or df.empty or not interaction_id_search:
        return df
    filtered = df.copy()
    id_cols = [col for col in filtered.columns if 'id' in col.lower() and 'interaction' in col.lower()]
    if id_cols:
        filtered = filtered[filtered[id_cols[0]].astype(str).str.contains(str(interaction_id_search), case=False, na=False)]
    return filtered


def to_excel_download(df_dict: dict, filename: str = "validation_outputs.xlsx") -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output


# ----------------------------------------
# 4. Google Sheets URL input
# ----------------------------------------
st.subheader("📊 Data Source")
google_sheets_url = st.text_input(
    "🔗 Enter Google Sheets URL",
    placeholder="https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit#gid=0",
    help="Paste the shareable link to your Google Sheet. Make sure the sheet is set to 'Anyone with the link can view'."
)

if not google_sheets_url:
    st.info("👆 Enter your Google Sheets URL above to load the dashboard data.")
    st.info("💡 **Tip:** Make sure your Google Sheet is shared with 'Anyone with the link can view' permission.")
    st.stop()

# Validate URL format
if "docs.google.com/spreadsheets" not in google_sheets_url:
    st.error("❌ Invalid Google Sheets URL. Please enter a valid Google Sheets link.")
    st.stop()

# Load and process data
try:
    with st.spinner("🔄 Loading data from Google Sheets..."):
        data = load_and_process(google_sheets_url)
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("💡 Make sure your Google Sheet is shared with 'Anyone with the link can view' permission.")
    st.stop()

users_lookup       = data["users_lookup"]
dcc_check_view     = data["dcc_check_view"]
ecc_check_view     = data["ecc_check_view"]
qr_check_view      = data["qr_check_view"]
bbos_check_view    = data["bbos_check_view"]
stock_check_view   = data["stock_check_view"]
stock_check_enhanced = data.get("stock_check_enhanced", pd.DataFrame())
attendance_view    = data["attendance_view"]
inter_type_counts  = data["inter_type_counts"]
inter_agent_counts = data["inter_agent_counts"]
inter_gender_counts = data["inter_gender_counts"]
inter_age_counts    = data["inter_age_counts"]
inter_main_brand_counts = data["inter_main_brand_counts"]
inter_pack_purchase_counts = data["inter_pack_purchase_counts"]
tasks_summary      = data["tasks_summary"]
tasks_detailed     = data.get("tasks_detailed", pd.DataFrame())
tasks_per_day      = data["tasks_per_day"]
tasks_status_counts = data["tasks_status_counts"]
tasks_shift_counts  = data["tasks_shift_counts"]
inter_per_day      = data["inter_per_day"]
min_date           = data["min_date"]
max_date           = data["max_date"]
inter_daily        = data["inter_daily"]
tasks              = data["tasks"]
inter              = data["inter"]
# New data
performance_table  = data.get("performance_table", pd.DataFrame())
checkinout_table   = data.get("checkinout_table", pd.DataFrame())
active_agents      = data.get("active_agents", 0)
inactive_agents    = data.get("inactive_agents", 0)
dcc_total          = data.get("dcc_total", 0)
ecc_total          = data.get("ecc_total", 0)
qr_total           = data.get("qr_total", 0)
bbos_total         = data.get("bbos_total", 0)
places_tasks       = data.get("places_tasks", pd.DataFrame())
inter_by_place     = data.get("inter_by_place", pd.DataFrame())
main_brand         = data.get("main_brand", pd.DataFrame())
# Debug info
checkin_col        = data.get("checkin_col", None)
checkout_col       = data.get("checkout_col", None)

# ----------------------------------------
# 5. Sidebar filters
# ----------------------------------------
# Dark/Light mode toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌓 Theme")
dark_mode_toggle = st.sidebar.toggle(
    "Dark Mode",
    value=st.session_state.dark_mode,
    help="Toggle between dark mode (white text) and light mode (black text)"
)
if dark_mode_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode_toggle
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🔎 Global Filters")

# Date filter
if min_date and max_date:
    date_filter = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_filter, tuple):
        date_range = date_filter
    else:
        date_range = (date_filter, date_filter)
else:
    date_range = None

# Agent filter
agents = sorted([a for a in users_lookup["username"].dropna().unique()])
agent_filter = st.sidebar.multiselect("👤 Agent Name", agents, default=agents)

# Zone filter
zones = sorted([z for z in users_lookup["zone"].dropna().unique()])
zone_filter = st.sidebar.multiselect("🌍 Zone", zones, default=zones)

# Area filter
areas = sorted([a for a in users_lookup["area"].dropna().unique()])
area_filter = st.sidebar.multiselect("📍 Area", areas, default=areas)

# Place name/code filter - show name and code together
place_names = []
if "place_display" in inter.columns:
    place_names = sorted([p for p in inter["place_display"].dropna().unique() if str(p) != "nan" and str(p) != ""])
elif "place_name" in inter.columns:
    place_names = sorted([p for p in inter["place_name"].dropna().unique() if str(p) != "nan" and str(p) != ""])
elif "place-name" in tasks.columns:
    # Check if it's coordinates
    place_data = tasks["place-name"].astype(str)
    is_coords = place_data.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
    if not is_coords:
        place_names = sorted([p for p in tasks["place-name"].dropna().unique() if str(p) != "nan" and str(p) != ""])
elif len([c for c in tasks.columns if 'place' in c.lower()]) > 0:
    place_col = [c for c in tasks.columns if 'place' in c.lower()][0]
    place_data = tasks[place_col].astype(str)
    is_coords = place_data.str.contains(r'^\d+\.\d+', regex=True, na=False).any()
    if not is_coords:
        place_names = sorted([p for p in tasks[place_col].dropna().unique() if str(p) != "nan" and str(p) != ""])

place_filter = st.sidebar.multiselect("🏢 Place Name/Code", place_names, default=place_names if place_names else [])

# Interaction ID search
interaction_id_search = st.sidebar.text_input("🔍 Search by Interaction ID", "")

# Shift duration filter
shift_filter_options = ["All Shifts", "Less than 1 hour", "1-8 hours", "More than 8 hours", "Invalid/Unknown"]
shift_filter = st.sidebar.selectbox("⏱️ Shift Duration", shift_filter_options, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Filters apply to all validation tables and most charts.")

# ----------------------------------------
# 6. Apply filters
# ----------------------------------------
dcc_filtered = filter_by_zone_area(dcc_check_view, zone_filter, area_filter)
dcc_filtered = filter_by_date(dcc_filtered, "day_key", date_range)
dcc_filtered = filter_by_agents(dcc_filtered, agent_filter, username_col="username")

ecc_filtered = filter_by_zone_area(ecc_check_view, zone_filter, area_filter)
ecc_filtered = filter_by_date(ecc_filtered, "day_key", date_range)
ecc_filtered = filter_by_agents(ecc_filtered, agent_filter, username_col="username")

qr_filtered = filter_by_zone_area(qr_check_view, zone_filter, area_filter)
qr_filtered = filter_by_date(qr_filtered, "day_key", date_range)
qr_filtered = filter_by_agents(qr_filtered, agent_filter, username_col="username")

bbos_filtered = filter_by_zone_area(bbos_check_view, zone_filter, area_filter)
bbos_filtered = filter_by_date(bbos_filtered, "day_key", date_range)
bbos_filtered = filter_by_agents(bbos_filtered, agent_filter, username_col="username")

stock_filtered = filter_by_zone_area(stock_check_view, zone_filter, area_filter)
stock_filtered = filter_by_date(stock_filtered, "stock_day", date_range)
stock_filtered = filter_by_agents(stock_filtered, agent_filter, alt_agent_col="agent-name")

attendance_filtered = filter_by_zone_area(attendance_view, zone_filter, area_filter)
attendance_filtered = filter_by_date(attendance_filtered, "login_date", date_range)
attendance_filtered = filter_by_agents(attendance_filtered, agent_filter, username_col="username")

tasks_per_day_f = filter_by_date(tasks_per_day, "date", date_range)
inter_per_day_f = filter_by_date(inter_per_day, "date", date_range)

# ----------------------------------------
# 7. KPIs
# ----------------------------------------
st.subheader("📌 Overview")

# First row: Agent metrics
col1, col2, col3, col4 = st.columns(4)

total_agents = users_lookup["user-id"].nunique()
total_interactions = inter_daily["interactions_count"].sum() if not inter_daily.empty else 0
total_tasks = tasks["task-date"].shape[0] if not tasks.empty else 0

col1.metric("Total Agents", total_agents)
col2.metric("Active Agents", active_agents)
col3.metric("Inactive Agents", inactive_agents)
col4.metric("Total Interactions", int(total_interactions))

# Second row: Task metrics
col5, col6, col7, col8 = st.columns(4)
col5.metric("DCC Tasks", int(dcc_total))
col6.metric("ECC Tasks", int(ecc_total))
col7.metric("QR Tasks", int(qr_total))
col8.metric("BBOS Tasks", int(bbos_total))

# Top 20 Agents Chart
st.markdown("### Top 20 Agents by Interactions")
if not inter_agent_counts.empty:
    top_20_agents = inter_agent_counts.head(20)
    fig_top_agents = px.bar(
        top_20_agents,
        x="user-name",
        y="interactions_count",
        title="Top 20 Agents by Total Interactions",
        labels={"user-name": "Agent Name", "interactions_count": "Total Interactions"},
        color="interactions_count",
        color_continuous_scale=[[0, BRAND_COLORS["black"]], [0.5, BRAND_COLORS["dark_gray"]], [1, BRAND_COLORS["red"]]],
    )
    fig_top_agents.update_layout(
        xaxis_tickangle=45, 
        font_family="Century Gothic",
        showlegend=False
    )
    st.plotly_chart(fig_top_agents, use_container_width=True)
else:
    st.info("No agent interaction data available.")

# Places vs Tasks chart
if not places_tasks.empty:
    st.markdown("### Places vs Tasks")
    fig_places = px.bar(
        places_tasks.head(20),
        x="place_name",
        y="tasks_count",
        title="Top 20 Places by Task Count",
        labels={"place_name": "Place Name", "tasks_count": "Tasks Count"},
        color_discrete_sequence=[BRAND_COLORS["red"]],
    )
    fig_places.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
    st.plotly_chart(fig_places, use_container_width=True)

st.markdown("---")

# ----------------------------------------
# 8. Tabs
# ----------------------------------------
tab_overview, tab_performance, tab_valid, tab_stock, tab_att, tab_inter, tab_tasks = st.tabs(
    [
        "📈 Overview Charts",
        "⚡ Performance",
        "🎯 Validation (DCC / ECC / QR / BBOS)",
        "📦 Stock Validation",
        "🕒 Attendance vs Interactions",
        "👥 Interactions Analytics",
        "📋 Tasks Analytics",
    ]
)

# ---------- Overview ----------
with tab_overview:
    st.subheader("High-level Data Quality & Performance")

    colA, colB = st.columns(2)

    if not dcc_filtered.empty:
        dcc_top = (
            dcc_filtered.groupby("username", as_index=False)["dcc_difference"]
            .sum()
            .sort_values("dcc_difference", ascending=False)
        )
        fig_dcc = px.bar(
            dcc_top,
            x="username",
            y="dcc_difference",
            title="DCC Difference per Agent (Tasks - Interactions)",
            labels={"dcc_difference": "DCC Difference", "username": "Agent"},
            color_discrete_sequence=PALETTE_MAIN,
        )
        fig_dcc.update_layout(xaxis_tickangle=45)
        colA.plotly_chart(fig_dcc, use_container_width=True)
    else:
        colA.info("No DCC data available for current filters.")

    if not stock_filtered.empty:
        stock_top = (
            stock_filtered.groupby("agent-name", as_index=False)["used_minus_interactions"]
            .sum()
            .sort_values("used_minus_interactions", ascending=False)
        )
        fig_stock = px.bar(
            stock_top,
            x="agent-name",
            y="used_minus_interactions",
            title="Stock Used - Interactions per Agent",
            labels={"used_minus_interactions": "Used - Interactions"},
            color_discrete_sequence=PALETTE_ALT,
        )
        fig_stock.update_layout(xaxis_tickangle=45)
        colB.plotly_chart(fig_stock, use_container_width=True)
    else:
        colB.info("No stock data available for current filters.")

    st.markdown("### Activity Over Time")
    colC, colD = st.columns(2)

    if not inter_per_day_f.empty:
        fig_inter_time = px.line(
            inter_per_day_f,
            x="date",
            y="interactions_count",
            title="Interactions Over Time",
            labels={"interactions_count": "Interactions", "date": "Date"},
            color_discrete_sequence=PALETTE_MAIN,
        )
        colC.plotly_chart(fig_inter_time, use_container_width=True)
    else:
        colC.info("No interaction time-series for current filters.")

    if not tasks_per_day_f.empty:
        fig_tasks_time = px.line(
            tasks_per_day_f,
            x="date",
            y="tasks_count",
            title="Tasks Over Time",
            labels={"tasks_count": "Tasks", "date": "Date"},
            color_discrete_sequence=PALETTE_ALT,
        )
        colD.plotly_chart(fig_tasks_time, use_container_width=True)
    else:
        colD.info("No tasks time-series for current filters.")

    st.markdown("### Top DCC Mismatches (Rows)")
    st.dataframe(dcc_filtered.head(20), use_container_width=True)

# ---------- Performance ----------
with tab_performance:
    st.subheader("⚡ Performance Metrics")
    
    # Debug info: Show what columns were detected
    with st.expander("🔍 Debug: Column Detection Info", expanded=True):
        col_debug1, col_debug2 = st.columns(2)
        
        with col_debug1:
            st.caption("**Check-in column detected:**")
            if checkin_col:
                st.success(f"✅ Found: `{checkin_col}`")
                # Show sample values
                if not tasks.empty and checkin_col in tasks.columns:
                    sample_vals = tasks[checkin_col].dropna().head(3).tolist()
                    if sample_vals:
                        st.caption("Sample values:")
                        for i, val in enumerate(sample_vals, 1):
                            st.code(f"{i}. {val}")
            else:
                st.error("❌ Not found")
                st.caption("Please check the column names below and tell me the exact name of your check-in column.")
        
        with col_debug2:
            st.caption("**Check-out column detected:**")
            if checkout_col:
                st.success(f"✅ Found: `{checkout_col}`")
                # Show sample values
                if not tasks.empty and checkout_col in tasks.columns:
                    sample_vals = tasks[checkout_col].dropna().head(3).tolist()
                    if sample_vals:
                        st.caption("Sample values:")
                        for i, val in enumerate(sample_vals, 1):
                            st.code(f"{i}. {val}")
            else:
                st.error("❌ Not found")
                st.caption("Please check the column names below and tell me the exact name of your check-out column.")
        
        st.markdown("---")
        
        st.caption("**All columns with 'check' or 'time' in name:**")
        time_check_cols = [c for c in tasks.columns if 'check' in c.lower() or 'time' in c.lower()]
        if time_check_cols:
            st.success(f"Found {len(time_check_cols)} columns:")
            for col in time_check_cols:
                st.code(f"• {col}")
        else:
            st.warning("No columns found with 'check' or 'time'")
        
        st.markdown("---")
        st.caption("**All Tasks columns:**")
        st.text(", ".join(tasks.columns.tolist()))
    
    # Performance table: date, agent name, shift, place name
    st.markdown("#### Performance Table: Date, Agent, Shift, Place")
    if not performance_table.empty:
        # Apply filters
        perf_filtered = performance_table.copy()
        
        # Show original count
        original_count = len(perf_filtered)
        
        if date_range and "date" in perf_filtered.columns:
            perf_filtered = filter_by_date(perf_filtered, "date", date_range)
        
        if agent_filter and "agent_name" in perf_filtered.columns:
            perf_filtered = perf_filtered[perf_filtered["agent_name"].isin(agent_filter)]
        
        if place_filter and "place_name" in perf_filtered.columns:
            # Filter by place name - try exact match first, then partial match
            place_mask = perf_filtered["place_name"].isin(place_filter)
            if not place_mask.any() and place_filter:
                # Try partial match if exact match fails
                for place in place_filter:
                    place_mask = place_mask | perf_filtered["place_name"].str.contains(str(place), case=False, na=False)
            perf_filtered = perf_filtered[place_mask]
        
        # Apply shift filter using category column
        if shift_filter != "All Shifts" and "category" in perf_filtered.columns:
            if shift_filter == "Less than 1 hour":
                perf_filtered = perf_filtered[perf_filtered["category"] == "Less than 1 hour"]
            elif shift_filter == "1-8 hours":
                perf_filtered = perf_filtered[perf_filtered["category"] == "1-8 hours"]
            elif shift_filter == "More than 8 hours":
                perf_filtered = perf_filtered[perf_filtered["category"] == "More than 8 hours"]
            elif shift_filter == "Invalid/Unknown":
                perf_filtered = perf_filtered[perf_filtered["category"].isin(["Unknown", "Invalid"])]
        
        if not perf_filtered.empty:
            st.dataframe(perf_filtered, use_container_width=True, height=400)
            st.caption(f"Showing {len(perf_filtered)} of {original_count} records")
            
            # Chart: Performance by shift category
            if "category" in perf_filtered.columns:
                shift_cat_dist = perf_filtered.groupby("category").size().reset_index(name="count")
                fig_shift_cat = px.bar(
                    shift_cat_dist,
                    x="category",
                    y="count",
                    title="Performance by Shift Category",
                    labels={"category": "Shift Category", "count": "Count"},
                    color="category",
                    color_discrete_map={
                        "Less than 1 hour": PALETTE_MAIN[0],
                        "1-8 hours": BRAND_COLORS["black"],
                        "More than 8 hours": BRAND_COLORS["red"],
                        "Unknown": BRAND_COLORS["dark_gray"],
                        "Invalid": BRAND_COLORS["dark_gray"]
                    }
                )
                fig_shift_cat.update_layout(font_family="Century Gothic", showlegend=False)
                st.plotly_chart(fig_shift_cat, use_container_width=True)
            
            st.markdown("---")
            
            # Show detailed breakdown by category
            if "category" in perf_filtered.columns:
                st.markdown("### 📊 Breakdown by Shift Category")
                
                col_cat1, col_cat2, col_cat3 = st.columns(3)
                
                # Less than 1 hour
                with col_cat1:
                    st.markdown("#### ⚡ Less than 1 hour")
                    less_1h = perf_filtered[perf_filtered["category"] == "Less than 1 hour"]
                    if not less_1h.empty:
                        st.metric("Count", len(less_1h))
                        st.dataframe(less_1h[["date", "agent_name", "shift", "place_name"]].head(10), use_container_width=True)
                    else:
                        st.info("No shifts in this category")
                
                # 1-8 hours (Normal)
                with col_cat2:
                    st.markdown("#### ✅ 1-8 hours")
                    normal_shifts = perf_filtered[perf_filtered["category"] == "1-8 hours"]
                    if not normal_shifts.empty:
                        st.metric("Count", len(normal_shifts))
                        st.dataframe(normal_shifts[["date", "agent_name", "shift", "place_name"]].head(10), use_container_width=True)
                    else:
                        st.info("No shifts in this category")
                
                # More than 8 hours
                with col_cat3:
                    st.markdown("#### ⚠️ More than 8 hours")
                    more_8h = perf_filtered[perf_filtered["category"] == "More than 8 hours"]
                    if not more_8h.empty:
                        st.metric("Count", len(more_8h))
                        st.dataframe(more_8h[["date", "agent_name", "shift", "place_name"]].head(10), use_container_width=True)
                        
                        # Show full list if there are problem shifts
                        if len(more_8h) > 10:
                            with st.expander(f"Show all {len(more_8h)} shifts >8 hours"):
                                st.dataframe(more_8h, use_container_width=True)
                    else:
                        st.info("No shifts in this category")
        else:
            st.warning(f"⚠️ No data matches the current filters. Original table had {original_count} records.")
            st.info("💡 Try adjusting your filters (date range, agent, place, or shift) to see data.")
            # Show a sample of unfiltered data for debugging
            if original_count > 0:
                st.markdown("**Sample of available data (unfiltered):**")
                st.dataframe(performance_table.head(10), use_container_width=True)
    else:
        st.info("No performance data available. Ensure tasks have 'task-date' and 'user-id' columns.")
        # Debug info
        if not tasks.empty:
            st.caption(f"📊 Tasks table has {len(tasks)} rows and columns: {', '.join(tasks.columns[:10].tolist())}")
    
    st.markdown("---")
    
    # Check-in/Check-out table
    st.markdown("#### Check-in/Check-out Times: Date, Place, Agent, Check-in, Check-out")
    if not checkinout_table.empty:
        # Apply filters
        cio_filtered = checkinout_table.copy()
        if date_range and "date" in cio_filtered.columns:
            cio_filtered = filter_by_date(cio_filtered, "date", date_range)
        if agent_filter and "username" in cio_filtered.columns:
            cio_filtered = cio_filtered[cio_filtered["username"].isin(agent_filter)]
        if place_filter and "place_name" in cio_filtered.columns:
            cio_filtered = cio_filtered[cio_filtered["place_name"].isin(place_filter)]
        
        # Apply shift duration filter if available
        if shift_filter != "All Shifts" and "shift_duration" in cio_filtered.columns:
            if shift_filter == "Normal (≤12 hours)":
                cio_filtered = cio_filtered[cio_filtered["shift_duration"] <= 12]
            elif shift_filter == "Problem (>12 hours)":
                cio_filtered = cio_filtered[cio_filtered["shift_duration"] > 12]
        
        st.dataframe(cio_filtered, use_container_width=True, height=400)
        st.caption(f"Showing {len(cio_filtered)} records")
        
        # Status distribution
        if "status" in cio_filtered.columns:
            status_dist = cio_filtered.groupby("status").size().reset_index(name="count")
            fig_status_dist = px.pie(
                status_dist,
                names="status",
                values="count",
                title="Task Status Distribution",
                color_discrete_sequence=[BRAND_COLORS["black"], BRAND_COLORS["red"], BRAND_COLORS["dark_gray"]],
            )
            fig_status_dist.update_layout(font_family="Century Gothic")
            st.plotly_chart(fig_status_dist, use_container_width=True)
        
        # Show problem durations
        if "shift_duration" in cio_filtered.columns:
            problem_durations = cio_filtered[cio_filtered["shift_duration"] > 12]
            if not problem_durations.empty:
                st.markdown("#### ⚠️ Shifts >12 hours (likely haven't checked out)")
                st.dataframe(problem_durations, use_container_width=True)
    else:
        st.info("No check-in/check-out data available. Ensure tasks have check-in and check-out time columns.")
        # Debug check-in columns
        if not tasks.empty:
            checkin_related = [col for col in tasks.columns if 'check' in col.lower()]
            if checkin_related:
                st.caption(f"Found check-in related columns: {', '.join(checkin_related)}")
            else:
                st.caption("No check-in/check-out columns found in tasks table")

# ---------- Validation (DCC / ECC / QR / BBOS) ----------
with tab_valid:
    st.subheader("🎯 Validation by Metric")

    # DCC
    st.markdown("#### DCC (Direct Consumer Contact)")
    st.dataframe(dcc_filtered, use_container_width=True, height=250)
    if not dcc_filtered.empty:
        by_zone = (
            dcc_filtered.groupby(["zone", "dcc_match_flag"], as_index=False)["user-id"]
            .count()
            .rename(columns={"user-id": "rows"})
        )
        fig_zone = px.bar(
            by_zone,
            x="zone",
            y="rows",
            color="dcc_match_flag",
            barmode="stack",
            title="DCC Validation Status by Zone",
            color_discrete_sequence=PALETTE_MAIN,
        )
        st.plotly_chart(fig_zone, use_container_width=True)

        by_agent_day = dcc_filtered.pivot_table(
            index="username",
            columns="day_key",
            values="dcc_difference",
            aggfunc="sum",
            fill_value=0,
        )
        if not by_agent_day.empty:
            fig_heat = px.imshow(
                by_agent_day,
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower",
                title="DCC Difference Heatmap (Agent x Day)",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    dcc_xlsx = to_excel_download({"DCC_validation": dcc_filtered})
    st.download_button(
        "⬇️ Download DCC Validation (filtered)",
        data=dcc_xlsx,
        file_name="dcc_validation_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    # ECC
    st.markdown("#### ECC Validation")
    st.dataframe(ecc_filtered, use_container_width=True, height=250)
    if not ecc_filtered.empty:
        by_zone_e = (
            ecc_filtered.groupby(["zone", "ecc_match_flag"], as_index=False)["user-id"]
            .count()
            .rename(columns={"user-id": "rows"})
        )
        fig_zone_e = px.bar(
            by_zone_e,
            x="zone",
            y="rows",
            color="ecc_match_flag",
            barmode="stack",
            title="ECC Validation Status by Zone",
            color_discrete_sequence=PALETTE_ALT,
        )
        st.plotly_chart(fig_zone_e, use_container_width=True)

    ecc_xlsx = to_excel_download({"ECC_validation": ecc_filtered})
    st.download_button(
        "⬇️ Download ECC Validation (filtered)",
        data=ecc_xlsx,
        file_name="ecc_validation_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    # QR
    st.markdown("#### QR Validation")
    st.dataframe(qr_filtered, use_container_width=True, height=250)
    if not qr_filtered.empty:
        by_zone_q = (
            qr_filtered.groupby(["zone", "qr_match_flag"], as_index=False)["user-id"]
            .count()
            .rename(columns={"user-id": "rows"})
        )
        fig_zone_q = px.bar(
            by_zone_q,
            x="zone",
            y="rows",
            color="qr_match_flag",
            barmode="stack",
            title="QR Validation Status by Zone",
            color_discrete_sequence=PALETTE_MAIN,
        )
        st.plotly_chart(fig_zone_q, use_container_width=True)

    qr_xlsx = to_excel_download({"QR_validation": qr_filtered})
    st.download_button(
        "⬇️ Download QR Validation (filtered)",
        data=qr_xlsx,
        file_name="qr_validation_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    # BBOS
    st.markdown("#### BBOS Validation")
    st.dataframe(bbos_filtered, use_container_width=True, height=250)
    if not bbos_filtered.empty:
        by_zone_b = (
            bbos_filtered.groupby(["zone", "bbos_match_flag"], as_index=False)["user-id"]
            .count()
            .rename(columns={"user-id": "rows"})
        )
        fig_zone_b = px.bar(
            by_zone_b,
            x="zone",
            y="rows",
            color="bbos_match_flag",
            barmode="stack",
            title="BBOS Validation Status by Zone",
            color_discrete_sequence=PALETTE_ALT,
        )
        st.plotly_chart(fig_zone_b, use_container_width=True)

    bbos_xlsx = to_excel_download({"BBOS_validation": bbos_filtered})
    st.download_button(
        "⬇️ Download BBOS Validation (filtered)",
        data=bbos_xlsx,
        file_name="bbos_validation_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---------- Stock ----------
with tab_stock:
    st.subheader("📦 Stock Validation (Used vs Pack Purchase Interactions)")

    # Enhanced stock validation with pack purchase interactions
    stock_export_data = stock_filtered  # Default to standard validation
    
    if not stock_check_enhanced.empty:
        stock_enhanced_filtered = stock_check_enhanced.copy()
        stock_enhanced_filtered = filter_by_zone_area(stock_enhanced_filtered, zone_filter, area_filter)
        stock_enhanced_filtered = filter_by_date(stock_enhanced_filtered, "stock_day", date_range)
        stock_enhanced_filtered = filter_by_agents(stock_enhanced_filtered, agent_filter, alt_agent_col="agent-name")
        stock_export_data = stock_enhanced_filtered  # Update export data
        
        # Add place-status if available
        if "place-status" in stock_enhanced_filtered.columns:
            st.markdown("#### Stock Validation with Place Status")
        else:
            st.markdown("#### Stock Validation: Used vs Pack Purchase Interactions")
        
        # Show relevant columns
        display_cols = ["agent-name", "user-id", "zone", "area", "stock_day", "used", "pack_purchase_interactions", "used_minus_pack_interactions", "pack_stock_flag"]
        if "place-status" in stock_enhanced_filtered.columns:
            display_cols.insert(4, "place-status")
        
        st.dataframe(stock_enhanced_filtered[display_cols], use_container_width=True, height=400)
        
        # Mismatch analysis
        mismatches = stock_enhanced_filtered[stock_enhanced_filtered["pack_stock_flag"] == "Mismatch"]
        if not mismatches.empty:
            st.markdown("#### ⚠️ Stock Mismatches (Used ≠ Pack Purchase Interactions)")
            st.dataframe(mismatches[display_cols], use_container_width=True)
            
            by_agent_mismatch = (
                mismatches.groupby("agent-name", as_index=False)["used_minus_pack_interactions"]
            .sum()
                .sort_values("used_minus_pack_interactions", ascending=False)
        )
            fig_mismatch = px.bar(
                by_agent_mismatch,
            x="agent-name",
                y="used_minus_pack_interactions",
                title="Stock Mismatches by Agent",
                color_discrete_sequence=[BRAND_COLORS["red"]],
            )
            fig_mismatch.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
            st.plotly_chart(fig_mismatch, use_container_width=True)
        else:
            st.success("✅ No stock mismatches found!")
        
        # Place status distribution if available
        if "place-status" in stock_enhanced_filtered.columns:
            place_status_dist = stock_enhanced_filtered.groupby("place-status").size().reset_index(name="count")
            fig_place_status = px.bar(
                place_status_dist,
                x="place-status",
                y="count",
                title="Places by Status (Active/Inactive/Closed)",
                color_discrete_sequence=[BRAND_COLORS["black"], BRAND_COLORS["red"], BRAND_COLORS["dark_gray"]],
            )
            fig_place_status.update_layout(font_family="Century Gothic")
            st.plotly_chart(fig_place_status, use_container_width=True)
    else:
        st.dataframe(stock_filtered, use_container_width=True, height=400)
        st.info("Enhanced stock validation with pack purchase interactions not available. Showing standard validation.")

    stock_xlsx = to_excel_download({"Stock_validation": stock_export_data})
    st.download_button(
        "⬇️ Download Stock Validation (filtered)",
        data=stock_xlsx,
        file_name="stock_validation_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---------- Attendance ----------
with tab_att:
    st.subheader("🕒 Attendance vs Interactions")

    st.dataframe(attendance_filtered, use_container_width=True, height=400)

    flagged = attendance_filtered[
        attendance_filtered["activity_flag"] == "Attended_no_interactions"
    ]
    if not flagged.empty:
        st.markdown("#### Agents who logged in but had **no interactions**")
        st.dataframe(flagged, use_container_width=True)
        fig_att = px.scatter(
            flagged,
            x="login_date",
            y="username",
            size="login_count",
            color="zone",
            title="Agents with Attendance but No Interactions",
            color_discrete_sequence=PALETTE_MAIN,
        )
        st.plotly_chart(fig_att, use_container_width=True)

        by_zone_att = (
            flagged.groupby("zone")
            .size()
            .reset_index(name="issues")
            .sort_values("issues", ascending=False)
        )
        fig_zone_att = px.bar(
            by_zone_att,
            x="zone",
            y="issues",
            title="Attendance Issues by Zone",
            color_discrete_sequence=PALETTE_ALT,
        )
        st.plotly_chart(fig_zone_att, use_container_width=True)
    else:
        st.success("No 'attended but no interactions' issues for the current filter.")

    att_xlsx = to_excel_download({"Attendance_validation": attendance_filtered})
    st.download_button(
        "⬇️ Download Attendance Validation (filtered)",
        data=att_xlsx,
        file_name="attendance_validation_filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---------- Interactions ----------
with tab_inter:
    st.subheader("👥 Interactions Analytics")
    
    # Apply filters to interactions
    inter_filtered = inter.copy()
    if date_range:
        inter_filtered = filter_by_date(inter_filtered, "interaction_day", date_range)
    if agent_filter and "user-name" in inter_filtered.columns:
        inter_filtered = inter_filtered[inter_filtered["user-name"].isin(agent_filter)]
    if place_filter:
        # Use place_display if available, otherwise place_name
        place_col_filter = "place_display" if "place_display" in inter_filtered.columns else "place_name"
        if place_col_filter in inter_filtered.columns:
            inter_filtered = inter_filtered[inter_filtered[place_col_filter].isin(place_filter)]
    if interaction_id_search and "interaction_id" in inter_filtered.columns:
        inter_filtered = filter_by_interaction_id(inter_filtered, interaction_id_search)

    colI1, colI2 = st.columns(2)

    if not inter_type_counts.empty:
        fig_it = px.bar(
            inter_type_counts,
            x="Consumer interactions",
            y="count",
            title="Interactions by Type",
            color_discrete_sequence=PALETTE_MAIN,
        )
        fig_it.update_layout(font_family="Century Gothic")
        colI1.plotly_chart(fig_it, use_container_width=True)
    else:
        colI1.info("No interaction type data.")

    if not inter_agent_counts.empty:
        fig_ic = px.bar(
            inter_agent_counts.head(20),
            x="user-name",
            y="interactions_count",
            title="Top Agents by Interactions",
            color_discrete_sequence=PALETTE_ALT,
        )
        fig_ic.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
        colI2.plotly_chart(fig_ic, use_container_width=True)
    else:
        colI2.info("No interactions per agent data.")

    st.markdown("### 🏷️ Main Brands vs Occasional Brands - Insights")
    
    # Show main brands list with counts and insights
    if "Main Brand" in inter_filtered.columns:
        # Get all brands with their counts
        brands_with_counts = inter_filtered[inter_filtered["Main Brand"].notna()]["Main Brand"].value_counts().reset_index()
        brands_with_counts.columns = ["Brand", "Count"]
        
        # Get the reference list of main brands from the main_brand sheet
        main_brands_set = set()
        if not main_brand.empty:
            # Try to find the brand column in main_brand sheet
            brand_col = None
            for col in main_brand.columns:
                if 'brand' in str(col).lower():
                    brand_col = col
                    break
            
            if brand_col is None and len(main_brand.columns) > 0:
                brand_col = main_brand.columns[0]
            
            if brand_col:
                main_brands_set = set(main_brand[brand_col].dropna().astype(str).str.strip().str.lower().unique())
        
        # Classify each brand as Main or Occasional
        brands_with_counts["Brand_lower"] = brands_with_counts["Brand"].astype(str).str.strip().str.lower()
        brands_with_counts["Type"] = brands_with_counts["Brand_lower"].apply(
            lambda x: "Main" if x in main_brands_set else "Occasional"
        )
        
        if not brands_with_counts.empty:
            colI3, colI4 = st.columns(2)
            
            # Main brands
            main_brands_df = brands_with_counts[brands_with_counts["Type"] == "Main"].sort_values("Count", ascending=False)
            colI3.markdown("#### 🎯 Main Brands")
            if not main_brands_df.empty:
                colI3.dataframe(main_brands_df[["Brand", "Count"]], use_container_width=True, height=250)
                colI3.metric("Total Main Brand Interactions", main_brands_df["Count"].sum())
                colI3.caption(f"Top brand: {main_brands_df.iloc[0]['Brand']} ({main_brands_df.iloc[0]['Count']} interactions)")
            else:
                colI3.info("No main brands data.")
            
            # Occasional brands
            occasional_brands_df = brands_with_counts[brands_with_counts["Type"] == "Occasional"].sort_values("Count", ascending=False)
            colI4.markdown("#### 🔄 Occasional Brands")
            if not occasional_brands_df.empty:
                colI4.dataframe(occasional_brands_df[["Brand", "Count"]], use_container_width=True, height=250)
                colI4.metric("Total Occasional Brand Interactions", occasional_brands_df["Count"].sum())
                colI4.caption(f"Top occasional: {occasional_brands_df.iloc[0]['Brand']} ({occasional_brands_df.iloc[0]['Count']} interactions)")
            else:
                colI4.info("No occasional brands identified.")
            
            # Comparison chart
            st.markdown("#### 📊 Brand Type Comparison")
            type_summary = brands_with_counts.groupby("Type").agg({"Count": "sum", "Brand": "nunique"}).reset_index()
            type_summary.columns = ["Type", "Total Interactions", "Number of Brands"]
            
            fig_brand_comparison = px.bar(
                type_summary,
                x="Type",
                y="Total Interactions",
                text="Number of Brands",
                title="Main vs Occasional Brands: Total Interactions",
                color="Type",
                color_discrete_map={"Main": BRAND_COLORS["black"], "Occasional": BRAND_COLORS["red"]}
            )
            fig_brand_comparison.update_layout(font_family="Century Gothic")
            st.plotly_chart(fig_brand_comparison, use_container_width=True)
        else:
            st.info("No brand data available.")
    else:
        st.info("Brand data not available in interactions.")
    
    st.markdown("---")
    st.markdown("### 📦 ECC & BBOS from Pack Purchase")
    st.caption("**BBOS:** Any pack purchase with '+' sign | **ECC:** Text after the '+' sign")
    
    colI5, colI6 = st.columns(2)
    
    # BBOS Analysis - Count by Agent
    colI5.markdown("#### BBOS Count by Agent")
    if "has_bbos" in inter_filtered.columns:
        bbos_data = inter_filtered[inter_filtered["has_bbos"] == True]
        if not bbos_data.empty:
            # Count BBOS per agent (each pack purchase with + = 1 BBOS)
            bbos_summary = bbos_data.groupby("user-name").size().reset_index(name="bbos_count")
            bbos_summary = bbos_summary.sort_values("bbos_count", ascending=False)
            
            fig_bbos = px.bar(
                bbos_summary.head(20),
                x="user-name",
                y="bbos_count",
                title="BBOS Count by Agent",
                labels={"bbos_count": "BBOS Count", "user-name": "Agent"},
                color_discrete_sequence=[BRAND_COLORS["red"]],
            )
            fig_bbos.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
            colI5.plotly_chart(fig_bbos, use_container_width=True)
            colI5.metric("Total BBOS", len(bbos_data))
        else:
            colI5.info("No pack purchases with '+' sign found.")
    else:
        colI5.warning("BBOS column not found. Check Pack Purchase parsing.")
    
    # ECC Analysis - What items come after the +
    colI6.markdown("#### ECC Items")
    if "ecc_item" in inter_filtered.columns:
        ecc_data = inter_filtered[inter_filtered["ecc_item"].notna() & (inter_filtered["ecc_item"].astype(str).str.strip() != "")]
        if not ecc_data.empty:
            ecc_summary = ecc_data.groupby("ecc_item").size().reset_index(name="count")
            ecc_summary = ecc_summary.sort_values("count", ascending=False)
            
            fig_ecc = px.bar(
                ecc_summary.head(20),
                x="ecc_item",
                y="count",
                title="Top ECC Items",
                labels={"count": "Count", "ecc_item": "ECC Item"},
                color_discrete_sequence=[BRAND_COLORS["black"]],
            )
            fig_ecc.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
            colI6.plotly_chart(fig_ecc, use_container_width=True)
            colI6.metric("Unique ECC Items", len(ecc_summary))
        else:
            colI6.info("No ECC items found.")
    else:
        colI6.warning("ECC column not found. Check Pack Purchase parsing.")

    st.markdown("### Interactions by Place: DCC & ECC")
    # Create interaction by place table with place name (not coordinates)
    if "place_name" in inter_filtered.columns or "place_display" in inter_filtered.columns:
        place_col = "place_display" if "place_display" in inter_filtered.columns else "place_name"
        inter_by_place_table = inter_filtered.groupby(place_col).agg({
            "is_DCC": "sum",
            "is_ECC": "sum"
        }).reset_index()
        inter_by_place_table = inter_by_place_table.rename(columns={
            place_col: "place_name",
            "is_DCC": "dcc_count",
            "is_ECC": "ecc_count"
        })
        inter_by_place_table = inter_by_place_table[inter_by_place_table["place_name"] != ""].sort_values("dcc_count", ascending=False)
        
        if not inter_by_place_table.empty:
            # Apply place filter
            if place_filter:
                inter_place_filtered = inter_by_place_table[inter_by_place_table["place_name"].isin(place_filter)]
            else:
                inter_place_filtered = inter_by_place_table
            
            st.dataframe(inter_place_filtered, use_container_width=True, height=300)
            
            fig_place_dcc_ecc = px.bar(
                inter_place_filtered.head(20),
                x="place_name",
                y=["dcc_count", "ecc_count"],
                title="DCC & ECC Interactions by Place (Top 20)",
                barmode="group",
                labels={"value": "Count", "place_name": "Place Name"},
                color_discrete_sequence=[BRAND_COLORS["black"], BRAND_COLORS["red"]],
            )
            fig_place_dcc_ecc.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
            st.plotly_chart(fig_place_dcc_ecc, use_container_width=True)
        else:
            st.info("No place-based interaction data available.")
    else:
        st.info("No place data available in interactions.")

    st.markdown("### Customer Profile & Behavior")
    colI5, colI6 = st.columns(2)

    if not inter_gender_counts.empty:
        fig_gender = px.pie(
            inter_gender_counts,
            names="Gender",
            values="count",
            title="Interactions by Gender",
            color_discrete_sequence=PALETTE_MAIN,
        )
        fig_gender.update_layout(font_family="Century Gothic")
        colI5.plotly_chart(fig_gender, use_container_width=True)
    else:
        colI5.info("No gender data available.")

    if not inter_age_counts.empty:
        fig_age = px.bar(
            inter_age_counts,
            x="Age Range",
            y="count",
            title="Interactions by Age Range",
            color_discrete_sequence=PALETTE_ALT,
        )
        fig_age.update_layout(font_family="Century Gothic")
        colI6.plotly_chart(fig_age, use_container_width=True)
    else:
        colI6.info("No age range data.")

    st.markdown("### Brand & Conversion")
    colI7, colI8 = st.columns(2)

    if not inter_main_brand_counts.empty:
        fig_brand = px.bar(
            inter_main_brand_counts.head(10),
            x="Main Brand",
            y="count",
            title="Top Main Brands in Interactions",
            color_discrete_sequence=PALETTE_MAIN,
        )
        fig_brand.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
        colI7.plotly_chart(fig_brand, use_container_width=True)
    else:
        colI7.info("No main brand data.")

    if not inter_pack_purchase_counts.empty:
        fig_pack = px.bar(
            inter_pack_purchase_counts,
            x="Pack Purchase",
            y="count",
            title="Pack Purchase Results",
            color_discrete_sequence=PALETTE_ALT,
        )
        fig_pack.update_layout(xaxis_tickangle=45, font_family="Century Gothic")
        colI8.plotly_chart(fig_pack, use_container_width=True)
    else:
        colI8.info("No pack purchase data.")
    
    # Photo Carousel - OPTIMIZED FOR APPSHEET URLS
    st.markdown("### 📸 Interaction Photos")
    
    # Try multiple column names for photos - prioritize "URL" first
    photo_col_candidates = ["url"]
    photo_col = None
    for col in photo_col_candidates:
        if col in inter_filtered.columns:
            photo_col = col
            break
    
    if photo_col:
        # Filter for valid photo URLs (including AppSheet URLs)
        photos_with_data = inter_filtered[
            (inter_filtered[photo_col].notna()) & 
            (inter_filtered[photo_col].astype(str).str.strip() != "") &
            (inter_filtered[photo_col].astype(str) != "nan") &
            (inter_filtered[photo_col].astype(str).str.contains("http", case=False, na=False))
        ].copy()
        
        if not photos_with_data.empty:
            st.success(f"✅ Found {len(photos_with_data)} photos with URLs")
            
            # Check if using AppSheet URLs
            sample_url = str(photos_with_data.iloc[0][photo_col])
            if "appsheet.com" in sample_url.lower():
                st.info("📱 Using AppSheet image URLs")
            
            # Add pagination controls to show all photos
            total_photos = len(photos_with_data)
            photos_per_page = 30
            total_pages = (total_photos + photos_per_page - 1) // photos_per_page
            
            # Page selector
            col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
            with col_page2:
                if total_pages > 1:
                    page_number = st.selectbox(
                        "Select Page", 
                        range(1, total_pages + 1),
                        format_func=lambda x: f"Page {x} of {total_pages}"
                    )
                else:
                    page_number = 1
                    st.info(f"Showing all {total_photos} photos")
            
            # Calculate start and end indices
            start_idx = (page_number - 1) * photos_per_page
            end_idx = min(start_idx + photos_per_page, total_photos)
            photos_to_show = photos_with_data.iloc[start_idx:end_idx]
            
            st.caption(f"Showing photos {start_idx + 1} to {end_idx} of {total_photos}")
            
            # Create 3-column grid
            num_cols = 3
            
            for idx in range(0, len(photos_to_show), num_cols):
                cols = st.columns(num_cols)
                
                for col_idx in range(num_cols):
                    if idx + col_idx < len(photos_to_show):
                        row = photos_to_show.iloc[idx + col_idx]
                        
                        with cols[col_idx]:
                            photo_url = str(row[photo_col]).strip()
                            interaction_id = row.get("interaction_id", "N/A")
                            place = row.get("place_display", row.get("place_name", "N/A"))
                            agent = row.get("user-name", "N/A")
                            date = str(row.get("interaction_day", "N/A"))
                            
                            # Display using st.image (works with AppSheet URLs)
                            try:
                                st.image(photo_url, use_container_width=True, caption=f"ID: {interaction_id}")
                                # Add info below image in a compact format
                                st.markdown(f"**Place:** {place}")
                                st.markdown(f"**Agent:** {agent} | **Date:** {date}")
                            except Exception as e:
                                # If image fails, show clickable link
                                st.warning(f"Image preview unavailable")
                                st.markdown(f"[🔗 Click to view photo]({photo_url})")
                                st.caption(f"ID: {interaction_id} | Place: {place}")
                                st.caption(f"Agent: {agent} | Date: {date}")
                            
                            st.markdown("---")  # Separator between photos
        else:
            st.warning("⚠️ No valid photo URLs found")
            # Debug: Show sample values
            if not inter_filtered.empty and photo_col in inter_filtered.columns:
                sample_photos = inter_filtered[photo_col].dropna().head(3).tolist()
                if sample_photos:
                    st.caption(f"**Sample photo column values:**")
                    for i, url in enumerate(sample_photos[:3], 1):
                        st.code(f"{i}. {str(url)[:100]}...")
                    st.caption("💡 URLs should contain 'http' or 'https'")
    else:
        st.info(f"📷 Photo column not found. Looking for: {', '.join(photo_col_candidates[:5])}...")
        if not inter_filtered.empty:
            available_cols = [c for c in inter_filtered.columns if 'photo' in c.lower() or 'image' in c.lower() or 'url' in c.lower()]
            if available_cols:
                st.caption(f"**Found similar columns:** {', '.join(available_cols)}")
            else:
                st.caption(f"**Sample columns:** {', '.join(inter_filtered.columns.tolist()[:10])}")

# ---------- Tasks ----------
with tab_tasks:
    st.subheader("📋 Tasks Analytics")
    
    # Detailed tasks table with place information
    st.markdown("#### Detailed Tasks Table")
    if not tasks_detailed.empty:
        td_filtered = filter_by_zone_area(tasks_detailed, zone_filter, area_filter)
        td_filtered = filter_by_date(td_filtered, "date", date_range)
        td_filtered = filter_by_agents(td_filtered, agent_filter, username_col="username")
        if place_filter and "place_name" in td_filtered.columns:
            td_filtered = td_filtered[td_filtered["place_name"].isin(place_filter)]
        
        st.dataframe(td_filtered, use_container_width=True, height=400)
        st.caption(f"Showing {len(td_filtered)} tasks")
    else:
        st.info("No detailed tasks data available.")

    st.markdown("---")

    ts_filtered = filter_by_zone_area(tasks_summary, zone_filter, area_filter)
    ts_filtered = filter_by_agents(ts_filtered, agent_filter, username_col="username")

    st.markdown("#### Task Summary per Agent")
    st.dataframe(ts_filtered, use_container_width=True)

    if not ts_filtered.empty:
        colT1, colT2 = st.columns(2)

        fig_tasks = px.bar(
            ts_filtered.sort_values("tasks_count", ascending=False),
            x="username",
            y="tasks_count",
            title="Number of Tasks per Agent",
            color_discrete_sequence=PALETTE_MAIN,
        )
        fig_tasks.update_layout(xaxis_tickangle=45)
        colT1.plotly_chart(fig_tasks, use_container_width=True)

        # Group by distance: above 100 (problem) vs below 100 (normal)
        if "avg_in_distance" in ts_filtered.columns and "avg_out_distance" in ts_filtered.columns:
            ts_filtered["distance_category"] = "Normal"
            ts_filtered.loc[
                (ts_filtered["avg_in_distance"] > 100) | (ts_filtered["avg_out_distance"] > 100),
                "distance_category"
            ] = "Problem (>100)"
            
            # Distance grouping chart
            dist_group = ts_filtered.groupby("distance_category").size().reset_index(name="count")
            fig_dist_group = px.bar(
                dist_group,
                x="distance_category",
                y="count",
                title="Distance Categories (Normal vs Problem >100)",
                color_discrete_sequence=[BRAND_COLORS["black"], BRAND_COLORS["red"]],
            )
            fig_dist_group.update_layout(font_family="Century Gothic")
            colT2.plotly_chart(fig_dist_group, use_container_width=True)
            
            # Show problem cases
            problems = ts_filtered[ts_filtered["distance_category"] == "Problem (>100)"]
            if not problems.empty:
                st.markdown("#### ⚠️ Agents with Distance Problems (>100)")
                problem_cols = ["username", "zone", "area", "avg_in_distance", "avg_out_distance", "tasks_count"]
                existing_problem_cols = [c for c in problem_cols if c in problems.columns]
                st.dataframe(problems[existing_problem_cols], use_container_width=True)
                
                # Show detailed tasks for problem agents if available
                if not tasks_detailed.empty:
                    problem_agents = problems["username"].unique()
                    problem_tasks = tasks_detailed[tasks_detailed["username"].isin(problem_agents)]
                    if not problem_tasks.empty:
                        with st.expander(f"📋 Show detailed tasks for agents with distance problems ({len(problem_tasks)} tasks)"):
                            st.dataframe(problem_tasks, use_container_width=True)
        else:
            colT2.info("Distance data not available.")

        st.markdown("### Task Status & Shifts")
        colT3, colT4 = st.columns(2)

        # Status chart
        if not tasks_status_counts.empty:
            valid_status = tasks_status_counts.dropna(subset=["status"])
            if not valid_status.empty and valid_status["count"].sum() > 0:
                fig_status = px.bar(
                    valid_status,
                    x="status",
                    y="count",
                    title="Tasks by Status",
                    color_discrete_sequence=PALETTE_MAIN,
                )
                colT3.plotly_chart(fig_status, use_container_width=True)
            else:
                colT3.info("No status data available yet.")
        else:
            colT3.info("No status data available yet.")

        # Shift chart – filter out >12 hour shifts (likely errors/haven't checked out)
        if not tasks.empty and "shift" in tasks.columns:
            # Try to parse shift as numeric
            tasks_for_shift = tasks.copy()
            try:
                tasks_for_shift["shift_numeric"] = pd.to_numeric(tasks_for_shift["shift"], errors='coerce')
                # Filter to reasonable shifts (≤12 hours)
                tasks_normal_shift = tasks_for_shift[tasks_for_shift["shift_numeric"] <= 12]
                if not tasks_normal_shift.empty:
                    shift_counts_filtered = tasks_normal_shift.groupby("shift").size().reset_index(name="count")
                    shift_counts_filtered = shift_counts_filtered.dropna(subset=["shift"])
                    if not shift_counts_filtered.empty:
                        fig_shift = px.bar(
                            shift_counts_filtered,
                            x="shift",
                            y="count",
                            title="Tasks by Shift (≤12 hours only)",
                            color_discrete_sequence=PALETTE_ALT,
                        )
                        fig_shift.update_layout(font_family="Century Gothic")
                        colT4.plotly_chart(fig_shift, use_container_width=True)
                    else:
                        colT4.info("No valid shift data (≤12 hours) available.")
                else:
                    colT4.info("No shift data ≤12 hours available.")
            except:
                # If numeric parsing fails, show all shifts
                if not tasks_shift_counts.empty:
                    valid_shift = tasks_shift_counts.dropna(subset=["shift"])
                    if not valid_shift.empty and valid_shift["count"].sum() > 0:
                        fig_shift = px.bar(
                            valid_shift,
                            x="shift",
                            y="count",
                            title="Tasks by Shift",
                            color_discrete_sequence=PALETTE_ALT,
                        )
                        fig_shift.update_layout(font_family="Century Gothic")
                        colT4.plotly_chart(fig_shift, use_container_width=True)
                    else:
                        colT4.info("No shift data available yet.")
                else:
                    colT4.info("No shift data available yet.")
        else:
            colT4.info("No shift data available yet.")
    else:
        st.info("No tasks summary available for current filters.")
