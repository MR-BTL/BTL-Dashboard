"""Main Streamlit dashboard application."""

import streamlit as st
import pandas as pd
from functools import reduce

from config import APP_TITLE, APP_SUBTITLE
from components.ui import apply_theme, display_logo, mini_card
from utils.data_processing import load_and_process
from utils.filters import filter_date, filter_in, filter_text_contains
from utils.data_optimization import safe_fillna, limit_display_rows
from utils.charts import style_chart
from utils.exports import to_excel_download
import plotly.express as px


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
display_logo()

st.title(f"📊 {APP_TITLE}")
st.caption(APP_SUBTITLE)
st.markdown("---")

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

st.sidebar.markdown("---")
st.sidebar.header("🔎 Global Filters")

date_range = None
if min_date and max_date:
    dr = st.sidebar.date_input("📅 Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    date_range = dr if isinstance(dr, tuple) else (dr, dr)

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

agent_filter = st.sidebar.multiselect("👤 Agents", agents, default=[])
zone_filter = st.sidebar.multiselect("🌍 Zones", zones, default=[])
area_filter = st.sidebar.multiselect("📍 Areas", areas, default=[])
channel_filter = st.sidebar.multiselect("📡 Channels", channels, default=[])
sv_filter = st.sidebar.multiselect("🧑‍💼 Supervisors (SV)", svs, default=[])

task_status_options = []
if not tasks.empty and "status_norm" in tasks.columns:
    task_status_options = sorted(tasks["status_norm"].dropna().unique().tolist())
task_status_filter = st.sidebar.multiselect("✅ Task Status", task_status_options, default=[])

shift_bucket_options = []
if not tasks.empty and "shift_bucket" in tasks.columns:
    shift_bucket_options = sorted(tasks["shift_bucket"].dropna().unique().tolist())
shift_bucket_filter = st.sidebar.multiselect("⏱️ Shift Bucket", shift_bucket_options, default=[])

place_options = []
if not tasks.empty and "place_display" in tasks.columns:
    place_options = sorted([p for p in tasks["place_display"].dropna().astype(str).unique() if p.strip() != "" and p != "nan"])
place_filter = st.sidebar.multiselect("🏢 Places", place_options, default=[])

interaction_types = []
if not inter.empty and "interaction_type" in inter.columns:
    interaction_types = sorted([x for x in inter["interaction_type"].dropna().astype(str).unique() if x.strip() != "" and x != "nan"])
interaction_type_filter = st.sidebar.multiselect("👥 Interaction Type", interaction_types, default=[])

age_options = unique_sorted(inter, "age_range")
age_filter = st.sidebar.multiselect("🎂 Age Range", age_options, default=[])

interaction_id_search = st.sidebar.text_input("🔍 Interaction ID contains", "")

st.sidebar.markdown("---")
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
    for f in active_filters[:5]:
        st.sidebar.caption(f"• {f}")
    if len(active_filters) > 5:
        st.sidebar.caption(f"... and {len(active_filters) - 5} more")
else:
    st.sidebar.caption("No filters applied (showing all data)")

st.sidebar.markdown("---")
st.sidebar.caption("All filters apply across KPIs, tables and charts.")

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

tasks_f = tasks if not tasks.empty else pd.DataFrame()
if not tasks_f.empty:
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
    if date_range or valid_agents or task_status_filter or shift_bucket_filter or place_filter:
        tasks_f = tasks_f.copy()

inter_f = inter if not inter.empty else pd.DataFrame()
if not inter_f.empty:
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
    if date_range or valid_agents or interaction_type_filter or age_filter or place_filter or interaction_id_search:
        inter_f = inter_f.copy()

stock_f = stock_agg.copy() if not stock_agg.empty else stock_agg
if not stock_f.empty:
    if date_range:
        stock_f = filter_date(stock_f, "stock_day", date_range)
    if valid_agents:
        stock_f = filter_in(stock_f, "agent_name", list(valid_agents))

stock_check_f = stock_check.copy() if not stock_check.empty else stock_check
if not stock_check_f.empty:
    if date_range:
        stock_check_f = filter_date(stock_check_f, "stock_day", date_range)
    if valid_agents:
        stock_check_f = filter_in(stock_check_f, "agent_name", list(valid_agents))

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
qr_f = filter_validation(qr_check)
bbos_f = filter_validation(bbos_check)

att_f = attendance_view.copy()
if not att_f.empty:
    if date_range and "day" in att_f.columns:
        att_f = filter_date(att_f, "day", date_range)
    if valid_agents:
        att_f = filter_in(att_f, "agent_name", list(valid_agents))

st.subheader("📌 Overview (fully filter-driven)")

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

completed_tasks = int((tasks_f["status_norm"] == "completed").sum()) if (not tasks_f.empty and "status_norm" in tasks_f.columns) else 0
pending_tasks = int((tasks_f["status_norm"] == "pending").sum()) if (not tasks_f.empty and "status_norm" in tasks_f.columns) else 0
in_progress_tasks = int(tasks_f["status_norm"].isin(["in progress","in_progress","inprogress"]).sum()) if (not tasks_f.empty and "status_norm" in tasks_f.columns) else 0

dcc_total = float(tasks_f["dcc"].sum()) if (not tasks_f.empty and "dcc" in tasks_f.columns) else 0
ecc_total = float(tasks_f["ecc"].sum()) if (not tasks_f.empty and "ecc" in tasks_f.columns) else 0
qr_total = float(tasks_f["qr"].sum()) if (not tasks_f.empty and "qr" in tasks_f.columns) else 0
bbos_total = float(tasks_f["bbos"].sum()) if (not tasks_f.empty and "bbos" in tasks_f.columns) else 0

working_days = int(tasks_f["task_day"].nunique()) if (not tasks_f.empty and "task_day" in tasks_f.columns) else 0
total_shift_hours = float(tasks_f["shift_hours"].sum()) if (not tasks_f.empty and "shift_hours" in tasks_f.columns) else 0
avg_shift_hours = float(tasks_f["shift_hours"].mean()) if (not tasks_f.empty and "shift_hours" in tasks_f.columns) else 0

zones_count = int(u["zone"].dropna().nunique()) if not u.empty else 0
areas_count = int(u["area"].dropna().nunique()) if not u.empty else 0

avg_interactions_per_agent = (total_interactions / active_agents) if active_agents > 0 else 0
shift_efficiency = (total_interactions / total_shift_hours) if total_shift_hours > 0 else 0

unique_brands = 0
if not inter_f.empty and "main_brand" in inter_f.columns:
    unique_brands = int(inter_f["main_brand"].dropna().nunique())

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

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview Dashboard",
    "⚡ Operations & Performance",
    "🎯 Interactions & Brands",
    "📦 Stock & Inventory",
])

# Import tab components - we'll create these next
# For now, keeping the tab logic here but organized
# This will be moved to separate component files

with tab1:
    from components.overview import render_overview_tab
    render_overview_tab(tasks_f, inter_f, stock_f, active_agents, avg_interactions_per_agent, 
                       total_shift_hours, shift_efficiency, total_tasks, completed_tasks)

with tab2:
    from components.operations import render_operations_tab
    render_operations_tab(tasks_f, inter_f, supervisor_perf, sv_filter, u, 
                         stock_check_f, dcc_f, ecc_f, qr_f, bbos_f, att_f)

with tab3:
    from components.interactions import render_interactions_tab
    render_interactions_tab(inter_f)

with tab4:
    from components.stock import render_stock_tab
    render_stock_tab(stock_f, stock_check_f)

st.markdown("---")
st.subheader("⬇️ Export (filtered)")

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
