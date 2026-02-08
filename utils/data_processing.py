"""Main data loading and processing module."""

import pandas as pd
import numpy as np
import streamlit as st
from utils.google_sheets import load_file_from_url, safe_read_sheet
from utils.helpers import standardize_columns
from utils.data_transformers import (
    normalize_users, normalize_tasks, normalize_interactions, normalize_stocklog
)
from utils.validations import (
    build_metric_validation, build_stock_validation, build_attendance_validation
)
from utils.data_optimization import optimize_dtypes


@st.cache_data(show_spinner=False, ttl=600)
def load_and_process(google_sheets_url: str) -> dict:
    """Load and process data from Google Sheets."""
    file = load_file_from_url(google_sheets_url)
    xls = pd.ExcelFile(file)

    users = safe_read_sheet(xls, "Users")
    tasks = safe_read_sheet(xls, "Tasks")
    inter = safe_read_sheet(xls, "interactions")
    stocklog = safe_read_sheet(xls, "Stocklog")
    sv_tasks = safe_read_sheet(xls, "sv-tasks")
    login = safe_read_sheet(xls, "Login")
    main_brand = safe_read_sheet(xls, "main")
    purchase = safe_read_sheet(xls, "purchase")

    users = standardize_columns(users)
    tasks = standardize_columns(tasks)
    inter = standardize_columns(inter)
    stocklog = standardize_columns(stocklog)
    sv_tasks = standardize_columns(sv_tasks)
    login = standardize_columns(login)
    main_brand = standardize_columns(main_brand)
    purchase = standardize_columns(purchase)

    users, users_lookup = normalize_users(users)
    tasks = normalize_tasks(tasks, users_lookup)
    inter = normalize_interactions(inter, users_lookup)
    stock_agg = normalize_stocklog(stocklog)

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
        tasks_summary = pd.DataFrame(columns=["agent_name", "tasks_count", "completed", "pending", "in_progress", "avg_shift_hours", "working_days"])

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
        inter_summary = pd.DataFrame(columns=["agent_name", "interactions", "dcc", "ecc", "qr", "bbos", "bbos_from_pack"])

    dcc_check = build_metric_validation("dcc", tasks, inter)
    ecc_check = build_metric_validation("ecc", tasks, inter)
    qr_check = build_metric_validation("qr", tasks, inter)
    bbos_check = build_metric_validation("bbos", tasks, inter)

    stock_check = build_stock_validation(stock_agg, inter)
    attendance_view = build_attendance_validation(tasks, inter)

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

    supervisor_perf = pd.DataFrame()
    if not tasks.empty and "sv" in tasks.columns:
        supervisor_perf = tasks.groupby("sv", dropna=False, sort=False).agg(
            team_size=("agent_name", "nunique"),
            total_tasks=("task_day", "count"),
            completed_tasks=("status_norm", lambda s: (s == "completed").sum()),
            avg_completion_rate=("status_norm", lambda s: (s == "completed").sum() / len(s) * 100 if len(s) > 0 else 0),
            total_shift_hours=("shift_hours", "sum"),
            avg_shift_hours=("shift_hours", "mean"),
            working_days=("task_day", "nunique"),
        ).reset_index()
        supervisor_perf["avg_completion_rate"] = supervisor_perf["avg_completion_rate"].round(1)
    
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
