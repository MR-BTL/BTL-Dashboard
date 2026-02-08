"""Data transformation functions for normalizing different data types."""

import pandas as pd
import numpy as np
from utils.helpers import (
    standardize_columns, clean_user_id, to_datetime_any, 
    pick_first_existing_col
)


def normalize_users(users: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize users DataFrame and create lookup table."""
    if users.empty:
        return users, pd.DataFrame()
    
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

    wanted_user_cols = ["user_id", "username", "zone", "area", "channel", "sv", "role"]
    for c in wanted_user_cols:
        if c not in users.columns:
            users[c] = np.nan
    
    users_lookup = users[wanted_user_cols].drop_duplicates()
    return users, users_lookup


def normalize_tasks(tasks: pd.DataFrame, users_lookup: pd.DataFrame) -> pd.DataFrame:
    """Normalize tasks DataFrame."""
    if tasks.empty:
        return tasks
    
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

    if "task_date" in tasks.columns:
        tasks["task_dt"] = to_datetime_any(tasks["task_date"])
        tasks["task_day"] = tasks["task_dt"].dt.date
    else:
        tasks["task_dt"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index, dtype='datetime64[ns]')
        tasks["task_day"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index)

    if "status" in tasks.columns:
        tasks["status_norm"] = (
            tasks["status"].astype(str).str.strip().str.lower()
            .replace({"nan": "", "none": ""})
        )
    else:
        tasks["status_norm"] = pd.Series([""] * len(tasks), index=tasks.index, dtype='string')

    checkin_col = pick_first_existing_col(tasks, [
        "check_in_time", "checkin_time", "checkin", "check_in", "check_in_datetime",
        "check_in_time_", "check_in_time__"
    ])
    checkout_col = pick_first_existing_col(tasks, [
        "check_out_time", "checkout_time", "checkout", "check_out", "check_out_datetime",
        "check_out_time_", "check_out_time__"
    ])

    checkin_date_col = pick_first_existing_col(tasks, ["check_in_date", "checkin_date", "check_in"])
    checkout_date_col = pick_first_existing_col(tasks, ["check_out_date", "checkout_date", "check_out"])

    tasks["checkin_dt"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index, dtype='datetime64[ns]')
    tasks["checkout_dt"] = pd.Series([pd.NaT] * len(tasks), index=tasks.index, dtype='datetime64[ns]')

    if checkin_col:
        tasks["checkin_dt"] = to_datetime_any(tasks[checkin_col])
    if checkout_col:
        tasks["checkout_dt"] = to_datetime_any(tasks[checkout_col])

    if tasks["checkin_dt"].isna().all() and checkin_date_col and "check_in_time" in tasks.columns:
        tasks["checkin_dt"] = to_datetime_any(
            tasks[checkin_date_col].astype(str) + " " + tasks["check_in_time"].astype(str)
        )
    if tasks["checkout_dt"].isna().all() and checkout_date_col and "check_out_time" in tasks.columns:
        tasks["checkout_dt"] = to_datetime_any(
            tasks[checkout_date_col].astype(str) + " " + tasks["check_out_time"].astype(str)
        )

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

    for m in ["dcc", "ecc", "qr", "bbos"]:
        col = m
        if col not in tasks.columns:
            if m.upper() in tasks.columns:
                tasks[m] = tasks[m.upper()]
            else:
                tasks[m] = pd.Series([0] * len(tasks), index=tasks.index, dtype='float64')
        tasks[m] = pd.to_numeric(tasks[m], errors="coerce").fillna(0)
    
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
    
    if "in_distance" not in tasks.columns:
        tasks["in_distance"] = np.nan
    if "out_distance" not in tasks.columns:
        tasks["out_distance"] = np.nan
    
    if "in_distance" in tasks.columns and "out_distance" in tasks.columns:
        tasks["total_distance"] = tasks["in_distance"].fillna(0) + tasks["out_distance"].fillna(0)
        tasks["total_distance"] = tasks["total_distance"].replace(0, np.nan)
    elif "in_distance" in tasks.columns:
        tasks["total_distance"] = tasks["in_distance"]
    elif "out_distance" in tasks.columns:
        tasks["total_distance"] = tasks["out_distance"]
    else:
        tasks["total_distance"] = np.nan

    if "place_code" not in tasks.columns:
        tasks["place_code"] = pd.Series([""] * len(tasks), index=tasks.index)
    if "place_name" not in tasks.columns:
        tasks["place_name"] = pd.Series([""] * len(tasks), index=tasks.index)
    
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

    if not users_lookup.empty:
        tasks = tasks.merge(
            users_lookup[["user_id", "username", "zone", "area", "channel", "sv", "role"]],
            on="user_id",
            how="left",
            suffixes=("", "_u")
        )
        if "user_name" in tasks.columns:
            tasks["agent_name"] = tasks["user_name"].fillna(tasks["username"])
        else:
            tasks["agent_name"] = tasks["username"]
    
    return tasks


def normalize_interactions(inter: pd.DataFrame, users_lookup: pd.DataFrame) -> pd.DataFrame:
    """Normalize interactions DataFrame."""
    if inter.empty:
        return inter
    
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
    inter = inter.rename(columns={k: v for k, v in alias.items() if k in inter.columns})

    if "user_id" in inter.columns:
        inter["user_id"] = clean_user_id(inter["user_id"])

    if "day" in inter.columns:
        inter["day_dt"] = to_datetime_any(inter["day"])
        inter["interaction_day"] = inter["day_dt"].dt.date
    elif "date" in inter.columns:
        inter["day_dt"] = to_datetime_any(inter["date"])
        inter["interaction_day"] = inter["day_dt"].dt.date
    else:
        inter["day_dt"] = pd.Series([pd.NaT] * len(inter), index=inter.index, dtype='datetime64[ns]')
        inter["interaction_day"] = pd.Series([pd.NaT] * len(inter), index=inter.index)

    ci_col = pick_first_existing_col(inter, ["consumer_interactions", "consumerinteractions", "consumer_interaction"])
    if not ci_col:
        ci_col = "consumer_interactions" if "consumer_interactions" in inter.columns else None

    if ci_col:
        inter["interaction_type"] = inter[ci_col].astype(str).str.strip().str.upper()
    else:
        inter["interaction_type"] = pd.Series([""] * len(inter), index=inter.index, dtype='string')

    if "place_name" not in inter.columns:
        inter["place_name"] = pd.Series([""] * len(inter), index=inter.index)
    if "place_code" not in inter.columns:
        inter["place_code"] = pd.Series([""] * len(inter), index=inter.index)
    
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

    if "pack_purchase" not in inter.columns:
        inter["pack_purchase"] = ""
    pp = inter["pack_purchase"].astype(str).fillna("").str.strip()
    inter["has_bbos"] = pp.str.contains(r"\+", regex=True, na=False)
    inter["ecc_item"] = np.where(inter["has_bbos"], pp.str.split("+", n=1).str[1].fillna("").str.strip(), "")

    if not users_lookup.empty:
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

    inter["is_dcc"] = (inter["interaction_type"] == "DCC").astype(int)
    inter["is_ecc"] = (inter["interaction_type"] == "ECC").astype(int)
    inter["is_qr"] = (inter["interaction_type"] == "QR").astype(int)
    inter["is_bbos"] = (inter["interaction_type"] == "BBOS").astype(int)
    
    if "main_brand" in inter.columns:
        inter["main_brand"] = inter["main_brand"].astype(str).str.strip().str.title()
        inter["main_brand"] = inter["main_brand"].replace({"Nan": "", "None": "", "": np.nan})
    
    occasional_col = pick_first_existing_col(inter, ["occasional_brand", "occasional brand", "occasionalbrand"])
    if occasional_col:
        inter["occasional_brand"] = inter[occasional_col].astype(str).str.strip().str.title()
        inter["occasional_brand"] = inter["occasional_brand"].replace({"Nan": "", "None": "", "": np.nan})
    elif "occasional_brand" not in inter.columns:
        inter["occasional_brand"] = np.nan

    return inter


def normalize_stocklog(stocklog: pd.DataFrame) -> pd.DataFrame:
    """Normalize stocklog DataFrame and create aggregated view."""
    if stocklog.empty:
        return pd.DataFrame(columns=["agent_name", "stock_day", "qty_release", "qty_return", "qty_used"])
    
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

    lower_cols = {c.lower(): c for c in stock_pivot.columns}
    
    def get_tx(name: str, aliases: list = None) -> pd.Series:
        names_to_check = [name]
        if aliases:
            names_to_check.extend(aliases)
        for n in names_to_check:
            if n in lower_cols:
                return stock_pivot[lower_cols[n]].astype(float)
        return pd.Series([0.0] * len(stock_pivot), index=stock_pivot.index)

    stock_pivot["qty_release"] = get_tx("release")
    stock_pivot["qty_return"] = get_tx("back", aliases=["return"])
    qty_used_from_tx = get_tx("used")
    stock_pivot["qty_used"] = np.where(
        qty_used_from_tx > 0,
        qty_used_from_tx,
        stock_pivot["qty_release"] - stock_pivot["qty_return"]
    )
    
    return stock_pivot.copy()
