"""Data validation functions."""

import pandas as pd
import numpy as np


def build_metric_validation(metric: str, tasks_df: pd.DataFrame, inter_df: pd.DataFrame) -> pd.DataFrame:
    """Build validation DataFrame comparing tasks vs interactions for a metric."""
    if tasks_df.empty and inter_df.empty:
        return pd.DataFrame()

    t = pd.DataFrame(columns=["agent_name", "day", "tasks_total"])
    i = pd.DataFrame(columns=["agent_name", "day", "inter_total"])

    if not tasks_df.empty and "task_day" in tasks_df.columns:
        t = (
            tasks_df.groupby(["agent_name", "task_day"], as_index=False)[metric]
            .sum()
            .rename(columns={"task_day": "day", metric: "tasks_total"})
        )

    inter_flag = f"is_{metric}"
    if not inter_df.empty and "interaction_day" in inter_df.columns and inter_flag in inter_df.columns:
        i = (
            inter_df.groupby(["agent_name", "interaction_day"], as_index=False)[inter_flag]
            .sum()
            .rename(columns={"interaction_day": "day", inter_flag: "inter_total"})
        )

    v = pd.merge(t, i, on=["agent_name", "day"], how="outer")
    v["tasks_total"] = v["tasks_total"].fillna(0)
    v["inter_total"] = v["inter_total"].fillna(0)
    v["difference"] = v["tasks_total"] - v["inter_total"]
    v["match_flag"] = np.where(v["difference"] == 0, "Match", "Mismatch")
    return v


def build_stock_validation(stock_agg: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    """Build stock validation DataFrame."""
    if stock_agg.empty and inter.empty:
        return pd.DataFrame(columns=[
            "agent_name", "stock_day", "qty_release", "qty_return", "qty_used",
            "interactions_total", "bbos_pack", "bbos_type",
            "diff_used_vs_interactions", "stock_flag_interactions",
            "diff_used_vs_bbos_pack", "stock_flag_bbos_pack"
        ])
    
    if "qty_used" not in stock_agg.columns:
        stock_agg["qty_used"] = stock_agg["qty_release"] - stock_agg["qty_return"]
    else:
        stock_agg["qty_used"] = stock_agg["qty_used"].fillna(stock_agg["qty_release"] - stock_agg["qty_return"])
    
    inter_agent_day = (
        inter.groupby(["agent_name", "interaction_day"], as_index=False)
        .agg(
            interactions_total=("interaction_type", "count"),
            bbos_pack=("has_bbos", "sum"),
            bbos_type=("is_bbos", "sum"),
        )
        .rename(columns={"interaction_day": "stock_day"})
    )
    stock_check = stock_agg.merge(inter_agent_day, on=["agent_name", "stock_day"], how="left")
    stock_check[["interactions_total", "bbos_pack", "bbos_type", "qty_used"]] = \
        stock_check[["interactions_total", "bbos_pack", "bbos_type", "qty_used"]].fillna(0)

    stock_check["diff_used_vs_interactions"] = stock_check["qty_used"] - stock_check["interactions_total"]
    stock_check["stock_flag_interactions"] = np.where(
        stock_check["diff_used_vs_interactions"] >= 0, 
        "OK", 
        "Interactions Exceed Used Stock"
    )

    stock_check["diff_used_vs_bbos_pack"] = stock_check["qty_used"] - stock_check["bbos_pack"]
    stock_check["stock_flag_bbos_pack"] = np.where(
        stock_check["diff_used_vs_bbos_pack"] >= 0, 
        "OK", 
        "BBOS Pack Exceed Used Stock"
    )
    
    return stock_check


def build_attendance_validation(tasks: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    """Build attendance validation DataFrame."""
    if tasks.empty:
        task_day_any = pd.DataFrame(columns=["agent_name", "day", "tasks_any"])
    else:
        task_day_any = (
            tasks.groupby(["agent_name", "task_day"], as_index=False)
            .size()
            .rename(columns={"size": "tasks_any", "task_day": "day"})
        )

    if inter.empty:
        inter_day_any = pd.DataFrame(columns=["agent_name", "day", "interactions_any"])
    else:
        inter_day_any = (
            inter.groupby(["agent_name", "interaction_day"], as_index=False)
            .size()
            .rename(columns={"size": "interactions_any", "interaction_day": "day"})
        )

    attendance_view = task_day_any.merge(inter_day_any, on=["agent_name", "day"], how="left")
    attendance_view["interactions_any"] = attendance_view["interactions_any"].fillna(0)
    attendance_view["activity_flag"] = np.where(
        (attendance_view["tasks_any"] > 0) & (attendance_view["interactions_any"] == 0),
        "Worked_no_interactions",
        "OK"
    )
    return attendance_view
