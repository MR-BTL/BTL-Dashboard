"""Operations & Performance tab component."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from functools import reduce
from utils.data_optimization import safe_fillna, limit_display_rows
from utils.charts import style_chart
from utils.exports import to_excel_download


def render_operations_tab(tasks_f, inter_f, supervisor_perf, sv_filter, u, 
                          stock_check_f, dcc_f, ecc_f, qr_f, bbos_f, att_f):
    """Render the Operations & Performance tab."""
    st.subheader("⚡ Operations & Performance")
    
    _render_supervisor_performance(supervisor_perf, sv_filter, u)
    st.markdown("---")
    _render_agent_performance(tasks_f, inter_f)
    _render_distance_analysis(tasks_f)
    _render_validations_section(dcc_f, ecc_f, qr_f, bbos_f, att_f)
    _render_consolidated_summary(tasks_f, inter_f, stock_check_f, dcc_f, ecc_f, qr_f, bbos_f, att_f)


def _render_supervisor_performance(supervisor_perf, sv_filter, u):
    """Render supervisor performance section."""
    st.markdown("### Supervisor Performance")
    if not supervisor_perf.empty and "sv" in supervisor_perf.columns:
        sv_filtered = supervisor_perf.copy()
        if sv_filter:
            sv_filtered = sv_filtered[sv_filtered["sv"].isin(sv_filter)]
        
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
            
            with st.expander("📋 View Supervisor Performance Details"):
                display_cols = [c for c in ["sv", "team_size", "total_tasks", "completed_tasks", 
                                           "avg_completion_rate", "total_interactions", 
                                           "total_shift_hours", "avg_shift_hours", "working_days"]
                               if c in sv_filtered.columns]
                if display_cols:
                    sort_col = "total_interactions" if "total_interactions" in display_cols else "total_tasks"
                    sv_display = sv_filtered[display_cols].sort_values(sort_col, ascending=False)
                    st.dataframe(sv_display, use_container_width=True, height=400)
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


def _render_agent_performance(tasks_f, inter_f):
    """Render agent performance section."""
    st.markdown("### Agent Performance")
    if not tasks_f.empty or not inter_f.empty:
        agent_perf_list = []
        
        if not tasks_f.empty:
            perf_tasks = tasks_f.groupby("agent_name").agg(
                tasks=("task_day", "count"),
                completed=("status_norm", lambda s: (s == "completed").sum()),
                pending=("status_norm", lambda s: (s == "pending").sum()),
                in_progress=("status_norm", lambda s: s.isin(["in progress", "in_progress", "inprogress"]).sum()),
                working_days=("task_day", lambda s: pd.Series(s).nunique()),
                avg_shift=("shift_hours", "mean"),
                total_shift=("shift_hours", "sum"),
                dcc=("dcc", "sum"), ecc=("ecc", "sum"), qr=("qr", "sum"), bbos=("bbos", "sum"),
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
            perf = safe_fillna(perf, value=0)
            
            if "tasks" in perf.columns and perf["tasks"].sum() > 0:
                perf["completion_rate"] = np.where(perf["tasks"] > 0, (perf["completed"] / perf["tasks"]) * 100, 0).round(1)
            
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
            
            st.markdown("#### Shift Analysis")
            
            if not tasks_f.empty and "status_norm" in tasks_f.columns:
                status_counts = tasks_f["status_norm"].value_counts().reset_index()
                status_counts.columns = ["status", "count"]
                fig = px.pie(status_counts, names="status", values="count",
                            title="Tasks by Status")
                fig = style_chart(fig, "Tasks by Status")
                st.plotly_chart(fig, use_container_width=True)
            
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
                    csv = perf_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Agent Performance Table",
                        data=csv,
                        file_name="agent_performance.csv",
                        mime="text/csv"
                    )
    else:
        st.info("No performance data available for the selected filters.")


def _render_distance_analysis(tasks_f):
    """Render distance analysis section."""
    st.markdown("---")
    st.markdown("#### Distance Analysis")
    
    if tasks_f.empty:
        st.info("Distance fields (in-distance, out-distance) not found in the data.")
        return
    
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
        combined_condition = reduce(lambda x, y: x | y, distance_conditions)
        distance_tasks = tasks_f[combined_condition].copy()
    else:
        distance_tasks = pd.DataFrame()
    
    if distance_tasks.empty:
        st.info("No distance data available for the selected filters.")
        return
    
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
    
    col_chart_dist1, col_chart_dist2 = st.columns(2)
    
    with col_chart_dist1:
        if "agent_name" in distance_tasks.columns and "total_distance" in distance_tasks.columns:
            agent_dist = distance_tasks.groupby("agent_name").agg(
                avg_distance=("total_distance", "mean"),
                total_distance=("total_distance", "sum"),
                task_count=("total_distance", "count")
            ).reset_index()
            agent_dist = agent_dist.sort_values("total_distance", ascending=False).head(15)
            
            if not agent_dist.empty:
                fig = px.bar(agent_dist, x="agent_name", y="total_distance",
                            title="Total Distance by Agent (Top 15)",
                            labels={"total_distance": "Total Distance", "agent_name": "Agent"})
                fig = style_chart(fig, "Total Distance by Agent (Top 15)")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    with col_chart_dist2:
        if "zone" in distance_tasks.columns and "total_distance" in distance_tasks.columns:
            zone_dist = distance_tasks.groupby("zone").agg(
                avg_distance=("total_distance", "mean"),
                total_distance=("total_distance", "sum"),
                task_count=("total_distance", "count")
            ).reset_index()
            zone_dist = zone_dist.sort_values("total_distance", ascending=False)
            
            if not zone_dist.empty:
                fig = px.bar(zone_dist, x="zone", y="total_distance",
                            title="Total Distance by Zone",
                            labels={"total_distance": "Total Distance", "zone": "Zone"})
                fig = style_chart(fig, "Total Distance by Zone")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    if "task_day" in distance_tasks.columns and "total_distance" in distance_tasks.columns:
        daily_dist = distance_tasks.groupby("task_day").agg(
            avg_distance=("total_distance", "mean"),
            total_distance=("total_distance", "sum"),
            task_count=("total_distance", "count")
        ).reset_index()
        daily_dist = daily_dist.sort_values("task_day")
        
        if not daily_dist.empty:
            fig = px.line(daily_dist, x="task_day", y=["avg_distance", "total_distance"],
                        title="Distance Trends Over Time",
                        labels={"value": "Distance", "task_day": "Date", "variable": "Metric"})
            fig = style_chart(fig, "Distance Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Distance Summary by Agent")
    if "agent_name" in distance_tasks.columns:
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
        
        if "total_total_distance" in dist_summary_df.columns:
            dist_summary_df = dist_summary_df.sort_values("total_total_distance", ascending=False)
        elif "total_in_distance" in dist_summary_df.columns:
            dist_summary_df = dist_summary_df.sort_values("total_in_distance", ascending=False)
        
        st.dataframe(dist_summary_df, use_container_width=True, height=400)
        csv = dist_summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Distance Summary Table",
            data=csv,
            file_name="distance_summary_by_agent.csv",
            mime="text/csv"
        )
    
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
            csv = dist_detail.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Detailed Distance Data",
                data=csv,
                file_name="distance_detailed.csv",
                mime="text/csv"
            )


def _render_validations_section(dcc_f, ecc_f, qr_f, bbos_f, att_f):
    """Render validations section."""
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

            summary = vdf["match_flag"].value_counts().reset_index()
            summary.columns = ["match_flag", "rows"]
            fig = px.bar(summary, x="match_flag", y="rows", title=f"{title} – Match vs Mismatch")
            fig = style_chart(fig, f"{title} – Match vs Mismatch")
            st.plotly_chart(fig, use_container_width=True)

            mismatches = vdf[vdf["match_flag"] == "Mismatch"].sort_values("difference", ascending=False)
            if not mismatches.empty:
                st.markdown("#### Top Mismatches")
                st.dataframe(mismatches.head(50), use_container_width=True, height=300)

            xlsx = to_excel_download({title[:25]: vdf})
            st.download_button(
                f"⬇️ Download {title} (filtered)",
                data=xlsx,
                file_name=f"{title.replace(' ', '_').lower()}_filtered.xlsx",
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
                csv = issues_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Attendance Issues Table",
                    data=csv,
                    file_name="attendance_issues.csv",
                    mime="text/csv"
                )


def _render_consolidated_summary(tasks_f, inter_f, stock_check_f, dcc_f, ecc_f, qr_f, bbos_f, att_f):
    """Render consolidated summary table."""
    st.markdown("---")
    st.subheader("📊 Consolidated Agent Summary")
    st.caption("Key metrics from performance, stock, validations, and attendance data")

    consolidated_list = []

    if not tasks_f.empty or not inter_f.empty:
        agent_perf_consolidated = []
        
        if not tasks_f.empty:
            perf_tasks_cons = tasks_f.groupby("agent_name").agg(
                tasks=("task_day", "count"),
                completed=("status_norm", lambda s: (s == "completed").sum()),
                pending=("status_norm", lambda s: (s == "pending").sum()),
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
            consolidated_base = safe_fillna(consolidated_base, value=0)
            
            if "tasks" in consolidated_base.columns and consolidated_base["tasks"].sum() > 0:
                consolidated_base["completion_rate"] = np.where(
                    consolidated_base["tasks"] > 0,
                    (consolidated_base["completed"] / consolidated_base["tasks"]) * 100,
                    0
                ).round(1)
            
            consolidated_list.append(consolidated_base)

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

    validation_cols = {}
    for metric, val_df in [("dcc", dcc_f), ("ecc", ecc_f), ("qr", qr_f), ("bbos", bbos_f)]:
        if not val_df.empty and "agent_name" in val_df.columns:
            mismatches = val_df[val_df["match_flag"] == "Mismatch"].groupby("agent_name").size().reset_index(name=f"{metric}_mismatches")
            validation_cols[f"{metric}_mismatches"] = mismatches

    for col_name, val_df in validation_cols.items():
        if consolidated_list:
            consolidated_list[0] = consolidated_list[0].merge(val_df, on="agent_name", how="left")
        else:
            consolidated_list.append(val_df)

    if not att_f.empty and "agent_name" in att_f.columns:
        attendance_issues = att_f[att_f["activity_flag"] == "Worked_no_interactions"].groupby("agent_name").size().reset_index(name="attendance_issues_count")
        if consolidated_list:
            consolidated_list[0] = consolidated_list[0].merge(attendance_issues, on="agent_name", how="left")
        else:
            consolidated_list.append(attendance_issues)

    if consolidated_list:
        consolidated_final = reduce(lambda x, y: pd.merge(x, y, on="agent_name", how="outer"), consolidated_list) if len(consolidated_list) > 1 else consolidated_list[0]
        consolidated_final = safe_fillna(consolidated_final, value=0)
        
        val_cols = [c for c in consolidated_final.columns if "mismatches" in c.lower()]
        if val_cols:
            consolidated_final["total_validation_issues"] = consolidated_final[val_cols].sum(axis=1)
        
        display_order = [
            "agent_name", "tasks", "completed", "pending", "completion_rate", 
            "interactions", "working_days", "avg_shift_hours",
            "total_stock_used", "total_interactions_stock", "stock_utilization_rate", "stock_issues_count",
            "dcc_mismatches", "ecc_mismatches", "qr_mismatches", "bbos_mismatches", "total_validation_issues",
            "attendance_issues_count"
        ]
        
        available_cols = [c for c in display_order if c in consolidated_final.columns]
        consolidated_display = consolidated_final[available_cols].copy()
        
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
