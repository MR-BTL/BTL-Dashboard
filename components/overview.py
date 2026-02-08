"""Overview Dashboard tab component."""

import streamlit as st
import pandas as pd
import plotly.express as px
from functools import reduce
from utils.data_optimization import safe_fillna, optimize_chart_data
from utils.charts import style_chart


def render_overview_tab(tasks_f, inter_f, stock_f, active_agents, avg_interactions_per_agent,
                        total_shift_hours, shift_efficiency, total_tasks, completed_tasks):
    """Render the Overview Dashboard tab."""
    st.subheader("📊 Overview Dashboard")
    
    st.markdown("### Top Performing Agents")
    colA, colB = st.columns(2)
    
    if not inter_f.empty or not tasks_f.empty:
        agent_perf = []
        
        if not inter_f.empty:
            inter_agents = inter_f.groupby("agent_name").size().reset_index(name="interactions")
            agent_perf.append(inter_agents)
        
        if not tasks_f.empty:
            task_agents = tasks_f.groupby("agent_name").size().reset_index(name="tasks")
            agent_perf.append(task_agents)
        
        if agent_perf:
            agent_combined = reduce(lambda x, y: pd.merge(x, y, on="agent_name", how="outer"), agent_perf)
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
    st.markdown("### Performance Metrics")
    
    avg_task_duration, task_durations_data = _calculate_task_durations(tasks_f)
    zone_perf_data = _calculate_zone_performance(tasks_f, inter_f)
    agent_activity_data, agent_activity = _calculate_agent_activity(tasks_f, inter_f)
    stock_util_data, stock_utilization_rate = _calculate_stock_utilization(stock_f)
    agent_engagement_rate = _calculate_agent_engagement(tasks_f, active_agents)
    low_performers_data = _calculate_low_performers(agent_activity_data, agent_activity, active_agents)
    best_zone, worst_zone = _calculate_zone_comparison(zone_perf_data)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        _render_task_duration_chart(task_durations_data, avg_task_duration)
    
    with col_chart2:
        _render_zone_performance_chart(zone_perf_data)
    
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        _render_stock_utilization_chart(stock_util_data, stock_utilization_rate)
    
    with col_chart4:
        _render_shift_hours_trend(tasks_f)
    
    _render_low_performers_table(low_performers_data)
    
    interactions_per_task = (len(inter_f) / len(tasks_f)) if not tasks_f.empty and not inter_f.empty else 0
    
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


def _calculate_task_durations(tasks_f):
    """Calculate task duration metrics."""
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
    return avg_task_duration, task_durations_data


def _calculate_zone_performance(tasks_f, inter_f):
    """Calculate zone performance data."""
    zone_perf_data = pd.DataFrame()
    if not inter_f.empty and "zone" in inter_f.columns:
        zone_inter = inter_f.groupby("zone").size().reset_index(name="interactions")
        zone_perf_data = zone_inter
    if not tasks_f.empty and "zone" in tasks_f.columns:
        zone_tasks = tasks_f.groupby("zone").size().reset_index(name="tasks")
        if not zone_perf_data.empty:
            zone_perf_data = zone_perf_data.merge(zone_tasks, on="zone", how="outer")
            zone_perf_data = safe_fillna(zone_perf_data, value=0)
        else:
            zone_perf_data = zone_tasks
    return zone_perf_data


def _calculate_agent_activity(tasks_f, inter_f):
    """Calculate agent activity metrics."""
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
            agent_activity_data = safe_fillna(agent_activity_data, value=0)
            agent_activity_data["total_activity"] = agent_activity_data["interactions"].fillna(0) + agent_activity_data["tasks"].fillna(0)
        else:
            agent_activity_data = tasks_by_agent
            agent_activity_data["total_activity"] = agent_activity_data["tasks"]
        for _, row in tasks_by_agent.iterrows():
            agent_activity[row["agent_name"]] = agent_activity.get(row["agent_name"], 0) + row["tasks"]
    return agent_activity_data, agent_activity


def _calculate_stock_utilization(stock_f):
    """Calculate stock utilization metrics."""
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
    return stock_util_data, stock_utilization_rate


def _calculate_agent_engagement(tasks_f, total_agents):
    """Calculate agent engagement rate."""
    agent_engagement_rate = 0
    if total_agents > 0 and not tasks_f.empty and "agent_name" in tasks_f.columns:
        agents_with_completed = tasks_f[tasks_f["status_norm"] == "completed"]["agent_name"].nunique()
        agent_engagement_rate = (agents_with_completed / total_agents) * 100
    return agent_engagement_rate


def _calculate_low_performers(agent_activity_data, agent_activity, active_agents):
    """Calculate low performers list."""
    low_performers_data = pd.DataFrame()
    if agent_activity and active_agents > 0:
        avg_activity = sum(agent_activity.values()) / len(agent_activity)
        low_performers_list = [agent for agent, count in agent_activity.items() if count < avg_activity]
        if low_performers_list and not agent_activity_data.empty:
            low_performers_data = agent_activity_data[agent_activity_data["agent_name"].isin(low_performers_list)].copy()
            sort_col = "total_activity" if "total_activity" in low_performers_data.columns else "interactions" if "interactions" in low_performers_data.columns else "tasks"
            low_performers_data = low_performers_data.sort_values(sort_col, ascending=True)
    return low_performers_data


def _calculate_zone_comparison(zone_perf_data):
    """Calculate best and worst zones."""
    best_zone = "N/A"
    worst_zone = "N/A"
    if not zone_perf_data.empty:
        if "interactions" in zone_perf_data.columns:
            best_zone = zone_perf_data.loc[zone_perf_data["interactions"].idxmax(), "zone"]
            worst_zone = zone_perf_data.loc[zone_perf_data["interactions"].idxmin(), "zone"]
        elif "tasks" in zone_perf_data.columns:
            best_zone = zone_perf_data.loc[zone_perf_data["tasks"].idxmax(), "zone"]
            worst_zone = zone_perf_data.loc[zone_perf_data["tasks"].idxmin(), "zone"]
    return best_zone, worst_zone


def _render_task_duration_chart(task_durations_data, avg_task_duration):
    """Render task duration distribution chart."""
    st.markdown("#### Task Duration Distribution")
    if not task_durations_data.empty and "duration_hours" in task_durations_data.columns:
        chart_data = optimize_chart_data(task_durations_data, max_points=5000)
        fig = px.histogram(chart_data, x="duration_hours", nbins=20,
                         title=f"Task Duration Distribution (Avg: {avg_task_duration:.1f}h)",
                         labels={"duration_hours": "Duration (hours)", "count": "Number of Tasks"})
        fig = style_chart(fig, f"Task Duration Distribution (Avg: {avg_task_duration:.1f}h)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No task duration data available")


def _render_zone_performance_chart(zone_perf_data):
    """Render zone performance chart."""
    st.markdown("#### Zone Performance")
    if not zone_perf_data.empty:
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


def _render_stock_utilization_chart(stock_util_data, stock_utilization_rate):
    """Render stock utilization chart."""
    st.markdown("#### Stock Utilization by Agent")
    if not stock_util_data.empty:
        stock_util_sorted = stock_util_data.sort_values("utilization_rate", ascending=True).head(15)
        fig = px.bar(stock_util_sorted, x="agent_name", y="utilization_rate",
                    title=f"Stock Utilization Rate (Overall: {stock_utilization_rate:.1f}%)",
                    labels={"utilization_rate": "Utilization Rate (%)", "agent_name": "Agent"})
        fig = style_chart(fig, f"Stock Utilization Rate (Overall: {stock_utilization_rate:.1f}%)")
        fig.update_layout(xaxis_tickangle=45, yaxis_title="Utilization Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No stock utilization data available")


def _render_shift_hours_trend(tasks_f):
    """Render shift hours trend chart."""
    st.markdown("#### Average Shift Hours Trend")
    if not tasks_f.empty and "task_day" in tasks_f.columns and "shift_hours" in tasks_f.columns:
        shift_trend = tasks_f.groupby("task_day")["shift_hours"].mean().reset_index()
        shift_trend.columns = ["date", "avg_shift_hours"]
        avg_shift = tasks_f["shift_hours"].mean()
        fig = px.line(shift_trend, x="date", y="avg_shift_hours",
                     title=f"Average Shift Hours Over Time (Overall: {avg_shift:.1f}h)",
                     labels={"avg_shift_hours": "Avg Shift Hours", "date": "Date"})
        fig = style_chart(fig, f"Average Shift Hours Over Time (Overall: {avg_shift:.1f}h)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No shift hours data available")


def _render_low_performers_table(low_performers_data):
    """Render low performers table."""
    st.markdown("#### Low Performers (Below Average)")
    if not low_performers_data.empty:
        display_cols = [c for c in ["agent_name", "interactions", "tasks", "total_activity"] if c in low_performers_data.columns]
        if display_cols:
            st.dataframe(low_performers_data[display_cols],
                       use_container_width=True, height=300)
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
