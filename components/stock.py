"""Stock & Inventory tab component."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.charts import style_chart
from utils.data_optimization import limit_display_rows


def render_stock_tab(stock_f, stock_check_f):
    """Render the Stock & Inventory tab."""
    st.subheader("📦 Stock & Inventory")
    
    if stock_f.empty and stock_check_f.empty:
        st.info("No stock data available for the selected filters.")
        return
    
    _render_stock_overview(stock_f)
    st.markdown("---")
    _render_stock_analysis(stock_check_f)
    st.markdown("---")
    _render_stock_alerts(stock_check_f)


def _render_stock_overview(stock_f):
    """Render stock overview section."""
    st.markdown("### Stock Overview")
    
    if stock_f.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
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
        total_release = float(stock_f["qty_release"].sum()) if "qty_release" in stock_f.columns else 0
        total_return = float(stock_f["qty_return"].sum()) if "qty_return" in stock_f.columns else 0
        total_used = float(stock_f["qty_used"].sum()) if "qty_used" in stock_f.columns else 0
        net_stock = total_release - total_return - total_used
        
        st.metric("Total Released", f"{total_release:,.0f}")
        st.metric("Total Returned", f"{total_return:,.0f}")
        st.metric("Total Used", f"{total_used:,.0f}")
        st.metric("Net Stock", f"{net_stock:,.0f}")


def _render_stock_analysis(stock_check_f):
    """Render stock analysis section."""
    st.markdown("### Used Stock (Release - Back) vs Interactions Analysis")
    
    if stock_check_f.empty:
        return
    
    colA, colB = st.columns(2)
    
    with colA:
        by_agent = stock_check_f.groupby("agent_name").agg(
            used=("qty_used", "sum"),
            interactions=("interactions_total", "sum"),
            diff=("diff_used_vs_interactions", "sum")
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


def _render_stock_alerts(stock_check_f):
    """Render stock alerts section."""
    st.markdown("### Stock Alerts")
    
    if stock_check_f.empty:
        return
    
    mism = stock_check_f[stock_check_f["stock_flag_interactions"] != "OK"]
    
    if not mism.empty:
        st.warning(f"⚠️ Found {len(mism)} stock issue(s): Interactions exceed used stock (Release - Back)")
        
        alert_cols = [c for c in ["agent_name", "stock_day", "qty_release", "qty_return", "qty_used", 
                                   "interactions_total", "diff_used_vs_interactions", "stock_flag_interactions"]
                     if c in mism.columns]
        if alert_cols:
            mism_display = mism[alert_cols].sort_values("diff_used_vs_interactions")
            st.dataframe(mism_display, use_container_width=True, height=300)
            csv = mism_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Stock Alerts Table",
                data=csv,
                file_name="stock_alerts.csv",
                mime="text/csv"
            )
    else:
        st.success("✅ No stock issues: All interactions are within used stock limits (Release - Back).")
    
    with st.expander("📋 View Detailed Stock Data"):
        if not stock_check_f.empty:
            cols = [c for c in [
                "agent_name", "stock_day", "qty_release", "qty_return", "qty_used",
                "interactions_total", "diff_used_vs_interactions", "stock_flag_interactions",
                "bbos_pack", "diff_used_vs_bbos_pack", "stock_flag_bbos_pack",
            ] if c in stock_check_f.columns]
            if cols:
                stock_display = stock_check_f[cols].sort_values(["stock_day", "agent_name"], ascending=[False, True])
                st.dataframe(stock_display, use_container_width=True, height=500)
                csv = stock_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Detailed Stock Data Table",
                    data=csv,
                    file_name="stock_detailed.csv",
                    mime="text/csv"
                )
