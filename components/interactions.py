"""Interactions & Brands tab component."""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.charts import style_chart
from utils.data_optimization import limit_display_rows


def render_interactions_tab(inter_f):
    """Render the Interactions & Brands tab."""
    st.subheader("🎯 Interactions & Brands")
    
    if inter_f.empty:
        st.info("No interactions found for selected filters.")
        return
    
    _render_brand_performance(inter_f)
    _render_demographics(inter_f)
    _render_interaction_details(inter_f)


def _render_brand_performance(inter_f):
    """Render brand performance section."""
    st.markdown("### Brand Performance")
    
    if "main_brand" not in inter_f.columns:
        st.info("No brand data available.")
        return
    
    brand_data = inter_f[inter_f["main_brand"].notna() & (inter_f["main_brand"] != "")]
    
    if brand_data.empty:
        st.info("No brand data available for selected filters.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_brands = brand_data.groupby("main_brand").size().reset_index(name="count")
        top_brands = top_brands.sort_values("count", ascending=False).head(15)
        fig = px.bar(top_brands, x="main_brand", y="count",
                    title="Top 15 Brands by Interactions")
        fig = style_chart(fig, "Top 15 Brands by Interactions")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        brand_dist = brand_data["main_brand"].value_counts().head(10).reset_index()
        brand_dist.columns = ["brand", "count"]
        fig = px.pie(brand_dist, names="brand", values="count",
                    title="Top 10 Brands Distribution")
        fig = style_chart(fig, "Top 10 Brands Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    if "interaction_day" in brand_data.columns:
        brand_trends = brand_data.groupby(["interaction_day", "main_brand"]).size().reset_index(name="count")
        brand_trends_pivot = brand_trends.pivot_table(index="interaction_day", columns="main_brand", values="count", fill_value=0).reset_index()
        top_brands_list = top_brands["main_brand"].head(5).tolist()
        if top_brands_list:
            fig = px.line(brand_trends_pivot, x="interaction_day", y=top_brands_list,
                        title="Top 5 Brands Trends Over Time",
                        labels={"value": "Interactions", "interaction_day": "Date", "variable": "Brand"})
            fig = style_chart(fig, "Top 5 Brands Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)


def _render_demographics(inter_f):
    """Render demographics section."""
    st.markdown("### Demographics")
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        if "gender" in inter_f.columns:
            gender_counts = inter_f["gender"].value_counts().reset_index()
            gender_counts.columns = ["gender", "count"]
            fig = px.pie(gender_counts, names="gender", values="count",
                        title="Interactions by Gender")
            fig = style_chart(fig, "Interactions by Gender")
            st.plotly_chart(fig, use_container_width=True)
    
    with colB:
        if "age_range" in inter_f.columns:
            age_counts = inter_f["age_range"].value_counts().reset_index()
            age_counts.columns = ["age_range", "count"]
            fig = px.bar(age_counts, x="age_range", y="count",
                        title="Interactions by Age Range")
            fig = style_chart(fig, "Interactions by Age Range")
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with colC:
        if "interaction_type" in inter_f.columns:
            type_counts = inter_f["interaction_type"].value_counts().reset_index()
            type_counts.columns = ["interaction_type", "count"]
            fig = px.pie(type_counts, names="interaction_type", values="count",
                        title="Interactions by Type")
            fig = style_chart(fig, "Interactions by Type")
            st.plotly_chart(fig, use_container_width=True)
    
    colD, colE = st.columns(2)
    
    with colD:
        if "pack_purchase" in inter_f.columns:
            purchase_data = inter_f[inter_f["pack_purchase"].notna() & (inter_f["pack_purchase"] != "")]
            if not purchase_data.empty:
                purchase_counts = purchase_data["pack_purchase"].value_counts().head(10).reset_index()
                purchase_counts.columns = ["pack_purchase", "count"]
                fig = px.bar(purchase_counts, x="pack_purchase", y="count",
                            title="Top 10 Pack Purchases")
                fig = style_chart(fig, "Top 10 Pack Purchases")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    with colE:
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


def _render_interaction_details(inter_f):
    """Render detailed interactions table."""
    with st.expander("📋 View Detailed Interactions"):
        show_cols = [c for c in [
            "interaction_day", "interaction_id", "agent_name", "place_display", "zone", "area", 
            "channel", "sv", "role", "interaction_type", "gender", "age_range", 
            "main_brand", "pack_purchase", "ecc_item", "url", "location"
        ] if c in inter_f.columns]
        if show_cols:
            inter_display = inter_f[show_cols].sort_values("interaction_day", ascending=False)
            inter_display_limited = limit_display_rows(inter_display, max_rows=1000)
            st.dataframe(inter_display_limited, use_container_width=True, height=500)
            if len(inter_display) > 1000:
                st.caption(f"⚠️ Showing first 1,000 of {len(inter_display):,} rows. Use download button for full data.")
            csv = inter_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Interactions Table",
                data=csv,
                file_name="interactions_detailed.csv",
                mime="text/csv"
            )
