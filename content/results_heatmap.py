"""Heatmap Results Page."""
import streamlit as st
import numpy as np
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from src.common.common import page_setup

params = page_setup()
st.title("Heatmap")

st.markdown(
    """
Hierarchically clustered heatmap of protein-level abundance (Z-score normalized).
Proteins and samples are ordered by similarity.
"""
)

if (
    "pivot_df" not in st.session_state
    or "expr_df" not in st.session_state
    or "group_map" not in st.session_state
):
    st.info("Abundance data not loaded. Please visit the Abundance page first.")
    st.page_link("content/results_abundance.py", label="Go to Abundance", icon="📋")
    st.stop()

expr_df = st.session_state["expr_df"]

top_n = st.slider("Number of proteins", 20, 200, 50, key="heatmap_top_n")

var_series = expr_df.var(axis=1)
top_proteins = var_series.sort_values(ascending=False).head(top_n).index
heatmap_df = expr_df.loc[top_proteins]
heatmap_z = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0).div(heatmap_df.std(axis=1), axis=0)
heatmap_z = heatmap_z.replace([np.inf, -np.inf], np.nan).dropna()

if not heatmap_z.empty:
    row_linkage = linkage(pdist(heatmap_z.values), method="average")
    row_order = leaves_list(row_linkage)

    col_linkage = linkage(pdist(heatmap_z.T.values), method="average")
    col_order = leaves_list(col_linkage)

    heatmap_clustered = heatmap_z.iloc[row_order, col_order]

    fig_heatmap = px.imshow(
        heatmap_clustered,
        labels=dict(x="Sample", y="Protein", color="Z-score"),
        aspect="auto",
        color_continuous_scale=[[0.0, "#3b6fb6"], [0.5, "white"], [1.0, "#b40426"]],
        zmin=-3, zmax=3
    )

    fig_heatmap.update_layout(
        height=700,
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )

    fig_heatmap.update_xaxes(tickfont=dict(size=10))
    fig_heatmap.update_yaxes(tickfont=dict(size=8))

    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.warning("Insufficient data to generate the heatmap.")

st.markdown("---")
st.markdown("**Other visualizations:**")
col1, col2 = st.columns(2)
with col1:
    st.page_link("content/results_volcano.py", label="Volcano Plot", icon="🌋")
with col2:
    st.page_link("content/results_pca.py", label="PCA", icon="📊")
