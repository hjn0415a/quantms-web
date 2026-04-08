"""Volcano Plot Results Page."""
import streamlit as st
import plotly.express as px
import numpy as np
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data

params = page_setup()
st.title("Volcano Plot")

st.markdown(
    """
Visualize differential expression analysis with a volcano plot.
Points represent proteins colored by significance status.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

result = get_abundance_data(st.session_state["workspace"])
if result is None:
    st.info("Abundance data not available. Please run the workflow and configure sample groups first.")
    st.page_link("content/results_abundance.py", label="Go to Abundance", icon="📋")
    st.stop()

pivot_df, expr_df, group_map = result

if pivot_df.empty:
    st.info("No data available for volcano plot.")
    st.stop()

volcano_df = pivot_df.copy()
volcano_df = volcano_df.dropna(subset=["log2FC", "p-adj"])

volcano_df["neg_log10_padj"] = -np.log10(volcano_df["p-adj"])

fc_thresh = st.slider(
    "log2 Fold Change threshold",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
)

p_thresh = st.slider(
    "p-adj (FDR) threshold",
    min_value=0.001,
    max_value=0.1,
    value=0.05,
    step=0.001,
)

volcano_df["Significance"] = "Not significant"
volcano_df.loc[
    (volcano_df["p-adj"] <= p_thresh) & (volcano_df["log2FC"] >= fc_thresh),
    "Significance",
] = "Up-regulated"

volcano_df.loc[
    (volcano_df["p-adj"] <= p_thresh) & (volcano_df["log2FC"] <= -fc_thresh),
    "Significance",
] = "Down-regulated"

fig_volcano = px.scatter(
    volcano_df,
    x="log2FC",
    y="neg_log10_padj",
    color="Significance",
    hover_data=["ProteinName", "log2FC", "p-value", "p-adj"],
    color_discrete_map={
        "Up-regulated": "red",
        "Down-regulated": "blue",
        "Not significant": "lightgrey",
    }
)

fig_volcano.add_vline(x=fc_thresh, line_dash="dash")
fig_volcano.add_vline(x=-fc_thresh, line_dash="dash")
fig_volcano.add_hline(y=-np.log10(p_thresh), line_dash="dash")

# Make x-axis symmetric around zero
max_abs_fc = volcano_df["log2FC"].abs().max()
x_range = [-max_abs_fc * 1.1, max_abs_fc * 1.1]  # 10% padding

fig_volcano.update_layout(
    xaxis_title="log2 Fold Change",
    yaxis_title="-log10(p-adj)",
    xaxis_range=x_range,
    height=600,
)

st.plotly_chart(fig_volcano, use_container_width=True)

up_count = (volcano_df["Significance"] == "Up-regulated").sum()
down_count = (volcano_df["Significance"] == "Down-regulated").sum()
st.markdown(f"**Up-regulated:** {up_count} | **Down-regulated:** {down_count}")

st.markdown("---")
st.markdown("**Other visualizations:**")
col1, col2 = st.columns(2)
with col1:
    st.page_link("content/results_pca.py", label="PCA", icon="📊")
with col2:
    st.page_link("content/results_heatmap.py", label="Heatmap", icon="🔥")
