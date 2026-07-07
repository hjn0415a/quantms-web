"""Clustered Heatmap Results Page."""
import streamlit as st
import numpy as np
import polars as pl
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column, get_sample_group_map
from openms_insight import ClusteredHeatmap

params = page_setup()
st.title("Clustered Heatmap")

st.markdown(
    """
A real grid heatmap (rows = proteins, columns = samples) with hierarchical
clustering dendrograms on both axes and a sample-group color bar, powered
by OpenMS-Insight.
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
id_col = get_id_column(st.session_state["workspace"], pivot_df)
sample_group_map = get_sample_group_map(st.session_state["workspace"], pivot_df, group_map)

if expr_df.empty:
    st.info("No data available for heatmap.")
    st.stop()

top_n = st.slider("Number of proteins (Highest Variance)", 10, 200, 30, key="clustered_heatmap_top_n")

var_series = expr_df.var(axis=1)
top_proteins = var_series.sort_values(ascending=False).head(top_n).index
heatmap_df = expr_df.loc[top_proteins]

heatmap_z = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0).div(heatmap_df.std(axis=1), axis=0)
heatmap_z = heatmap_z.replace([np.inf, -np.inf], np.nan).dropna()

if heatmap_z.empty:
    st.warning("Insufficient data to generate the heatmap.")
    st.stop()

heatmap_z_reset = heatmap_z.reset_index()
heatmap_lazy = pl.from_pandas(heatmap_z_reset).lazy()

sample_cols = expr_df.columns.tolist()
metadata_pl = pl.DataFrame(
    [{"sample_id": s, "group": sample_group_map[s]} for s in sample_cols if s in sample_group_map],
    schema={"sample_id": pl.String, "group": pl.String},
)

# Assign group annotation-bar colors in sorted-group order (matching how
# ClusteredHeatmap._preprocess() orders unique groups internally).
group_palette = [
    "#00BFC4",  # teal
    "#F8766D",  # salmon
    "#7CAE00",  # yellow-green
    "#C77CFF",  # lavender purple
    "#E7B800",  # gold/amber
    "#619CFF",  # blue
    "#FF61C3",  # pink/magenta
    "#00BA38",  # green
    "#FF8C42",  # orange
    "#00B0F6",  # sky blue
]
unique_groups = sorted(set(sample_group_map.values()))
group_colors = {g: group_palette[i % len(group_palette)] for i, g in enumerate(unique_groups)}

heatmap_component = ClusteredHeatmap(
    cache_id="quantms_clustered_heatmap",
    cache_path=str(st.session_state["workspace"]),
    id_col=id_col,
    data=heatmap_lazy,
    metadata=metadata_pl,
    row_cluster=True,
    col_cluster=True,
    title="Protein Abundance Heatmap (Z-score, clustered)",
    x_label="Samples",
    y_label="Proteins",
    colorscale=[[0, "#6699E0"], [0.5, "#FFFFFF"], [1, "#E06666"]],
    reversescale=False,
    intensity_label="Z-score",
    group_colors=group_colors,
)

state_manager = st.session_state.get("state")
# Scale height with the number of proteins so row labels stay readable -
# BaseComponent otherwise defaults to a flat 400px, too short for a
# dendrogram+heatmap composite with more than a handful of rows.
heatmap_height = max(600, min(1400, 300 + top_n * 20))
heatmap_component(state_manager=state_manager, height=heatmap_height)

st.markdown("---")
st.markdown("**Other visualizations:**")
col1, col2 = st.columns(2)
with col1:
    st.page_link("content/results_volcano.py", label="Volcano Plot", icon="🌋")
with col2:
    st.page_link("content/results_heatmap.py", label="Heatmap (original)", icon="🔥")
