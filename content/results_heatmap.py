"""Heatmap Results Page."""
import streamlit as st
import numpy as np
import polars as pl
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column, get_sample_group_map
from openms_insight import Heatmap

params = page_setup()
st.title("Heatmap")

st.markdown(
    """
Interactive hierarchically clustered heatmap of protein-level abundance (Z-score normalized).
Powered by OpenMS-Insight multi-resolution engine.
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

sample_cols = expr_df.columns.tolist()

# UI settings (number of top variance proteins)
top_n = st.slider("Number of proteins (Highest Variance)", 20, 200, 50, key="heatmap_top_n")

# Process data (variance selection -> Z-score normalization)
var_series = expr_df.var(axis=1)
top_proteins = var_series.sort_values(ascending=False).head(top_n).index
heatmap_df = expr_df.loc[top_proteins]

# Compute Z-scores and clean missing/invalid values
heatmap_z = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0).div(heatmap_df.std(axis=1), axis=0)
heatmap_z = heatmap_z.replace([np.inf, -np.inf], np.nan).dropna()

if not heatmap_z.empty:
    # Melt and convert data to Polars to satisfy OpenMS-Insight component requirements
    # Restore the id column from the index as a regular column
    heatmap_z_reset = heatmap_z.reset_index()

    # Unpivot the wide-format matrix into long-format (X, Y, Intensity)
    melted_df = heatmap_z_reset.melt(
        id_vars=[id_col],
        value_vars=sample_cols,
        var_name="Sample",
        value_name="Z_score"
    )

    # Add sample group mapping if available for heatmap categories
    if sample_group_map:
        melted_df["Group"] = melted_df["Sample"].map(sample_group_map)

    # Pack the Pandas DataFrame into a Polars LazyFrame
    heatmap_pl_lazy = pl.from_pandas(melted_df).lazy()

    # Initialize the OpenMS-Insight Heatmap component and map attributes
    heatmap_component = Heatmap(
        cache_id="quantms_protein_heatmap",
        x_column="Sample",
        y_column=id_col,
        data=heatmap_pl_lazy,
        intensity_column="Z_score",
        title="Protein Abundance Heatmap (Z-score)",
        x_label="Samples",
        y_label="Proteins",
        colorscale="RdBu",
        reversescale=True,
        log_scale=False,           # Z-score can be negative, so log scale must stay off
        intensity_label="Z-score",
        category_column=None,
        min_points=10000,          # Generous point-count ceiling so the full grid renders
    )

    # Render the component
    state_manager = st.session_state.get("state")
    heatmap_component(state_manager=state_manager)
else:
    st.warning("Insufficient data to generate the heatmap.")

st.markdown("---")
st.markdown("**Other visualizations:**")
col1, col2 = st.columns(2)
with col1:
    st.page_link("content/results_volcano.py", label="Volcano Plot", icon="🌋")
with col2:
    st.page_link("content/results_pca.py", label="PCA", icon="📊")
