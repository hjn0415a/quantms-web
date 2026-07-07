"""Volcano Plot Results Page."""
import streamlit as st
import polars as pl
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column
from openms_insight import VolcanoPlot

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

# 1. Check if statistical analysis results are available in the session state
if "statistics_df" not in st.session_state or st.session_state["statistics_df"] is None:
    st.info("Statistical analysis data not found. Please run the statistical engine first.")
    st.page_link("content/statistical.py", label="Go to Statistical Inference", icon="🔬")
    st.stop()

# Retrieve the completed statistical analysis DataFrame
statistics_df = st.session_state["statistics_df"]

if statistics_df.empty:
    st.info("No data available for volcano plot.")
    st.stop()

result = get_abundance_data(st.session_state["workspace"])
if result is None:
    st.info("Abundance data not available. Please run the workflow and configure sample groups first.")
    st.page_link("content/results_abundance.py", label="Go to Abundance", icon="📋")
    st.stop()

pivot_df, expr_df, group_map = result
id_col = get_id_column(st.session_state["workspace"], pivot_df)

# 2. Clean data and convert to Polars for component input
volcano_df = statistics_df.dropna(subset=["log2FC", "p-adj"]).copy()
volcano_pl_lazy = pl.from_pandas(volcano_df).lazy()

# 3. Configure UI sliders (changing thresholds does not invalidate cache)
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

# 4. Initialize the OpenMS-Insight VolcanoPlot component
volcano_plot_component = VolcanoPlot(
    cache_id="quantms_volcano_plot",
    data=volcano_pl_lazy,
    log2fc_column="log2FC",
    pvalue_column="p-adj",
    label_column=id_col,
    up_color="#E74C3C",
    down_color="#3498DB",
    ns_color="#95A5A6",
    show_threshold_lines=True,
    threshold_line_style="dash",
)

# 5. Render the component
state_manager = st.session_state.get("state")  # Inject the project state management object

volcano_plot_component(
    state_manager=state_manager,
    fc_threshold=fc_thresh,
    p_threshold=p_thresh,
    max_labels=10,  # Display labels for the top N significant proteins
    height=600,
)

# 6. Keep the existing statistical summary and bottom links
up_count = ((volcano_df["p-adj"] <= p_thresh) & (volcano_df["log2FC"] >= fc_thresh)).sum()
down_count = ((volcano_df["p-adj"] <= p_thresh) & (volcano_df["log2FC"] <= -fc_thresh)).sum()
st.markdown(f"**Up-regulated:** {up_count} | **Down-regulated:** {down_count}")

st.markdown("---")
st.markdown("**Other visualizations:**")
col1, col2 = st.columns(2)
with col1:
    st.page_link("content/results_pca.py", label="PCA", icon="📊")
with col2:
    st.page_link("content/results_heatmap.py", label="Heatmap", icon="🔥")
