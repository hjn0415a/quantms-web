"""Filtering Page."""

from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column, get_sample_group_map

# Import filtering functions from openms_insight package
from openms_insight.analysis.filter import (
    filter_low_abundance,
    filter_low_repeatability,
    filter_low_variance,
)

STAT_COLUMNS = ["log2FC", "p-value", "p-adj", "stat"]


def strip_stat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep preprocessing tables intensity-only before statistical analysis."""
    return df.drop(columns=[c for c in STAT_COLUMNS if c in df.columns], errors="ignore")

params = page_setup()
st.title("Data Filtering")

st.markdown(
    """
Filter out low-quality proteins from your dataset based on abundance, repeatability, or variance thresholds.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

result = get_abundance_data(st.session_state["workspace"])
if result is None:
    st.info(
        "Abundance data not available. Please run the workflow and configure sample groups first."
    )
    st.page_link(
        "content/results_abundance.py", label="Go to Abundance", icon="📋"
    )
    st.stop()

pivot_df, expr_df, group_map = result
pivot_df = strip_stat_columns(pivot_df)
id_col = get_id_column(st.session_state["workspace"], pivot_df)
sample_group_map = get_sample_group_map(st.session_state["workspace"], pivot_df, group_map)

# 1. Identify actual sample columns dynamically
sample_cols = [
    c
    for c in pivot_df.columns
    if c not in [id_col, "PeptideSequence", "log2FC", "p-value", "p-adj"]
]

# --- SECTION 1: Original Data View ---
st.subheader("Original Abundance Table")
st.markdown(
    f"Currently displaying **{pivot_df.shape[0]}** proteins and **{len(sample_cols)}** samples before filtering."
)
st.dataframe(pivot_df, use_container_width=True)

st.markdown("---")

# --- SECTION 2: Filter Configuration ---
st.subheader("Configure Filter Engine")

# Prepare Polars Metadata DataFrame required by openms_insight functions
metadata_rows = [{"sample_id": s, "group": sample_group_map[s]} for s in sample_cols if s in sample_group_map]
metadata_pl = pl.DataFrame(
    metadata_rows, schema={"sample_id": pl.String, "group": pl.String}
)

# User selection for filtering strategy
filter_method = st.selectbox(
    "Select Filtering Method",
    options=["Low Abundance", "Low Repeatability", "Low Variance"],
    index=0,
    help="Choose the statistical criteria to prune unreliable protein entries.",
)

# Render threshold sliders dynamically based on the selected filter method
if filter_method == "Low Abundance":
    st.markdown(
        "**Low Abundance Filter**: Keeps rows where at least one group's median is above the selected percentile threshold."
    )
    threshold = st.slider(
        "Threshold Percentile (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=5.0,
    )

elif filter_method == "Low Repeatability":
    st.markdown(
        "**Low Repeatability Filter**: Keeps rows where at least one group has a missing value ratio within the allowed maximum."
    )
    threshold = st.slider(
        "Max Missing Ratio",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Allowed missing value (zero or null) ratio per group.",
    )

elif filter_method == "Low Variance":
    st.markdown(
        "**Low Variance Filter**: Keeps rows where at least one group's variance is above the selected percentile threshold."
    )
    threshold = st.slider(
        "Threshold Percentile (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=5.0,
    )

# --- SECTION 3: Filter Execution and Collected Results View ---
if st.button("Apply Filter", type="primary"):
    # Convert the original Pandas DataFrame into a Polars LazyFrame graph
    quant_lazy = pl.from_pandas(pivot_df).lazy()

    # Route execution to the chosen openms_insight engine function
    if filter_method == "Low Abundance":
        filtered_lazy = filter_low_abundance(
            quantification_data=quant_lazy,
            metadata=metadata_pl,
            group_column="group",
            threshold_percentile=threshold,
        )
    elif filter_method == "Low Repeatability":
        # Convert percent slider input to ratio expected by the function (e.g., 50.0% -> 0.5)
        filtered_lazy = filter_low_repeatability(
            quantification_data=quant_lazy,
            metadata=metadata_pl,
            group_column="group",
            max_missing_ratio=threshold / 100.0,
        )
    elif filter_method == "Low Variance":
        filtered_lazy = filter_low_variance(
            quantification_data=quant_lazy,
            metadata=metadata_pl,
            group_column="group",
            threshold_percentile=threshold,
        )

    # Collect the evaluated lazy graph and convert back to Pandas for visualization
    filtered_df = strip_stat_columns(filtered_lazy.collect().to_pandas())
    st.session_state["filtered_df"] = filtered_df

    # Layout response metrics and the filtered matrix
    st.success(f"Successfully applied **{filter_method}** filter!")

    # Display dataset scale compression stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Proteins", pivot_df.shape[0])
    col2.metric("Filtered Proteins", filtered_df.shape[0])
    col3.metric(
        "Removed Proteins", pivot_df.shape[0] - filtered_df.shape[0], delta=None
    )

    st.subheader("Filtered Abundance Table")
    if filtered_df.empty:
        st.warning(
            "The filtered table is empty. Try relaxing the threshold constraints."
        )
    else:
        st.dataframe(filtered_df, use_container_width=True)