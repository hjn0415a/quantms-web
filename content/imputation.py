"""Imputation Page."""

from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column, get_sample_group_map

# Import imputation algorithms from openms_insight engine
from openms_insight.analysis.imputation import impute_mar, impute_smallest_value

STAT_COLUMNS = ["log2FC", "p-value", "p-adj", "stat"]


def strip_stat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep preprocessing tables intensity-only before statistical analysis."""
    return df.drop(columns=[c for c in STAT_COLUMNS if c in df.columns], errors="ignore")

params = page_setup()
st.title("Missing Value Imputation")

st.markdown(
    """
Handle missing values (zeros or nulls) in your quantification matrix using biological group-aware (MAR) or absolute lowest limit (MNAR) techniques.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

# Load base dataset and clean dictionary keys
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

# 1. Pipeline Checkpoint: Fetch upstream filtered data if available, fallback to raw pivot matrix
if "filtered_df" in st.session_state and st.session_state["filtered_df"] is not None:
    base_df = strip_stat_columns(st.session_state["filtered_df"])
    st.session_state["filtered_df"] = base_df
    st.info(
        "🔄 **Upstream Pipeline Detected**: Using data processed from the **Filtering** step."
    )
else:
    base_df = pivot_df
    st.warning(
        "⚠️ **Raw Input Active**: No filtering history found. Operating on the original unfiltered table."
    )

# 2. Identify actual sample columns dynamically based on the current active matrix
sample_cols = [
    c for c in base_df.columns if c not in [id_col, "PeptideSequence", "log2FC", "p-value", "p-adj"]
]

# --- SECTION 1: Input Matrix Summary ---
st.subheader("Input Matrix Overview")
st.markdown(
    f"Currently analyzing **{base_df.shape[0]}** rows across **{len(sample_cols)}** samples before imputation."
)
st.dataframe(base_df, use_container_width=True)

st.markdown("---")

# --- SECTION 2: Imputation Configuration ---
st.subheader("Configure Imputation Engine")

# Build Polars structural metadata DataFrame
metadata_rows = [{"sample_id": s, "group": sample_group_map[s]} for s in sample_cols if s in sample_group_map]
metadata_pl = pl.DataFrame(
    metadata_rows, schema={"sample_id": pl.String, "group": pl.String}
)

# User selection for core missingness assumption strategy
impute_category = st.selectbox(
    "Select Imputation Class",
    options=["MAR (Missing At Random)", "MNAR (Missing Not At Random)"],
    index=0,
    help="MAR uses group metrics (Mean/Median). MNAR shifts values below the limit of detection.",
)

# Render algorithmic options sub-menus based on the parent selection
if impute_category == "MAR (Missing At Random)":
    st.markdown(
        "**Group Character Imputation**: Fills missing metrics leveraging sample properties belonging to the same group."
    )
    strategy_opt = st.radio(
        "Mathematical Strategy",
        options=["median", "mean"],
        index=0,
        horizontal=True,
    )

elif impute_category == "MNAR (Missing Not At Random)":
    st.markdown(
        "**Smallest Value Imputation**: Replaces missing items with the minimum values detected to reflect technical dropout limits."
    )
    scope_opt = st.radio(
        "Detection Minimum Scope",
        options=["row", "global"],
        index=0,
        horizontal=True,
        help="'row' targets current protein minimum; 'global' searches the entire mass spectrometry matrix profile.",
    )

# --- SECTION 3: Imputation Execution ---
if st.button("Apply Imputation", type="primary"):
    # Initialize optimization pipeline graph via lazy loading conversion
    quant_lazy = pl.from_pandas(base_df).lazy()

    # Route configuration matrix parameters to designated engine function channels
    if impute_category == "MAR (Missing At Random)":
        imputed_lazy = impute_mar(
            quantification_data=quant_lazy,
            metadata=metadata_pl,
            group_column="group",
            strategy=strategy_opt,
        )
    elif impute_category == "MNAR (Missing Not At Random)":
        imputed_lazy = impute_smallest_value(
            quantification_data=quant_lazy, metadata=metadata_pl, scope=scope_opt
        )

    # Resolve lazy graph optimization tree and push to display data frame structure
    imputed_df = strip_stat_columns(imputed_lazy.collect().to_pandas())

    # 💾 Save current output into Session State for down-stream processing (Normalization, Statistics)
    st.session_state["imputed_df"] = imputed_df

    st.success(f"Successfully finalized **{impute_category}** imputation step!")

    # Calculate and display a quick performance matrix check
    st.subheader("Imputed Result Table")
    st.dataframe(imputed_df, use_container_width=True)