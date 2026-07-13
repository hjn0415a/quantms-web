"""Normalization Page."""

from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column, get_sample_group_map
# Import normalization engine functions from openms_insight
from openms_insight.analysis.normalization import (
    normalize_samples,
    scale_data,
    transform_data,
)

STAT_COLUMNS = ["log2FC", "p-value", "p-adj", "stat"]


def strip_stat_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Keep preprocessing tables intensity-only before statistical analysis."""
    if df is None:
        return None
    return df.drop(columns=[c for c in STAT_COLUMNS if c in df.columns], errors="ignore")

params = page_setup()
st.title("Data Normalization & Scaling")

st.markdown(
    """
Standardize and transform your protein abundance profiles to correct for technical variations and optimize statistical distributions.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

# Load primary database assets
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

filtered_df = strip_stat_columns(st.session_state.get("filtered_df"))
imputed_df = strip_stat_columns(st.session_state.get("imputed_df"))
normalized_df = strip_stat_columns(st.session_state.get("normalized_df"))
if filtered_df is not None:
    st.session_state["filtered_df"] = filtered_df
if imputed_df is not None:
    st.session_state["imputed_df"] = imputed_df
if normalized_df is not None:
    st.session_state["normalized_df"] = normalized_df

# --- STEP 1: Upstream Pipeline Tracker (Fallback Architecture) ---
if (
    "imputed_df" in st.session_state
    and st.session_state["imputed_df"] is not None
):
    base_df = imputed_df
    st.info(
        "🔄 **Upstream Pipeline Detected**: Using data processed from the **Imputation** step."
    )
elif (
    "filtered_df" in st.session_state
    and st.session_state["filtered_df"] is not None
):
    base_df = filtered_df
    st.warning(
        "⚠️ **Imputation Skipped**: Using data processed from the **Filtering** step."
    )
else:
    base_df = pivot_df
    st.warning(
        "⚠️ **Raw Input Active**: No preprocessing history found. Operating on the original unfiltered table."
    )

# 2. Extract actual active sample columns dynamically
sample_cols = [
    c for c in base_df.columns if c not in [id_col, "PeptideSequence", "log2FC", "p-value", "p-adj"]
]

# --- SECTION 1: Active Input Table Preview ---
st.subheader("Input Table Overview")
st.markdown(
    f"Currently displaying **{base_df.shape[0]}** rows and **{len(sample_cols)}** samples entering the normalization block."
)
st.dataframe(base_df, use_container_width=True)

st.markdown("### Pipeline Overview")
st.caption("Data flows in order: Filtering -> Imputation -> Normalization")

step_rows = [
    {
        "Step": "Filtering",
        "Status": "Done" if filtered_df is not None else "Not run",
        "Rows": filtered_df.shape[0] if filtered_df is not None else "-",
        "Cols": filtered_df.shape[1] if filtered_df is not None else "-",
    },
    {
        "Step": "Imputation",
        "Status": "Done" if imputed_df is not None else "Not run",
        "Rows": imputed_df.shape[0] if imputed_df is not None else "-",
        "Cols": imputed_df.shape[1] if imputed_df is not None else "-",
    },
    {
        "Step": "Normalization",
        "Status": "Done" if normalized_df is not None else "Not run",
        "Rows": normalized_df.shape[0] if normalized_df is not None else "-",
        "Cols": normalized_df.shape[1] if normalized_df is not None else "-",
    },
]
st.dataframe(pd.DataFrame(step_rows), hide_index=True, use_container_width=True)

with st.expander("Show step tables", expanded=False):
    if filtered_df is not None:
        st.markdown("#### Filtering output")
        st.dataframe(filtered_df.head(10), use_container_width=True)
    if imputed_df is not None:
        st.markdown("#### Imputation output")
        st.dataframe(imputed_df.head(10), use_container_width=True)
    if normalized_df is not None:
        st.markdown("#### Normalization output")
        st.dataframe(normalized_df.head(10), use_container_width=True)
    if filtered_df is None and imputed_df is None and normalized_df is None:
        st.info("No preprocessing outputs yet. Start from Filtering.")

st.markdown("---")

# --- SECTION 2: Normalization Parameter Configuration ---
st.subheader("Configure Preprocessing & Scaling Chains")

# Prepare structural Polars metadata DataFrame required by backend functions
metadata_rows = [{"sample_id": s, "group": sample_group_map[s]} for s in sample_cols if s in sample_group_map]
metadata_pl = pl.DataFrame(
    metadata_rows, schema={"sample_id": pl.String, "group": pl.String}
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧬 1. Mathematical Transformation")
    transform_strategy = st.selectbox(
        "Select Transformation",
        options=["None", "log2", "log10", "square_root", "cube_root"],
        index=0,
        help="Compress data dynamic range and stabilize heteroscedastic variance profiles.",
    )

with col2:
    st.markdown("### 🧪 2. Sample Normalization")
    norm_strategy = st.selectbox(
        "Select Normalization",
        options=["None", "sum", "median", "pqn", "reference_feature", "quantile"],
        index=0,
        help="Perform column-wise corrections to account for variable sample loading concentrations.",
    )

    # Conditionally display target input field for reference feature matching
    ref_feature_input = None
    if norm_strategy == "reference_feature":
        ref_feature_input = st.text_input(
            "Reference Protein Name (ID)",
            value="",
            placeholder="e.g., P01234 or GAPDH",
            help=f"Enter the exact unique identifier string matching a key inside the '{id_col}' column.",
        )

with col3:
    st.markdown("### 📊 3. Row Scaling")
    scaling_strategy = st.selectbox(
        "Select Scaling Mode",
        options=["None", "mean_centering", "auto_scaling", "pareto_scaling", "range_scaling"],
        index=0,
        help="Adjust individual feature weights to make low and high abundance proteins comparable.",
    )


# --- SECTION 3: Normalization Pipe Sequential Execution ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Apply Normalization Pipelines", type="primary"):

    # Validate reference feature selection if active before hitting polars execution layers
    if norm_strategy == "reference_feature" and not ref_feature_input:
        st.error(
            "❌ Validation Error: Please provide a valid Reference Protein Name to use the 'reference_feature' strategy."
        )
        st.stop()

    # Convert pandas memory buffer into optimization lazy dataframe tree graph
    processing_lazy = pl.from_pandas(base_df).lazy()

    # Execute Chain 1: Transform Matrix Data
    try:
        processing_lazy = transform_data(
            quantification_data=processing_lazy,
            metadata=metadata_pl,
            strategy=transform_strategy,
        )

        # Execute Chain 2: Normalize Sample Intensities (Columns)
        processing_lazy = normalize_samples(
            quantification_data=processing_lazy,
            metadata=metadata_pl,
            strategy=norm_strategy,
            id_col=id_col,
            reference_feature=ref_feature_input if norm_strategy == "reference_feature" else None,
        )

        # Execute Chain 3: Scale Individual Features (Rows)
        processing_lazy = scale_data(
            quantification_data=processing_lazy,
            metadata=metadata_pl,
            strategy=scaling_strategy,
        )

        # Finalize and collect pipeline query graph optimizations
        normalized_df = strip_stat_columns(processing_lazy.collect().to_pandas())

        # 💾 Save processing checkpoint inside Session State for Downstream (Statistics Block)
        st.session_state["normalized_df"] = normalized_df

        st.success("Successfully executed all selected normalization pipelines!")

        # Display the finalized transformation matrix view
        st.subheader("Normalized Abundance Table")
        st.dataframe(normalized_df, use_container_width=True)

    except ValueError as val_err:
        # Gracefully handle validation failures raised from the engine layers (e.g., missing reference protein)
        st.error(f"Engine Configuration Error: {str(val_err)}")
    except Exception as e:
        st.error(f"An unexpected pipeline error occurred: {str(e)}")