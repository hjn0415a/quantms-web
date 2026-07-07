"""Statistical Inference Page."""

from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column, get_sample_group_map
# Import statistics engine functions from openms_insight
from openms_insight.analysis.statistics import calculate_statistical_tests, adjust_fdr_lazy

params = page_setup()
st.title("Statistical Inference")

st.markdown(
    """
Run differential expression analysis to identify statistically significant proteins across your biological groups.
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
id_col = get_id_column(st.session_state["workspace"], pivot_df)
sample_group_map = get_sample_group_map(st.session_state["workspace"], pivot_df, group_map)

# --- STEP 1: Upstream Pipeline Tracker (Fallback Architecture) ---
if (
    "normalized_df" in st.session_state
    and st.session_state["normalized_df"] is not None
):
    base_df = st.session_state["normalized_df"]
    st.info(
        "🔄 **Upstream Pipeline Detected**: Using data processed from the **Normalization** step."
    )
elif (
    "imputed_df" in st.session_state
    and st.session_state["imputed_df"] is not None
):
    base_df = st.session_state["imputed_df"]
    st.warning(
        "⚠️ **Normalization Skipped**: Using data processed from the **Imputation** step."
    )
elif (
    "filtered_df" in st.session_state
    and st.session_state["filtered_df"] is not None
):
    base_df = st.session_state["filtered_df"]
    st.warning(
        "⚠️ **Preprocessing Skipped**: Using data processed from the **Filtering** step."
    )
else:
    base_df = pivot_df
    st.warning(
        "⚠️ **Raw Input Active**: No preprocessing history found. Operating on the original table."
    )

# 2. Extract actual active sample columns and detect unique biological groups
sample_cols = [
    c for c in base_df.columns if c not in [id_col, "PeptideSequence", "log2FC", "p-value", "p-adj"]
]
unique_groups = sorted(list(set([sample_group_map[s] for s in sample_cols if s in sample_group_map])))
group_count = len(unique_groups)

# --- SECTION 1: Active Input Table Preview ---
st.subheader("Input Table Overview")
st.markdown(
    f"Currently analyzing **{base_df.shape[0]}** rows across **{len(sample_cols)}** samples belonging to **{group_count} groups** ({', '.join(unique_groups)})."
)
st.dataframe(base_df, use_container_width=True)

st.markdown("---")

# --- SECTION 2: Dynamic Statistical Parameter Configuration ---
st.subheader("Configure Statistical Engine")

# Prepare structural Polars metadata DataFrame required by backend functions
metadata_rows = [{"sample_id": s, "group": sample_group_map[s]} for s in sample_cols if s in sample_group_map]
metadata_pl = pl.DataFrame(
    metadata_rows, schema={"sample_id": pl.String, "group": pl.String}
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔬 1. Hypothesis Testing Method")
    
    # Route available method options dynamically based on the group count
    if group_count == 2:
        method_options = ["limma_like", "welch", "paired"]
        help_text = "'limma_like' uses Empirical Bayes variance shrinking. 'welch' is for unequal variances. 'paired' is for dependent samples."
    elif group_count >= 3:
        method_options = ["limma_like", "anova"]
        help_text = "'limma_like' supports multi-group design matrices. 'anova' computes standard row-wise One-way ANOVA."
    else:
        st.error("❌ Statistical testing requires at least 2 unique sample groups.")
        st.stop()
        
    selected_method = st.selectbox(
        "Select Statistical Test",
        options=method_options,
        index=0,
        help=help_text
    )

with col2:
    st.markdown("### 🛡️ 2. Multiple Testing Correction (FDR)")
    selected_fdr = st.selectbox(
        "Select FDR Adjustment Strategy",
        options=["BH", "Bonferroni", "None"],
        index=0,
        help="'BH' (Benjamini-Hochberg) controls False Discovery Rate. 'Bonferroni' is strict Family-Wise Error Rate control."
    )

# --- SECTION 3: Statistical Query Execution ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Run Statistical Analysis", type="primary"):
    
    # Convert active pandas dataframe into polars lazyframe graph
    stats_lazy = pl.from_pandas(base_df).lazy()
    
    try:
        # Execute Chain 1: Calculate core statistics (Adds log2FC, stat, p-value)
        stats_lazy = calculate_statistical_tests(
            quantification_data=stats_lazy,
            metadata=metadata_pl,
            method=selected_method
        )
        
        # Execute Chain 2: Adjust Multiple Testing (Adds p-adj)
        stats_lazy = adjust_fdr_lazy(
            quantification_data=stats_lazy,
            strategy=selected_fdr
        )
        
        # Resolve lazy graph optimization tree and bring back to pandas memory
        statistics_df = stats_lazy.collect().to_pandas()
        
        # 💾 Save processing checkpoint inside Session State for Downstream (e.g., Volcano plot, Volcano/Heatmap UI)
        st.session_state["statistics_df"] = statistics_df
        
        st.success(f"Successfully calculated **{selected_method}** test with **{selected_fdr}** FDR correction!")
        
        # Display the finalized statistics table view
        st.subheader("Statistical Analysis Results")
        st.markdown(f"Generated framework containing columns: `{id_col}`, `log2FC`, `stat`, `p-value`, `p-adj`")
        st.dataframe(statistics_df, use_container_width=True)
        
    except ValueError as val_err:
        st.error(f"Engine Validation Fallure: {str(val_err)}")
    except Exception as e:
        st.error(f"An unexpected pipeline error occurred: {str(e)}")