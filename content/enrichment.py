"""Pathway Analysis Page."""

from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data, get_id_column
# Import GO Enrichment modules from openms_insight engine
from openms_insight.analysis.enrichment import calculate_go_enrichment

params = page_setup()
st.title("GO Enrichment Analysis")

st.markdown(
    """
Identify overrepresented biological themes (BP, CC, MF) within your differentially expressed protein features using MyGene.info and Fisher's Exact Test.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

# --- STEP 1: Upstream Statistics Checkpoint ---
if (
    "statistics_df" in st.session_state
    and st.session_state["statistics_df"] is not None
):
    final_statistics_report = st.session_state["statistics_df"]
    st.info(
        "🔄 **Upstream Pipeline Detected**: Using analyzed matrices from the **Statistical Inference** step."
    )
else:
    st.warning(
        "⚠️ **Missing Prerequisites**: Statistical inference data not detected. Please run hypothesis testing first."
    )
    st.page_link(
        "content/statistical.py", label="Go to Statistical Inference", icon="🔬"
    )
    st.stop()

# --- STEP 2: Preprocessing Mapping Key Configuration ---
# Identify target identifier columns dynamically
abundance_result = get_abundance_data(st.session_state["workspace"])
id_col = get_id_column(st.session_state["workspace"], abundance_result[0]) if abundance_result else "ProteinName"
if id_col not in final_statistics_report.columns:
    st.error(f"❌ Structural Error: Column '{id_col}' is missing from the active matrix context.")
    st.stop()

# --- SECTION 1: Parameter Setup & Dynamic Cutoff Labels ---
st.subheader("Configure Enrichment Thresholds")

# Check if target p-value should be adjusted or raw based on previous selections (Fallback safely to 'p-adj')
target_p_col = "p-adj" if "p-adj" in final_statistics_report.columns else "p-value"
p_label = (
    "Adjusted P-value (p-adj) Cutoff"
    if target_p_col == "p-adj"
    else "Raw P-value (p-value) Cutoff"
)

ui_go_col1, ui_go_col2 = st.columns(2)

with ui_go_col1:
    p_cutoff = st.number_input(
        f"🔬 {p_label}",
        min_value=0.0001,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.4f",
        help="Proteins with significance metrics below this value are mapped to the foreground cohort.",
    )

with ui_go_col2:
    fc_cutoff = st.number_input(
        "📈 Absolute Difference Cutoff (|log2FC|)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        help="Proteins with absolute log2 fold change greater than or equal to this threshold will be selected.",
    )

# --- SECTION 2: Execution and Interactive View Charts ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Run GO Enrichment Analysis", type="primary", key="run_go_analysis"):
    
    with st.spinner("Querying MyGene.info API & executing hyper-geometric calculation loops..."):
        # Convert internal pandas DataFrame to openms_insight Polars DataFrame expectation
        stats_pl = pl.from_pandas(final_statistics_report)
        
        status, output = calculate_go_enrichment(
            final_report=stats_pl,
            id_col=id_col,
            target_p_col=target_p_col,
            p_cutoff=p_cutoff,
            fc_cutoff=fc_cutoff,
        )

    # Route response structures based on analysis output status code
    if status == "empty_data":
        st.error("❌ No valid statistical rows found containing standard columns to run GO alignment.")
        
    elif status == "insufficient_proteins":
        st.warning(
            f"⚠️ Not enough significant proteins found to construct target datasets. "
            f"(Criteria: {target_p_col} < {p_cutoff:.4f}, |log2FC| ≥ {fc_cutoff:.2f})."
        )
        st.info(f"💡 Found significant proteins count: **{output}**. Try relaxing your p-value or log2FC filters.")
        
    elif status == "success":
        st.success("⭕ GO Enrichment Analysis completed successfully!")
        
        # Display operational matrix scale
        st.markdown(
            f"📊 **Analysis Profile Scope**: Mapped **{output['fg_count']}** significant foreground profiles out of **{output['bg_count']}** reference background items."
        )
        
        # Build multi-tab interface layer for ontology subcategories
        tabs = st.tabs([
            "🧬 Biological Process (BP)",
            "🔬 Cellular Component (CC)",
            "🧪 Molecular Function (MF)"
        ])
        categories_data = output["categories"]
        
        for idx, go_type in enumerate(["BP", "CC", "MF"]):
            with tabs[idx]:
                fig = categories_data[go_type]["fig"]
                df_go = categories_data[go_type]["df"]
                
                if fig is not None and df_go is not None:
                    # Render plotly bar figures generated straight from backend engine
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader(f"📊 {go_type} Results Dataframe")
                    st.dataframe(df_go, use_container_width=True)
                else:
                    st.info(f"No statistically overrepresented terms identified for Category: **{go_type}**")