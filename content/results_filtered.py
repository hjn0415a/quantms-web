"""Filtered PSMs Results Page."""
import streamlit as st
from pathlib import Path
from streamlit_plotly_events import plotly_events
from src.common.common import page_setup
from src.common.results_helpers import idxml_to_df, get_workflow_dir, create_psm_scatter_plot

params = page_setup()
st.title("Filtered PSMs")

st.markdown(
    """
View FDR-controlled peptide identifications after **IDFilter** processing.
These are high-confidence PSMs that passed the specified FDR threshold.
"""
)

st.info("Explore the PSM scatterplot along with the detailed PSM table.")

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

workflow_dir = get_workflow_dir(st.session_state["workspace"])
filter_dir = workflow_dir / "results" / "filter_results"

if not filter_dir.exists():
    st.info("No filtered results available yet. Please run the workflow first.")
    st.page_link("content/workflow_run.py", label="Go to Run", icon="🚀")
    st.stop()

filter_files = sorted(filter_dir.glob("*.idXML"))

if not filter_files:
    st.warning("No filtering output files found.")
    st.stop()

selected_filter = st.selectbox("Select filtering result file", filter_files)

df_f = idxml_to_df(selected_filter)

if df_f.empty:
    st.info("No peptide hits found in filtering result.")
    st.stop()

st.dataframe(df_f, use_container_width=True)

df_plot_f = df_f.reset_index()

fig3 = create_psm_scatter_plot(df_plot_f)

clicked3 = plotly_events(fig3, click_event=True, hover_event=False, override_height=550, key="filter_plot")

if clicked3:
    idx3 = clicked3[0]["pointNumber"]
    st.subheader("Selected Filtering Peptide Match")
    st.dataframe(df_f.iloc[[idx3]], use_container_width=True)

st.markdown("---")
st.markdown("**Next step:** View abundance quantification")
st.page_link("content/results_abundance.py", label="Go to Abundance", icon="📋")
