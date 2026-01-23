"""Rescoring (Percolator) Results Page."""
import streamlit as st
from pathlib import Path
from streamlit_plotly_events import plotly_events
from src.common.common import page_setup
from src.common.results_helpers import idxml_to_df, get_workflow_dir, create_psm_scatter_plot

params = page_setup()
st.title("Rescoring Results")

st.markdown(
    """
View PSMs after **Percolator** statistical validation. Percolator uses machine learning
to re-score PSMs and estimate false discovery rates (FDR) for more accurate results.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

workflow_dir = get_workflow_dir(st.session_state["workspace"])
perc_dir = workflow_dir / "results" / "percolator_results"

if not perc_dir.exists():
    st.info("No rescoring results available yet. Please run the workflow first.")
    st.page_link("content/workflow_run.py", label="Go to Run", icon="🚀")
    st.stop()

perc_files = sorted(perc_dir.glob("*.idXML"))

if not perc_files:
    st.warning("No rescoring output files found.")
    st.stop()

selected_perc = st.selectbox("Select rescoring result file", perc_files)

df_p = idxml_to_df(selected_perc)

if df_p.empty:
    st.info("No peptide hits found in rescoring result.")
    st.stop()

st.dataframe(df_p, use_container_width=True)

df_plot_p = df_p.reset_index()

fig2 = create_psm_scatter_plot(df_plot_p)

clicked2 = plotly_events(fig2, click_event=True, hover_event=False, override_height=550, key="perc_plot")

if clicked2:
    idx = clicked2[0]["pointNumber"]
    st.subheader("Selected Rescoring Peptide Match")
    st.dataframe(df_p.iloc[[idx]], use_container_width=True)

st.markdown("---")
st.markdown("**Next step:** View filtered PSMs")
st.page_link("content/results_filtered.py", label="Go to Filtered PSMs", icon="🎯")
