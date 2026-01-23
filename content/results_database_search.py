"""Database Search (Comet) Results Page."""
import streamlit as st
from pathlib import Path
from streamlit_plotly_events import plotly_events
from src.common.common import page_setup
from src.common.results_helpers import idxml_to_df, get_workflow_dir, create_psm_scatter_plot

params = page_setup()
st.title("Database Search Results")

st.markdown(
    """
View peptide-spectrum matches (PSMs) identified by **Comet** database search.
Each PSM represents a match between an observed spectrum and a peptide sequence.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

workflow_dir = get_workflow_dir(st.session_state["workspace"])
comet_dir = workflow_dir / "results" / "comet_results"

if not comet_dir.exists():
    st.info("No database search results available yet. Please run the workflow first.")
    st.page_link("content/workflow_run.py", label="Go to Run", icon="🚀")
    st.stop()

comet_files = sorted(comet_dir.glob("*.idXML"))

if not comet_files:
    st.warning("No identification output files found.")
    st.stop()

selected_file = st.selectbox("Select identification result file", comet_files)

df = idxml_to_df(selected_file)

if df.empty:
    st.info("No peptide hits found.")
    st.stop()

st.dataframe(df, use_container_width=True)

df_plot = df.reset_index()

fig = create_psm_scatter_plot(df_plot)

clicked = plotly_events(fig, click_event=True, hover_event=False, override_height=550, key="comet_plot")

if clicked:
    row_index = clicked[0]["pointNumber"]
    st.subheader("Selected Peptide Match")
    st.dataframe(df.iloc[[row_index]], use_container_width=True)

st.markdown("---")
st.markdown("**Next step:** View rescoring results")
st.page_link("content/results_rescoring.py", label="Go to Rescoring", icon="📈")
