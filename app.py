import streamlit as st
from pathlib import Path
import json
# For some reason the windows version only works if this is imported here
import pyopenms

if "settings" not in st.session_state:
        with open("settings.json", "r") as f:
            st.session_state.settings = json.load(f)

if __name__ == '__main__':
    pages = {
        "Welcome": [
            st.Page(Path("content", "quickstart.py"), title="Quickstart", icon="👋"),
        ],
        "Workflow": [
            st.Page(Path("content", "workflow_fileupload.py"), title="File Upload", icon="📁"),
            st.Page(Path("content", "workflow_configure.py"), title="Configure", icon="⚙️"),
            st.Page(Path("content", "workflow_run.py"), title="Run", icon="🚀"),
        ],
        "Results": [
            st.Page(Path("content", "results_database_search.py"), title="Database Search", icon="🔬"),
            st.Page(Path("content", "results_rescoring.py"), title="Rescoring", icon="📈"),
            st.Page(Path("content", "results_filtered.py"), title="Filtered PSMs", icon="🎯"),
            st.Page(Path("content", "results_abundance.py"), title="Abundance", icon="📋"),
            st.Page(Path("content", "results_volcano.py"), title="Volcano", icon="🌋"),
            st.Page(Path("content", "results_pca.py"), title="PCA", icon="📊"),
            st.Page(Path("content", "results_heatmap.py"), title="Heatmap", icon="🔥"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()

