import os
# Polars' default (CPU-core-count-sized) native thread pool access-violation
# crashes the whole process on some high-core-count Windows machines when
# invoked from Streamlit's script-runner thread. Must be set before polars
# is imported anywhere (including transitively via openms_insight).
os.environ.setdefault("POLARS_MAX_THREADS", "1")

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
    
        ],
        "Differential Protein Analysis": [
            st.Page(Path("content", "filtering.py"), title="Filtering", icon="🧹"),
            st.Page(Path("content", "imputation.py"), title="Imputation", icon="🩹"),
            st.Page(Path("content", "normalization.py"), title="Normalization", icon="⚖️"),
            st.Page(Path("content", "statistical.py"), title="Statistical", icon="🔢"),
            st.Page(Path("content", "results_volcano.py"), title="Volcano", icon="🌋"),
            st.Page(Path("content", "results_pca.py"), title="PCA", icon="📊"),
            st.Page(Path("content", "results_heatmap.py"), title="Heatmap", icon="🔥"),
            st.Page(Path("content", "results_heatmap_clustered.py"), title="Clustered Heatmap", icon="🧬"),
            st.Page(Path("content", "enrichment.py"), title="Pathway Analysis", icon="📉"),
        ]
    }

    pg = st.navigation(pages)
    pg.run()

