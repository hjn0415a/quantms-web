"""
DDA Label-Free Quantification Quickstart Page.

This page provides an overview of the DDA-LFQ workflow and guidance
for getting started with the analysis pipeline.
"""

import streamlit as st
from src.common.common import page_setup

page_setup(page="main")

st.markdown("# DDA Label-Free Quantification")

st.markdown(
    """
This application provides a complete **Data-Dependent Acquisition (DDA) Label-Free Quantification**
workflow for proteomics data analysis. The pipeline enables identification and quantification of
proteins from mass spectrometry data.
"""
)

st.info(
    "This workflow mirrors the **dda-lfq branch of the quantms Nextflow workflow** "
    "for DDA-based label-free quantification."
)

st.markdown("## Workflow Overview")

st.markdown(
    """
The analysis pipeline consists of five main stages:

| Stage | Tool | Description |
|-------|------|-------------|
| **1. Identification** | Comet | Peptide-spectrum matching using a protein database |
| **2. Rescoring** | Percolator | Statistical validation using machine learning |
| **3. Filtering** | IDFilter | FDR-controlled filtering of identifications |
| **4. Quantification** | ProteomicsLFQ | Label-free quantification across samples |
| **5. Statistical Analysis** | Built-in | Volcano plots, PCA, and heatmaps |
"""
)

st.markdown("## Getting Started")

st.markdown(
    """
Follow these steps to run your analysis:

### 1. Upload Files
Upload your mzML mass spectrometry files and a protein FASTA database.
"""
)
st.page_link("content/workflow_fileupload.py", label="Go to File Upload", icon="📁")

st.markdown(
    """
### 2. Configure Parameters
Set up search parameters, sample groups, and analysis settings.
"""
)
st.page_link("content/workflow_configure.py", label="Go to Configure", icon="⚙️")

st.markdown(
    """
### 3. Run Workflow
Execute the analysis pipeline and monitor progress.
"""
)
st.page_link("content/workflow_run.py", label="Go to Run", icon="🚀")

st.markdown(
    """
### 4. Explore Results
View identification results, quantification tables, and statistical visualizations.
"""
)

st.markdown("#### Overview")

st.markdown(
    """
After running the workflow, you can explore:

- **Database Search**: View Comet PSM identification results
- **Rescoring**: Examine Percolator statistical validation output
- **Filtered PSMs**: Inspect FDR-controlled peptide identifications
- **Abundance**: Protein and PSM quantification tables with statistics
- **Volcano Plot**: Differential expression analysis visualization
- **PCA**: Principal component analysis of sample relationships
- **Heatmap**: Hierarchically clustered expression patterns
"""
)

st.markdown("## Workspaces")

st.markdown(
    """
**Workspaces** store your inputs, parameters, and results for each analysis session.

- In **online mode**: The workspace ID is embedded in the URL for easy sharing
- In **local mode**: Create and manage separate workspaces for different projects

Your workspace persists between sessions, allowing you to return to a running or completed
workflow at any time.
"""
)
