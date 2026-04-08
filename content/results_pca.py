"""PCA Results Page."""
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data

params = page_setup()
st.title("PCA Analysis")

st.markdown(
    """
Principal Component Analysis (PCA) of protein-level abundance.
Samples are colored by group assignment to visualize clustering.
"""
)

if "workspace" not in st.session_state:
    st.warning("Please initialize your workspace first.")
    st.stop()

result = get_abundance_data(st.session_state["workspace"])
if result is None:
    st.info("Abundance data not available. Please run the workflow and configure sample groups first.")
    st.page_link("content/results_abundance.py", label="Go to Abundance", icon="📋")
    st.stop()

pivot_df, expr_df, group_map = result

top_n = 500

top_proteins = (
    pivot_df
    .dropna(subset=["p-adj"])
    .sort_values("p-adj", ascending=True)
    .head(top_n)["ProteinName"]
)

expr_df_pca = expr_df.loc[
    expr_df.index.intersection(top_proteins)
]

if expr_df_pca.shape[0] < 2:
    st.info("Not enough proteins after p-value filtering for PCA.")
    st.stop()

X = expr_df_pca.T
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    pcs,
    columns=["PC1", "PC2"],
    index=X.index
)

norm_map = {
    k.replace(".mzML", ""): v
    for k, v in group_map.items()
}
pca_df["Group"] = pca_df.index.map(norm_map)

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Group",
    text=pca_df.index,
)

fig_pca.update_traces(textposition="top center")
fig_pca.update_layout(
    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    height=600,
)

st.plotly_chart(fig_pca, use_container_width=True)

st.markdown(f"**Proteins used:** {expr_df_pca.shape[0]} (top {top_n} by p-adj)")

st.markdown("---")
st.markdown("**Other visualizations:**")
col1, col2 = st.columns(2)
with col1:
    st.page_link("content/results_volcano.py", label="Volcano Plot", icon="🌋")
with col2:
    st.page_link("content/results_heatmap.py", label="Heatmap", icon="🔥")
