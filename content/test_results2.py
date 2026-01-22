import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from src.common.common import page_setup

params = page_setup()
st.title("📊 Quantification")

# ================================
# Load shared results
# ================================
if (
    "pivot_df" not in st.session_state
    or "expr_df" not in st.session_state
    or "group_map" not in st.session_state
):
    st.info("ProteomicsLFQ results not available yet.")
    st.stop()

pivot_df = st.session_state["pivot_df"]
expr_df = st.session_state["expr_df"]
group_map = st.session_state["group_map"]

volcano_tab, pca_tab, heatmap_tab = st.tabs(["🌋 Statistical Analysis Volcano Plot", "📊 PCA", "🔥 Heatmap",])

# ================================
# 🌋 Volcano Plot
# ================================
with volcano_tab:

    st.markdown("### 🌋 Volcano Plot")

    if pivot_df.empty:
        st.info("No data available for volcano plot.")
        st.stop()

    volcano_df = pivot_df.copy()
    volcano_df = volcano_df.dropna(subset=["log2FC", "p-value"])

    volcano_df["neg_log10_p"] = -np.log10(volcano_df["p-value"])

    # Thresholds
    fc_thresh = st.slider(
        "log2 Fold Change threshold",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    p_thresh = st.slider(
        "p-value threshold",
        min_value=0.001,
        max_value=0.1,
        value=0.05,
        step=0.001,
    )

    volcano_df["Significance"] = "Not significant"
    volcano_df.loc[
        (volcano_df["p-value"] <= p_thresh) & (volcano_df["log2FC"] >= fc_thresh),
        "Significance",
    ] = "Up-regulated"

    volcano_df.loc[
        (volcano_df["p-value"] <= p_thresh) & (volcano_df["log2FC"] <= -fc_thresh),
        "Significance",
    ] = "Down-regulated"

    fig_volcano = px.scatter(
        volcano_df,
        x="log2FC",
        y="neg_log10_p",
        color="Significance",
        hover_data=["ProteinName", "p-value"],
    )

    fig_volcano.add_vline(x=fc_thresh, line_dash="dash")
    fig_volcano.add_vline(x=-fc_thresh, line_dash="dash")
    fig_volcano.add_hline(y=-np.log10(p_thresh), line_dash="dash")

    fig_volcano.update_layout(
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        height=600,
    )

    st.plotly_chart(fig_volcano, use_container_width=True)

with pca_tab:
    st.markdown("### 📊 PCA (Protein-level abundance)")

    top_n = 500

    top_proteins = (
        pivot_df
        .dropna(subset=["p-value"])
        .sort_values("p-value", ascending=True)
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

with heatmap_tab:
    st.markdown("### 🔥 Heatmap (Protein-level abundance, Z-score)")

    top_n = st.slider("Number of proteins", 20, 200, 50, key="heatmap_top_n")

    # Data preparation and Z-score normalization
    var_series = expr_df.var(axis=1)
    top_proteins = var_series.sort_values(ascending=False).head(top_n).index
    heatmap_df = expr_df.loc[top_proteins]
    heatmap_z = heatmap_df.sub(heatmap_df.mean(axis=1), axis=0).div(heatmap_df.std(axis=1), axis=0)
    heatmap_z = heatmap_z.replace([np.inf, -np.inf], np.nan).dropna()

    # Hierarchical clustering
    if not heatmap_z.empty:
        # Row clustering (proteins)
        row_linkage = linkage(pdist(heatmap_z.values), method="average")
        row_order = leaves_list(row_linkage)
        
        # Column clustering (samples)
        col_linkage = linkage(pdist(heatmap_z.T.values), method="average")
        col_order = leaves_list(col_linkage)

        # Reorder data  based on clustering results
        heatmap_clustered = heatmap_z.iloc[row_order, col_order]

        # Visualization
        fig_heatmap = px.imshow(
            heatmap_clustered,
            labels=dict(x="Sample", y="Protein", color="Z-score"),
            aspect="auto",
            color_continuous_scale=[[0.0, "#3b6fb6"], [0.5, "white"], [1.0, "#b40426"]],
            zmin=-3, zmax=3 # Fix Z-score range for easier comparison
        )

        fig_heatmap.update_layout(
            height=700,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        # Adjust font size to avoid overlapping labels
        fig_heatmap.update_xaxes(tickfont=dict(size=10))
        fig_heatmap.update_yaxes(tickfont=dict(size=8))

        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning("Insufficient data to generate the heatmap.")