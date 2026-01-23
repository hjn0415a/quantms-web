"""Helper functions for results pages."""
import pandas as pd
from pathlib import Path
from pyopenms import IdXMLFile, PeptideIdentificationList


def get_workflow_dir(workspace):
    """Get the workflow directory path."""
    return Path(workspace, "topp-workflow")


def idxml_to_df(idxml_file):
    """Parse idXML file and return DataFrame with peptide hits."""
    proteins = []
    peptides = PeptideIdentificationList()
    IdXMLFile().load(str(idxml_file), proteins, peptides)
    peptides = [peptides.at(i) for i in range(peptides.size())]

    records = []
    for pep in peptides:
        rt = pep.getRT()
        mz = pep.getMZ()
        for h in pep.getHits():
            protein_refs = [ev.getProteinAccession() for ev in h.getPeptideEvidences()]
            records.append({
                "RT": rt,
                "m/z": mz,
                "Sequence": h.getSequence().toString(),
                "Charge": h.getCharge(),
                "Score": h.getScore(),
                "Proteins": ",".join(protein_refs) if protein_refs else None,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["Charge"] = df["Charge"].astype(str)
        df["Charge_num"] = df["Charge"].astype(int)
    return df


def create_psm_scatter_plot(df_plot):
    """Create a scatter plot for PSM visualization."""
    import plotly.express as px

    fig = px.scatter(
        df_plot,
        x="RT",
        y="m/z",
        color="Score",
        custom_data=["index", "Sequence", "Proteins"],
        color_continuous_scale=["#a6cee3", "#1f78b4", "#08519c", "#08306b"],
    )
    fig.update_traces(
        marker=dict(size=6, opacity=0.8),
        hovertemplate='<b>Index: %{customdata[0]}</b><br>'
                    + 'RT: %{x:.2f}<br>'
                    + 'm/z: %{y:.4f}<br>'
                    + 'Score: %{marker.color:.3f}<br>'
                    + 'Sequence: %{customdata[1]}<br>'
                    + 'Proteins: %{customdata[2]}<br>'
                    + '<extra></extra>'
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="Score"),
        hovermode="closest"
    )
    return fig
