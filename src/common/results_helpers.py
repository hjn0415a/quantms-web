"""Helper functions for results pages."""
import re
import pandas as pd
import polars as pl
import numpy as np
import streamlit as st
from pathlib import Path
from scipy.stats import ttest_ind
from pyopenms import IdXMLFile, MSExperiment, MzMLFile
from src.workflow.ParameterManager import ParameterManager
from statsmodels.stats.multitest import multipletests

def get_workflow_dir(workspace):
    """Get the workflow directory path."""
    return Path(workspace, "topp-workflow")


def idxml_to_df(idxml_file):
    """Parse idXML file and return DataFrame with peptide hits."""
    proteins = []
    peptides = []
    IdXMLFile().load(str(idxml_file), proteins, peptides)

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


def extract_scan_from_ref(spec_ref: str) -> int:
    """Extract scan number from spectrum reference string.

    Format: "controllerType=0 controllerNumber=1 scan=1234"
    """
    match = re.search(r'scan=(\d+)', spec_ref)
    return int(match.group(1)) if match else 0


def extract_scan_number(native_id: str) -> int:
    """Extract scan number from native ID."""
    match = re.search(r'scan=(\d+)', native_id)
    return int(match.group(1)) if match else 0


def extract_filename_from_idxml(idxml_path: Path) -> str:
    """Derive mzML filename from idXML filename."""
    stem = idxml_path.stem
    for suffix in ['_comet', '_per', '_filter']:
        stem = stem.replace(suffix, '')
    return f"{stem}.mzML"


def parse_idxml(idxml_path: Path) -> tuple[pl.DataFrame, list[str]]:
    """Parse idXML and return DataFrame for openms_insight.

    Returns:
        Tuple of (id_df, spectra_data list of source filenames)
    """
    proteins = []
    peptides = []
    IdXMLFile().load(str(idxml_path), proteins, peptides)

    # Derive mzML filename from idXML filename (e.g., 02COVID_filter.idXML -> 02COVID.mzML)
    spectra_data = [extract_filename_from_idxml(idxml_path)]

    # Build filename to index mapping
    filename_to_index = {Path(f).name: i for i, f in enumerate(spectra_data)}

    records = []
    for pep in peptides:
        # Get spectrum reference from meta value (key may be bytes or string)
        spec_ref = ""
        if pep.metaValueExists("spectrum_reference"):
            spec_ref = pep.getMetaValue("spectrum_reference")
            if isinstance(spec_ref, bytes):
                spec_ref = spec_ref.decode()
        scan_id = extract_scan_from_ref(spec_ref)

        # Get file index from id_merge_index or derive from filename
        file_index = pep.getMetaValue("id_merge_index") if pep.metaValueExists("id_merge_index") else 0
        filename = spectra_data[file_index] if file_index < len(spectra_data) else ""

        for h in pep.getHits():
            records.append({
                "id_idx": len(records),
                "scan_id": scan_id,
                "file_index": file_index,
                "filename": Path(filename).name if filename else "",
                "sequence": h.getSequence().toString(),
                "charge": h.getCharge(),
                "mz": pep.getMZ(),
                "rt": pep.getRT(),
                "score": h.getScore(),
                "protein_accession": ";".join([ev.getProteinAccession() for ev in h.getPeptideEvidences()]),
            })

    return pl.DataFrame(records), spectra_data


def build_spectra_cache(mzml_dir: Path, filename_to_index: dict) -> tuple[pl.DataFrame, dict]:
    """Extract MS2 spectra from mzML files and return DataFrame.

    Args:
        mzml_dir: Directory containing mzML files
        filename_to_index: Dict mapping filename to file_index

    Returns:
        Tuple of (spectra_df, updated filename_to_index)
    """
    records = []
    peak_id = 0

    for mzml_path in sorted(mzml_dir.glob("*.mzML")):
        # Get or create file index
        if mzml_path.name not in filename_to_index:
            filename_to_index[mzml_path.name] = len(filename_to_index)
        file_index = filename_to_index[mzml_path.name]

        exp = MSExperiment()
        MzMLFile().load(str(mzml_path), exp)

        for spec in exp:
            if spec.getMSLevel() != 2:
                continue
            scan_id = extract_scan_number(spec.getNativeID())
            mz_array, int_array = spec.get_peaks()

            for mz, intensity in zip(mz_array, int_array):
                records.append({
                    "peak_id": peak_id,
                    "file_index": file_index,
                    "scan_id": scan_id,
                    "mass": float(mz),      # Use "mass" not "mz"
                    "intensity": float(intensity),
                })
                peak_id += 1

    return pl.DataFrame(records), filename_to_index


@st.cache_data
def load_abundance_data(workspace_path: str, csv_mtime: float) -> tuple | None:
    """Load CSV, compute stats (log2FC, p-value), build pivot_df and expr_df.

    Args:
        workspace_path: Path to the workspace directory
        csv_mtime: Modification time of CSV file (used as cache key)

    Returns:
        Tuple of (pivot_df, expr_df, group_map) or None if data unavailable
    """
    workflow_dir = get_workflow_dir(Path(workspace_path))
    quant_dir = workflow_dir / "results" / "quant_results"

    if not quant_dir.exists():
        return None

    csv_files = sorted(quant_dir.glob("*.csv"))
    if not csv_files:
        return None

    csv_file = csv_files[0]

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return None

    if df.empty:
        return None

    # Get group mapping from parameters
    param_manager = ParameterManager(workflow_dir)
    params = param_manager.get_parameters_from_json()
    group_map = {
        key[11:]: value  # Remove "mzML-group-" prefix
        for key, value in params.items()
        if key.startswith("mzML-group-") and value
    }

    if not group_map:
        return None

    df["Sample"] = df["Reference"].str.replace(".mzML", "", regex=False)
    df["Group"] = df["Reference"].map(group_map)
    df = df.dropna(subset=["Group"])

    groups = sorted(df["Group"].unique())

    if len(groups) < 2:
        return None

    group1, group2 = groups[:2]

    # Compute statistics per protein
    stats_rows = []
    for protein, protein_df in df.groupby("ProteinName"):
        g1_vals = protein_df[protein_df["Group"] == group1]["Intensity"].values
        g2_vals = protein_df[protein_df["Group"] == group2]["Intensity"].values

        if len(g1_vals) < 2 or len(g2_vals) < 2:
            pval = np.nan
        else:
            _, pval = ttest_ind(g1_vals, g2_vals, equal_var=False)

        mean_g1 = np.mean(g1_vals) if len(g1_vals) > 0 else np.nan
        mean_g2 = np.mean(g2_vals) if len(g2_vals) > 0 else np.nan

        log2fc = np.log2(mean_g2 / mean_g1) if mean_g1 > 0 else np.nan

        stats_rows.append({
            "ProteinName": protein,
            "log2FC": log2fc,
            "p-value": pval,
        })

    stats_df = pd.DataFrame(stats_rows)

    if not stats_df.empty:
        mask = stats_df["p-value"].notna()
        if mask.any():
            _, p_adj, _, _ = multipletests(stats_df.loc[mask, "p-value"], method="fdr_bh")
            stats_df.loc[mask, "p-adj"] = p_adj
        else:
            stats_df["p-adj"] = np.nan

    # Order samples by group (group2 first, then group1)
    sample_group_df = df[["Sample", "Group"]].drop_duplicates()
    group2_samples = sample_group_df[sample_group_df["Group"] == group2]["Sample"].tolist()
    group1_samples = sample_group_df[sample_group_df["Group"] == group1]["Sample"].tolist()
    all_samples = group2_samples + group1_samples

    # Build pivot table
    pivot_list = []
    for protein, group_df in df.groupby("ProteinName"):
        peptides = ";".join(group_df["PeptideSequence"].unique())
        intensity_dict = group_df.groupby("Sample")["Intensity"].sum().to_dict()
        intensity_dict_complete = {
            sample: intensity_dict.get(sample, 0)
            for sample in all_samples
        }
        row = {
            "ProteinName": protein,
            **intensity_dict_complete,
            "PeptideSequence": peptides,
        }
        pivot_list.append(row)

    pivot_df = pd.DataFrame(pivot_list)
    pivot_df = pivot_df.merge(stats_df, on="ProteinName", how="left")
    pivot_df = pivot_df[["ProteinName", "log2FC", "p-value", "p-adj"] + all_samples + ["PeptideSequence"]]

    # Build expression matrix (log2-transformed)
    expr_df = pivot_df.set_index("ProteinName")[all_samples]
    expr_df = expr_df.replace(0, np.nan)
    expr_df = np.log2(expr_df + 1)
    expr_df = expr_df.dropna()

    return pivot_df, expr_df, group_map


def get_abundance_data(workspace: Path) -> tuple | None:
    """Wrapper that handles cache key (workspace + CSV mtime).

    Args:
        workspace: Path to the workspace directory

    Returns:
        Tuple of (pivot_df, expr_df, group_map) or None if data unavailable
    """
    workflow_dir = get_workflow_dir(workspace)
    quant_dir = workflow_dir / "results" / "quant_results"

    if not quant_dir.exists():
        return None

    csv_files = sorted(quant_dir.glob("*.csv"))
    if not csv_files:
        return None

    csv_mtime = csv_files[0].stat().st_mtime
    return load_abundance_data(str(workspace), csv_mtime)
