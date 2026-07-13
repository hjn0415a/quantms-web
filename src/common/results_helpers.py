"""Helper functions for results pages."""
import re
import pandas as pd
import polars as pl
import numpy as np
import streamlit as st
from pathlib import Path
from pyopenms import IdXMLFile, MSExperiment, MzMLFile
from src.workflow.ParameterManager import ParameterManager

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
def load_abundance_data(workspace_path: str, csv_mtime: float, params_mtime: float = 0.0) -> tuple | None:
    """Load CSV and build abundance matrices for downstream preprocessing.

    Args:
        workspace_path: Path to the workspace directory
        csv_mtime: Modification time of CSV file (used as cache key)
        params_mtime: Modification time of params.json (used as cache key so
            changing group assignments in Configure invalidates the cache
            even when the CSV itself hasn't changed)

    Returns:
        Tuple of (pivot_df, expr_df, group_map) or None if data unavailable
    """
    workflow_dir = get_workflow_dir(Path(workspace_path))
    quant_dir = workflow_dir / "results" / "quant_results"

    parameter_manager = ParameterManager(workflow_dir, "TOPP Workflow")

    workflow_params = parameter_manager.get_parameters_from_json()
    analysis_mode = workflow_params.get("analysis-mode", "LFQ")

    if analysis_mode == "LFQ":
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

        # Get optional group mapping from parameters.
        # Group information is not required at this stage; statistical testing
        # happens in the Statistical page.
        param_manager = ParameterManager(workflow_dir)
        params = param_manager.get_parameters_from_json()
        group_map = {
            key[11:]: value  # Remove "mzML-group-" prefix
            for key, value in params.items()
            if key.startswith("mzML-group-") and value
        }

        df["Sample"] = df["Reference"].str.replace(".mzML", "", regex=False)

        # Build sample display order.
        if group_map:
            sample_group_df = df[["Sample", "Reference"]].drop_duplicates()
            sample_group_df["Group"] = sample_group_df["Reference"].map(group_map)
            grouped_samples = []
            for grp in sorted(sample_group_df["Group"].dropna().unique()):
                grouped_samples.extend(
                    sample_group_df[sample_group_df["Group"] == grp]["Sample"].tolist()
                )
            remaining_samples = [
                s for s in sorted(df["Sample"].unique()) if s not in grouped_samples
            ]
            all_samples = grouped_samples + remaining_samples
        else:
            all_samples = sorted(df["Sample"].unique())

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
        pivot_df = pivot_df[["ProteinName"] + all_samples + ["PeptideSequence"]]

        # Build expression matrix (log2-transformed)
        expr_df = pivot_df.set_index("ProteinName")[all_samples]
        expr_df = expr_df.replace(0, np.nan)
        expr_df = np.log2(expr_df + 1)
        expr_df = expr_df.dropna()

        return pivot_df, expr_df, group_map

    else:
        if not quant_dir.exists():
            return None

        csv_files = sorted(quant_dir.glob("*.csv"))
        if not csv_files:
            return None

        csv_file = csv_files[0]

        try:
            df = pd.read_csv(csv_file, sep="\t", comment="#", engine="python")
        except Exception:
            return None

        if df.empty:
            return None

        # ratio column removal
        df = df.loc[:, ~df.columns.str.contains('ratio', case=False)]

        # exclude_indices = st.session_state.get("tmt_exclude_indices", [])
        # group_map = st.session_state.get("tmt_group_map", {})
        # Get group mapping from parameters
        parameter_manager = ParameterManager(Path(workflow_dir), "TOPP Workflow")
        params = parameter_manager.get_parameters_from_json()
        group_map = {}
        for key, value in params.items():
            if key.startswith("TMT-group-") and value:
                # Extract the numeric part from keys like "TMT-group-sample1"
                match = re.search(r'sample(\d+)', key)
                if match:
                    # Subtract 1 to convert to a 0-based index (0, 1, 2...).
                    # If your samples are already 0-based, remove the -1 adjustment.
                    index = str(int(match.group(1)) - 1)
                    group_map[index] = value

        # 1. Extract keys labeled as "skip" from group_map as integer list
        exclude_indices = [
            int(k) for k, v in group_map.items() if v.lower() == "skip"
        ]

        # 2. Remove "skip" entries from group_map (keep only actual group info)
        group_map = {
            int(k): v for k, v in group_map.items() if v.lower() != "skip"
        }

        start_column_offset = 4

        # st.write("exclude_indices:", exclude_indices)
        # st.write("group_map:", group_map)

        if exclude_indices:
            # st.write("Current columns:", df.columns.tolist())
            # st.write("Number of columns:", len(df.columns))
            # st.write("Exclude indices:", exclude_indices)
            # st.write("Offset:", start_column_offset)
            cols_to_drop = [df.columns[i + start_column_offset] for i in exclude_indices]
            df_cleaned = df.drop(columns=cols_to_drop)
        else:
            df_cleaned = df.copy()

        current_cols = df_cleaned.columns.tolist()
        sample_cols = current_cols[start_column_offset:]

        # Ensure sample columns are numeric for downstream preprocessing/statistics.
        pivot_df = df_cleaned.copy()
        if sample_cols:
            pivot_df[sample_cols] = pivot_df[sample_cols].apply(pd.to_numeric, errors='coerce')

        protein_col = pivot_df.columns[0]
        expr_df = pivot_df.set_index(protein_col)[sample_cols]
        expr_df = expr_df.replace(0, np.nan)
        expr_df = np.log2(expr_df + 1)
        expr_df = expr_df.dropna()

        return pivot_df, expr_df, group_map


def get_abundance_data(workspace: Path) -> tuple | None:
    """Wrapper that handles cache key (workspace + CSV mtime + params mtime).

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

    params_file = workflow_dir / "params.json"
    params_mtime = params_file.stat().st_mtime if params_file.exists() else 0.0

    return load_abundance_data(str(workspace), csv_mtime, params_mtime)


def get_id_column(workspace: Path, pivot_df: pd.DataFrame) -> str:
    """Resolve the protein/row identifier column for the active analysis mode.

    LFQ reports always use "ProteinName"; TMT reports use whatever the
    report's first column is actually named (e.g. "protein").
    """
    workflow_dir = get_workflow_dir(workspace)
    analysis_mode = ParameterManager(workflow_dir, "TOPP Workflow").get_parameters_from_json().get("analysis-mode", "LFQ")
    return "ProteinName" if analysis_mode == "LFQ" else pivot_df.columns[0]


def get_sample_group_map(workspace: Path, pivot_df: pd.DataFrame, group_map: dict) -> dict:
    """Normalize group_map into {actual_sample_column_name: group_name}.

    LFQ group_map keys are already clean sample names (optionally with a
    ".mzML" suffix). TMT group_map keys are 0-based channel indices that must
    be matched against the report's actual "sampleN[...]" column names.
    """
    workflow_dir = get_workflow_dir(workspace)
    analysis_mode = ParameterManager(workflow_dir, "TOPP Workflow").get_parameters_from_json().get("analysis-mode", "LFQ")

    if analysis_mode == "LFQ":
        return {
            k[:-5] if k.endswith(".mzML") else k: v
            for k, v in group_map.items()
        }

    actual_sample_names = pivot_df.columns.tolist()
    norm_map = {}
    for k, v in group_map.items():
        try:
            sample_idx = int(k) + 1
        except (TypeError, ValueError):
            continue
        target_substring = f"sample{sample_idx}["
        real_full_name = next((name for name in actual_sample_names if target_substring in name), None)
        if real_full_name:
            norm_map[real_full_name] = v if v and v.strip() else "Unassigned"
    return norm_map
