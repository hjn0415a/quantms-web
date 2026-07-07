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
def load_abundance_data(workspace_path: str, csv_mtime: float, params_mtime: float = 0.0) -> tuple | None:
    """Load CSV, compute stats (log2FC, p-value), build pivot_df and expr_df.

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

        if not group_map:
            st.warning("⚠️ Group mapping information is missing. Please configure sample groups in the Setup page.")
            return None

        if exclude_indices:
            # st.write("Current columns:", df.columns.tolist())
            # st.write("Number of columns:", len(df.columns))
            # st.write("Exclude indices:", exclude_indices)
            # st.write("Offset:", start_column_offset)
            cols_to_drop = [df.columns[i + start_column_offset] for i in exclude_indices]
            df_cleaned = df.drop(columns=cols_to_drop)
        else:
            df_cleaned = df.copy()

        if group_map:
            # Create new row data (defaulting to empty strings)
            # Create a list with the same length as the column order of df_cleaned
            new_row = [""] * len(df_cleaned.columns)
            new_row[0] = "Group"

            # Get the column names of the current dataframe as a list
            current_cols = df_cleaned.columns.tolist()
            original_cols = df.columns.tolist()

            for col_name in current_cols[start_column_offset:]:
                # Check the original index position of this column
                original_idx = original_cols.index(col_name) - start_column_offset
                col_pos = current_cols.index(col_name)
                new_row[col_pos] = group_map.get(original_idx, "NA")

            # Insert the row at the top of the dataframe
            # Create a new DF and concatenate to prepend the row to existing data
            group_df = pd.DataFrame([new_row], columns=df_cleaned.columns)
            df_with_groups = pd.concat([group_df, df_cleaned], ignore_index=True)

            # drop_msg = f"{len(exclude_indices)} channels dropped" if exclude_indices else "No channels dropped"
            # st.success(f"✅ {drop_msg} and Group names have been inserted at the top of the data.")

            # st.write("### Data Preview with Group Information")
            # st.dataframe(df_with_groups.head(10))

            if group_map and len(set(group_map.values())) >= 2:
                # Prepare data for calculation
                # Extract group information from row 0 of df_with_groups (the newly added Group row)
                # Actual sample data starts from the 5th column (index 4)
                group_info_row = df_with_groups.iloc[0]

                # Get unique group names (excluding NA)
                unique_groups = sorted([g for g in set(group_map.values()) if g != "NA"])
                g1_name, g2_name = unique_groups[0], unique_groups[1]

                # Extract numerical data for statistical calculation (from row 1 and column index 4 onwards)
                # Convert to numeric type (to prevent calculation errors)
                numeric_data = df_with_groups.iloc[1:, 4:].apply(pd.to_numeric, errors='coerce')

                # Column indexing by group
                # Categorize columns based on the values in the Group row
                g1_cols = [col for col in numeric_data.columns if group_info_row[col] == g1_name]
                g2_cols = [col for col in numeric_data.columns if group_info_row[col] == g2_name]

                # Calculate log2FC and p-value for each row
                def run_stats(row):
                    v1 = row[g1_cols].dropna()
                    v2 = row[g2_cols].dropna()

                    # log2FC (Group2 / Group1)
                    m1, m2 = v1.mean(), v2.mean()
                    l2fc = np.log2(m2 / m1) if m1 > 0 and m2 > 0 else np.nan

                    # p-value (T-test)
                    if len(v1) > 1 and len(v2) > 1:
                        _, pval = ttest_ind(v1, v2, equal_var=False)
                    else:
                        pval = np.nan
                    return pd.Series([l2fc, pval])

                stats_results = numeric_data.apply(run_stats, axis=1)
                stats_results.columns = ['log2FC', 'p-value']
                # Add Adjusted p-value (FDR) calculation
                if not stats_results['p-value'].isna().all():
                    # Select only rows that contain p-values
                    mask = stats_results['p-value'].notna()
                    # Apply Benjamini-Hochberg (BH) correction
                    _, p_adj, _, _ = multipletests(stats_results.loc[mask, 'p-value'], method='fdr_bh')
                    stats_results.loc[mask, 'p-adj'] = p_adj
                else:
                    stats_results['p-adj'] = np.nan

                # Construct the final dataframe (Based on df_cleaned - excluding the group row)
                # Insert calculation results into the 2nd and 3rd column positions
                pivot_df = df_cleaned.copy()
                pivot_df.insert(1, "log2FC", stats_results['log2FC'].values)
                pivot_df.insert(2, "p-value", stats_results['p-value'].values)
                pivot_df.insert(3, "p-adj", stats_results['p-adj'].values)

                # st.success(f"Analysis Complete: {g1_name} (n={len(g1_cols)}) vs {g2_name} (n={len(g2_cols)})")

                # Set the first column ('protein') of final_df as the index
                protein_col = pivot_df.columns[0]
                sample_cols = current_cols[start_column_offset:] # Identify actual sample column names

                # Select sample columns and create a matrix
                expr_df = pivot_df.set_index(protein_col)[sample_cols]

                # Replace 0 with NaN (to prevent log transformation errors)
                expr_df = expr_df.replace(0, np.nan)

                # Log2 transformation (data normalization)
                expr_df = np.log2(expr_df + 1)

                # Remove proteins (rows) with any missing values
                expr_df = expr_df.dropna()

                return pivot_df, expr_df, group_map
            else:
                st.warning("⚠️ At least two distinct groups are required for statistical analysis.")
        else:
            st.warning("⚠️ No group mapping information is set. Please check the Configure page.")
        return None


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
