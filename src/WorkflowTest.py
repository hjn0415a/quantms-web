import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
from pyopenms import IdXMLFile
from scipy.stats import ttest_ind
import numpy as np
import mygene

from collections import defaultdict        
from scipy.stats import fisher_exact         
from src.workflow.WorkflowManager import WorkflowManager
from src.common.common import page_setup
from src.common.results_helpers import get_abundance_data
from src.common.results_helpers import parse_idxml, build_spectra_cache
from openms_insight import Table, Heatmap, LinePlot, SequenceView

# params = page_setup()
class WorkflowTest(WorkflowManager):

    def __init__(self) -> None:
        super().__init__("TOPP Workflow", st.session_state["workspace"])

    def upload(self) -> None:
        t = st.tabs(["MS data (mzML)", "FASTA database"])

        with t[0]:
            self.ui.upload_widget(
                key="mzML-files",
                name="MS data",
                file_types="mzML",
                fallback=[str(f) for f in Path("example-data", "mzML").glob("*.mzML")],
            )

        with t[1]:
            self.ui.upload_widget(
                key="fasta-file",
                name="Protein FASTA database",
                file_types=("fasta", "fa"),
                fallback=[str(f) for f in Path("example-data", "db").glob("*.fasta")],
            )

    @st.fragment
    def configure(self) -> None:
        # reactive=True so Group Selection tab updates when selection changes
        self.ui.select_input_file("mzML-files", multiple=True, reactive=True)
        self.ui.select_input_file("fasta-file", multiple=False)

        t = st.tabs(["**Identification**", "**Rescoring**", "**Filtering**", "**Library Generation**", "**Quantification**", "**Group Selection**"])

        with t[0]:
            # Checkbox for decoy generation
            # reactive=True ensures the parent configure() fragment re-runs when checkbox changes,
            # so conditional UI (DecoyDatabase settings) updates immediately
            self.ui.input_widget(
                key="generate-decoys",
                default=True,
                name="Generate Decoy Database",
                widget_type="checkbox",
                help="Generate reversed decoy sequences for FDR calculation. Disable if your FASTA already contains decoys.",
                reactive=True,
            )

            # Reload params to get current checkbox value after it was saved
            self.params = self.parameter_manager.get_parameters_from_json()

            # Show DecoyDatabase settings if generating decoys
            if self.params.get("generate-decoys", True):
                st.info("""
                **Decoy Database Settings:**
                * **method**: How decoy sequences are generated from target protein sequences.
                  *Reverse* creates decoys by reversing each sequence, while *shuffle* randomly
                  rearranges the amino acids. Both methods preserve the amino acid composition
                  of the original protein, ensuring decoys have similar properties to real sequences
                  for accurate false discovery rate (FDR) estimation.
                """)
                self.ui.input_TOPP(
                    "DecoyDatabase",
                    custom_defaults={
                        "decoy_string": "rev_",
                        "decoy_string_position": "prefix",
                        "method": "reverse",
                    },
                    include_parameters=["method"],
                )

            comet_info = """
            **Identification (Comet):**
            * **enzyme**: The enzyme used for peptide digestion.
            * **missed_cleavages**: Number of possible cleavage sites missed by the enzyme. It has no effect if enzyme is unspecific cleavage.
            * **fixed_modifications**: Fixed modifications, specified using Unimod (www.unimod.org) terms, e.g. 'Carbamidomethyl (C)' or 'Oxidation (M)'
            * **variable_modifications**: Variable modifications, specified using Unimod (www.unimod.org) terms, e.g. 'Carbamidomethyl (C)' or 'Oxidation (M)'
            * **instrument**: Type of instrument (high_res or low_res). Use 'high_res' for high-resolution MS2 (Orbitrap, TOF), 'low_res' for ion trap.
            * **fragment_mass_tolerance**: Fragment mass tolerance for MS2 matching.
            * **fragment_bin_offset**: Offset for binning MS2 spectra. Typically 0.0 for high-res, 0.4 for low-res instruments.
            """
            if not self.params.get("generate-decoys", True):
                comet_info += """* **PeptideIndexing:decoy_string**: String that was appended (or prefixed - see 'decoy_string_position' flag below) to the accessions
                    in the protein database to indicate decoy proteins.
            """
            st.info(comet_info)

            comet_include = [":enzyme", "missed_cleavages", "fixed_modifications", "variable_modifications",
                             "instrument", "fragment_mass_tolerance", "fragment_error_units", "fragment_bin_offset"]
            if not self.params.get("generate-decoys", True):
                # Only show decoy_string when not generating decoys
                comet_include.append("PeptideIndexing:decoy_string")

            self.ui.input_TOPP(
                "CometAdapter",
                custom_defaults={
                    "threads": 8,
                    "instrument": "high_res",
                    "missed_cleavages": 2,
                    "min_peptide_length": 6,
                    "max_peptide_length": 40,
                    "num_hits": 1,
                    "num_enzyme_termini": "fully",
                    "isotope_error": "0/1",
                    "precursor_charge": "2:4",
                    "precursor_mass_tolerance": 20.0,
                    "fragment_mass_tolerance": 0.02,
                    "fragment_bin_offset": 0.0,
                    "max_variable_mods_in_peptide": 3,
                    "minimum_peaks": 1,
                    "clip_nterm_methionine": "true",
                    "PeptideIndexing:IL_equivalent": "true",
                    "PeptideIndexing:unmatched_action": "warn",
                    "PeptideIndexing:decoy_string": "rev_",
                },
                include_parameters=comet_include,
                exclude_parameters=["second_enzyme"],
            )

        with t[1]:
            st.info("""
            **Rescoring (Percolator):**
            * **post_processing_tdc**: Use target-decoy competition to assign q-values and PEPs.
            * **score_type**: Type of the peptide main score
            * **subset_max_train**: Only train an SVM on a subset of <x> PSMs, and use the resulting score vector to evaluate the other PSMs. Recommended when analyzing huge numbers (>1 million) of PSMs. When set to 0, all PSMs are used for training as normal.
            """)
            # decoy_pattern is always derived from upstream, never shown
            percolator_include = ["post_processing_tdc", "score_type", "subset_max_train"]

            self.ui.input_TOPP(
                "PercolatorAdapter",
                custom_defaults={
                    "threads": 8,
                    "subset_max_train": 300000,
                    "decoy_pattern": "rev_",
                    "score_type": "pep",
                    "post_processing_tdc": "true",
                },
                include_parameters=percolator_include,
                exclude_parameters=["out_type"],
            )

        with t[2]:
            st.info("""
            **Filtering (IDFilter):**
            * **score:type_peptide**: Score used for filtering. If empty, the main score is used.
            * **score:psm**: The score which should be reached by a peptide hit to be kept. (use 'NAN' to disable this filter)
            """)
            self.ui.input_TOPP(
                "IDFilter",
                custom_defaults={
                    "threads": 2,
                    "score:type_peptide": "q-value",
                    "score:psm": 0.10,
                },
                # include_parameters=["type_peptide", "score:psm"]
                exclude_parameters=["type_protein"],
            )

        with t[3]:  # Library Generation
            st.info("""
            **Spectral Library Generation (EasyPQP):**
            Generate a spectral library from filtered PSMs for targeted proteomics (DIA/SWATH).
            """)

            self.ui.input_widget(
                key="generate-library",
                default=False,
                name="Generate Spectral Library",
                widget_type="checkbox",
                help="Enable spectral library generation using EasyPQP.",
                reactive=True,
            )
            self.params = self.parameter_manager.get_parameters_from_json()

            if self.params.get("generate-library", False):
                # FDR options
                self.ui.input_widget(
                    key="library-use-fdr",
                    default=False,
                    name="Apply additional FDR filtering",
                    widget_type="checkbox",
                    help="If disabled (recommended), uses --nofdr since IDFilter already applied FDR.",
                    reactive=True,
                )
                self.params = self.parameter_manager.get_parameters_from_json()

                if self.params.get("library-use-fdr", False):
                    self.ui.input_widget(
                        key="library-psm-fdr",
                        default=0.01,
                        name="PSM FDR Threshold",
                        widget_type="number",
                        min_value=0.001,
                        max_value=1.0,
                        step_size=0.01,
                    )

                # Decoy options
                self.ui.input_widget(
                    key="library-generate-decoys",
                    default=True,
                    name="Generate Decoy Library",
                    widget_type="checkbox",
                    reactive=True,
                )
                self.params = self.parameter_manager.get_parameters_from_json()

                if self.params.get("library-generate-decoys", True):
                    self.ui.input_widget(
                        key="library-decoy-method",
                        default="shuffle",
                        name="Decoy Method",
                        widget_type="selectbox",
                        options=["shuffle", "reverse", "pseudo-reverse", "shift"],
                    )

        with t[4]:
            st.info("""
            **Quantification (ProteomicsLFQ):**
            * **intThreshold**: Peak intensity threshold applied in seed detection.
            * **psmFDR**: FDR threshold for sub-protein level (e.g. 0.05=5%). Use -FDR_type to choose the level. Cutoff is applied at the highest level. If Bayesian inference was chosen, it is equivalent with a peptide FDR
            * **proteinFDR**: Protein FDR threshold (0.05=5%).
            """)
            self.ui.input_TOPP(
                "ProteomicsLFQ",
                custom_defaults={
                    "threads": 12,
                    "targeted_only": "true",
                    "feature_with_id_min_score": 0.1,
                    "Seeding:intThreshold": 1000.0,
                    "psmFDR": 0.01,
                    "proteinFDR": 0.01,
                    "picked_proteinFDR": "true",
                },
                include_parameters=["intThreshold", "psmFDR", "proteinFDR"],
            )

        with t[5]:
            st.markdown("### 🧪 Sample Group Assignment")
            st.info(
                "Enter a group name for each mzML file.\n\n"
                "Examples: case, control"
            )

            mzml_keys = self.params.get("mzML-files")

            if not mzml_keys:
                st.warning("No mzML files available. Please upload mzML files first.")
                return

            try:
                mzml_files = self.file_manager.get_files(mzml_keys)
            except ValueError:
                st.warning("Selected mzML files are not available.")
                return

            # Current mzML filenames
            current_filenames = {Path(mz).name for mz in mzml_files}

            # Per-file group input using input_widget (auto-saves to params.json)
            for mz in mzml_files:
                filename = Path(mz).name
                self.ui.input_widget(
                    key=f"mzML-group-{filename}",
                    default="",
                    name=f"Group for {filename}",
                    widget_type="text",
                    help="e.g. case, control",
                )

            # Reload params to get current values
            self.params = self.parameter_manager.get_parameters_from_json()

            # Clean up orphaned group params from previously selected files
            orphaned_keys = [
                k for k in self.params.keys()
                if k.startswith("mzML-group-") and k[11:] not in current_filenames
            ]
            for key in orphaned_keys:
                del self.params[key]
            if orphaned_keys:
                self.parameter_manager.save_parameters()

    def execution(self) -> bool:
        """
        Refactored TOPP workflow execution:
        - Per-sample: CometAdapter -> PercolatorAdapter -> IDFilter
        - Cross-sample: ProteomicsLFQ (single combined output)
        """
        # ================================
        # 0️⃣ Input validation
        # ================================
        if not self.params.get("mzML-files"):
            st.error("No mzML files selected.")
            return False

        if not self.params.get("fasta-file"):
            st.error("No FASTA file selected.")
            return False

        in_mzML = self.file_manager.get_files(self.params["mzML-files"])
        fasta_file = self.file_manager.get_files([self.params["fasta-file"]])[0]

        if len(in_mzML) < 1:
            st.error("At least one mzML file is required.")
            return False
        
        fasta_path = Path(fasta_file)

        self.logger.log(f"📂 Loaded {len(in_mzML)} sample(s)")

        if self.params.get("generate-decoys", True):
            decoy_fasta = fasta_path.with_suffix(".decoy.fasta")
            # Get decoy_string from DecoyDatabase params
            decoy_string = self.params.get("DecoyDatabase", {}).get("decoy_string", "rev_")

            if not decoy_fasta.exists():
                self.logger.log("🧬 Generating decoy database...")
                st.info("Generating decoy FASTA database...")
                if not self.executor.run_topp(
                    "DecoyDatabase",
                    {"in": [str(fasta_path)], "out": [str(decoy_fasta)]},
                ):
                    self.logger.log("Workflow stopped due to error")
                    return False
                self.logger.log("✅ Decoy database ready")
            st.success(f"Using decoy FASTA: {decoy_fasta.name}")
            database_fasta = decoy_fasta
        else:
            # Get decoy_string from CometAdapter params
            decoy_string = self.params.get("CometAdapter", {}).get("PeptideIndexing:decoy_string", "rev_")
            self.logger.log("📄 Using existing FASTA database")
            st.info(f"Using original FASTA: {fasta_path.name}")
            database_fasta = fasta_path
        
        # ================================
        # 1️⃣ Directory setup
        # ================================
        results_dir = Path(self.workflow_dir, "results")
        comet_dir = results_dir / "comet_results"
        perc_dir = results_dir / "percolator_results"
        filter_dir = results_dir / "filter_results"
        quant_dir = results_dir / "quant_results"

        results_dir = Path(self.workflow_dir, "input-files")

        for d in [comet_dir, perc_dir, filter_dir, quant_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger.log("📁 Output directories created")

        # # ================================
        # # 2️⃣ File path definitions (per sample)
        # # ================================
        comet_results = []
        percolator_results = []
        filter_results = []

        for mz in in_mzML:
            stem = Path(mz).stem
            comet_results.append(str(comet_dir / f"{stem}_comet.idXML"))
            percolator_results.append(str(perc_dir / f"{stem}_per.idXML"))
            filter_results.append(str(filter_dir / f"{stem}_filter.idXML"))

        # ================================
        # 3️⃣ Per-file processing
        # ================================
        for i, mz in enumerate(in_mzML):
            stem = Path(mz).stem
            st.info(f"Processing sample: {stem}")

        self.logger.log("🔬 Starting per-sample processing...")

        # --- CometAdapter ---
        self.logger.log("🔎 Running peptide search...")
        with st.spinner(f"CometAdapter ({stem})"):
            comet_extra_params = {"database": str(database_fasta)}
            if self.params.get("generate-decoys", True):
                # Propagate decoy_string from DecoyDatabase
                comet_extra_params["PeptideIndexing:decoy_string"] = decoy_string

            if not self.executor.run_topp(
                "CometAdapter",
                {
                    "in": in_mzML,
                    "out": comet_results,
                },
                comet_extra_params,
            ):
                self.logger.log("Workflow stopped due to error")
                return False

        # Get fragment tolerance from CometAdapter parameters for visualization
        comet_params = self.parameter_manager.get_topp_parameters("CometAdapter")
        frag_tol = comet_params.get("fragment_mass_tolerance", 0.02)
        frag_tol_is_ppm = comet_params.get("fragment_error_units", "Da") != "Da"

        # Build visualization cache for Comet results
        results_dir_path = Path(self.workflow_dir, "results")
        cache_dir = results_dir_path / "insight_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Get mzML directory
        mzml_dir = Path(in_mzML[0]).parent

        # Build spectra cache (once, shared by all stages)
        spectra_df = None
        filename_to_index = {}

        for idxml_file in comet_results:
            idxml_path = Path(idxml_file)
            cache_id_prefix = idxml_path.stem

            # Parse idXML to DataFrame
            id_df, spectra_data = parse_idxml(idxml_path)

            # Build spectra cache (only once)
            if spectra_df is None:
                filename_to_index = {Path(f).name: i for i, f in enumerate(spectra_data)}
                spectra_df, filename_to_index = build_spectra_cache(mzml_dir, filename_to_index)

            # Initialize Table component (caches itself)
            Table(
                cache_id=f"table_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                interactivity={"file": "file_index", "spectrum": "scan_id", "identification": "id_idx"},
                column_definitions=[
                    {"field": "sequence", "title": "Sequence"},
                    {"field": "charge", "title": "Z", "sorter": "number"},
                    {"field": "mz", "title": "m/z", "sorter": "number"},
                    {"field": "rt", "title": "RT", "sorter": "number"},
                    {"field": "score", "title": "Score", "sorter": "number"},
                    {"field": "protein_accession", "title": "Proteins"},
                ],
                initial_sort=[{"column": "score", "dir": "asc"}],
                index_field="id_idx",
            )

            # Initialize Heatmap component
            Heatmap(
                cache_id=f"heatmap_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                x_column="rt",
                y_column="mz",
                intensity_column="score",
                interactivity={"identification": "id_idx"},
            )

            # Initialize SequenceView component
            seq_view = SequenceView(
                cache_id=f"seqview_{cache_id_prefix}",
                sequence_data=id_df.lazy().select(["id_idx", "sequence", "charge", "file_index", "scan_id"]).rename({
                    "id_idx": "sequence_id",
                    "charge": "precursor_charge",
                }),
                peaks_data=spectra_df.lazy(),
                filters={
                    "identification": "sequence_id",
                    "file": "file_index",
                    "spectrum": "scan_id",
                },
                interactivity={"peak": "peak_id"},
                cache_path=str(cache_dir),
                deconvolved=False,
                annotation_config={
                    "ion_types": ["b", "y"],
                    "neutral_losses": True,
                    "tolerance": frag_tol,
                    "tolerance_ppm": frag_tol_is_ppm,
                },
            )

            # Initialize LinePlot from SequenceView
            LinePlot.from_sequence_view(
                seq_view,
                cache_id=f"lineplot_{cache_id_prefix}",
                cache_path=str(cache_dir),
                title="Annotated Spectrum",
                styling={
                    "unhighlightedColor": "#CCCCCC",
                    "highlightColor": "#E74C3C",
                    "selectedColor": "#F3A712",
                },
            )

        self.logger.log("✅ Peptide search complete")

        # --- PercolatorAdapter ---
        self.logger.log("📊 Running rescoring...")
        with st.spinner(f"PercolatorAdapter ({stem})"):
            if not self.executor.run_topp(
                "PercolatorAdapter",
                {
                    "in": comet_results,
                    "out": percolator_results,
                },
                {"decoy_pattern": decoy_string},  # Always propagated from upstream
            ):
                self.logger.log("Workflow stopped due to error")
                return False

        # Build visualization cache for Percolator results
        for idxml_file in percolator_results:
            idxml_path = Path(idxml_file)
            cache_id_prefix = idxml_path.stem

            # Parse idXML to DataFrame
            id_df, spectra_data = parse_idxml(idxml_path)

            # Initialize Table component (caches itself)
            Table(
                cache_id=f"table_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                interactivity={"file": "file_index", "spectrum": "scan_id", "identification": "id_idx"},
                column_definitions=[
                    {"field": "sequence", "title": "Sequence"},
                    {"field": "charge", "title": "Z", "sorter": "number"},
                    {"field": "mz", "title": "m/z", "sorter": "number"},
                    {"field": "rt", "title": "RT", "sorter": "number"},
                    {"field": "score", "title": "Score", "sorter": "number"},
                    {"field": "protein_accession", "title": "Proteins"},
                ],
                initial_sort=[{"column": "score", "dir": "asc"}],
                index_field="id_idx",
            )

            # Initialize Heatmap component
            Heatmap(
                cache_id=f"heatmap_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                x_column="rt",
                y_column="mz",
                intensity_column="score",
                interactivity={"identification": "id_idx"},
            )

            # Initialize SequenceView component
            seq_view = SequenceView(
                cache_id=f"seqview_{cache_id_prefix}",
                sequence_data=id_df.lazy().select(["id_idx", "sequence", "charge", "file_index", "scan_id"]).rename({
                    "id_idx": "sequence_id",
                    "charge": "precursor_charge",
                }),
                peaks_data=spectra_df.lazy(),
                filters={
                    "identification": "sequence_id",
                    "file": "file_index",
                    "spectrum": "scan_id",
                },
                interactivity={"peak": "peak_id"},
                cache_path=str(cache_dir),
                deconvolved=False,
                annotation_config={
                    "ion_types": ["b", "y"],
                    "neutral_losses": True,
                    "tolerance": frag_tol,
                    "tolerance_ppm": frag_tol_is_ppm,
                },
            )

            # Initialize LinePlot from SequenceView
            LinePlot.from_sequence_view(
                seq_view,
                cache_id=f"lineplot_{cache_id_prefix}",
                cache_path=str(cache_dir),
                title="Annotated Spectrum",
                styling={
                    "unhighlightedColor": "#CCCCCC",
                    "highlightColor": "#E74C3C",
                    "selectedColor": "#F3A712",
                },
            )

        self.logger.log("✅ Rescoring complete")

        # if not Path(percolator_results[i]).exists():
        #     st.error(f"PercolatorAdapter failed for {stem}")
        #     st.stop()

        # --- IDFilter ---
        self.logger.log("🔧 Filtering identifications...")
        with st.spinner(f"IDFilter ({stem})"):
            if not self.executor.run_topp(
                "IDFilter",
                {
                    "in": percolator_results,
                    "out": filter_results,
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False

        # Build visualization cache for Filter results
        for idxml_file in filter_results:
            idxml_path = Path(idxml_file)
            cache_id_prefix = idxml_path.stem

            # Parse idXML to DataFrame
            id_df, spectra_data = parse_idxml(idxml_path)

            # Initialize Table component (caches itself)
            Table(
                cache_id=f"table_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                interactivity={"file": "file_index", "spectrum": "scan_id", "identification": "id_idx"},
                column_definitions=[
                    {"field": "sequence", "title": "Sequence"},
                    {"field": "charge", "title": "Z", "sorter": "number"},
                    {"field": "mz", "title": "m/z", "sorter": "number"},
                    {"field": "rt", "title": "RT", "sorter": "number"},
                    {"field": "score", "title": "Score", "sorter": "number"},
                    {"field": "protein_accession", "title": "Proteins"},
                ],
                initial_sort=[{"column": "score", "dir": "asc"}],
                index_field="id_idx",
            )

            # Initialize Heatmap component
            Heatmap(
                cache_id=f"heatmap_{cache_id_prefix}",
                data=id_df.lazy(),
                cache_path=str(cache_dir),
                x_column="rt",
                y_column="mz",
                intensity_column="score",
                interactivity={"identification": "id_idx"},
            )

            # Initialize SequenceView component
            seq_view = SequenceView(
                cache_id=f"seqview_{cache_id_prefix}",
                sequence_data=id_df.lazy().select(["id_idx", "sequence", "charge", "file_index", "scan_id"]).rename({
                    "id_idx": "sequence_id",
                    "charge": "precursor_charge",
                }),
                peaks_data=spectra_df.lazy(),
                filters={
                    "identification": "sequence_id",
                    "file": "file_index",
                    "spectrum": "scan_id",
                },
                interactivity={"peak": "peak_id"},
                cache_path=str(cache_dir),
                deconvolved=False,
                annotation_config={
                    "ion_types": ["b", "y"],
                    "neutral_losses": True,
                    "tolerance": frag_tol,
                    "tolerance_ppm": frag_tol_is_ppm,
                },
            )

            # Initialize LinePlot from SequenceView
            LinePlot.from_sequence_view(
                seq_view,
                cache_id=f"lineplot_{cache_id_prefix}",
                cache_path=str(cache_dir),
                title="Annotated Spectrum",
                styling={
                    "unhighlightedColor": "#CCCCCC",
                    "highlightColor": "#E74C3C",
                    "selectedColor": "#F3A712",
                },
            )

        self.logger.log("✅ Filtering complete")

        # if not Path(filter_results[i]).exists():
        #     st.error(f"IDFilter failed for {stem}")
        #     st.stop()

        # ================================
        # EasyPQP Spectral Library Generation (optional)
        # ================================
        if self.params.get("generate-library", False):
            self.logger.log("📚 Building spectral library with EasyPQP...")
            st.info("Building spectral library with EasyPQP...")
            library_dir = Path(self.workflow_dir, "results", "library")
            library_dir.mkdir(parents=True, exist_ok=True)

            psms_files, peaks_files = [], []

            for filter_idxml in filter_results:
                original_stem = Path(filter_idxml).stem.replace("_filter", "")
                matching_mzml = next((m for m in in_mzML if Path(m).stem == original_stem), None)
                if not matching_mzml:
                    self.logger.log(f"Warning: No matching mzML found for {filter_idxml}")
                    continue

                # easypqp library requires specific extensions for file recognition:
                # - PSM files must contain 'psmpkl' → use .psmpkl extension
                # - Peak files must contain 'peakpkl' → use .peakpkl extension
                # After splitext(), stem will be just "{mzML_stem}" matching PSM base_name
                psms_out = str(library_dir / f"{original_stem}.psmpkl")
                peaks_out = str(library_dir / f"{original_stem}.peakpkl")

                convert_cmd = [
                    "easypqp", "convert",
                    "--pepxml", filter_idxml,
                    "--spectra", matching_mzml,
                    "--psms", psms_out,
                    "--peaks", peaks_out
                ]
                if self.executor.run_command(convert_cmd):
                    psms_files.append(psms_out)
                    peaks_files.append(peaks_out)

            if psms_files:
                # easypqp library outputs TSV format (despite common .pqp extension)
                library_tsv = str(library_dir / "spectral_library.tsv")
                library_cmd = ["easypqp", "library", "--out", library_tsv]

                if not self.params.get("library-use-fdr", False):
                    # --nofdr only skips FDR recalculation, NOT threshold filtering
                    # Set all thresholds to 1.0 to bypass filtering for pre-filtered input
                    library_cmd.extend([
                        "--nofdr",
                        "--psm_fdr_threshold", "1.0",
                        "--peptide_fdr_threshold", "1.0",
                        "--protein_fdr_threshold", "1.0"
                    ])
                else:
                    # Apply user-specified FDR filtering
                    library_cmd.extend([
                        "--psm_fdr_threshold",
                        str(self.params.get("library-psm-fdr", 0.01)),
                        "--peptide_fdr_threshold",
                        str(self.params.get("library-peptide-fdr", 0.01)),
                        "--protein_fdr_threshold",
                        str(self.params.get("library-protein-fdr", 0.01))
                    ])

                for psms, peaks in zip(psms_files, peaks_files):
                    library_cmd.extend([psms, peaks])

                if self.executor.run_command(library_cmd):
                    self.logger.log("✅ Spectral library created")
                    st.success("Spectral library created")
                else:
                    self.logger.log("Warning: Failed to build spectral library")
            else:
                self.logger.log("Warning: No PSMs converted for library generation")

        st.success(f"✓ {stem} identification completed")

        # ================================
        # 4️⃣ ProteomicsLFQ (cross-sample)
        # ================================
        self.logger.log("📈 Running cross-sample quantification...")
        st.info("Running ProteomicsLFQ (cross-sample quantification)")

        quant_mztab = str(quant_dir / "openms_quant.mzTab")
        quant_cxml = str(quant_dir / "openms.consensusXML")
        quant_msstats = str(quant_dir / "openms_msstats.csv")

        with st.spinner("ProteomicsLFQ"):
                combined_in = " ".join(in_mzML)
                combined_ids = " ".join(filter_results)
                self.logger.log(f"COMBINED_IN {combined_in}", 1)
                self.logger.log(f"COMBINED_IN_TYPE {type(combined_in).__name__}", 1)
                self.logger.log(f"FILTER_RESULTS = {filter_results}", 1)
                self.logger.log(f"FILTER_RESULTS_LEN = {len(filter_results)}", 1)

                # ✅ Streamlit output (debug view)
                st.markdown("### 🔍 ProteomicsLFQ Input Debug")
                st.write("**combined_in:**", combined_in)
                st.write("**combined_in type:**", type(combined_in).__name__)

                st.write("**combined_ids:**", combined_ids)
                st.write("**combined_ids type:**", type(combined_ids).__name__)

                if not self.executor.run_topp(
                        "ProteomicsLFQ",
                        {
                            "in": [in_mzML],
                            "ids": [filter_results],
                            "out": [quant_mztab],
                            "out_cxml": [quant_cxml],
                            "out_msstats": [quant_msstats],
                        },
                        {
                            "fasta": str(database_fasta),
                            "psmFDR": 0.5,
                            "proteinFDR": 0.5,
                            "threads": 12,
                            # Disable FAIMS/IM handling to avoid segfault in OpenMS 3.5.0
                            "PeptideQuantification:extract:IM_window": "0.0",
                            "PeptideQuantification:faims:merge_features": "false",
                        }
                    ):
                    self.logger.log("Workflow stopped due to error")
                    return False
        self.logger.log("✅ Quantification complete")

        # ======================================================
        # ⚠️ 5️⃣ GO Enrichment Analysis (INLINE IN EXECUTION)
        # ======================================================
        workspace_path = Path(self.workflow_dir).parent
        res = get_abundance_data(workspace_path)
        if res is not None:
            pivot_df, _, _ = res
            self.logger.log("✅ pivot_df loaded, starting GO enrichment...")
            self._run_go_enrichment(pivot_df, results_dir)
        else:
            st.warning("GO enrichment skipped: abundance data not available.")

        # ================================
        # 5️⃣ Final report
        # # ================================
        st.success("🎉 TOPP workflow completed successfully")
        st.write("📁 Results directory:")   
        st.code(str(results_dir)) 

        return True
    
    def _run_go_enrichment(self, pivot_df: pd.DataFrame, results_dir: Path):
        p_cutoff = 0.05
        fc_cutoff = 1.0

        analysis_df = pivot_df.dropna(subset=["p-value", "log2FC"]).copy()

        if analysis_df.empty:
            st.error("No valid statistical data found for GO enrichment.")
            self.logger.log("❗ analysis_df is empty")
        else:
            with st.spinner("Fetching GO terms from MyGene.info API..."):
                mg = mygene.MyGeneInfo()

                def get_clean_uniprot(name):
                    parts = str(name).split("|")
                    return parts[1] if len(parts) >= 2 else parts[0]

                analysis_df["UniProt"] = analysis_df["ProteinName"].apply(get_clean_uniprot)

                bg_ids = analysis_df["UniProt"].dropna().astype(str).unique().tolist()
                fg_ids = analysis_df[
                    (analysis_df["p-value"] < p_cutoff) &
                    (analysis_df["log2FC"].abs() >= fc_cutoff)
                ]["UniProt"].dropna().astype(str).unique().tolist()
                self.logger.log("✅ get_clean_uniprot applied")

                if len(fg_ids) < 3:
                    st.warning(
                        f"Not enough significant proteins "
                        f"(p < {p_cutoff}, |log2FC| ≥ {fc_cutoff}). "
                        f"Found: {len(fg_ids)}"
                    )
                    self.logger.log("❗ Not enough significant proteins")
                else:
                    res_list = mg.querymany(
                        bg_ids, scopes="uniprot", fields="go", as_dataframe=False
                    )
                    res_go = pd.DataFrame(res_list)
                    if "notfound" in res_go.columns:
                        res_go = res_go[res_go["notfound"] != True]

                    def extract_go_terms(go_data, go_type):
                        if not isinstance(go_data, dict) or go_type not in go_data:
                            return []
                        terms = go_data[go_type]
                        if isinstance(terms, dict):
                            terms = [terms]
                        return list({t.get("term") for t in terms if "term" in t})

                    for go_type in ["BP", "CC", "MF"]:
                        res_go[f"{go_type}_terms"] = res_go["go"].apply(
                            lambda x: extract_go_terms(x, go_type)
                        )

                    annotated_ids = set(res_go["query"].astype(str))
                    fg_set = annotated_ids.intersection(fg_ids)
                    bg_set = annotated_ids
                    self.logger.log(f"✅ fg_set bg_set are set")

                    def run_go(go_type):
                        go2fg = defaultdict(set)
                        go2bg = defaultdict(set)

                        for _, row in res_go.iterrows():
                            uid = str(row["query"])
                            for term in row[f"{go_type}_terms"]:
                                go2bg[term].add(uid)
                                if uid in fg_set:
                                    go2fg[term].add(uid)

                        records = []
                        N_fg = len(fg_set)
                        N_bg = len(bg_set)

                        for term, fg_genes in go2fg.items():
                            a = len(fg_genes)
                            if a == 0:
                                continue
                            b = N_fg - a
                            c = len(go2bg[term]) - a
                            d = N_bg - (a + b + c)

                            _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                            records.append({
                                "GO_Term": term,
                                "Count": a,
                                "GeneRatio": f"{a}/{N_fg}",
                                "p_value": p,
                            })

                        df = pd.DataFrame(records)
                        if df.empty:
                            return None, None

                        df["-log10(p)"] = -np.log10(df["p_value"].replace(0, 1e-10))
                        df = df.sort_values("p_value").head(20)

                        # ✅ Plotly Figure
                        fig = px.bar(
                            df,
                            x="-log10(p)",
                            y="GO_Term",
                            orientation="h",
                            title=f"GO Enrichment ({go_type})",
                        )

                        self.logger.log(f"✅ Plotly Figure generated")

                        fig.update_layout(
                            yaxis=dict(autorange="reversed"),
                            height=500,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )

                        return fig, df

                    go_results = {}

                    for go_type in ["BP", "CC", "MF"]:
                        fig, df_go = run_go(go_type)
                        if fig is not None:
                            go_results[go_type] = {
                                "fig": fig,
                                "df": df_go
                            }
                    self.logger.log(f"✅ go_type generated")

                    go_dir = results_dir / "go-terms"
                    go_dir.mkdir(parents=True, exist_ok=True)

                    import json
                    go_data = {}
                    
                    for go_type in ["BP", "CC", "MF"]:
                        if go_type in go_results:
                            fig = go_results[go_type]["fig"]
                            df = go_results[go_type]["df"]
                            
                            go_data[go_type] = {
                                "fig_json": fig.to_json(),  # Figure → JSON string
                                "df_dict": df.to_dict(orient="records")  # DataFrame → list of dicts
                            }
                    
                    go_json_file = go_dir / "go_results.json"
                    with open(go_json_file, "w") as f:
                        json.dump(go_data, f)
                    st.session_state["go_results"] = go_results
                    st.session_state["go_ready"] = True if go_data else False
                    self.logger.log("✅ GO enrichment analysis complete")
        

    @st.fragment
    def results(self) -> None:

        st.title("📊 Results")

        comet_tab, perc_tab, filter_tab, lfq_tab = st.tabs([
            "🔍 Identification",
            "🔍 Rescoring",
            "🔍 Filtering",
            "🔍 Quantification"
        ])

        # ================================
        # 🔍 CometAdapter
        # ================================
        with comet_tab:

            comet_dir = Path(self.workflow_dir, "results", "comet_results")
            comet_files = sorted(comet_dir.glob("*.idXML"))

            if not comet_files:
                st.warning("⚠ No Identification output files found.")
                return

            selected_file = st.selectbox("📁 Select Identification result file", comet_files)

            def idxml_to_df(idxml_file):
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

            df = idxml_to_df(selected_file)

            if df.empty:
                st.info("No peptide hits found.")
                return
                
            st.dataframe(df, use_container_width=True)

            df_plot = df.reset_index()

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

            clicked = plotly_events(fig, click_event=True, hover_event=False, override_height=550, key="comet_plot")

            if clicked:
                row_index = clicked[0]["pointNumber"]
                st.subheader("📌 Selected Peptide Match")
                st.dataframe(df.iloc[[row_index]], use_container_width=True)

        # ================================
        # 🔍 PercolatorAdapter RESULTS
        # ================================
        with perc_tab:

            perc_dir = Path(self.workflow_dir, "results", "percolator_results")
            perc_files = sorted(perc_dir.glob("*.idXML"))

            if not perc_files:
                st.warning("⚠ No Rescoring output files found.")
                return

            selected_perc = st.selectbox("📁 Select Rescoring result file", perc_files)

            df_p = idxml_to_df(selected_perc)

            if df_p.empty:
                st.info("No peptide hits found in Rescoring result.")
                return

            st.dataframe(df_p, use_container_width=True)

            df_plot_p = df_p.reset_index()

            fig2 = px.scatter(
                df_plot_p,
                x="RT",
                y="m/z",
                color="Score",
                custom_data=["index", "Sequence", "Proteins"],
                color_continuous_scale=["#a6cee3", "#1f78b4", "#08519c", "#08306b"],
            )
            fig2.update_traces(
                marker=dict(size=6, opacity=0.8),
                hovertemplate='<b>Index: %{customdata[0]}</b><br>'
                            + 'RT: %{x:.2f}<br>'
                            + 'm/z: %{y:.4f}<br>'
                            + 'Score: %{marker.color:.3f}<br>'
                            + 'Sequence: %{customdata[1]}<br>'
                            + 'Proteins: %{customdata[2]}<br>'
                            + '<extra></extra>'
            )
            fig2.update_layout(
                coloraxis_colorbar=dict(title="Score"),
                hovermode="closest"
            )

            clicked2 = plotly_events(fig2, click_event=True, hover_event=False, override_height=550, key="perc_plot")

            if clicked2:
                idx = clicked2[0]["pointNumber"]
                st.subheader("📌 Selected Rescoring Peptide Match")
                st.dataframe(df_p.iloc[[idx]], use_container_width=True)

        # ================================
        # 🔍 IDFilter RESULTS
        # ================================
        with filter_tab:

            filter_dir = Path(self.workflow_dir, "results", "filter_results")
            filter_files = sorted(filter_dir.glob("*.idXML"))

            if not filter_files:
                st.warning("⚠ No Filtering output files found.")
                return
            
            st.info("Here you can explore the PSM scatterplot along with the detailed PSM table.")

            selected_filter = st.selectbox("📁 Select Filtering result file", filter_files)

            df_f = idxml_to_df(selected_filter)

            if df_f.empty:
                st.info("No peptide hits found in Filtering result.")
                return

            st.dataframe(df_f, use_container_width=True)

            df_plot_f = df_f.reset_index()

            fig3 = px.scatter(
                df_plot_f,
                x="RT",
                y="m/z",
                color="Score",
                custom_data=["index", "Sequence", "Proteins"],
                color_continuous_scale=["#a6cee3", "#1f78b4", "#08519c", "#08306b"],
            )
            fig3.update_traces(
                marker=dict(size=6, opacity=0.8),
                hovertemplate='<b>Index: %{customdata[0]}</b><br>'
                            + 'RT: %{x:.2f}<br>'
                            + 'm/z: %{y:.4f}<br>'
                            + 'Score: %{marker.color:.3f}<br>'
                            + 'Sequence: %{customdata[1]}<br>'
                            + 'Proteins: %{customdata[2]}<br>'
                            + '<extra></extra>'
            )
            fig3.update_layout(
                coloraxis_colorbar=dict(title="Score"),
                hovermode="closest"
            )

            clicked3 = plotly_events(fig3, click_event=True, hover_event=False, override_height=550, key="filter_plot")

            if clicked3:
                idx3 = clicked3[0]["pointNumber"]
                st.subheader("📌 Selected Filtering Peptide Match")
                st.dataframe(df_f.iloc[[idx3]], use_container_width=True)

        # ================================
        # 📊 ProteomicsLFQ RESULTS
        # ================================
        with lfq_tab:

            results_dir = Path(self.workflow_dir, "results")
            proteomicslfq_dir = results_dir / "quant_results"

            if not proteomicslfq_dir.exists():
                st.warning("❗ Quantification directory not found. Please run the analysis first.")
                return

            csv_files = sorted(proteomicslfq_dir.glob("*.csv"))

            if not csv_files:
                st.info("No CSV files found in the Quantification directory.")
                return

            csv_file = csv_files[0]

            # Protein / PSM table tab
            protein_tab, psm_tab, volcano_tab, pca_tab, heatmap_tab = st.tabs(["🧬 Protein Table", "📄 PSM-level Quantification Table", "🌋 Statistical Analysis Volcano Plot", "📊 PCA", "🔥 Heatmap",])

            try:
                df = pd.read_csv(csv_file)

                if df.empty:
                    st.info("No data found in this file.")
                    return

                # Protein-level Table
                with protein_tab:

                    st.markdown("### 🧬 Protein-Level Abundance Table")

                    st.info(
                        "💡INFO\n\n"
                        "This protein-level table is generated by grouping all PSMs that map to the "
                        "same protein and aggregating their intensities across samples.\n"
                        "Additionally, log2 fold change and p-values are calculated between sample groups."
                    )

                    # --------------------------------------------------
                    # 1️⃣ Load group information from self.params
                    # --------------------------------------------------
                    group_map = {
                        key[11:]: value  # Remove "mzML-group-" prefix
                        for key, value in self.params.items()
                        if key.startswith("mzML-group-") and value
                    }

                    if not group_map:
                        st.warning("No group information found. Please define sample groups first.")
                        return

                    # --------------------------------------------------
                    # 2️⃣ Sample → Group mapping
                    # --------------------------------------------------
                    df["Sample"] = df["Reference"].str.replace(".mzML", "", regex=False)
                    df["Group"] = df["Reference"].map(group_map)

                    df = df.dropna(subset=["Group"])

                    # --------------------------------------------------
                    # 3️⃣ Determine comparison groups
                    # --------------------------------------------------
                    groups = sorted(df["Group"].unique())

                    if len(groups) < 2:
                        # st.write("Group map:", group_map)
                        # st.write("Reference examples:", df["Reference"].unique()[:5])
                        # st.write("Sample examples:", df["Sample"].unique()[:5])
                        # st.write("Groups after mapping:", df["Group"].unique())
                        st.warning("At least two groups are required for statistical comparison.")
                        return

                    group1, group2 = groups[:2]

                    st.info(f"Statistical comparison: **{group2} vs {group1}**")

                    # --------------------------------------------------
                    # 4️⃣ Protein-level statistics
                    # --------------------------------------------------
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

                        stats_rows.append(
                            {
                                "ProteinName": protein,
                                "log2FC": log2fc,
                                "p-value": pval,
                            }
                        )

                    stats_df = pd.DataFrame(stats_rows)

                    # --------------------------------------------------
                    # 5️⃣ Build protein abundance table (pivot)
                    # --------------------------------------------------
                    all_samples = sorted(df["Sample"].unique())
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

                    # --------------------------------------------------
                    # 6️⃣ Merge statistics + reorder columns
                    # --------------------------------------------------
                    pivot_df = pivot_df.merge(stats_df, on="ProteinName", how="left")

                    pivot_df = pivot_df[
                        ["ProteinName", "log2FC", "p-value"] + all_samples + ["PeptideSequence"]
                    ]

                    # ================================
                    # Common matrix for PCA / Heatmap
                    # ================================
                    expr_df = pivot_df.set_index("ProteinName")[all_samples]
                    expr_df = expr_df.replace(0, np.nan)   # treat missing values (not detected)
                    expr_df = np.log2(expr_df + 1)         # log2 transform
                    expr_df = expr_df.dropna()             # remove proteins with missing values (row-wise)

                    st.session_state["pivot_df"] = pivot_df
                    st.session_state["expr_df"] = expr_df
                    st.session_state["group_map"] = group_map

                    # --------------------------------------------------
                    # 7️⃣ Display
                    # --------------------------------------------------
                    st.dataframe(
                        pivot_df.sort_values("p-value"),
                        use_container_width=True,
                    )

                # PSM-level Table
                with psm_tab:
                    st.markdown(f"### 📄 PSM-level Quantification Table")
                    st.info("💡INFO \n\n This table shows the PSM-level quantification data, including protein IDs,peptide sequences, charge states, and intensities across samples.Each row represents one peptide-spectrum match detected from the MS/MS analysis.")
                    st.dataframe(df, use_container_width=True)


            except Exception as e:
                st.error(f"Failed to load {csv_file.name}: {e}")