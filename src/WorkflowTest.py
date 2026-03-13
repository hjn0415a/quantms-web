import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
from pyopenms import IdXMLFile
from scipy.stats import ttest_ind
import numpy as np

from src.workflow.WorkflowManager import WorkflowManager
from src.common.results_helpers import parse_idxml, build_spectra_cache
from openms_insight import Table, Heatmap, LinePlot, SequenceView


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

        t = st.tabs([
            "**IsobaricAnalyzer**", 
            "**CometAdapter**", 
            "**PercolatorAdapter**", 
            "**IDFilter**", 
            "**IDMapper**", 
            "**FileMerger**", 
            "**ProteinInference**",
            "**IDFilter**", 
            "**IDConflictResolver**",
            "**ProteinQuantifier**",
            "**Group Selection**"])

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
                             "instrument", "fragment_mass_tolerance", "fragment_error_units", "fragment_bin_offset", "PeptideIndexing:IL_equivalent"]
            if not self.params.get("generate-decoys", True):
                # Only show decoy_string when not generating decoys
                comet_include.append("PeptideIndexing:decoy_string")

            self.ui.input_TOPP(
                "IsobaricAnalyzer",
                custom_defaults={
                    "type": "tmt11plex",
                    "threads": 8,
                    "extraction:select_activation": "auto",
                    "extraction:reporter_mass_shift": 0.002,
                    "extraction:min_reporter_intensity": 0.0,
                    "extraction:min_precursor_purity": 0.0,
                    "extraction:precursor_isotope_deviation": 10.0,
                    "quantification:isotope_correction": "false",
                }
            )

        with t[1]:
            st.info("""
            **Rescoring (Percolator):**
            * **post_processing_tdc**: Use target-decoy competition to assign q-values and PEPs.
            * **score_type**: Type of the peptide main score
            * **subset_max_train**: Only train an SVM on a subset of <x> PSMs, and use the resulting score vector to evaluate the other PSMs. Recommended when analyzing huge numbers (>1 million) of PSMs. When set to 0, all PSMs are used for training as normal.
            """)
            self.ui.input_TOPP(
                "CometAdapter",
                custom_defaults={
                    "instrument": "high_res",
                    "missed_cleavages": 2,
                    "min_peptide_length": 6,
                    "max_peptide_length": 40,
                    "num_hits": 1,
                    "enzyme": "Trypsin/P",
                    "isotope_error": "0/1",
                    "precursor_charge": "2:4",
                    "max_variable_mods_in_peptide": 3,
                    "precursor_mass_tolerance": 4.5,
                    "fragment_mass_tolerance": 0.015,
                },
                include_parameters=comet_include,
                exclude_parameters=["second_enzyme"],
            )

        with t[2]:
            st.info("""
            **Filtering (IDFilter):**
            * **score:type_peptide**: Score used for filtering. If empty, the main score is used.
            * **score:psm**: The score which should be reached by a peptide hit to be kept. (use 'NAN' to disable this filter)
            """)
            self.ui.input_TOPP(
                "PercolatorAdapter",
                custom_defaults={
                    "max_threads": 8,
                    "use_all_psms": "true",
                    "subset_max_train": 300000,
                    "decoy_pattern": "DECOY_",
                    "score_type": "pep"
                },
                include_parameters=["post_processing_tdc"],
            )

        with t[3]:
            self.ui.input_TOPP(
                "IDFilter",
                tool_instance_name="IDFilter-strict",
                custom_defaults={
                    "score:type_peptitde": "q-value",
                    "score:psm": 0.10,
                }
            )

        with t[4]:
            st.info("""
            **Quantification (ProteomicsLFQ):**
            * **intThreshold**: Peak intensity threshold applied in seed detection.
            * **psmFDR**: FDR threshold for sub-protein level (e.g. 0.05=5%). Use -FDR_type to choose the level. Cutoff is applied at the highest level. If Bayesian inference was chosen, it is equivalent with a peptide FDR
            * **proteinFDR**: Protein FDR threshold (0.05=5%).
            """)
            self.ui.input_TOPP(
                "IDMapper",
                custom_defaults={
                    "threads": 8,
                }
            )
        with t[5]:
            self.ui.input_TOPP(
                "FileMerger",
                custom_defaults={
                    "in_type": "consensusXML",
                    "append_method": "append_cols",
                    "threads": 8,
                },
                include_parameters=["annotate_file_origin"]
            )
        with t[6]:
            self.ui.input_TOPP(
                "ProteinInference",
                custom_defaults={
                    "threads": 8,
                    "picked_decoy_string": "DECOY_",
                    "protein_fdr": True,
                    "Algorithm:score_type": "PEP",
                }
            )

        with t[7]:
            self.ui.input_TOPP(
                "IDFilter",
                tool_instance_name="IDFilter-lenient",
                custom_defaults={
                    "score:type_protein": "q-value",
                    "score:protein": 0.01,
                    "score:psm": 0.01,
                    "delete_unreferenced_peptide_hits": "true",
                    "remove_decoys": "true",
                }
            )

        with t[8]:
            self.ui.input_TOPP(
                "IDConflictResolver",
                custom_defaults={
                    "threads": 4,
                }
            )

        with t[9]:
            self.ui.input_TOPP(
                "ProteinQuantifier",
                custom_defaults={
                    "threads": 8,
                },
            )

        with t[10]:
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
        Isobaric proteomics workflow (list-based execution):
        IsobaricAnalyzer
        → CometAdapter
        → PercolatorAdapter
        → IDFilter (PSM)
        → IDMapper
        → FileMerger
        → ProteinInference
        → IDFilter (Protein)
        → IDConflictResolver
        → MSStatsConverter
        → ProteinQuantifier
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
        iso_dir = results_dir / "isobaric_consensusXML"
        comet_dir = results_dir / "comet"
        perc_dir = results_dir / "percolator"
        psm_filter_dir = results_dir / "psm_filter"
        map_dir = results_dir / "idmapper"
        merge_dir = results_dir / "merged"
        protein_dir = results_dir / "protein"
        msstats_dir = results_dir / "msstats"
        quant_dir = results_dir / "quant"

        for d in [
            iso_dir, comet_dir, perc_dir, psm_filter_dir,
            map_dir, merge_dir, protein_dir, msstats_dir, quant_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)


        self.logger.log("📁 Output directories created")

        # ================================
        # 2️⃣ File path definitions (per sample)
        # ================================
        iso_consensus = []
        comet_results = []
        percolator_results = []
        psm_filtered = []
        mapped_ids = []

        for mz in in_mzML:
            stem = Path(mz).stem
            iso_consensus.append(str(iso_dir / f"{stem}_iso.consensusXML"))
            comet_results.append(str(comet_dir / f"{stem}_comet.idXML"))
            percolator_results.append(str(perc_dir / f"{stem}_comet_perc.idXML"))
            psm_filtered.append(str(psm_filter_dir / f"{stem}_comet_perc_filter.idXML"))
            mapped_ids.append(str(map_dir / f"{stem}_comet_perc_filter_map.consensusXML"))

        merged_id = str(merge_dir / "ID_mapper_merge.consensusXML")
        protein_id = str(protein_dir / "ID_mapper_merge_epi.consensusXML")
        protein_filter = str(protein_dir / "ID_mapper_merge_epi_filter.consensusXML")
        protein_resolved = str(protein_dir / "ID_mapper_merge_epi_filter_resconf.consensusXML")
        msstats_input = str(msstats_dir / "msstats_input.csv")
        consensus_out = str(quant_dir / "openms_design_protein_openms.csv")

        # ================================
        # 3️⃣ Per-file processing
        # ================================
        for i, mz in enumerate(in_mzML):
            stem = Path(mz).stem
            st.info(f"Processing sample: {stem}")

        self.logger.log("🔬 Starting per-sample processing...")

        # --- IsobaricAnalyzer ---
        self.logger.log("🏷️ Running isobaric labeling analysis...")
        with st.spinner("IsobaricAnalyzer"):
            if not self.executor.run_topp(
                "IsobaricAnalyzer",
                {
                    "in": in_mzML,
                    "out": iso_consensus,
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ IsobaricAnalyzer complete")

        # --- CometAdapter ---
        self.logger.log("🔎 Running peptide search...")
        with st.spinner(f"CometAdapter"):
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
        with st.spinner(f"PercolatorAdapter"):
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

        # --- IDFilter ---
        self.logger.log("🔧 Filtering identifications...")
        with st.spinner(f"IDFilter"):
            if not self.executor.run_topp(
                "IDFilter",
                {
                    "in": percolator_results,
                    "out": psm_filtered,
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False

        # Build visualization cache for Filter results
        for idxml_file in psm_filtered:
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

        # ================================
        # ✨ NEW: 4️⃣ IDMapper (per-file: iso_consensus + psm_filtered → mapped_ids)
        # ================================
        self.logger.log("🗺️ Mapping IDs to isobaric consensus features...")
        for iso, psm, mapped in zip(iso_consensus, psm_filtered, mapped_ids):
            iso_stem = Path(iso).stem
            with st.spinner(f"IDMapper ({iso_stem})"):
                if not self.executor.run_topp(
                    "IDMapper",
                    {
                        "in": [iso],
                        "id": [psm],
                        "out": [mapped],
                    },
                ):
                    self.logger.log("Workflow stopped due to error")
                    return False
        self.logger.log("✅ IDMapper complete")

        # ================================
        # ✨ NEW: 5️⃣ FileMerger (mapped_ids → merged_id)
        # ================================
        self.logger.log("🔗 Merging mapped consensus files...")
        with st.spinner("FileMerger"):
            if not self.executor.run_topp(
                "FileMerger",
                {
                    "in": mapped_ids,
                    "out": [merged_id],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ FileMerger complete")

        # ================================
        # ✨ NEW: 6️⃣ ProteinInference / Epifany (merged_id → protein_id)
        # ================================
        self.logger.log("🧩 Running protein inference...")
        with st.spinner("Epifany (ProteinInference)"):
            if not self.executor.run_topp(
                "Epifany",
                {
                    "in": [merged_id],
                    "out": [protein_id],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ ProteinInference complete")

        # ================================
        # ✨ NEW: 7️⃣ IDFilter (Protein level) (protein_id → protein_filter)
        # ================================
        self.logger.log("🔬 Filtering proteins...")
        with st.spinner("IDFilter (Protein)"):
            if not self.executor.run_topp(
                "IDFilter",
                {
                    "in": [protein_id],
                    "out": [protein_filter],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ IDFilter (Protein) complete")

        # ================================
        # ✨ NEW: 8️⃣ IDConflictResolver (protein_filter → protein_resolved)
        # ================================
        self.logger.log("⚖️ Resolving ID conflicts...")
        with st.spinner("IDConflictResolver"):
            if not self.executor.run_topp(
                "IDConflictResolver",
                {
                    "in": [protein_filter],
                    "out": [protein_resolved],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ IDConflictResolver complete")

        # ================================
        # ✨ NEW: 9️⃣ MSStatsConverter (protein_resolved → msstats_input)
        # ================================
        # self.logger.log("📋 Converting to MSstats format...")
        # with st.spinner("MSStatsConverter"):
        #     if not self.executor.run_topp(
        #         "MSstatsConverter",
        #         {
        #             "in": [protein_resolved],
        #             "out": [msstats_input],
        #         },
        #     ):
        #         self.logger.log("Workflow stopped due to error")
        #         return False
        # self.logger.log("✅ MSStatsConverter complete")

        # ================================
        # ✨ NEW: 🔟 ProteinQuantifier (protein_resolved → consensus_out)
        # ================================
        self.logger.log("📐 Running protein quantification...")
        with st.spinner("ProteinQuantifier"):
            if not self.executor.run_topp(
                "ProteinQuantifier",
                {
                    "in": [protein_resolved],
                    "out": [consensus_out],
                },
            ):
                self.logger.log("Workflow stopped due to error")
                return False
        self.logger.log("✅ ProteinQuantifier complete")

        self.logger.log("🎉 Isobaric proteomics workflow completed successfully!")
        st.success("✅ Isobaric proteomics workflow completed!")
        return True

    @st.fragment
    def results(self) -> None:

        st.title("📊 Results")

        search_engine = self.params.get("search-engine", "comet")
        engine_suffix = "comet" if search_engine == "comet" else "sage"

        ident_tab, perc_tab, filter_tab, lfq_tab = st.tabs([
            "🔍 Identification",
            "🔍 Rescoring",
            "🔍 Filtering",
            "🔍 Quantification"
        ])

        # ================================
        # 🔍 Identification (Comet / Sage)
        # ================================
        with ident_tab:

            search_dir = Path(self.workflow_dir, "results", f"{engine_suffix}_results")
            search_files = sorted(search_dir.glob("*.idXML"))

            if not search_files:
                st.warning("⚠ No Identification output files found.")
                return

            selected_file = st.selectbox("📁 Select Identification result file", search_files)

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

            clicked = plotly_events(fig, click_event=True, hover_event=False, override_height=550, key="sage_plot")

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