"""Round 70 — Sphinx docs 구조 검증 (빌드 없이)."""

from __future__ import annotations

import importlib
import runpy
from argparse import _SubParsersAction
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"


class TestDocsStructure:
    def test_conf_exists(self) -> None:
        assert (DOCS / "source" / "conf.py").exists()

    def test_conf_release_tracks_package_version(self) -> None:
        """Sphinx release metadata must track the package version."""
        from naviertwin import __version__

        namespace = runpy.run_path(str(DOCS / "source" / "conf.py"))

        assert namespace["release"] == __version__
        assert namespace["version"] == ".".join(__version__.split(".")[:2])

    def test_index_exists(self) -> None:
        idx = DOCS / "source" / "index.rst"
        assert idx.exists()
        content = idx.read_text(encoding="utf-8")
        assert "toctree" in content

    def test_api_stubs(self) -> None:
        api = DOCS / "source" / "api"
        assert api.exists()
        # 주요 패키지 존재
        for pkg in [
            "cfd_reader",
            "dimensionality_reduction",
            "active_learning",
            "operator_learning",
            "gui",
            "post_process_facade",
        ]:
            assert (api / f"{pkg}.rst").exists()

    def test_operator_learning_api_docs_include_implemented_subpackages(self) -> None:
        """Operator-learning docs should expose shipped customer-facing operators."""
        page = (DOCS / "source" / "api" / "operator_learning.rst").read_text(
            encoding="utf-8"
        )

        for automodule in [
            "naviertwin.core.operator_learning",
            "naviertwin.core.operator_learning.kan",
        ]:
            assert f".. automodule:: {automodule}" in page

    def test_cfd_reader_api_docs_include_advanced_helpers(self) -> None:
        """CFD reader docs should expose shipped format-specific helper APIs."""
        page = (DOCS / "source" / "api" / "cfd_reader.rst").read_text(
            encoding="utf-8"
        )

        for automodule in [
            "naviertwin.core.cfd_reader",
            "naviertwin.core.cfd_reader.foamlib_case",
            "naviertwin.core.cfd_reader.cgns_advanced",
            "naviertwin.core.cfd_reader.fluent_cas_ext",
        ]:
            assert f".. automodule:: {automodule}" in page

    def test_gui_api_docs_are_discoverable(self) -> None:
        """GUI API docs should expose the shipped desktop public surface."""
        index = (DOCS / "source" / "index.rst").read_text(encoding="utf-8")
        page = (DOCS / "source" / "api" / "gui.rst").read_text(encoding="utf-8")

        assert "api/gui" in index
        for automodule in [
            "naviertwin.gui",
            "naviertwin.gui.panels",
            "naviertwin.gui.widgets",
        ]:
            assert f".. automodule:: {automodule}" in page

    def test_makefile(self) -> None:
        mf = DOCS / "Makefile"
        assert mf.exists()
        content = mf.read_text(encoding="utf-8")
        assert "html:" in content

    def test_overview_has_korean(self) -> None:
        ov = DOCS / "source" / "overview.rst"
        content = ov.read_text(encoding="utf-8").replace("\n", " ")
        assert "디지털" in content and "트윈" in content

    def test_docs_cli_reference_lists_public_subcommands(self) -> None:
        """CLI reference must stay in sync with parser public subcommands."""
        from naviertwin.main import _build_parser

        parser = _build_parser()
        subparsers = [
            action for action in parser._actions if isinstance(action, _SubParsersAction)
        ]
        assert len(subparsers) == 1

        cli = DOCS / "source" / "cli.rst"
        index = DOCS / "source" / "index.rst"
        content = cli.read_text(encoding="utf-8")
        index_content = index.read_text(encoding="utf-8")

        assert "   cli" in index_content
        assert "naviertwin --version" in content
        for command in sorted(subparsers[0].choices):
            assert command in content
            assert f"naviertwin {command}" in content
        assert "Expected:" in content

    def test_customer_docs_do_not_hardcode_current_package_version(self) -> None:
        """Customer-facing docs should not stale when the package version changes."""
        from naviertwin import __version__

        customer_docs = [
            ROOT / "README.md",
            DOCS / "source" / "cli.rst",
        ]

        for path in customer_docs:
            assert __version__ not in path.read_text(encoding="utf-8")

    def test_gui_docs_track_customer_facing_tabs_and_tools(self) -> None:
        """GUI docs must reflect currently exposed commercial desktop workflows."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        gui = (DOCS / "source" / "gui.rst").read_text(encoding="utf-8")
        overview = (DOCS / "source" / "overview.rst").read_text(encoding="utf-8")
        combined = readme + "\n" + gui + "\n" + overview

        assert "10 탭" in readme
        assert "10 개 탭 구조" in gui
        assert "PySide6 기반 10 탭 인터페이스" in overview
        for token in [
            "Post-Tools",
            "29개 facade",
            "Explain",
            "Kernel SHAP",
            "symbolic expression",
            "Attention viz",
            "Active Learning",
            "Online Update",
            "4D-Var",
            "UKF",
            "NSGA-II",
            "SIMP",
            "SINDy",
            "ONNX",
            "TorchScript",
            "파이프라인 데모",
            "API 서버",
        ]:
            assert token in combined
        assert "GUI: Import / Analyze / Reduce / Model / Twin / Export 6 탭" not in readme
        assert "8 개 탭 구조" not in gui
        assert "6+ 탭 인터페이스" not in overview

    def test_post_process_facade_docs_are_discoverable(self) -> None:
        """Facade API docs must advertise the reusable operation contract."""
        index = (DOCS / "source" / "index.rst").read_text(encoding="utf-8")
        page = (DOCS / "source" / "api" / "post_process_facade.rst").read_text(
            encoding="utf-8"
        )

        assert "api/post_process_facade" in index
        for token in [
            "PostProcessFacade",
            "list_operations()",
            "describe(op_name)",
            "run(op_name, **kwargs)",
            "surface forces",
            "plane flux",
            "EOF",
        ]:
            assert token in page

    def test_public_api_package_initializers_are_not_placeholders(self) -> None:
        """Customer-visible API packages should export real implemented symbols."""
        expected = {
            "naviertwin.gui": ["MainWindow"],
            "naviertwin.gui.panels": [
                "ImportPanel",
                "AnalyzePanel",
                "ReducePanel",
                "ModelPanel",
                "TwinPanel",
                "ExplainabilityPanel",
                "ExportPanel",
                "SimulationPanel",
                "PostProcessPanel",
            ],
            "naviertwin.gui.widgets": [
                "VtkViewer",
                "ModelCompareWidget",
                "LossCurveWidget",
            ],
            "naviertwin.core.cfd_reader": [
                "BaseReader",
                "CFDDataset",
                "ReaderFactory",
                "OpenFOAMReader",
                "VTKReader",
                "FluentReader",
                "CGNSReader",
                "GmshReader",
                "SU2Reader",
                "read_foam_dict",
                "modify_transport_properties",
                "parameter_sweep",
                "sample_field_at_points",
                "iter_zones",
                "list_zones",
                "parse_section_ids",
                "list_zone_names",
            ],
            "naviertwin.core.report": ["ReportGenerator", "HTMLReport", "MarkdownReport"],
            "naviertwin.core.multi_fidelity": ["AdditiveCoKriging", "freeze_layers"],
            "naviertwin.core.sensitivity": ["saltelli_sample", "sobol_indices"],
            "naviertwin.core.physics_correction": [
                "HybridROM",
                "project_linear_constraint",
            ],
            "naviertwin.core.physnemo": ["PINNSolver", "PhysicsNEMOWrapper"],
            "naviertwin.core.operator_learning.fno": ["FNO1D", "TFNO2D", "WNO1D"],
            "naviertwin.core.operator_learning.deeponet": [
                "DeepONet",
                "PIDeepONet",
            ],
            "naviertwin.core.operator_learning.unet": ["UNet2D", "UNet3D"],
            "naviertwin.core.operator_learning.koopman": ["KNO", "FlowDMD"],
            "naviertwin.core.operator_learning.kan": ["KANO1D"],
            "naviertwin.core.operator_learning.latent_operator": [
                "LDeepONet",
                "PILatentNO",
            ],
            "naviertwin.core.dimensionality_reduction": [
                "BaseReducer",
                "SnapshotPOD",
                "ConstrainedPOD",
                "DiffusionMaps",
                "Autoencoder",
            ],
            "naviertwin.core.dimensionality_reduction.linear": [
                "SnapshotPOD",
                "BalancedPOD",
                "ConstrainedPOD",
                "deim",
                "gnat_solve",
                "weighted_pod",
            ],
            "naviertwin.core.dimensionality_reduction.nonlinear": [
                "DiffusionMaps",
                "Autoencoder",
                "VAE",
                "isomap",
                "lle",
                "TuckerDecomposition",
            ],
            "naviertwin.core.surrogate": [
                "BaseSurrogate",
                "RBFSurrogate",
                "KrigingSurrogate",
                "RFFRegression",
                "batch_predict",
                "normalized_rmse",
            ],
            "naviertwin.core.online_learning": [
                "OnlineKriging",
                "OnlineNN",
                "active_loop",
                "select_next_samples",
            ],
            "naviertwin.core.active_learning": [
                "expected_improvement",
                "acquisition_expected_improvement",
                "query_expected_improvement",
                "top_variance_query",
                "greedy_batch_acquisition",
            ],
            "naviertwin.core.gnn": [
                "BaseGNN",
                "GNNSurrogate",
                "MeshGraphNets",
                "HAMLET",
            ],
            "naviertwin.core.gnn.gnn_surrogate": ["GNNSurrogate"],
            "naviertwin.core.gnn.meshgraphnets": ["MeshGraphNets"],
            "naviertwin.core.gnn.graph_transformer": ["HAMLET"],
            "naviertwin.core.time_series": [
                "BaseTimeSeries",
                "EchoStateNetwork",
                "LSTMForecaster",
                "TransformerForecaster",
                "NeuralODEForecaster",
                "LatentDynamicsForecaster",
                "TNO",
            ],
            "naviertwin.core.time_series.esn": ["EchoStateNetwork"],
            "naviertwin.core.time_series.lstm": ["LSTMForecaster"],
            "naviertwin.core.time_series.transformer": ["TransformerForecaster"],
            "naviertwin.core.time_series.neural_ode": ["NeuralODEForecaster"],
            "naviertwin.core.time_series.latent_dynamics": [
                "LatentDynamicsForecaster",
            ],
            "naviertwin.core.time_series.temporal_no": ["TNO"],
            "naviertwin.core.generative": [
                "BaseGenerative",
                "ConditionalVAE",
                "DiffusionPDE",
                "WaveletDiffusionNO",
                "langevin_sample",
                "euler_maruyama",
            ],
            "naviertwin.core.generative.conditional_gen": ["ConditionalVAE"],
            "naviertwin.core.generative.diffusion_pde": ["DiffusionPDE"],
            "naviertwin.core.generative.wavelet_diffusion": ["WaveletDiffusionNO"],
            "naviertwin.core.equivariant": [
                "BaseEquivariant",
                "C4EquivariantFNO2D",
                "SO2Canonicalizer",
                "SO2EquivariantOperator",
                "EGNN",
            ],
            "naviertwin.core.equivariant.group_equiv_fno": ["C4EquivariantFNO2D"],
            "naviertwin.core.equivariant.physics_embedded": [
                "SO2Canonicalizer",
                "SO2EquivariantOperator",
                "EGNN",
            ],
            "naviertwin.core.numerics": [
                "chebyshev_points",
                "chebyshev_diff_matrix",
                "clenshaw_curtis_weights",
                "integrate_cc",
            ],
            "naviertwin.core.sampling": [
                "generate_sweep",
                "halton",
                "latin_hypercube",
                "mc_integral",
                "poisson_disk_2d",
                "select_sensors",
                "SensorDMDPipeline",
            ],
            "naviertwin.core.benchmarks": [
                "generate_burgers_dataset",
                "generate_heat_dataset",
                "generate_cavity_dataset",
                "ghia_u_centerline",
            ],
            "naviertwin.core.turbulence": [
                "energy_spectrum_1d",
                "energy_spectrum_2d",
                "kolmogorov_slope",
                "eddy_viscosity",
                "production_rate",
                "k_epsilon_step",
            ],
            "naviertwin.core.flow_control": [
                "GaussianPolicy",
                "reinforce_update",
            ],
            "naviertwin.core.solver_interfaces": [
                "solve_burgers_1d",
                "solve_heat_1d",
                "minmod",
                "fvm_upwind_1d",
                "fvm_musclhancock_1d",
                "total_mass",
                "LBMD2Q9",
                "cubic_spline_kernel",
                "sph_density_1d",
                "sph_gradient_1d",
            ],
            "naviertwin.core.digital_twin": [
                "TwinEngine",
                "NavierTwinPipeline",
                "PipelineState",
                "StreamingDigitalTwin",
                "batch_predict_fields",
                "build_manifest",
                "save_manifest",
                "build_pipeline",
                "validate_config",
                "compare_models",
                "rank_table",
            ],
            "naviertwin.core.data_assimilation": [
                "EnKF",
                "EnKFSimple",
                "ParticleFilter",
                "KalmanFilter",
                "RLS",
                "ekf_step",
                "envar_analysis",
                "fd_gradient",
                "four_dvar_linear",
                "iekf_step",
                "mhe_estimate",
                "nonlinear_rts",
                "run_filter",
                "smooth_particles",
                "ukf_step",
            ],
            "naviertwin.core.validation": [
                "rmse",
                "r2_score",
                "relative_l2_error",
                "max_error",
                "compute_all_metrics",
                "AnalyticSolution",
                "couette_flow",
                "poiseuille_flow_2d",
                "poiseuille_pipe",
                "spectral_poiseuille",
                "compare_against_analytic",
                "kfold_scores",
                "grid_search",
                "wasserstein_1d",
                "mmd_gaussian",
                "kl_divergence_gaussian",
                "field_diff_stats",
                "hotspot_indices",
                "band_mask",
                "field_sanity_check",
                "detect_outliers_iqr",
                "detect_outliers_zscore",
                "psnr",
                "nrmse",
                "ssim",
                "channel_rmse",
                "channel_relative_error",
                "aggregated_rmse",
                "multi_output_r2",
                "cross_channel_correlation",
                "top_k_worst_channels",
                "per_sample_error_norm",
                "taylor_green_2d",
                "kinetic_energy_decay",
            ],
            "naviertwin.core.export": [
                "NTwinReader",
                "NTwinWriter",
                "load_dataset",
                "save_dataset",
                "export_to_onnx",
                "verify_onnx",
                "export_to_torchscript",
                "trace_and_save",
                "verify_script_matches",
                "dynamic_quantize",
                "compare_inference",
                "model_size_bytes",
                "suggest_mlmodel_name",
                "suggest_tflite_name",
            ],
            "naviertwin.core.optimization": [
                "BayesianOptimizer",
                "NSGA2",
                "SurrogateOptimizer",
                "PolynomialChaos",
                "AdamOpt",
                "SGDOpt",
                "armijo_backtrack",
                "bfgs_minimize",
                "bobyqa_lite",
                "check_wolfe",
                "ga",
                "mads_minimize",
                "pareto_mask",
                "propagate_mc",
                "sa",
                "simp_1d",
                "simp_2d",
                "sqp_eq",
                "successive_halving",
                "trust_region_minimize",
                "weight_grid",
            ],
            "naviertwin.core.flow_analysis.statistics": [
                "compute_fft",
                "two_point_correlation",
                "continuous_wavelet",
            ],
            "naviertwin.core.flow_analysis.vortex": [
                "compute_q_criterion",
                "compute_lambda2",
                "compute_ftle_2d",
            ],
            "naviertwin.core.flow_analysis.boundary_layer": [
                "compute_yplus",
                "boundary_layer_thicknesses",
                "skin_friction",
            ],
            "naviertwin.core.flow_analysis.thermofluids": [
                "reynolds",
                "prandtl",
                "entropy_generation_2d",
            ],
            "naviertwin.core.flow_analysis.modal": [
                "DMDAnalyzer",
                "KoopmanAnalysis",
                "compute_spod",
                "pde_find_1d",
            ],
            "naviertwin.core.flow_analysis": [
                "BaseFlowAnalyzer",
                "mean_field",
                "welch_psd",
                "pressure_force",
                "mass_flux",
                "interp_field",
                "cart_to_cyl",
                "slice_axis_aligned",
                "safe_eval",
                "RunningMoments",
                "savgol_filter",
                "percentile",
                "eof_decomposition",
                "ks_test_normal",
                "trigger_average",
                "gradient_2d",
                "find_critical_points",
                "anisotropy_tensor",
                "connected_components_2d",
                "tet_volume",
            ],
        }

        for module_name, symbols in expected.items():
            module = importlib.import_module(module_name)
            init_path = ROOT / "src" / Path(*module_name.split(".")) / "__init__.py"
            text = init_path.read_text(encoding="utf-8")
            assert "구현 예정" not in text
            exported = set(getattr(module, "__all__", []))
            for symbol in symbols:
                assert hasattr(module, symbol)
                assert symbol in exported
