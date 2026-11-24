"""
Unit tests for Stage 8 figures and summary documentation.
"""

from pathlib import Path
import config


def test_all_figures_generated():
    """Verify all 6 required project figures exist on disk."""
    expected_figures = [
        "class_distribution.png",
        "feature_maps_vs_classical.png",
        "kernel_heatmaps_comparison.png",
        "settings_comparison.png",
        "kernel_variance_vs_qubits.png",
        "imbalance_study.png",
    ]
    for fig_name in expected_figures:
        fig_path = config.FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure file is unexpectedly small: {fig_path}"


def test_summary_markdown_exists():
    """Verify that results/SUMMARY.md exists and contains required sections."""
    summary_path = config.RESULTS_DIR / "SUMMARY.md"
    assert summary_path.exists(), f"Missing file: {summary_path}"

    with open(summary_path, "r") as f:
        content = f.read()

    assert "Executive Summary: Answers to Core Research Questions" in content
    assert "Table 1: Classical Baselines Benchmarking" in content
    assert "Table 2: Ideal Quantum Kernel Benchmarking" in content
    assert "Table 3: Noisy Simulation Distortion" in content
    assert "Table 4: Hardware Benchmark & Error Mitigation" in content
    assert "Table 5: Imbalance Study" in content
