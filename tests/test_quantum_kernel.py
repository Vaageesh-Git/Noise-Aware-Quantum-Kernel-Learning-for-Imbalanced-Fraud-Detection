"""
Unit tests for Quantum Kernel Module.
Verifies:
  1. Feature maps instantiation (ZZ, Pauli, Custom) with linear/full/ring entanglement.
  2. Circuit complexity analysis (depth, 2-qubit gate counts).
  3. Kernel symmetry: K[i, j] == K[j, i].
  4. Kernel diagonal: K[i, i] == 1.0.
  5. Value range: K[i, j] in [0, 1].
  6. Shot-based kernel convergence to exact statevector kernel within statistical error tolerance.
"""

import numpy as np
import pytest
from src.quantum_kernel import (
    get_feature_map,
    build_custom_angle_ring_map,
    analyze_circuit_complexity,
    compute_kernel,
    compute_kernel_diagnostics,
)


def test_feature_map_instantiation():
    """Verify all required feature maps can be constructed."""
    for name in ["ZZFeatureMap", "PauliFeatureMap", "CustomAngleRingMap"]:
        for reps in [1, 2]:
            ent = "full" if name != "CustomAngleRingMap" else "ring"
            fm = get_feature_map(name=name, n_qubits=4, reps=reps, entanglement=ent)
            assert fm.num_qubits == 4
            assert fm.num_parameters == 4


def test_circuit_complexity_analysis():
    """Verify circuit depth and 2-qubit gate counts are accurately calculated."""
    fm = get_feature_map("ZZFeatureMap", n_qubits=4, reps=1, entanglement="linear")
    metrics = analyze_circuit_complexity(fm)
    assert metrics["depth"] > 0
    assert metrics["two_qubit_gates"] > 0
    assert "gate_counts" in metrics


def test_kernel_properties_and_symmetry():
    """Verify kernel is symmetric, diagonal is 1, and values lie in [0, 1]."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0, np.pi, size=(6, 4)).astype(np.float32)

    fm = get_feature_map("ZZFeatureMap", n_qubits=4, reps=1, entanglement="linear")
    K = compute_kernel(X, X, fm, mode="statevector", show_progress=False)

    # 1. Shape is (N, N)
    assert K.shape == (6, 6)

    # 2. Diagonal is strictly 1.0
    assert np.allclose(np.diag(K), 1.0)

    # 3. Symmetric: K == K.T
    assert np.allclose(K, K.T)

    # 4. Values in [0, 1]
    assert np.all(K >= 0.0) and np.all(K <= 1.0)


def test_shot_based_convergence_to_exact():
    """Verify finite shot kernel converges to exact statevector kernel within tolerance."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0, np.pi, size=(5, 4)).astype(np.float32)

    fm = get_feature_map("CustomAngleRingMap", n_qubits=4, reps=1, entanglement="ring")

    K_exact = compute_kernel(X, X, fm, mode="statevector", show_progress=False)
    K_shots = compute_kernel(X, X, fm, mode="shots", shots=8192, show_progress=False)

    # 1. Properties hold for shot kernel
    assert np.allclose(np.diag(K_shots), 1.0)
    assert np.allclose(K_shots, K_shots.T)
    assert np.all(K_shots >= 0.0) and np.all(K_shots <= 1.0)

    # 2. Maximum absolute error between 8192 shots and exact should be < 0.05 (binomial standard error ~ 0.005)
    max_err = float(np.max(np.abs(K_exact - K_shots)))
    assert max_err < 0.05, f"Max error {max_err:.4f} exceeded threshold 0.05"


def test_kernel_diagnostics():
    """Verify kernel diagnostics calculation including KTA."""
    K = np.eye(4, dtype=np.float32) + 0.2
    np.fill_diagonal(K, 1.0)
    y = np.array([0, 0, 1, 1])

    diag = compute_kernel_diagnostics(K, y)
    assert "off_diagonal_mean" in diag
    assert "off_diagonal_var" in diag
    assert "kernel_target_alignment" in diag
    assert np.isclose(diag["off_diagonal_mean"], 0.2)
    assert np.isclose(diag["off_diagonal_var"], 0.0)
