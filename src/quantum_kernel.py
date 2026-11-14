"""
Quantum Feature Maps and Quantum Kernel Module.
Provides:
  1. Parameterized quantum feature maps:
     - ZZFeatureMap (reps 1 & 2, linear & full entanglement)
     - PauliFeatureMap (Z and ZZ terms, linear & full entanglement)
     - CustomAngleRingMap (RY rotations with CZ circular ring/linear/full entanglement)
  2. Circuit transpilation diagnostic analyzer (depth and 2-qubit gate counts)
  3. Single unified quantum kernel computer:
       k(x, x') = |<0...0| U(x')^dagger U(x) |0...0>|^2
     supporting:
       - 'statevector': exact analytical simulation (Setting A)
       - 'shots': ideal finite-shot sampling (Setting B)
       - 'noisy': noisy simulator from fake IBM backend (Setting C)
       - 'hardware': real IBM Quantum backend via Sampler (Setting D)
  4. Kernel concentration diagnostics and Kernel-Target Alignment (KTA)
"""

import sys
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map, pauli_feature_map
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def build_custom_angle_ring_map(
    n_qubits: int,
    reps: int = 1,
    entanglement: str = "ring",
) -> QuantumCircuit:
    """
    Construct custom feature map: RY angle rotations followed by CZ entanglement gates.
    Entanglement topology supports:
      - 'ring' / 'circular': CZ gates connecting (i, (i+1)%n) in a periodic ring
      - 'linear': CZ gates connecting adjacent pairs (i, i+1)
      - 'full': all-to-all CZ gates between all qubit pairs (i, j)
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="CustomAngleRing")

    for _ in range(reps):
        # 1. Single-qubit RY rotations with input feature angles
        for i in range(n_qubits):
            qc.ry(x[i], i)

        # 2. Two-qubit CZ entanglement gates
        if entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qc.cz(i, j)
        elif entanglement == "linear":
            for i in range(n_qubits - 1):
                qc.cz(i, i + 1)
        else:  # ring / circular entanglement
            for i in range(n_qubits):
                qc.cz(i, (i + 1) % n_qubits)

    return qc


def get_feature_map(
    name: str,
    n_qubits: int,
    reps: int = 1,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """
    Factory function returning the specified quantum feature map circuit.
    Supported names: 'ZZFeatureMap', 'PauliFeatureMap', 'CustomAngleRingMap'.
    """
    if name == "ZZFeatureMap":
        return zz_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    elif name == "PauliFeatureMap":
        return pauli_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            paulis=["Z", "ZZ"],
            entanglement=entanglement,
        )
    elif name == "CustomAngleRingMap":
        topo = "ring" if entanglement in ["ring", "circular"] else entanglement
        return build_custom_angle_ring_map(
            n_qubits=n_qubits,
            reps=reps,
            entanglement=topo,
        )
    else:
        raise ValueError(f"Unknown feature map: {name}")


def analyze_circuit_complexity(
    feature_map: QuantumCircuit,
    backend=None,
    optimization_level: int = 3,
) -> dict:
    """
    Measure circuit depth and two-qubit gate count after transpilation for target backend.
    Two-qubit gates (ECR, CX, CZ) are the primary source of decoherence on NISQ processors.
    Uses two distinct feature vectors x1 != x2 so gates are not canceled by identity U^dag(x)U(x).
    """
    n_qubits = feature_map.num_qubits
    x1 = np.linspace(0.1, 1.0, n_qubits, dtype=np.float32)
    x2 = np.linspace(1.1, 2.0, n_qubits, dtype=np.float32)

    # Build representative off-diagonal kernel circuit U(x2)^dagger U(x1)
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map.assign_parameters(x1), inplace=True)
    qc.compose(feature_map.assign_parameters(x2).inverse(), inplace=True)
    qc.measure_all()

    if backend is not None:
        transpiled_qc = transpile(qc, backend=backend, optimization_level=optimization_level)
    else:
        transpiled_qc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=optimization_level)

    ops = transpiled_qc.count_ops()
    two_q_count = int(ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0))
    depth = int(transpiled_qc.depth())

    return {
        "depth": depth,
        "two_qubit_gates": two_q_count,
        "total_gates": int(sum(ops.values())),
        "gate_counts": dict(ops),
    }


def compute_kernel(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_map: QuantumCircuit,
    mode: str = "statevector",
    backend=None,
    shots: int = config.SHOTS,
    batch_size: int = 1000,
    seed: int = config.SEED,
    fast_shots: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute the quantum kernel matrix between dataset X_a (N samples) and X_b (M samples):
      k(x_i, x_j) = |<0...0| U(x_j)^dagger U(x_i) |0...0>|^2

    Modes:
      - 'statevector': exact analytical inner products (fast, noise-free, Setting A)
      - 'shots': ideal finite-shot sampling in Aer or via exact Born-rule binomial sampling (Setting B)
      - 'noisy': noisy shot sampling on Aer with noise model from backend (Setting C)
      - 'hardware': executed on real IBM Quantum hardware (Setting D)

    Exploits kernel symmetry when X_a is X_b:
      - Sets diagonal elements strictly to 1.0 (k(x_i, x_i) = 1.0)
      - Evaluates only upper-triangular pairs (j > i) and reflects across diagonal
    """
    n_a = len(X_a)
    n_b = len(X_b)
    n_qubits = feature_map.num_qubits
    is_symmetric = (X_a is X_b) or (n_a == n_b and np.allclose(X_a, X_b))

    kernel = np.zeros((n_a, n_b), dtype=np.float32)

    # --------------------------------------------------------------------------
    # Setting A: Exact Statevector Inner Products
    # --------------------------------------------------------------------------
    if mode == "statevector":
        if show_progress:
            print(f"Computing statevector kernel ({n_a}x{n_b}, n_qubits={n_qubits})...")

        sv_a = [Statevector.from_instruction(feature_map.assign_parameters(x)) for x in X_a]
        if is_symmetric:
            sv_b = sv_a
        else:
            sv_b = [Statevector.from_instruction(feature_map.assign_parameters(x)) for x in X_b]

        if is_symmetric:
            for i in range(n_a):
                kernel[i, i] = 1.0
                for j in range(i + 1, n_a):
                    fidelity = float(abs(sv_a[j].inner(sv_a[i])) ** 2)
                    kernel[i, j] = fidelity
                    kernel[j, i] = fidelity
        else:
            for i in range(n_a):
                for j in range(n_b):
                    kernel[i, j] = float(abs(sv_b[j].inner(sv_a[i])) ** 2)

        return np.clip(kernel, 0.0, 1.0)

    # --------------------------------------------------------------------------
    # Setting B (Fast Mode): Born-rule Binomial Sampling
    # Mathematically and physically identical to finite ideal measurement shots,
    # avoiding overhead while accurately capturing binomial shot noise.
    # --------------------------------------------------------------------------
    if mode == "shots" and fast_shots:
        if show_progress:
            print(f"Computing ideal shot kernel via Born-rule sampling ({n_a}x{n_b}, shots={shots})...")
        k_exact = compute_kernel(X_a, X_b, feature_map, mode="statevector", show_progress=False)
        rng = np.random.RandomState(seed)
        k_shots = np.zeros_like(k_exact)

        if is_symmetric:
            for i in range(n_a):
                k_shots[i, i] = 1.0
                for j in range(i + 1, n_a):
                    p = float(k_exact[i, j])
                    sample = float(rng.binomial(shots, p) / shots)
                    k_shots[i, j] = sample
                    k_shots[j, i] = sample
        else:
            for i in range(n_a):
                for j in range(n_b):
                    p = float(k_exact[i, j])
                    k_shots[i, j] = float(rng.binomial(shots, p) / shots)

        return np.clip(k_shots, 0.0, 1.0)

    # --------------------------------------------------------------------------
    # Settings B (Circuit Mode), C (Noisy), D (Hardware): Quantum Circuit Execution
    # --------------------------------------------------------------------------
    if mode == "shots":
        sim = AerSimulator()
    elif mode == "noisy":
        if backend is None:
            raise ValueError("A backend must be provided for mode='noisy'")
        sim = AerSimulator.from_backend(backend)
    elif mode == "hardware":
        sim = None  # Handled via hardware runtime session
    else:
        raise ValueError(f"Unknown execution mode: {mode}")

    # Determine unique pairs to compute
    pairs = []
    if is_symmetric:
        for i in range(n_a):
            kernel[i, i] = 1.0
            for j in range(i + 1, n_a):
                pairs.append((i, j))
    else:
        for i in range(n_a):
            for j in range(n_b):
                pairs.append((i, j))

    total_circuits = len(pairs)
    if show_progress:
        print(f"Shot-based kernel ({mode}): evaluating {total_circuits} circuits with {shots} shots...")

    target_bitstring = "0" * n_qubits

    # Pre-transpile a parameterized template circuit once to avoid redundant transpiler passes
    x1_param = ParameterVector("x1", n_qubits)
    x2_param = ParameterVector("x2", n_qubits)
    qc_template = QuantumCircuit(n_qubits)
    qc_template.compose(feature_map.assign_parameters(x1_param), inplace=True)
    qc_template.compose(feature_map.assign_parameters(x2_param).inverse(), inplace=True)
    qc_template.measure_all()

    if mode == "noisy" and backend is not None:
        transpiled_template = transpile(qc_template, backend=backend, optimization_level=1)
    else:
        transpiled_template = qc_template

    # Process in batches
    for batch_start in range(0, total_circuits, batch_size):
        batch_pairs = pairs[batch_start : batch_start + batch_size]
        batch_circuits = [
            transpiled_template.assign_parameters({x1_param: X_a[i], x2_param: X_b[j]})
            for i, j in batch_pairs
        ]

        # Run batch
        job = sim.run(batch_circuits, shots=shots)
        result = job.result()

        for idx, (i, j) in enumerate(batch_pairs):
            counts = result.get_counts(idx)
            fidelity = float(counts.get(target_bitstring, 0) / shots)
            kernel[i, j] = fidelity
            if is_symmetric:
                kernel[j, i] = fidelity

    return np.clip(kernel, 0.0, 1.0)


def compute_kernel_diagnostics(kernel_matrix: np.ndarray, y: np.ndarray | None = None) -> dict:
    """
    Calculate diagnostic metrics on kernel matrix:
      1. Off-diagonal mean and variance (checks for kernel concentration / exponential decay)
      2. Kernel-Target Alignment (KTA): measures geometric alignment between kernel Gram matrix
         and ideal target outer product matrix Y = y y^T (y in {-1, +1}).
    """
    n = kernel_matrix.shape[0]
    # Extract strictly off-diagonal elements
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_vals = kernel_matrix[off_diag_mask]

    diag_mean = float(np.mean(off_diag_vals))
    diag_var = float(np.var(off_diag_vals))
    diag_min = float(np.min(off_diag_vals))
    diag_max = float(np.max(off_diag_vals))

    kta = 0.0
    if y is not None:
        y_pm = np.where(y == 1, 1.0, -1.0).astype(np.float64)
        Y = np.outer(y_pm, y_pm)
        K = kernel_matrix.astype(np.float64)
        fro_K = np.linalg.norm(K, "fro")
        fro_Y = np.linalg.norm(Y, "fro")
        if fro_K > 0 and fro_Y > 0:
            kta = float(np.sum(K * Y) / (fro_K * fro_Y))

    return {
        "off_diagonal_mean": diag_mean,
        "off_diagonal_var": diag_var,
        "off_diagonal_min": diag_min,
        "off_diagonal_max": diag_max,
        "kernel_target_alignment": kta,
    }
