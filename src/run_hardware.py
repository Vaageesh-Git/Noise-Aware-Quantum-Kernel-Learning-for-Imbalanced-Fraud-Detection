"""
Hardware Execution Pipeline (Setting D) on IBM Quantum.
Supports:
  1. Live execution on IBM Quantum hardware via Qiskit Runtime SamplerV2
  2. Safe --dry-run mode using FakeSherbrooke (127-qubit simulated backend)
  3. Interactive circuit & shot estimation with confirmation prompt
  4. Automatic least-busy backend query with optimization_level=3 transpilation
  5. Fault-tolerant batch execution with immediate streaming disk checkpointing
  6. Two quantum execution runs:
     - Run D1: Raw unmitigated hardware execution
     - Run D2: Hardware error-mitigated execution (Dynamical Decoupling XX + Gate Twirling)
  7. Classical Post-Processing option:
     - Diagonal preservation + Nearest Positive Semi-Definite (PSD) matrix projection
     (Explicitly marked as classical mathematical post-processing, not hardware mitigation)
  8. Kernel distortion analysis against Setting A (Exact Statevector)
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.quantum_kernel import (
    get_feature_map,
    compute_kernel,
    compute_kernel_diagnostics,
    analyze_circuit_complexity,
)
from src.evaluate import compute_classification_metrics

# Raw checkpoints directory
RAW_DIR = config.RESULTS_DIR / "hardware_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
KERNELS_DIR = config.RESULTS_DIR / "kernels"
KERNELS_DIR.mkdir(parents=True, exist_ok=True)


def project_to_nearest_psd(K: np.ndarray) -> np.ndarray:
    """
    Classical Post-Processing: Project symmetric kernel matrix onto the cone
    of Positive Semi-Definite (PSD) correlation matrices with unit diagonal.

    IMPORTANT NOTE FOR STUDENTS & REVIEWERS:
    This is CLASSICAL MATHEMATICAL POST-PROCESSING, NOT hardware quantum error mitigation.
    Hardware noise can introduce small negative eigenvalues that violate Mercer's theorem.
    This projection enforces positive semi-definiteness via eigenvalue clipping
    and congruence diagonal normalization (Higham, 2002).
    """
    # 1. Enforce exact symmetry
    K_sym = (K + K.T) / 2.0

    # 2. Spectral decomposition
    eigvals, eigvecs = np.linalg.eigh(K_sym)

    # 3. Clip negative eigenvalues (set floor at small positive epsilon)
    eigvals_clipped = np.maximum(eigvals, 1e-7)

    # 4. Reconstruct PSD matrix
    K_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # 5. Congruence normalization to guarantee diag = 1 while strictly preserving PSD property
    d = np.sqrt(np.diag(K_psd))
    d = np.where(d < 1e-12, 1.0, d)
    K_norm = K_psd / np.outer(d, d)
    np.fill_diagonal(K_norm, 1.0)

    return np.clip(K_norm, 0.0, 1.0).astype(np.float32)


def get_hardware_backend(dry_run: bool = True):
    """
    Retrieve backend for execution.
    If dry_run=True, returns FakeSherbrooke without network/token requirements.
    If dry_run=False, connects to IBM Quantum and selects the least busy operational backend.
    """
    if dry_run:
        backend = FakeSherbrooke()
        print(f"[DRY-RUN MODE] Using simulated 127-qubit backend: {backend.name}")
        return backend, None

    token = os.environ.get(config.IBM_TOKEN_ENV_VAR) or os.environ.get("QISKIT_IBM_TOKEN")
    try:
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        else:
            service = QiskitRuntimeService(channel="ibm_quantum")
    except Exception as e:
        print("=" * 70)
        print("ERROR: Could not authenticate with IBM Quantum!")
        print(f"Please set your IBM Quantum API token via environment variable: {config.IBM_TOKEN_ENV_VAR}")
        print("Example: export IBM_QUANTUM_TOKEN='your_api_token_here'")
        print(f"Details: {e}")
        print("=" * 70)
        sys.exit(1)

    print("Querying IBM Quantum for least busy backend with >= 4 qubits...")
    backend = service.least_busy(operational=True, min_num_qubits=4, simulator=False)
    status = backend.status()
    print(f"Selected IBM Backend: {backend.name}")
    print(f"  Pending jobs in queue: {status.pending_jobs}")
    print(f"  Operational: {status.operational}")
    return backend, service


def execute_hardware_batches(
    circuits: list[QuantumCircuit],
    backend,
    shots: int = config.HW_SHOTS,
    mitigation_mode: str = "none",
    batch_size: int = 100,
    run_label: str = "hardware",
    resume: bool = True,
) -> list[dict]:
    """
    Submit quantum circuits in batches to SamplerV2 with immediate disk checkpointing.
    Supports dynamical decoupling and gate twirling error mitigation.
    """
    sampler = SamplerV2(mode=backend)

    # Configure Error Mitigation
    if mitigation_mode == "mitigated":
        print("Enabling Qiskit Runtime Error Mitigation:")
        print("  - Dynamical Decoupling: sequence_type='XX'")
        print("  - Pauli Twirling: enable_gates=True, enable_measure=True")
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XX"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
    else:
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False

    all_counts = []
    total_batches = int(np.ceil(len(circuits) / batch_size))

    for b_idx in range(total_batches):
        checkpoint_path = RAW_DIR / f"{run_label}_{mitigation_mode}_batch_{b_idx}.json"

        # Check if batch can be resumed
        if resume and checkpoint_path.exists():
            print(f"  Resuming batch {b_idx + 1}/{total_batches} from checkpoint...")
            with open(checkpoint_path, "r") as f:
                batch_counts = json.load(f)
            all_counts.extend(batch_counts)
            continue

        start = b_idx * batch_size
        end = min(start + batch_size, len(circuits))
        batch_circs = circuits[start:end]

        print(f"  Submitting batch {b_idx + 1}/{total_batches} ({len(batch_circs)} circuits)...")
        job = sampler.run(batch_circs, shots=shots)
        result = job.result()

        # Extract bitstring counts
        batch_counts = []
        for pub_res in result:
            reg_name = list(pub_res.data.keys())[0]
            counts = getattr(pub_res.data, reg_name).get_counts()
            batch_counts.append(counts)

        # Immediately save raw batch results to disk
        with open(checkpoint_path, "w") as f:
            json.dump(batch_counts, f)

        all_counts.extend(batch_counts)

    return all_counts


def run_hardware_benchmarking(
    dry_run: bool = True,
    n_train: int = config.HW_TRAIN,
    n_test: int = config.HW_TEST,
    shots: int = config.HW_SHOTS,
    resume: bool = True,
    confirm: bool = True,
) -> pd.DataFrame:
    """
    Main hardware benchmark execution function.
    """
    print("=" * 70)
    print("STAGE 6: REAL HARDWARE BENCHMARKING (SETTING D - IBM QUANTUM)")
    print("=" * 70)

    # 1. Load Top-1 configuration from Stage 4
    top_file = config.RESULTS_DIR / "top_configurations.csv"
    if top_file.exists():
        top_df = pd.read_csv(top_file)
        top_cfg = top_df.iloc[0].to_dict()
        fm_name = top_cfg["feature_map"]
        n_qubits = int(top_cfg["n_qubits"])
        reps = int(top_cfg["reps"])
        ent = "ring" if fm_name == "CustomAngleRingMap" else "linear"
    else:
        fm_name = "CustomAngleRingMap"
        n_qubits = 4
        reps = 1
        ent = "ring"

    print(f"Target Configuration: {fm_name} | Qubits: {n_qubits} | Reps: {reps}")

    # 2. Extract hardware subset
    data_file = config.PROCESSED_DATA_DIR / f"quantum_subset_n{n_qubits}_rep0.npz"
    data = np.load(data_file)
    X_train = data["X_train"][:n_train]
    y_train = data["y_train"][:n_train]
    X_test = data["X_test"][:n_test]
    y_test = data["y_test"][:n_test]

    print(f"Hardware dataset slice: Train={len(X_train)} rows ({int(y_train.sum())} fraud), "
          f"Test={len(X_test)} rows ({int(y_test.sum())} fraud)")

    # 3. Calculate exact circuit & shot requirements
    n_train_pairs = n_train * (n_train - 1) // 2  # 190
    n_test_pairs = n_test * n_train               # 200
    total_circuits = n_train_pairs + n_test_pairs  # 390
    total_shots = total_circuits * shots          # 399,360

    print("-" * 70)
    print("RESOURCE ESTIMATION:")
    print(f"  Training kernel pairs (upper triangle): {n_train_pairs} circuits")
    print(f"  Testing kernel pairs:                   {n_test_pairs} circuits")
    print(f"  Total quantum circuits per run:         {total_circuits} circuits")
    print(f"  Shots per circuit:                      {shots}")
    print(f"  Total shot consumption per run:         {total_shots:,} shots")
    print(f"  Total for 2 runs (raw + mitigated):     {total_shots * 2:,} shots")
    print("-" * 70)

    # 4. Confirmation guard
    if not dry_run and confirm:
        user_input = input("Proceed with live IBM Quantum hardware submission? (y/N): ").strip().lower()
        if user_input not in ["y", "yes"]:
            print("Submission aborted by user. Exiting safely.")
            sys.exit(0)

    # 5. Acquire backend
    backend, _ = get_hardware_backend(dry_run=dry_run)
    fm = get_feature_map(fm_name, n_qubits=n_qubits, reps=reps, entanglement=ent)
    complexity = analyze_circuit_complexity(fm, backend=backend, optimization_level=3)
    print(f"Transpiled circuit complexity on {backend.name}: Depth={complexity['depth']}, 2Q Gates={complexity['two_qubit_gates']}")

    # 6. Build circuit list
    train_pairs = [(i, j) for i in range(n_train) for j in range(i + 1, n_train)]
    test_pairs = [(i, j) for i in range(n_test) for j in range(n_train)]

    all_pairs = train_pairs + test_pairs
    raw_circuits = []
    for i, j in all_pairs:
        # Determine whether train pair or test pair
        is_train_pair = len(raw_circuits) < n_train_pairs
        x_i = X_train[i] if is_train_pair else X_test[i]
        x_j = X_train[j]

        qc = QuantumCircuit(n_qubits)
        qc.compose(fm.assign_parameters(x_i), inplace=True)
        qc.compose(fm.assign_parameters(x_j).inverse(), inplace=True)
        qc.measure_all()
        raw_circuits.append(qc)

    print(f"Transpiling {len(raw_circuits)} circuits with optimization_level=3...")
    transpiled_circs = transpile(raw_circuits, backend=backend, optimization_level=3)

    # 7. Compute Reference Exact Kernel (Setting A)
    print("\nComputing exact statevector reference kernel on hardware slice...")
    K_train_exact = compute_kernel(X_train, X_train, fm, mode="statevector", show_progress=False)
    K_test_exact = compute_kernel(X_test, X_train, fm, mode="statevector", show_progress=False)

    target_bs = "0" * n_qubits

    def assemble_kernel_from_counts(counts_list: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        K_tr = np.eye(n_train, dtype=np.float32)
        K_te = np.zeros((n_test, n_train), dtype=np.float32)

        for idx, (i, j) in enumerate(train_pairs):
            prob = float(counts_list[idx].get(target_bs, 0) / shots)
            K_tr[i, j] = prob
            K_tr[j, i] = prob

        offset = len(train_pairs)
        for idx, (i, j) in enumerate(test_pairs):
            prob = float(counts_list[offset + idx].get(target_bs, 0) / shots)
            K_te[i, j] = prob

        return np.clip(K_tr, 0.0, 1.0), np.clip(K_te, 0.0, 1.0)

    # --------------------------------------------------------------------------
    # Run D1: Raw Unmitigated Hardware
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("RUN D1: RAW UNMITIGATED HARDWARE EXECUTION")
    print("-" * 70)
    counts_raw = execute_hardware_batches(
        transpiled_circs, backend=backend, shots=shots, mitigation_mode="none", run_label="hw_raw", resume=resume
    )
    K_train_raw, K_test_raw = assemble_kernel_from_counts(counts_raw)

    # --------------------------------------------------------------------------
    # Run D2: Hardware Error Mitigated (DD + Twirling)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("RUN D2: MITIGATED HARDWARE EXECUTION (Dynamical Decoupling + Twirling)")
    print("-" * 70)
    counts_mit = execute_hardware_batches(
        transpiled_circs, backend=backend, shots=shots, mitigation_mode="mitigated", run_label="hw_mitigated", resume=resume
    )
    K_train_mit, K_test_mit = assemble_kernel_from_counts(counts_mit)

    # --------------------------------------------------------------------------
    # Run D3: Mitigated + Classical PSD Post-Processing
    # --------------------------------------------------------------------------
    print("\nApplying Classical Positive Semi-Definite (PSD) Post-Processing...")
    K_train_post = project_to_nearest_psd(K_train_mit)
    K_test_post = np.clip(K_test_mit, 0.0, 1.0)

    # --------------------------------------------------------------------------
    # Evaluation and Comparison
    # --------------------------------------------------------------------------
    runs = [
        ("D1_raw_hardware", K_train_raw, K_test_raw),
        ("D2_mitigated_hardware", K_train_mit, K_test_mit),
        ("D3_mitigated_plus_classical_psd", K_train_post, K_test_post),
    ]

    records = []

    print("\n" + "=" * 70)
    print("HARDWARE EVALUATION RESULTS (Comparison against Exact Setting A)")
    print("=" * 70)

    for run_name, K_tr, K_te in runs:
        # Fidelity distortion metrics against exact kernel
        fro_diff = float(np.linalg.norm(K_tr - K_train_exact, "fro"))
        fro_exact = float(np.linalg.norm(K_train_exact, "fro"))
        rel_fro = float(fro_diff / fro_exact)

        triu_idx = np.triu_indices(n_train, k=1)
        vals_hw = K_tr[triu_idx]
        vals_ex = K_train_exact[triu_idx]
        pearson = float(np.corrcoef(vals_hw, vals_ex)[0, 1])
        var_hw = float(np.var(vals_hw))
        var_ex = float(np.var(vals_ex))
        var_ret = float(var_hw / max(var_ex, 1e-12))

        diag = compute_kernel_diagnostics(K_tr, y_train)

        # Train SVM (C=1.0 on small hardware slice)
        clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
        clf.fit(K_tr, y_train)
        y_pred = clf.predict(K_te)
        y_scores = clf.decision_function(K_te)
        metrics = compute_classification_metrics(y_test, y_pred, y_scores)

        # Save kernel
        np.savez_compressed(
            KERNELS_DIR / f"kernel_{run_name}.npz",
            K_train=K_tr,
            K_test=K_te,
            y_train=y_train,
            y_test=y_test,
        )

        record = {
            "setting": run_name,
            "backend": backend.name,
            "n_train": n_train,
            "n_test": n_test,
            "shots": shots,
            "frobenius_rel_error": rel_fro,
            "pearson_correlation": pearson,
            "variance_retention": var_ret,
            "kta": diag["kernel_target_alignment"],
            **metrics,
        }
        records.append(record)

        print(
            f"{run_name:<32} | Frob Err: {rel_fro:.4f} | Pearson: {pearson:.4f} | "
            f"Var Ret: {var_ret:.2%} | PR-AUC: {metrics['pr_auc']:.4f}"
        )

    print("\n" + ">" * 70)
    print("NOTE ON SAMPLE SIZE AND CLASSIFICATION METRICS:")
    print("Due to strict quantum hardware execution constraints, the hardware dataset")
    print(f"slice is small ({n_train} train, {n_test} test). Therefore, classification metrics")
    print("(PR-AUC, F1) serve as secondary behavioral indicators; the primary physical conclusions")
    print("are drawn from kernel matrix fidelity (Frobenius error, Pearson correlation, and PSD structure).")
    print("<" * 70)

    results_df = pd.DataFrame(records)
    out_csv = config.RESULTS_DIR / "hardware_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nHardware benchmarking results written to: {out_csv}")
    print("STAGE 6 COMPLETE: Hardware benchmarking pipeline ready.")
    print("=" * 70)
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6 IBM Quantum Hardware Execution")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in safe dry-run mode using FakeSherbrooke")
    parser.add_argument("--live", dest="dry_run", action="store_false", help="Submit live jobs to IBM Quantum hardware")
    parser.add_argument("--no-confirm", dest="confirm", action="store_false", help="Bypass confirmation prompt")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Force recomputation of batches")
    args = parser.parse_args()

    run_hardware_benchmarking(
        dry_run=args.dry_run,
        resume=args.resume,
        confirm=args.confirm,
    )
