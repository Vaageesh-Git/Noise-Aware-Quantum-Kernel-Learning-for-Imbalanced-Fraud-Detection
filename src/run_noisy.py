"""
Noisy Simulation Pipeline (Setting C) for Quantum Kernel Learning.
Evaluates quantum kernels under realistic physical hardware noise models
derived from IBM Quantum's 127-qubit FakeSherbrooke backend:
  - Thermal relaxation (T1 / T2 dephasing & decay)
  - Single-qubit depolarizing gate errors
  - Two-qubit cross-resonance / ECR entangling gate errors
  - Measurement readout assignment errors

Benchmarks:
  1. Top configurations identified from Stage 4 (CustomAngleRingMap, n=4 and n=6, reps=1)
  2. Contrast configuration (ZZFeatureMap, n=4, reps=1) to demonstrate the impact of 2Q gate depth
  3. Metric degradation analysis against Setting A (Exact Statevector):
     - Frobenius norm relative error
     - Pearson correlation coefficient
     - Kernel concentration (off-diagonal variance shrinkage toward noisy constant)
     - Test PR-AUC and ROC-AUC impact
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
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

KERNELS_DIR = config.RESULTS_DIR / "kernels"
KERNELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_kernel_fidelity_metrics(K_noisy: np.ndarray, K_exact: np.ndarray) -> dict:
    """
    Quantify distortion caused by physical device noise against the exact analytical kernel:
      1. Frobenius relative error: ||K_noisy - K_exact||_F / ||K_exact||_F
      2. Pearson correlation between all matrix entries
      3. Variance ratio: Var(off_diag(K_noisy)) / Var(off_diag(K_exact))
    """
    fro_diff = np.linalg.norm(K_noisy - K_exact, "fro")
    fro_exact = np.linalg.norm(K_exact, "fro")
    rel_fro_error = float(fro_diff / max(fro_exact, 1e-12))

    # Pearson correlation on upper triangle (non-trivial elements)
    n = K_noisy.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vals_noisy = K_noisy[triu_idx]
    vals_exact = K_exact[triu_idx]

    std_n = np.std(vals_noisy)
    std_e = np.std(vals_exact)
    if std_n > 1e-12 and std_e > 1e-12:
        pearson_corr = float(np.corrcoef(vals_noisy, vals_exact)[0, 1])
    else:
        pearson_corr = 0.0

    var_noisy = float(np.var(vals_noisy))
    var_exact = float(np.var(vals_exact))
    var_ratio = float(var_noisy / max(var_exact, 1e-12))

    return {
        "frobenius_rel_error": rel_fro_error,
        "pearson_correlation": pearson_corr,
        "off_diag_var_noisy": var_noisy,
        "off_diag_var_exact": var_exact,
        "variance_retention_ratio": var_ratio,
        "mean_off_diag_shift": float(np.mean(vals_noisy) - np.mean(vals_exact)),
    }


def tune_and_fit_precomputed_svm(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    c_grid: list[float] = config.SVM_C_GRID,
) -> tuple[dict, float, SVC]:
    """
    3-fold stratified cross-validation on precomputed training kernel to select best C,
    followed by test set evaluation.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c = c_grid[0]
    best_cv_score = -1.0

    for c_val in c_grid:
        fold_scores = []
        for tr_idx, val_idx in cv.split(K_train, y_train):
            K_fold_tr = K_train[np.ix_(tr_idx, tr_idx)]
            K_fold_val = K_train[np.ix_(val_idx, tr_idx)]
            y_fold_tr = y_train[tr_idx]
            y_fold_val = y_train[val_idx]

            clf = SVC(kernel="precomputed", C=c_val, class_weight="balanced")
            clf.fit(K_fold_tr, y_fold_tr)
            decision_scores = clf.decision_function(K_fold_val)
            fold_scores.append(average_precision_score(y_fold_val, decision_scores))

        mean_cv_score = float(np.mean(fold_scores))
        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_c = c_val

    final_clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    final_clf.fit(K_train, y_train)

    y_pred = final_clf.predict(K_test)
    y_scores = final_clf.decision_function(K_test)

    metrics = compute_classification_metrics(y_test, y_pred, y_scores)
    metrics["best_c"] = best_c
    metrics["best_cv_pr_auc"] = best_cv_score
    return metrics, best_c, final_clf


def run_noisy_experiments(n_repeats: int = 3) -> pd.DataFrame:
    """
    Execute Setting C noisy simulations using FakeSherbrooke backend.
    Runs top configurations from Stage 4 plus ZZFeatureMap contrast across n_repeats.
    """
    print("=" * 70)
    print("STAGE 5: NOISY QUANTUM SIMULATION (SETTING C - FAKE IBM SHERBROOKE)")
    print("=" * 70)

    # Initialize fake backend
    backend = FakeSherbrooke()
    print(f"Loaded noisy backend: {backend.name} (127 qubits, Eagle r3 topology)")

    # Test suite: Top configurations from Stage 4 + contrast ZZ map
    test_configs = [
        {"name": "CustomAngleRingMap", "n_qubits": 4, "reps": 1, "entanglement": "ring", "label": "Top-1 (Angle Ring 4Q)"},
        {"name": "CustomAngleRingMap", "n_qubits": 6, "reps": 1, "entanglement": "ring", "label": "Top-2 (Angle Ring 6Q)"},
        {"name": "ZZFeatureMap", "n_qubits": 4, "reps": 1, "entanglement": "linear", "label": "Contrast (ZZFeatureMap 4Q)"},
    ]

    all_records = []

    for cfg in test_configs:
        fm_name = cfg["name"]
        n = cfg["n_qubits"]
        reps = cfg["reps"]
        ent = cfg["entanglement"]
        label = cfg["label"]

        fm = get_feature_map(name=fm_name, n_qubits=n, reps=reps, entanglement=ent)
        complexity = analyze_circuit_complexity(fm, backend=backend)

        print("\n" + "-" * 70)
        print(f"BENCHMARKING NOISY: {label}")
        print(f"  Circuit Depth: {complexity['depth']} | 2Q Gates: {complexity['two_qubit_gates']}")
        print("-" * 70)

        for rep in range(n_repeats):
            # Load subset
            sub_file = config.PROCESSED_DATA_DIR / f"quantum_subset_n{n}_rep{rep}.npz"
            sub_data = np.load(sub_file)
            X_train, y_train = sub_data["X_train"], sub_data["y_train"]
            X_test, y_test = sub_data["X_test"], sub_data["y_test"]
            rep_seed = int(sub_data["seed"][0])

            # Load corresponding Setting A (Exact) kernel for baseline comparison
            exact_fn = f"kernel_settingA_{fm_name}_n{n}_r{reps}_rep{rep}.npz"
            exact_path = KERNELS_DIR / exact_fn
            if exact_path.exists():
                exact_data = np.load(exact_path)
                K_train_exact = exact_data["K_train"]
                K_test_exact = exact_data["K_test"]
            else:
                # Compute exact on the fly if file not cached
                K_train_exact = compute_kernel(X_train, X_train, fm, mode="statevector", show_progress=False)
                K_test_exact = compute_kernel(X_test, X_train, fm, mode="statevector", show_progress=False)

            # Compute Noisy Kernels via AerSimulator with FakeSherbrooke noise model
            print(f"  [Rep {rep+1}/{n_repeats}] Simulating noisy training kernel (300x300)...")
            K_train_noisy = compute_kernel(
                X_train, X_train, fm, mode="noisy", backend=backend, shots=config.SHOTS, show_progress=False
            )

            print(f"  [Rep {rep+1}/{n_repeats}] Simulating noisy test kernel (200x300)...")
            K_test_noisy = compute_kernel(
                X_test, X_train, fm, mode="noisy", backend=backend, shots=config.SHOTS, show_progress=False
            )

            # Distortion metrics against exact kernel
            fidelity_metrics = compute_kernel_fidelity_metrics(K_train_noisy, K_train_exact)
            diag_noisy = compute_kernel_diagnostics(K_train_noisy, y_train)

            # Fit precomputed SVM on noisy kernel
            metrics_noisy, best_c, _ = tune_and_fit_precomputed_svm(
                K_train_noisy, y_train, K_test_noisy, y_test, seed=rep_seed
            )

            # Save noisy kernel matrices
            noisy_fn = f"kernel_settingC_{fm_name}_n{n}_r{reps}_rep{rep}.npz"
            np.savez_compressed(
                KERNELS_DIR / noisy_fn,
                K_train=K_train_noisy,
                K_test=K_test_noisy,
                y_train=y_train,
                y_test=y_test,
            )

            record = {
                "setting": "C_noisy_simulation",
                "backend": backend.name,
                "label": label,
                "feature_map": fm_name,
                "n_qubits": n,
                "reps": reps,
                "repeat": rep,
                "depth": complexity["depth"],
                "two_qubit_gates": complexity["two_qubit_gates"],
                **fidelity_metrics,
                **diag_noisy,
                **metrics_noisy,
            }
            all_records.append(record)

            print(
                f"  -> Result: PR-AUC={metrics_noisy['pr_auc']:.4f} (C={best_c}) | "
                f"Frob Rel Error={fidelity_metrics['frobenius_rel_error']:.4f} | "
                f"Pearson Corr={fidelity_metrics['pearson_correlation']:.4f} | "
                f"Var Retention={fidelity_metrics['variance_retention_ratio']:.2%}"
            )

    results_df = pd.DataFrame(all_records)
    out_csv = config.RESULTS_DIR / "noisy_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nAll noisy experiment results saved to: {out_csv}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: NOISY SIMULATION DISTORTION & PERFORMANCE IMPACT")
    print("=" * 70)
    summary = results_df.groupby("label")[
        ["pr_auc", "roc_auc", "frobenius_rel_error", "pearson_correlation", "variance_retention_ratio"]
    ].mean()
    print(summary.round(4).to_string())

    print("\nSTAGE 5 COMPLETE: Noisy simulation benchmarking finished.")
    print("=" * 70)
    return results_df


if __name__ == "__main__":
    run_noisy_experiments()
