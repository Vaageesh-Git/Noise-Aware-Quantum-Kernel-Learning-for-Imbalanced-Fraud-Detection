"""
Ideal Quantum Kernel Benchmarking Pipeline (Settings A and B).
Evaluates quantum kernel SVMs under ideal simulation conditions:
  Setting A: Exact statevector simulation (analytical inner products, zero noise)
  Setting B: Finite shot simulation (isolates quantum projective measurement shot noise)

Benchmarking grid:
  - Feature Maps: ZZFeatureMap, PauliFeatureMap, CustomAngleRingMap
  - Repetitions: 1, 2
  - Qubit Counts: 4, 6
  - Subsets: 5 independent stratified repeats (N_train=300, N_test=200, 10% fraud)
  - Hyperparameter tuning: 3-fold stratified CV for SVM C in [0.1, 1.0, 10.0, 100.0]

Saves:
  - Kernel matrices to results/kernels/
  - Detailed results table to results/ideal_results.csv
  - Automatic ranking and identification of top 1-2 configurations based on mean PR-AUC
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

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

# Kernel storage directory
KERNELS_DIR = config.RESULTS_DIR / "kernels"
KERNELS_DIR.mkdir(parents=True, exist_ok=True)


def tune_and_fit_precomputed_svm(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    c_grid: list[float] = config.SVM_C_GRID,
) -> tuple[dict, float, SVC]:
    """
    Perform 3-fold stratified CV on the precomputed training kernel to select best C,
    fit final precomputed SVM on full K_train, and evaluate on K_test.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c = c_grid[0]
    best_cv_score = -1.0

    # Cross-validation over C values
    for c_val in c_grid:
        fold_scores = []
        for tr_idx, val_idx in cv.split(K_train, y_train):
            # Sub-matrices for CV fold
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

    # Fit final model on full training kernel
    final_clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    final_clf.fit(K_train, y_train)

    # Predict on test kernel
    y_pred = final_clf.predict(K_test)
    y_scores = final_clf.decision_function(K_test)

    metrics = compute_classification_metrics(y_test, y_pred, y_scores)
    metrics["best_c"] = best_c
    metrics["best_cv_pr_auc"] = best_cv_score
    return metrics, best_c, final_clf


def run_ideal_experiments() -> pd.DataFrame:
    """
    Execute full grid of ideal experiments for Settings A (Statevector) and B (Shots).
    """
    print("=" * 70)
    print("STAGE 4: IDEAL QUANTUM KERNEL EXPERIMENTS (SETTINGS A & B)")
    print("=" * 70)

    feature_map_configs = [
        {"name": "ZZFeatureMap", "entanglement": "linear"},
        {"name": "PauliFeatureMap", "entanglement": "linear"},
        {"name": "CustomAngleRingMap", "entanglement": "ring"},
    ]

    all_records = []

    for n in config.N_QUBITS_LIST:
        for fmc in feature_map_configs:
            fm_name = fmc["name"]
            ent = fmc["entanglement"]

            for reps in config.FEATURE_MAP_REPS:
                fm = get_feature_map(name=fm_name, n_qubits=n, reps=reps, entanglement=ent)
                complexity = analyze_circuit_complexity(fm)

                print(
                    f"\nEvaluating: {fm_name} | Qubits: {n} | Reps: {reps} | "
                    f"Depth: {complexity['depth']} | 2Q Gates: {complexity['two_qubit_gates']}"
                )

                for rep in range(config.N_REPEATS):
                    # Load pre-generated quantum subset
                    subset_file = config.PROCESSED_DATA_DIR / f"quantum_subset_n{n}_rep{rep}.npz"
                    sub_data = np.load(subset_file)
                    X_train, y_train = sub_data["X_train"], sub_data["y_train"]
                    X_test, y_test = sub_data["X_test"], sub_data["y_test"]
                    rep_seed = int(sub_data["seed"][0])

                    # ----------------------------------------------------------
                    # Setting A: Exact Statevector (Noise-free)
                    # ----------------------------------------------------------
                    K_train_a = compute_kernel(X_train, X_train, fm, mode="statevector", show_progress=False)
                    K_test_a = compute_kernel(X_test, X_train, fm, mode="statevector", show_progress=False)

                    diag_a = compute_kernel_diagnostics(K_train_a, y_train)
                    metrics_a, best_c_a, _ = tune_and_fit_precomputed_svm(
                        K_train_a, y_train, K_test_a, y_test, seed=rep_seed
                    )

                    # Save Setting A kernel matrices
                    kernel_fn_a = f"kernel_settingA_{fm_name}_n{n}_r{reps}_rep{rep}.npz"
                    np.savez_compressed(
                        KERNELS_DIR / kernel_fn_a,
                        K_train=K_train_a,
                        K_test=K_test_a,
                        y_train=y_train,
                        y_test=y_test,
                    )

                    record_a = {
                        "setting": "A_exact_statevector",
                        "feature_map": fm_name,
                        "n_qubits": n,
                        "reps": reps,
                        "entanglement": ent,
                        "repeat": rep,
                        "seed": rep_seed,
                        "depth": complexity["depth"],
                        "two_qubit_gates": complexity["two_qubit_gates"],
                        **diag_a,
                        **metrics_a,
                    }
                    all_records.append(record_a)

                    # ----------------------------------------------------------
                    # Setting B: Finite Shots (Shot Noise)
                    # ----------------------------------------------------------
                    K_train_b = compute_kernel(
                        X_train, X_train, fm, mode="shots", shots=config.SHOTS, seed=rep_seed, show_progress=False
                    )
                    K_test_b = compute_kernel(
                        X_test, X_train, fm, mode="shots", shots=config.SHOTS, seed=rep_seed, show_progress=False
                    )

                    diag_b = compute_kernel_diagnostics(K_train_b, y_train)
                    metrics_b, best_c_b, _ = tune_and_fit_precomputed_svm(
                        K_train_b, y_train, K_test_b, y_test, seed=rep_seed
                    )

                    # Save Setting B kernel matrices
                    kernel_fn_b = f"kernel_settingB_{fm_name}_n{n}_r{reps}_rep{rep}.npz"
                    np.savez_compressed(
                        KERNELS_DIR / kernel_fn_b,
                        K_train=K_train_b,
                        K_test=K_test_b,
                        y_train=y_train,
                        y_test=y_test,
                    )

                    record_b = {
                        "setting": "B_ideal_shots",
                        "feature_map": fm_name,
                        "n_qubits": n,
                        "reps": reps,
                        "entanglement": ent,
                        "repeat": rep,
                        "seed": rep_seed,
                        "depth": complexity["depth"],
                        "two_qubit_gates": complexity["two_qubit_gates"],
                        **diag_b,
                        **metrics_b,
                    }
                    all_records.append(record_b)

                    print(
                        f"  [Rep {rep+1}/5] PR-AUC: "
                        f"Setting A (Exact)={metrics_a['pr_auc']:.4f} (C={best_c_a}) | "
                        f"Setting B (Shots)={metrics_b['pr_auc']:.4f} (C={best_c_b}) | "
                        f"KTA={diag_a['kernel_target_alignment']:.4f}"
                    )

    results_df = pd.DataFrame(all_records)
    out_csv = config.RESULTS_DIR / "ideal_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nAll ideal experiment results saved to: {out_csv}")

    # ==========================================================================
    # Ranking and Identifying the Best 1-2 Configurations
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RANKING: QUANTUM FEATURE MAP CONFIGURATIONS (Mean across 5 repeats)")
    print("=" * 70)

    # Group by setting, feature_map, n_qubits, reps
    summary = (
        results_df.groupby(["setting", "feature_map", "n_qubits", "reps"])[
            ["pr_auc", "roc_auc", "f1", "kernel_target_alignment", "off_diagonal_var"]
        ]
        .mean()
        .reset_index()
    )

    # Sort Setting A by PR-AUC descending
    summary_a = summary[summary["setting"] == "A_exact_statevector"].sort_values(by="pr_auc", ascending=False)
    print("\n--- Top Configurations in Setting A (Exact Statevector) ---")
    print(summary_a.to_string(index=False))

    top_configs = summary_a.iloc[:2].to_dict(orient="records")
    print("\n" + "*" * 70)
    print("SELECTED TOP CONFIGURATIONS FOR STAGE 5 (NOISY) AND STAGE 6 (HARDWARE):")
    for idx, cfg in enumerate(top_configs, 1):
        print(
            f"  Top {idx}: {cfg['feature_map']} (n_qubits={cfg['n_qubits']}, reps={cfg['reps']}) "
            f"-> Mean PR-AUC = {cfg['pr_auc']:.4f}, Mean ROC-AUC = {cfg['roc_auc']:.4f}, "
            f"KTA = {cfg['kernel_target_alignment']:.4f}"
        )
    print("*" * 70)

    # Save top configuration info
    top_df = pd.DataFrame(top_configs)
    top_df.to_csv(config.RESULTS_DIR / "top_configurations.csv", index=False)

    print("\nSTAGE 4 COMPLETE: Ideal simulation benchmarks finished.")
    print("=" * 70)
    return results_df


if __name__ == "__main__":
    run_ideal_experiments()
