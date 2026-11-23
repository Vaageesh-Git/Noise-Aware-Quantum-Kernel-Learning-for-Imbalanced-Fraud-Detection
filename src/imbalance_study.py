"""
Class Imbalance Study for Quantum Kernel Fraud Detection.
Evaluates:
  Part 1: Four Imbalance Handling Strategies (on standard 10% fraud training split)
    - None (Natural unadjusted baseline)
    - Class Weights (Cost-sensitive inverse-frequency penalty in SVM objective)
    - Random Majority Undersampling (Balance classes by discarding legitimate samples)
    - SMOTE (Synthetic Minority Over-sampling Technique on reduced training features)
    Strict Rule: All sampling methods are fitted strictly on training data; test data is untouched.

  Part 2: Sensitivity Analysis across Varying Training Fraud Ratios
    - Fraud ratios: [0.50 (balanced), 0.20, 0.10 (standard), 0.05 (severe)]
    - Evaluates both Quantum Kernel SVM (CustomAngleRingMap) and Classical RBF-SVM
      across all 5 repeats to answer Research Question 4:
      Under what conditions does the quantum kernel match or beat classical baselines?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.quantum_kernel import get_feature_map, compute_kernel
from src.evaluate import compute_classification_metrics


def tune_and_fit_quantum_svm(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    class_weight: str | None = "balanced",
    c_grid: list[float] = config.SVM_C_GRID,
) -> dict:
    """
    Tune SVM C via 3-fold stratified CV on K_train and evaluate on K_test.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c = c_grid[0]
    best_score = -1.0

    for c_val in c_grid:
        scores = []
        for tr_idx, val_idx in cv.split(K_train, y_train):
            K_tr = K_train[np.ix_(tr_idx, tr_idx)]
            K_val = K_train[np.ix_(val_idx, tr_idx)]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            clf = SVC(kernel="precomputed", C=c_val, class_weight=class_weight)
            clf.fit(K_tr, y_tr)
            decision_scores = clf.decision_function(K_val)
            scores.append(average_precision_score(y_val, decision_scores))

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_val

    final_clf = SVC(kernel="precomputed", C=best_c, class_weight=class_weight)
    final_clf.fit(K_train, y_train)

    y_pred = final_clf.predict(K_test)
    y_scores = final_clf.decision_function(K_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_scores)
    metrics["best_c"] = best_c
    return metrics


def tune_and_fit_classical_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    class_weight: str | None = "balanced",
    c_grid: list[float] = [0.1, 1.0, 10.0],
) -> dict:
    """
    Tune Classical RBF-SVM C via 3-fold stratified CV on training data and evaluate on test data.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c = c_grid[0]
    best_score = -1.0

    for c_val in c_grid:
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            clf = SVC(kernel="rbf", C=c_val, gamma="scale", class_weight=class_weight, random_state=seed)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            df = clf.decision_function(X_train[val_idx])
            scores.append(average_precision_score(y_train[val_idx], df))

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_val

    final_clf = SVC(kernel="rbf", C=best_c, gamma="scale", class_weight=class_weight, random_state=seed)
    final_clf.fit(X_train, y_train)

    y_pred = final_clf.predict(X_test)
    y_scores = final_clf.decision_function(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_scores)
    metrics["best_c"] = best_c
    return metrics


def run_imbalance_study() -> pd.DataFrame:
    """
    Execute full imbalance investigation (Part 1 and Part 2) across all 5 repeats.
    """
    print("=" * 70)
    print("STAGE 7: IMBALANCE STUDY FOR QUANTUM KERNEL LEARNING")
    print("=" * 70)

    # Best configuration from Stage 4
    fm = get_feature_map("CustomAngleRingMap", n_qubits=4, reps=1, entanglement="ring")
    print("Using best quantum feature map: CustomAngleRingMap (4 qubits, reps=1)")

    all_records = []

    # ==========================================================================
    # PART 1: Four Imbalance Handling Strategies (at standard 10% fraud ratio)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("PART 1: COMPARING IMBALANCE STRATEGIES (Standard 10% Fraud Split)")
    print("-" * 70)

    strategies = ["none", "class_weights", "undersampling", "smote"]

    for rep in range(config.N_REPEATS):
        sub_file = config.PROCESSED_DATA_DIR / f"quantum_subset_n4_rep{rep}.npz"
        sub_data = np.load(sub_file)
        X_train_raw, y_train_raw = sub_data["X_train"], sub_data["y_train"]
        X_test, y_test = sub_data["X_test"], sub_data["y_test"]
        seed = int(sub_data["seed"][0])

        for strat in strategies:
            # 1. Prepare training data according to strategy
            if strat == "none":
                X_tr, y_tr = X_train_raw, y_train_raw
                cw = None
            elif strat == "class_weights":
                X_tr, y_tr = X_train_raw, y_train_raw
                cw = "balanced"
            elif strat == "undersampling":
                rus = RandomUnderSampler(random_state=seed)
                X_tr, y_tr = rus.fit_resample(X_train_raw, y_train_raw)
                cw = None
            elif strat == "smote":
                smote = SMOTE(random_state=seed)
                X_tr, y_tr = smote.fit_resample(X_train_raw, y_train_raw)
                # Clip synthetic angles into [0, pi]
                X_tr = np.clip(X_tr, 0.0, np.pi).astype(np.float32)
                cw = None

            # 2. Evaluate Classical RBF-SVM
            m_classical = tune_and_fit_classical_rbf(X_tr, y_tr, X_test, y_test, seed=seed, class_weight=cw)
            all_records.append({
                "experiment": "part1_strategies",
                "strategy": strat,
                "model_type": "Classical_RBF_SVM",
                "fraud_ratio": 0.10,
                "n_train_samples": len(y_tr),
                "repeat": rep,
                **m_classical,
            })

            # 3. Evaluate Quantum Kernel SVM
            K_tr = compute_kernel(X_tr, X_tr, fm, mode="statevector", show_progress=False)
            K_te = compute_kernel(X_test, X_tr, fm, mode="statevector", show_progress=False)
            m_quantum = tune_and_fit_quantum_svm(K_tr, y_tr, K_te, y_test, seed=seed, class_weight=cw)
            all_records.append({
                "experiment": "part1_strategies",
                "strategy": strat,
                "model_type": "Quantum_Kernel_SVM",
                "fraud_ratio": 0.10,
                "n_train_samples": len(y_tr),
                "repeat": rep,
                **m_quantum,
            })

            print(
                f"  [Rep {rep+1}/5 | {strat:<14}] "
                f"Quantum PR-AUC={m_quantum['pr_auc']:.4f} (Rec={m_quantum['recall']:.1%}) | "
                f"Classical PR-AUC={m_classical['pr_auc']:.4f} (Rec={m_classical['recall']:.1%})"
            )

    # ==========================================================================
    # PART 2: Sensitivity across Varying Training Fraud Ratios
    # ==========================================================================
    print("\n" + "-" * 70)
    print("PART 2: SENSITIVITY ACROSS FRAUD RATIOS [0.50, 0.20, 0.10, 0.05]")
    print("-" * 70)

    # Load full PCA features to draw custom fraud ratio subsets
    pca_data = np.load(config.PROCESSED_DATA_DIR / "pca_4_features.npz")
    X_train_full = pca_data["X_train_scaled"]
    y_train_full = np.load(config.PROCESSED_DATA_DIR / "full_features.npz")["y_train"]

    fraud_train_idx = np.where(y_train_full == 1)[0]
    legit_train_idx = np.where(y_train_full == 0)[0]

    for ratio in config.IMBALANCE_FRAUD_RATIOS:
        n_fraud = int(round(config.QUANTUM_TRAIN * ratio))
        n_legit = config.QUANTUM_TRAIN - n_fraud

        for rep in range(config.N_REPEATS):
            sub_file = config.PROCESSED_DATA_DIR / f"quantum_subset_n4_rep{rep}.npz"
            sub_data = np.load(sub_file)
            X_test, y_test = sub_data["X_test"], sub_data["y_test"]
            seed = int(sub_data["seed"][0])

            rng = np.random.RandomState(seed + int(ratio * 1000))
            chosen_fraud = rng.choice(fraud_train_idx, size=n_fraud, replace=False)
            chosen_legit = rng.choice(legit_train_idx, size=n_legit, replace=False)
            train_idx = rng.permutation(np.concatenate([chosen_fraud, chosen_legit]))

            X_tr = X_train_full[train_idx]
            y_tr = y_train_full[train_idx]

            # Classical RBF
            m_classical = tune_and_fit_classical_rbf(
                X_tr, y_tr, X_test, y_test, seed=seed, class_weight="balanced"
            )
            all_records.append({
                "experiment": "part2_fraud_ratios",
                "strategy": "class_weights",
                "model_type": "Classical_RBF_SVM",
                "fraud_ratio": ratio,
                "n_train_samples": len(y_tr),
                "repeat": rep,
                **m_classical,
            })

            # Quantum Kernel SVM
            K_tr = compute_kernel(X_tr, X_tr, fm, mode="statevector", show_progress=False)
            K_te = compute_kernel(X_test, X_tr, fm, mode="statevector", show_progress=False)
            m_quantum = tune_and_fit_quantum_svm(
                K_tr, y_tr, K_te, y_test, seed=seed, class_weight="balanced"
            )
            all_records.append({
                "experiment": "part2_fraud_ratios",
                "strategy": "class_weights",
                "model_type": "Quantum_Kernel_SVM",
                "fraud_ratio": ratio,
                "n_train_samples": len(y_tr),
                "repeat": rep,
                **m_quantum,
            })

            print(
                f"  [Ratio {ratio:.2f} | Rep {rep+1}/5] "
                f"Quantum PR-AUC={m_quantum['pr_auc']:.4f} (Rec={m_quantum['recall']:.1%}) | "
                f"Classical PR-AUC={m_classical['pr_auc']:.4f} (Rec={m_classical['recall']:.1%})"
            )

    results_df = pd.DataFrame(all_records)
    out_csv = config.RESULTS_DIR / "imbalance_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nAll imbalance study results written to: {out_csv}")

    # ==========================================================================
    # Summary Tables
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY PART 1: IMBALANCE STRATEGIES (Mean +/- Std over 5 repeats)")
    print("=" * 70)
    p1 = results_df[results_df["experiment"] == "part1_strategies"]
    s1 = p1.groupby(["strategy", "model_type"])[["pr_auc", "recall", "f1", "accuracy"]].mean()
    print(s1.round(4).to_string())

    print("\n" + "=" * 70)
    print("SUMMARY PART 2: FRAUD RATIO SENSITIVITY (Mean +/- Std over 5 repeats)")
    print("=" * 70)
    p2 = results_df[results_df["experiment"] == "part2_fraud_ratios"]
    s2 = p2.groupby(["fraud_ratio", "model_type"])[["pr_auc", "recall", "f1", "roc_auc"]].mean()
    print(s2.round(4).to_string())

    print("\nSTAGE 7 COMPLETE: Imbalance study completed.")
    print("=" * 70)
    return results_df


if __name__ == "__main__":
    run_imbalance_study()
