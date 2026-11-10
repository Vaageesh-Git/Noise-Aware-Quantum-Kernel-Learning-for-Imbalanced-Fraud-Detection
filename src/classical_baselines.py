"""
Classical Machine Learning Baselines for Imbalanced Fraud Detection.
Evaluates four classical reference models:
  1. Dummy Classifier (always predicts legitimate class 0)
  2. Support Vector Machine with RBF kernel (cost-sensitive class weights)
  3. Random Forest (balanced class weights)
  4. XGBoost (gradient boosted trees with scale_pos_weight)

Executed across two distinct experimental regimes:
  Regime (a): Full cleaned features (218 dims) on large training partition (reference ceiling)
  Regime (b): Exact same PCA-reduced features (4 and 6 dims) and exact same small quantum
              subsets across all 5 repeats (fair head-to-head comparison)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.evaluate import compute_classification_metrics
from sklearn.metrics import average_precision_score


def pr_auc_scorer(estimator, X: np.ndarray, y: np.ndarray) -> float:
    """
    Unified scoring function for GridSearchCV that safely retrieves 1D fraud scores
    from either predict_proba[:, 1] or decision_function.
    """
    if hasattr(estimator, "predict_proba"):
        prob = estimator.predict_proba(X)[:, 1]
    else:
        prob = estimator.decision_function(X)
    return float(average_precision_score(y, prob))


def train_eval_dummy(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train and evaluate dummy classifier that always predicts the majority class (legitimate = 0).
    Exposes the accuracy paradox in imbalanced datasets.
    """
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Probabilities for class 1 are constant (0.0)
    y_prob = np.zeros(len(y_test), dtype=np.float32)
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    metrics["best_params"] = "strategy=most_frequent"
    return metrics


def train_eval_svm_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict:
    """
    Train and evaluate RBF-kernel SVM with class_weight='balanced' and 3-fold CV tuning.
    Decision function values are used as continuous ranking scores for PR-AUC / ROC-AUC.
    """
    param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        param_grid=param_grid,
        cv=cv,
        scoring=pr_auc_scorer,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_scores = best_model.decision_function(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_scores)
    metrics["best_params"] = str(grid.best_params_)
    return metrics


def train_eval_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict:
    """
    Train and evaluate Random Forest with class_weight='balanced' and 3-fold CV tuning.
    """
    param_grid = {"max_depth": [3, 5, 8], "n_estimators": [50, 100]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=seed),
        param_grid=param_grid,
        cv=cv,
        scoring=pr_auc_scorer,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    metrics["best_params"] = str(grid.best_params_)
    return metrics


def train_eval_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict:
    """
    Train and evaluate XGBoost with scale_pos_weight tuned to class imbalance ratio.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    param_grid = {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_estimators=50,
            eval_metric="logloss",
        ),
        param_grid=param_grid,
        cv=cv,
        scoring=pr_auc_scorer,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    metrics["best_params"] = str(grid.best_params_)
    return metrics


def run_full_features_regime() -> list[dict]:
    """
    Run classical models on the full feature set (218 dimensions, 80,000 train rows).
    Provides reference ceiling for model capacity.
    """
    print("-" * 70)
    print("REGIME (A): FULL FEATURES & LARGE DATASET REFERENCE")
    print("-" * 70)

    full_path = config.PROCESSED_DATA_DIR / "full_features.npz"
    data = np.load(full_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = []

    # 1. Dummy Classifier
    print("Evaluating Dummy Classifier on full test set...")
    dummy_res = train_eval_dummy(X_train, y_train, X_test, y_test)
    dummy_res.update({"regime": "full_features", "model": "Dummy", "n_qubits": 218, "repeat": 0})
    results.append(dummy_res)
    print(f"  Dummy: Acc={dummy_res['accuracy']:.4f}, PR-AUC={dummy_res['pr_auc']:.4f}, F1={dummy_res['f1']:.4f}")

    # 2. Random Forest
    print("Training and evaluating Random Forest (full features)...")
    rf_res = train_eval_random_forest(X_train, y_train, X_test, y_test, seed=config.SEED)
    rf_res.update({"regime": "full_features", "model": "RandomForest", "n_qubits": 218, "repeat": 0})
    results.append(rf_res)
    print(f"  RandomForest: PR-AUC={rf_res['pr_auc']:.4f}, ROC-AUC={rf_res['roc_auc']:.4f}, F1={rf_res['f1']:.4f}")

    # 3. XGBoost
    print("Training and evaluating XGBoost (full features)...")
    xgb_res = train_eval_xgboost(X_train, y_train, X_test, y_test, seed=config.SEED)
    xgb_res.update({"regime": "full_features", "model": "XGBoost", "n_qubits": 218, "repeat": 0})
    results.append(xgb_res)
    print(f"  XGBoost: PR-AUC={xgb_res['pr_auc']:.4f}, ROC-AUC={xgb_res['roc_auc']:.4f}, F1={xgb_res['f1']:.4f}")

    # 4. SVM with RBF (stratified 10,000 sample for tractable O(N^2) training time)
    print("Training and evaluating SVM RBF (10,000 stratified train sample)...")
    rng = np.random.RandomState(config.SEED)
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    sample_fraud = rng.choice(fraud_idx, size=int(10000 * y_train.mean()), replace=False)
    sample_legit = rng.choice(legit_idx, size=10000 - len(sample_fraud), replace=False)
    sub_idx = rng.permutation(np.concatenate([sample_fraud, sample_legit]))

    svm_res = train_eval_svm_rbf(X_train[sub_idx], y_train[sub_idx], X_test, y_test, seed=config.SEED)
    svm_res.update({"regime": "full_features", "model": "SVM_RBF", "n_qubits": 218, "repeat": 0})
    results.append(svm_res)
    print(f"  SVM_RBF: PR-AUC={svm_res['pr_auc']:.4f}, ROC-AUC={svm_res['roc_auc']:.4f}, F1={svm_res['f1']:.4f}")

    return results


def run_fair_quantum_subsets_regime() -> list[dict]:
    """
    Run classical models on the exact same PCA-reduced features and small subsets
    as the quantum models across all n_qubits and all N_REPEATS.
    This guarantees a strictly fair, apples-to-apples benchmark.
    """
    print("-" * 70)
    print("REGIME (B): FAIR QUANTUM SUBSETS COMPARISON")
    print("-" * 70)

    results = []

    for n in config.N_QUBITS_LIST:
        print(f"\nEvaluating on n = {n} PCA features...")
        for rep in range(config.N_REPEATS):
            subset_file = config.PROCESSED_DATA_DIR / f"quantum_subset_n{n}_rep{rep}.npz"
            data = np.load(subset_file)
            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]
            rep_seed = int(data["seed"][0])

            # Dummy
            d_res = train_eval_dummy(X_train, y_train, X_test, y_test)
            d_res.update({"regime": "fair_quantum_subset", "model": "Dummy", "n_qubits": n, "repeat": rep})
            results.append(d_res)

            # SVM RBF
            svm_res = train_eval_svm_rbf(X_train, y_train, X_test, y_test, seed=rep_seed)
            svm_res.update({"regime": "fair_quantum_subset", "model": "SVM_RBF", "n_qubits": n, "repeat": rep})
            results.append(svm_res)

            # Random Forest
            rf_res = train_eval_random_forest(X_train, y_train, X_test, y_test, seed=rep_seed)
            rf_res.update({"regime": "fair_quantum_subset", "model": "RandomForest", "n_qubits": n, "repeat": rep})
            results.append(rf_res)

            # XGBoost
            xgb_res = train_eval_xgboost(X_train, y_train, X_test, y_test, seed=rep_seed)
            xgb_res.update({"regime": "fair_quantum_subset", "model": "XGBoost", "n_qubits": n, "repeat": rep})
            results.append(xgb_res)

            print(
                f"  [Qubits={n}, Rep={rep+1}/{config.N_REPEATS}] "
                f"PR-AUC: SVM={svm_res['pr_auc']:.4f}, RF={rf_res['pr_auc']:.4f}, XGB={xgb_res['pr_auc']:.4f}"
            )

    return results


def run_classical_baselines() -> pd.DataFrame:
    """
    Execute all classical baseline evaluations and save results to CSV.
    """
    print("=" * 70)
    print("STAGE 2: CLASSICAL BASELINES BENCHMARKING")
    print("=" * 70)

    # Regime A: Full features
    regime_a_results = run_full_features_regime()

    # Regime B: Fair quantum subsets
    regime_b_results = run_fair_quantum_subsets_regime()

    all_results = regime_a_results + regime_b_results
    results_df = pd.DataFrame(all_results)

    output_csv = config.RESULTS_DIR / "classical_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nAll classical results saved to: {output_csv}")

    # Print summary table of fair comparison
    print("\n" + "=" * 70)
    print("SUMMARY: FAIR QUANTUM SUBSET COMPARISON (Mean +/- Std over 5 repeats)")
    print("=" * 70)
    fair_df = results_df[results_df["regime"] == "fair_quantum_subset"]
    summary = fair_df.groupby(["n_qubits", "model"])[["pr_auc", "roc_auc", "f1", "accuracy"]].agg(["mean", "std"])
    print(summary.round(4))

    print("\nSTAGE 2 COMPLETE: Classical baselines successfully benchmarked.")
    print("=" * 70)
    return results_df


if __name__ == "__main__":
    run_classical_baselines()
