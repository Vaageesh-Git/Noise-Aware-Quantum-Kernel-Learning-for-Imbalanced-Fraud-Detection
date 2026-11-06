"""
Data Preparation Pipeline for Fraud Detection.
Performs:
  1. Data integrity check and memory-efficient loading of IEEE-CIS transaction & identity CSVs.
  2. Stratified pooling to POOL_SIZE rows.
  3. Class distribution analysis and visualization showing the 'accuracy paradox'.
  4. Leak-free stratified train/test split.
  5. Feature cleaning, missing value imputation, frequency encoding, and standardization.
  6. PCA reduction to n_qubits dimensions and [0, pi] angle scaling.
  7. Stratified small quantum subset generation across N_REPEATS independent random seeds.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path and suppress matplotlib cache warnings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config
from src.features import (
    engineer_time_and_amount,
    find_high_missing_columns,
    separate_column_types,
    impute_missing_values,
    frequency_encode,
    standardize_features,
    fit_pca_reduction,
    scale_to_quantum_range,
)


def verify_data_files() -> tuple[Path, Path]:
    """
    Check if IEEE-CIS CSV files exist at the configured data path or alternative path.
    If missing, print clear instructions where to place them and stop.
    """
    tr_path = config.TRANSACTION_CSV
    id_path = config.IDENTITY_CSV

    if not tr_path.exists() or not id_path.exists():
        # Check alternative folder
        if config.ALT_TRANSACTION_CSV.exists() and config.ALT_IDENTITY_CSV.exists():
            tr_path = config.ALT_TRANSACTION_CSV
            id_path = config.ALT_IDENTITY_CSV
        else:
            print("=" * 70)
            print("ERROR: IEEE-CIS dataset files not found!")
            print("Please place the following Kaggle IEEE-CIS files in the data/ folder:")
            print(f"  1. {config.TRANSACTION_CSV}")
            print(f"  2. {config.IDENTITY_CSV}")
            print("=" * 70)
            sys.exit(1)

    print(f"Dataset verified: \n  Transaction: {tr_path}\n  Identity: {id_path}")
    return tr_path, id_path


def load_and_pool_data(
    tr_path: Path,
    id_path: Path,
    pool_size: int = config.POOL_SIZE,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """
    Load raw transaction and identity CSVs with left join, downcast numeric dtypes
    to float32, and draw a stratified sample of pool_size rows.
    """
    print("Loading raw CSV files (this may take a few seconds)...")
    # Read transactions and identities
    df_tr = pd.read_csv(tr_path, low_memory=False)
    df_id = pd.read_csv(id_path, low_memory=False)

    print("Merging transaction and identity on TransactionID (left join)...")
    df = pd.merge(df_tr, df_id, on="TransactionID", how="left")
    del df_tr, df_id

    # Downcast float64 columns to float32 to save RAM
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    total_rows = len(df)
    total_fraud = int(df["isFraud"].sum())
    fraud_pct = (total_fraud / total_rows) * 100
    print(f"Full dataset: {total_rows:,} rows, {total_fraud:,} frauds ({fraud_pct:.2f}%).")

    # Sample stratified pool
    if pool_size and pool_size < total_rows:
        print(f"Drawing stratified sample of {pool_size:,} rows (seed={seed})...")
        df_pool, _ = train_test_split(
            df,
            train_size=pool_size,
            stratify=df["isFraud"],
            random_state=seed,
        )
    else:
        df_pool = df

    pool_fraud = int(df_pool["isFraud"].sum())
    print(f"Stratified pool: {len(df_pool):,} rows, {pool_fraud:,} frauds ({(pool_fraud/len(df_pool))*100:.2f}%).")
    return df_pool.reset_index(drop=True)


def plot_class_distribution(y: pd.Series, output_path: Path) -> None:
    """
    Plot and save the class distribution bar chart highlighting the severe class imbalance
    and illustrating the 'accuracy paradox' (a trivial dummy model gets ~96.5% accuracy).
    """
    counts = y.value_counts().sort_index()
    legit_count = int(counts.get(0, 0))
    fraud_count = int(counts.get(1, 0))
    total = legit_count + fraud_count
    legit_pct = (legit_count / total) * 100
    fraud_pct = (fraud_count / total) * 100
    dummy_acc = legit_pct

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], [legit_count, fraud_count], color=["#2b5c8f", "#d95f02"], width=0.5)

    # Annotate bar counts and percentages
    ax.text(0, legit_count + (total * 0.01), f"{legit_count:,}\n({legit_pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(1, fraud_count + (total * 0.01), f"{fraud_count:,}\n({fraud_pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("IEEE-CIS Fraud Detection: Class Imbalance & Accuracy Paradox", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Number of Transactions", fontsize=10)
    ax.set_ylim(0, total * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Callout note on accuracy paradox
    note = (
        f"Severe Imbalance ({fraud_pct:.1f}% Fraud)\n"
        f"A dummy classifier that predicts 100% legitimate\n"
        f"achieves {dummy_acc:.2f}% accuracy with 0% fraud recall!"
    )
    ax.text(
        0.55, 0.70, note,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2ae", edgecolor="#e6ab02", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved class distribution figure to: {output_path}")


def prepare_full_features(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Execute full feature preprocessing pipeline:
      1. Engineer time of day and log1p amount.
      2. Stratified train/test split.
      3. Drop high-missing columns (>50% missing on train split).
      4. Impute numeric and categorical features using training statistics.
      5. Frequency encode categoricals based on training frequencies.
      6. Standardize all features.
    """
    print("Engineering time of day and log-transformed amount...")
    df = engineer_time_and_amount(df)

    # Separate target and features
    y = df["isFraud"].values.astype(np.int32)
    feature_df = df.drop(columns=["isFraud"])

    print(f"Splitting data into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%)...")
    train_df, test_df, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    # Remove TransactionID identifier
    if "TransactionID" in train_df.columns:
        train_df = train_df.drop(columns=["TransactionID"])
        test_df = test_df.drop(columns=["TransactionID"])

    print("Identifying columns with >50% missing values in training set...")
    high_missing = find_high_missing_columns(train_df, threshold=config.MISSING_THRESHOLD)
    print(f"Dropping {len(high_missing)} high-missing columns...")
    train_df = train_df.drop(columns=high_missing)
    test_df = test_df.drop(columns=high_missing)

    numeric_cols, categorical_cols = separate_column_types(train_df, ignore_cols=[])
    print(f"Columns remaining: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical.")

    print("Imputing missing values (median for numeric, 'missing' for categorical)...")
    train_clean, test_clean = impute_missing_values(train_df, test_df, numeric_cols, categorical_cols)

    print("Frequency encoding categorical columns...")
    train_encoded, test_encoded = frequency_encode(train_clean, test_clean, categorical_cols)

    feature_names = train_encoded.columns.tolist()

    print("Standardizing all features via StandardScaler (fit on train only)...")
    X_train_full, X_test_full, _ = standardize_features(train_encoded, test_encoded)

    print(f"Full feature matrix shape: Train={X_train_full.shape}, Test={X_test_full.shape}")
    return X_train_full, X_test_full, y_train, y_test, feature_names


def generate_pca_and_quantum_sets(
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_qubits_list: list[int] = config.N_QUBITS_LIST,
    quantum_train_size: int = config.QUANTUM_TRAIN,
    quantum_test_size: int = config.QUANTUM_TEST,
    fraud_ratio: float = config.FRAUD_RATIO_QUANTUM,
    n_repeats: int = config.N_REPEATS,
    seed: int = config.SEED,
) -> dict:
    """
    For each n in n_qubits_list:
      1. Fit PCA to n components on training features.
      2. Scale components to [0, pi] and clip test set.
      3. Create stratified quantum subsets for n_repeats with exact fraud_ratio.
    Returns a structured dictionary with all arrays ready for modeling.
    """
    dataset_bundle = {
        "y_train": y_train,
        "y_test": y_test,
        "pca_data": {},
        "quantum_subsets": {},
    }

    for n in n_qubits_list:
        print(f"Fitting PCA({n} components) on training data...")
        X_train_pca, X_test_pca, pca_model = fit_pca_reduction(X_train_full, X_test_full, n_components=n, seed=seed)
        explained_var = float(np.sum(pca_model.explained_variance_ratio_))
        print(f"  PCA({n}) total explained variance ratio: {explained_var:.4f}")

        # Rescale strictly to [0, pi] for quantum angle encoding
        X_train_qscaled, X_test_qscaled = scale_to_quantum_range(
            X_train_pca,
            X_test_pca,
            min_val=config.FEATURE_MAP_MIN,
            max_val=config.FEATURE_MAP_MAX,
        )

        dataset_bundle["pca_data"][n] = {
            "X_train_scaled": X_train_qscaled,
            "X_test_scaled": X_test_qscaled,
            "explained_variance": explained_var,
        }

        # Build quantum subsets for n_repeats
        subsets_for_n = []
        for rep in range(n_repeats):
            rep_seed = seed + rep * 100
            rng = np.random.RandomState(rep_seed)

            # Build train subset
            n_train_fraud = int(round(quantum_train_size * fraud_ratio))
            n_train_legit = quantum_train_size - n_train_fraud

            fraud_train_idx = np.where(y_train == 1)[0]
            legit_train_idx = np.where(y_train == 0)[0]

            chosen_train_fraud = rng.choice(fraud_train_idx, size=n_train_fraud, replace=False)
            chosen_train_legit = rng.choice(legit_train_idx, size=n_train_legit, replace=False)
            train_sub_idx = rng.permutation(np.concatenate([chosen_train_fraud, chosen_train_legit]))

            # Build test subset
            n_test_fraud = int(round(quantum_test_size * fraud_ratio))
            n_test_legit = quantum_test_size - n_test_fraud

            fraud_test_idx = np.where(y_test == 1)[0]
            legit_test_idx = np.where(y_test == 0)[0]

            chosen_test_fraud = rng.choice(fraud_test_idx, size=n_test_fraud, replace=False)
            chosen_test_legit = rng.choice(legit_test_idx, size=n_test_legit, replace=False)
            test_sub_idx = rng.permutation(np.concatenate([chosen_test_fraud, chosen_test_legit]))

            sub_X_train = X_train_qscaled[train_sub_idx]
            sub_y_train = y_train[train_sub_idx]
            sub_X_test = X_test_qscaled[test_sub_idx]
            sub_y_test = y_test[test_sub_idx]

            print(
                f"  [Qubits={n}, Repeat={rep+1}/{n_repeats}] "
                f"Train: {len(sub_y_train)} rows ({int(sub_y_train.sum())} fraud, {sub_y_train.mean():.1%}), "
                f"Test: {len(sub_y_test)} rows ({int(sub_y_test.sum())} fraud, {sub_y_test.mean():.1%})"
            )

            subsets_for_n.append({
                "repeat": rep,
                "seed": rep_seed,
                "train_idx": train_sub_idx,
                "test_idx": test_sub_idx,
                "X_train": sub_X_train,
                "y_train": sub_y_train,
                "X_test": sub_X_test,
                "y_test": sub_y_test,
            })

        dataset_bundle["quantum_subsets"][n] = subsets_for_n

    return dataset_bundle


def save_processed_data(
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    dataset_bundle: dict,
    out_dir: Path = config.PROCESSED_DATA_DIR,
) -> None:
    """
    Save the prepared feature matrices and quantum subsets to compressed .npz files.
    """
    print(f"Saving prepared data arrays to: {out_dir}...")
    np.savez_compressed(
        out_dir / "full_features.npz",
        X_train=X_train_full,
        X_test=X_test_full,
        y_train=y_train,
        y_test=y_test,
    )

    for n, data_n in dataset_bundle["pca_data"].items():
        np.savez_compressed(
            out_dir / f"pca_{n}_features.npz",
            X_train_scaled=data_n["X_train_scaled"],
            X_test_scaled=data_n["X_test_scaled"],
            explained_variance=np.array([data_n["explained_variance"]]),
        )

        # Save quantum repeats
        subsets = dataset_bundle["quantum_subsets"][n]
        for item in subsets:
            rep = item["repeat"]
            np.savez_compressed(
                out_dir / f"quantum_subset_n{n}_rep{rep}.npz",
                X_train=item["X_train"],
                y_train=item["y_train"],
                X_test=item["X_test"],
                y_test=item["y_test"],
                train_idx=item["train_idx"],
                test_idx=item["test_idx"],
                seed=np.array([item["seed"]]),
            )

    print("Data preparation artifacts successfully written to disk.")


def run_pipeline() -> None:
    """
    Complete end-to-end execution of Stage 1 data preparation.
    """
    print("=" * 70)
    print("STAGE 1: IEEE-CIS FRAUD DETECTION DATA PREPARATION PIPELINE")
    print("=" * 70)

    tr_path, id_path = verify_data_files()
    df_pool = load_and_pool_data(tr_path, id_path)

    # Class distribution plot
    plot_class_distribution(df_pool["isFraud"], config.FIGURES_DIR / "class_distribution.png")

    # Full feature engineering & preprocessing
    X_train_full, X_test_full, y_train, y_test, _ = prepare_full_features(df_pool)

    # PCA reduction and quantum subsets
    dataset_bundle = generate_pca_and_quantum_sets(X_train_full, X_test_full, y_train, y_test)

    # Save all datasets
    save_processed_data(X_train_full, X_test_full, y_train, y_test, dataset_bundle)
    print("STAGE 1 COMPLETE: Ready for classical and quantum modeling.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
