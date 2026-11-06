"""
Feature Engineering and Preprocessing Module for Fraud Detection.
Implements leak-free feature transformations:
  1. Time of day and log-transformed transaction amount
  2. High-missing column dropping (based on training split only)
  3. Median numeric imputation and 'missing' string categorical imputation
  4. Frequency encoding for high-cardinality categorical variables
  5. Feature standardization via StandardScaler
  6. PCA dimensionality reduction to n components
  7. Rescaling PCA components to [0, pi] for quantum angle encoding
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def engineer_time_and_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw transaction timestamp into cyclical hour of day and apply log1p
    to transaction amount to compress heavy right-tailed financial values.
    """
    df = df.copy()
    # TransactionDT represents elapsed seconds; convert to cyclical hour of day (0-23)
    df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
    df.drop(columns=["TransactionDT"], inplace=True)

    # Financial transaction amounts follow a power law; log1p stabilizes variance
    df["TransactionAmt"] = np.log1p(df["TransactionAmt"].astype(np.float32))
    return df


def find_high_missing_columns(train_df: pd.DataFrame, threshold: float = 0.50) -> list[str]:
    """
    Identify columns in the training split with missing value fraction > threshold.
    Computing this on training data only prevents test data leakage.
    """
    missing_fraction = train_df.isna().mean()
    high_missing_cols = missing_fraction[missing_fraction > threshold].index.tolist()
    return high_missing_cols


def separate_column_types(df: pd.DataFrame, ignore_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Separate feature column names into numeric and categorical types.
    """
    candidate_cols = [c for c in df.columns if c not in ignore_cols]
    numeric_cols = []
    categorical_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def impute_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing values: training median for numerics, 'missing' for categoricals.
    Medians are derived strictly from training data to avoid look-ahead bias.
    """
    train_clean = train_df.copy()
    test_clean = test_df.copy()

    # Impute numeric features with training median
    for col in numeric_cols:
        med = train_clean[col].median()
        # Fallback to 0 if an entire column happens to be NaN
        fill_val = 0.0 if pd.isna(med) else med
        train_clean[col] = train_clean[col].fillna(fill_val).astype(np.float32)
        test_clean[col] = test_clean[col].fillna(fill_val).astype(np.float32)

    # Impute categorical features with explicit string 'missing'
    for col in categorical_cols:
        train_clean[col] = train_clean[col].fillna("missing").astype(str)
        test_clean[col] = test_clean[col].fillna("missing").astype(str)

    return train_clean, test_clean


def frequency_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical strings with their relative frequencies observed in train data.
    Unseen categories in test data receive a frequency of 0.0.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in categorical_cols:
        # Compute normalized frequency map from training set
        freq_map = train_encoded[col].value_counts(normalize=True).to_dict()
        train_encoded[col] = train_encoded[col].map(freq_map).fillna(0.0).astype(np.float32)
        test_encoded[col] = test_encoded[col].map(freq_map).fillna(0.0).astype(np.float32)

    return train_encoded, test_encoded


def standardize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize all features to zero mean and unit variance using StandardScaler.
    Fit strictly on training data; transform both train and test.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_train_scaled, X_test_scaled, scaler


def fit_pca_reduction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA to n components on the training features and transform both splits.
    PCA extracts the orthogonal axes of greatest variance for quantum encoding.
    """
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train).astype(np.float32)
    X_test_pca = pca.transform(X_test).astype(np.float32)
    return X_train_pca, X_test_pca, pca


def scale_to_quantum_range(
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    min_val: float = 0.0,
    max_val: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly scale PCA features into [min_val, max_val] (default [0, pi]) for rotation gates.
    Min and max bounds are computed strictly on training data, and test data is clipped.
    """
    train_min = X_train_pca.min(axis=0)
    train_max = X_train_pca.max(axis=0)

    # Avoid division by zero if a component has near zero variance
    range_span = np.where(train_max - train_min == 0, 1.0, train_max - train_min)

    # Scale training features to [0, 1] then to [min_val, max_val]
    X_train_scaled = (X_train_pca - train_min) / range_span * (max_val - min_val) + min_val

    # Scale test features using the exact training bounds
    X_test_scaled = (X_test_pca - train_min) / range_span * (max_val - min_val) + min_val

    # Clip test values so that out-of-bounds test samples stay within [min_val, max_val]
    X_train_scaled = np.clip(X_train_scaled, min_val, max_val).astype(np.float32)
    X_test_scaled = np.clip(X_test_scaled, min_val, max_val).astype(np.float32)

    return X_train_scaled, X_test_scaled
