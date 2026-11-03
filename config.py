"""
Central Configuration for Noise-Aware Quantum Kernel Learning Project.
All parameters, random seeds, sample sizes, circuit configurations, and paths
are defined in this single file for clarity, reproducibility, and ease of editing.
"""

from pathlib import Path
import os

# Set matplotlib cache directory to avoid permissions warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# ==============================================================================
# 1. Project Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"
IDENTITY_CSV = DATA_DIR / "train_identity.csv"

# Fallback path if files are located in ieee-fraud-detection folder
ALT_DATA_DIR = PROJECT_ROOT / "ieee-fraud-detection"
ALT_TRANSACTION_CSV = ALT_DATA_DIR / "train_transaction.csv"
ALT_IDENTITY_CSV = ALT_DATA_DIR / "train_identity.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Saved processed data paths
PROCESSED_DATA_DIR = RESULTS_DIR / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. General Experiment & Reproducibility Settings
# ==============================================================================
SEED = 42

# Total stratified pool size sampled from merged transaction + identity data
POOL_SIZE = 100000

# Stratified train/test split proportion for classical and feature preparation
TEST_SIZE = 0.20

# Missing value threshold: drop columns with more than this ratio of NaNs
MISSING_THRESHOLD = 0.50

# ==============================================================================
# 3. Quantum Experiment Settings
# ==============================================================================
# Feature dimension / qubit counts to benchmark
N_QUBITS_LIST = [4, 6]

# Quantum subset sizes (small due to O(N^2) quantum kernel circuit evaluations)
QUANTUM_TRAIN = 300
QUANTUM_TEST = 200

# Fraud ratio enforced in quantum subsets (10% positive fraud rate)
FRAUD_RATIO_QUANTUM = 0.10

# Number of independent random seeds/repeats for statistical confidence
N_REPEATS = 5

# Quantum feature map parameter scaling range
FEATURE_MAP_MIN = 0.0
FEATURE_MAP_MAX = 3.141592653589793  # pi

# Quantum execution shots
SHOTS = 2048

# Feature maps to evaluate
FEATURE_MAP_NAMES = ["ZZFeatureMap", "PauliFeatureMap", "CustomAngleRingMap"]
FEATURE_MAP_REPS = [1, 2]
ENTANGLEMENT_TYPES = ["linear", "full"]

# Classical SVM C parameter search grid for precomputed kernel
SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]

# ==============================================================================
# 4. Hardware and Noisy Simulation Settings
# ==============================================================================
# Tiny subset for real quantum hardware execution
HW_TRAIN = 20
HW_TEST = 10
HW_SHOTS = 1024

# Default simulated fake backend for noisy benchmarking (127-qubit IBM backend)
FAKE_BACKEND_NAME = "FakeSherbrooke"

# Environment variable name holding IBM Quantum API token
IBM_TOKEN_ENV_VAR = "IBM_QUANTUM_TOKEN"

# ==============================================================================
# 5. Imbalance Study Settings
# ==============================================================================
# Fraud ratios tested in Stage 7
IMBALANCE_FRAUD_RATIOS = [0.50, 0.20, 0.10, 0.05]
