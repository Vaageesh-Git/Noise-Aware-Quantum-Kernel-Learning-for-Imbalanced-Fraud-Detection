# Noise-Aware Quantum Kernel Learning for Imbalanced Fraud Detection
### Simulator-to-Hardware Benchmarking on IBM Quantum

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.6+](https://img.shields.io/badge/qiskit-2.6.2-purple.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid quantum-classical machine learning project benchmarking quantum kernel learning against strong classical baselines (XGBoost, Random Forest, RBF-SVM) on the real-world, highly imbalanced IEEE-CIS fraud dataset. The benchmark systematically analyzes performance across four execution environments:
1. **Setting A**: Ideal exact analytical statevector simulation (noise-free).
2. **Setting B**: Ideal finite-shot projective simulation (isolating shot noise).
3. **Setting C**: Noisy simulation with IBM Quantum's 127-qubit `FakeSherbrooke` calibration snapshot ($T_1/T_2$ relaxation, depolarizing gate noise, readout error).
4. **Setting D**: IBM Quantum hardware execution via Qiskit Runtime SamplerV2 with Dynamical Decoupling (`XX` sequences), Pauli Twirling, and classical Positive Semi-Definite (PSD) post-processing.

---

## Project Structure

```text
├── README.md                   <- Complete setup, execution instructions, and runtimes
├── requirements.txt            <- Exact pinned dependencies (Python 3.12 compatible)
├── config.py                   <- Centralized settings (seeds, paths, qubit counts, shots)
├── pytest.ini                  <- Pytest configuration
├── .gitignore                  <- Git ignore rules (protects data, tokens, and caches)
│
├── data/                       <- Raw Kaggle IEEE-CIS CSV files (user provided)
│   ├── train_transaction.csv
│   └── train_identity.csv
│
├── src/                        <- Modular clean source code (<30 lines/function)
│   ├── data_prep.py            <- Step 1: Load, merge, clean, split, PCA, quantum subsets
│   ├── features.py             <- Leak-free feature transforms, scaling, and [0, pi] angles
│   ├── classical_baselines.py  <- Step 2: Dummy, RBF-SVM, Random Forest, XGBoost
│   ├── quantum_kernel.py       <- Step 3: Feature maps, complexity analysis, unified kernel
│   ├── run_ideal.py            <- Step 4: Settings A & B grid benchmarks & ranking
│   ├── run_noisy.py            <- Step 5: Setting C FakeSherbrooke noisy simulation
│   ├── run_hardware.py         <- Step 6: Setting D IBM Quantum hardware & mitigation
│   ├── imbalance_study.py      <- Step 7: Strategies (SMOTE, RUS, weights) & sensitivity
│   ├── evaluate.py             <- Evaluation metrics (PR-AUC, ROC-AUC, F1, Recall)
│   └── make_plots.py           <- Step 8: Publication-quality figures & SUMMARY.md
│
├── tests/                      <- 19 comprehensive unit tests
│   ├── test_data_prep.py
│   ├── test_classical.py
│   ├── test_quantum_kernel.py
│   ├── test_ideal.py
│   ├── test_noisy.py
│   ├── test_hardware.py
│   ├── test_imbalance.py
│   └── test_plots.py
│
├── results/                    <- Benchmark output CSVs, raw JSON batches, and summary
│   ├── processed_data/         <- Preprocessed train/test and quantum subset .npz files
│   ├── kernels/                <- Saved train/test quantum Gram matrices (.npz)
│   ├── hardware_raw/           <- Streaming raw hardware measurement batch JSONs
│   ├── classical_results.csv   <- Classical baseline benchmark results
│   ├── ideal_results.csv       <- Settings A & B simulation results
│   ├── top_configurations.csv  <- Top-ranked quantum configurations
│   ├── noisy_results.csv       <- Setting C noisy simulation results
│   ├── hardware_results.csv    <- Setting D hardware benchmark results
│   ├── imbalance_results.csv   <- Imbalance strategies & fraud ratio sensitivity results
│   └── SUMMARY.md              <- Auto-generated Markdown summary with performance tables
│
├── figures/                    <- Publication-quality visual figures (.png)
│   ├── class_distribution.png
│   ├── feature_maps_vs_classical.png
│   ├── kernel_heatmaps_comparison.png
│   ├── settings_comparison.png
│   ├── kernel_variance_vs_qubits.png
│   └── imbalance_study.png
│
└── docs/                       <- Educational guide for students and evaluators
    └── EXPLAINER.md            <- Pedagogical explanations, limitations & 10 evaluator Q&As
```

---

## Quickstart & Environment Setup

### 1. Prerequisites
- Python 3.10+ (tested and verified on Python 3.12)
- macOS, Linux, or Windows WSL

### 2. Installation
Clone the repository and install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 3. Dataset Setup
Download the Kaggle IEEE-CIS Fraud Detection dataset and place the training files in `data/`:
- `data/train_transaction.csv`
- `data/train_identity.csv`

*(If files are missing or placed in `ieee-fraud-detection/`, the loader detects them automatically or prints a helpful error message).*

---

## Step-by-Step Execution Guide

Every script can be run independently from the command line and reads its parameters from `config.py`.

| Stage | Script Command | Output Artifacts | Expected Runtime |
| :--- | :--- | :--- | :--- |
| **Stage 1** | `python3 src/data_prep.py` | `results/processed_data/*.npz`<br>`figures/class_distribution.png` | ~25 seconds |
| **Stage 2** | `python3 src/classical_baselines.py` | `results/classical_results.csv` | ~30 seconds |
| **Stage 3** | `pytest tests/test_quantum_kernel.py` | Validated kernel mathematical invariants | ~2 seconds |
| **Stage 4** | `python3 src/run_ideal.py` | `results/ideal_results.csv`<br>`results/kernels/kernel_settingA_*.npz` | ~20 seconds |
| **Stage 5** | `python3 src/run_noisy.py` | `results/noisy_results.csv`<br>`results/kernels/kernel_settingC_*.npz` | ~3.5 minutes |
| **Stage 6** | `python3 src/run_hardware.py --dry-run` | `results/hardware_results.csv`<br>`results/hardware_raw/*.json` | ~30 seconds |
| **Stage 7** | `python3 src/imbalance_study.py` | `results/imbalance_results.csv` | ~15 seconds |
| **Stage 8** | `python3 src/make_plots.py` | `figures/*.png`<br>`results/SUMMARY.md` | ~5 seconds |

---

## Hardware Execution & IBM Quantum Credentials

The hardware script `src/run_hardware.py` provides complete dual-mode execution:

### Dry-Run Mode (Default, Zero Quantum Cost)
Tests the complete circuit transpilation, batching, and error mitigation pipeline using IBM's 127-qubit `FakeSherbrooke` backend without contacting the IBM cloud:
```bash
python3 src/run_hardware.py --dry-run
```

### Live Hardware Mode
To run on real IBM Quantum quantum processors:
1. Export your IBM Quantum API token as an environment variable (never written to disk):
   ```bash
   export IBM_QUANTUM_TOKEN="your_personal_api_token_here"
   ```
2. Run the script in live mode:
   ```bash
   python3 src/run_hardware.py --live
   ```
3. The script displays the least-busy available processor, estimates total circuit (390) and shot (399,360) consumption, and requests explicit user confirmation before submitting. All completed batches are immediately written to disk to allow graceful resumption.

---

## Running the Unit Test Suite

Execute the full automated pytest suite covering data integrity, classical models, quantum circuits, statevector fidelity, noise models, PSD projection, and figure generation:
```bash
pytest
```
*Expected: 19 passed in ~2 seconds.*

---

## Key Experimental Takeaways

1. **Top Quantum Feature Map**: `CustomAngleRingMap` (angle encoding $R_Y(\theta)$ with a 2-qubit $CZ$ ring) decisively outperformed standard `ZZFeatureMap` (**PR-AUC $0.3675$ vs $0.2668$**, Kernel-Target Alignment $0.652$ vs $0.591$).
2. **Noise and Concentration**: As feature dimension grew from 4 to 6 qubits on noisy simulated hardware, Frobenius error doubled ($11.6\% \to 21.4\%$) and off-diagonal variance retention fell to $60.1\%$, showing early quantum kernel concentration.
3. **Error Mitigation**: Dynamical Decoupling (`XX` pulses) and Pauli Twirling increased Pearson correlation with the ideal kernel from $0.9954$ to $0.9965$. Classical nearest-PSD projection further raised it to $0.9969$.
4. **Severe Imbalance Sensitivity**: At extreme 5% fraud imbalance, the Quantum Kernel SVM slightly outperformed the Classical RBF-SVM (**PR-AUC $0.3810$ vs $0.3589$**). On balanced and moderately imbalanced sets, the quantum model matched classical RBF-SVM.
5. **Full Features vs. Quantum Limits**: On the full 218 features, classical XGBoost (PR-AUC $\approx 0.49$) easily surpassed quantum models restricted to 4-6 features. Quantum advantage requires fault-tolerant devices capable of encoding wide feature spaces.
