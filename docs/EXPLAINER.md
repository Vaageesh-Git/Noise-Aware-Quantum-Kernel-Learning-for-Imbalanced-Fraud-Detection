# Noise-Aware Quantum Kernel Learning: A Student's Companion & Explainer

This document explains every concept, design decision, mathematical formula, and empirical finding in this project in clear, plain English. It is written for university students who need to run the code, understand how it works, and confidently explain their findings in an exam, presentation, or thesis defense.

---

## 1. Fundamentals in Plain English

### What is a Quantum Kernel?
In classical machine learning, a Support Vector Machine (SVM) finds a decision boundary (a hyperplane) that separates two classes (e.g., fraudulent vs. legitimate transactions). However, financial fraud is rarely linearly separable in the raw input space. 

Classical kernels (such as the Radial Basis Function / Gaussian RBF kernel) implicitly map data points into a higher-dimensional space where linear separation is possible:
$$k_{\text{RBF}}(x, x') = \exp(-\gamma \|x - x'\|^2)$$

A **Quantum Kernel** performs an analogous operation using the Hilbert space of quantum states. It maps a classical feature vector $x$ into a quantum state $|\psi(x)\rangle$ using a quantum circuit $U(x)$:
$$x \mapsto |\psi(x)\rangle = U(x)|0^{\otimes n}\rangle$$

The quantum kernel function measures the **quantum fidelity** (transition probability) between two quantum feature states:
$$\kappa(x, x') = |\langle \psi(x') | \psi(x) \rangle|^2 = |\langle 0^{\otimes n} | U^\dagger(x') U(x) | 0^{\otimes n} \rangle|^2$$

Physically, we prepare the state $U(x)|0\dots 0\rangle$, apply the inverse circuit $U^\dagger(x')$, and measure the probability of observing the all-zeros bitstring $|0\dots 0\rangle$. If $x = x'$, $U^\dagger(x) U(x) = I$, so the fidelity is exactly $1.0$. If $x$ and $x'$ are orthogonal in quantum state space, the fidelity is $0.0$.

---

### What is a Quantum Feature Map?
A **quantum feature map** is the parameterized quantum circuit $U(x)$ that translates classical numbers into quantum gate rotation angles and entanglement phases.
- **Single-Qubit Rotations**: Encode continuous coordinates as Bloch sphere rotations:
  $$R_Y(\theta) = \exp\left(-i \frac{\theta}{2} Y\right) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$
- **Two-Qubit Entanglement**: Gates like $CZ$, $CX$, or $ECR$ create non-local quantum correlations between qubits, allowing the kernel to capture multi-feature interactions that classical models might find difficult to compute.

In this project, we compared three feature maps:
1. **`ZZFeatureMap`**: Applies single-qubit rotations followed by pairwise phase interactions $\phi_{\{i,j\}}(x) = (\pi - x_i)(\pi - x_j)$ using $CNOT - R_Z - CNOT$ ladders.
2. **`PauliFeatureMap`**: Uses generalized Pauli operators ($Z$ and $ZZ$).
3. **`CustomAngleRingMap`**: Our custom architecture: single-qubit $R_Y(x_i)$ angle encoding followed by a circular ring of $CZ$ entangling gates between neighboring qubits $(i, (i+1) \pmod n)$.

---

### Why Must We Reduce Data to 4 or 6 Features with PCA?
The raw IEEE-CIS dataset contains over 400 transaction and identity attributes. Why do we reduce them to $n=4$ or $n=6$ components?
1. **NISQ Qubit Constraints**: Near-term quantum computers and exact quantum simulators have limited qubit numbers. Encoding 200 features would require 200 physical qubits, which exceeds the low-error coherence capacity of current hardware.
2. **Barren Plateaus & Quantum Kernel Concentration**: Mathematical proofs (Thanasilp et al., 2022; Cerezo et al., 2021) demonstrate that as the number of qubits grows in deeply entangled circuits, off-diagonal kernel entries decay exponentially toward a uniform constant:
   $$\kappa(x, x') \to \frac{1}{2^n}$$
   When this happens, the kernel loses all contrast (variance drops to zero), and the SVM cannot identify a separating hyperplane.
3. **Dimensional Scaling**: Principal Component Analysis (PCA) retains the orthogonal axes of highest variance ($39.6\%$ with 4 components, $46.9\%$ with 6 components) while keeping quantum simulation and hardware execution computationally tractable.

---

### What are Shots and Shot Noise?
A quantum computer cannot output continuous probabilities directly. When a qubit is measured in the computational basis, it collapses to either $|0\rangle$ or $|1\rangle$.

To estimate the probability $p = |\langle 0^{\otimes n} | U^\dagger(x') U(x) | 0^{\otimes n} \rangle|^2$, we must execute the circuit repeatedly $S$ times (where $S$ is the number of **shots**, e.g., $S = 2048$):
$$\hat{p} = \frac{N_{\text{all zeros}}}{S}$$

Because $S$ is finite, the estimated fidelity $\hat{p}$ fluctuates around the true value $p$ according to a binomial distribution with standard error:
$$\sigma_{\text{shot}} = \sqrt{\frac{p(1 - p)}{S}}$$
With $S = 2048$, this statistical uncertainty is approximately $\pm 0.01$, known as **shot noise**.

---

### What is Hardware Device Noise and Decoherence?
On real superconducting transmon quantum computers (like IBM Quantum's Eagle 127-qubit processors), qubits are physical microchips cooled to 15 millikelvin. They suffer from several sources of noise:
1. **$T_1$ Relaxation**: Qubits spontaneously decay from the excited state $|1\rangle$ to the ground state $|0\rangle$ due to thermal interactions with the environment.
2. **$T_2$ Dephasing**: Random environmental electromagnetic fluctuations destroy the fragile relative phase between quantum superpositions.
3. **Two-Qubit Gate Errors**: Physical entangling gates (e.g., $ECR$) take tens of nanoseconds to execute and have error rates $\sim 0.5\% - 2\%$, which is $\sim 10\times$ higher than single-qubit gates.
4. **Readout Assignment Bit-Flips**: The microwave measurement resonator occasionally misidentifies a $|1\rangle$ as $|0\rangle$ or vice versa.

In Setting C, we simulate these physical mechanisms using IBM's calibrated `FakeSherbrooke` backend. In Setting D, we observe them on physical hardware.

---

### What is Quantum Error Mitigation?
**Error Mitigation** refers to algorithmic and pulse-level techniques designed to suppress or cancel noise without the massive physical qubit overhead required for full fault-tolerant quantum error correction (QEC):
1. **Dynamical Decoupling (DD)**: When certain qubits are idle waiting for slower gates on other qubits to finish, environmental phase noise accumulates. DD inserts calibrated spin-echo refocusing pulses (e.g., periodic `XX` $\pi$-pulses) during idle periods to dynamically decouple the qubit from low-frequency environmental noise.
2. **Pauli Twirling**: Inserts random Pauli gates before and after entangling two-qubit gates, transforming arbitrary coherent systematic calibration errors into simple stochastic Pauli depolarizing channels, which are much easier to handle.

---

### What is Classical Positive Semi-Definite (PSD) Post-Processing?
Mercer's theorem states that any valid kernel matrix must be **symmetric and positive semi-definite (PSD)**:
$$v^T K v \ge 0 \quad \forall v \in \mathbb{R}^N \iff \lambda_i(K) \ge 0 \quad \forall i$$
While ideal quantum statevector kernels are mathematically guaranteed to be PSD, physical hardware noise and finite shot sampling can distort the matrix and introduce small negative eigenvalues ($\lambda < 0$). This causes classical quadratic programming solvers (like `libsvm` inside `sklearn.svm.SVC`) to fail or produce suboptimal margins.

To solve this, our script includes a classical mathematical projection:
1. Spectral decomposition: $K = V \Lambda V^T$
2. Clip negative eigenvalues: $\Lambda_+ = \max(\Lambda, \epsilon)$
3. Congruence diagonal normalization: $K_{\text{norm}} = D^{-1/2} (V \Lambda_+ V^T) D^{-1/2}$ where $D = \text{diag}(K_{\text{psd}})$
4. Guarantee $K_{ii} = 1.0$ and $\lambda_i \ge 0$.

> **CRITICAL DISTINCTION FOR STUDENTS:** This is **classical matrix post-processing**, NOT quantum hardware error mitigation. It does not alter physical quantum states; it merely projects an estimated noisy matrix onto the closest valid mathematical correlation matrix.

---

### Why is PR-AUC the Primary Metric (The Accuracy Paradox)?
In the IEEE-CIS fraud dataset, only **$3.5\%$** of transactions are fraudulent ($96.5\%$ are legitimate).

If a "model" simply predicts legitimate for every single transaction:
- **Accuracy = 96.50%**
- **Recall = 0.00%** (Zero frauds caught)
- **Precision = 0.00%**

This is the **Accuracy Paradox**: high accuracy is completely meaningless on imbalanced data. 

Similarly, the standard **ROC-AUC** metric is dominated by the massive number of True Negatives ($TN$), giving an overly optimistic picture of performance.

**Precision-Recall AUC (PR-AUC / Average Precision)** measures the trade-off strictly between Precision (fraction of flagged alerts that are truly fraud) and Recall (fraction of all frauds caught). For an uninformative baseline, PR-AUC equals the base fraud rate ($0.035$). A model with PR-AUC $> 0.35$ is demonstrating genuine predictive power.

---

## 2. File-by-File Walkthrough

| File | Purpose | Key Functions |
| :--- | :--- | :--- |
| **`config.py`** | Central configuration file. | Defines all hyperparameters, seeds, sample sizes, paths, and qubit lists in one place. |
| **`src/features.py`** | Feature engineering without leakage. | `engineer_time_and_amount`, `impute_missing_values`, `frequency_encode`, `scale_to_quantum_range`. |
| **`src/data_prep.py`** | Ingestion, cleaning, PCA, subsets. | `load_and_pool_data`, `plot_class_distribution`, `generate_pca_and_quantum_sets`. |
| **`src/evaluate.py`** | Consistent metric calculations. | `compute_classification_metrics` (Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrix). |
| **`src/classical_baselines.py`** | Classical models benchmark. | Evaluates Dummy, RBF-SVM, Random Forest, and XGBoost on full features (Regime A) and fair subsets (Regime B). |
| **`src/quantum_kernel.py`** | Quantum kernel core engine. | `get_feature_map`, `analyze_circuit_complexity`, unified `compute_kernel`, `compute_kernel_diagnostics`. |
| **`src/run_ideal.py`** | Ideal simulator benchmarks. | Evaluates Setting A (Statevector) and Setting B (Shots) across all maps, qubits, and 5 repeats. Tunes $C$ via 3-fold CV. |
| **`src/run_noisy.py`** | Setting C noisy simulation. | Simulates physical noise using `FakeSherbrooke`. Measures Frobenius error, Pearson correlation, and variance shrinkage. |
| **`src/run_hardware.py`** | Setting D hardware pipeline. | Supports `--dry-run` and live execution with Dynamical Decoupling, Pauli Twirling, and classical PSD post-processing. |
| **`src/imbalance_study.py`** | Imbalance strategies & sensitivity. | Compares `none`, `class_weights`, `undersampling`, and `SMOTE`, then benchmarks across fraud ratios $0.05 \to 0.50$. |
| **`src/make_plots.py`** | Visualization & report synthesis. | Generates publication figures in `figures/` and writes `results/SUMMARY.md`. |

---

## 3. Limitations and Honest Notes

As honest scientists and engineers, we must transparently document the scope and limitations of our benchmark:

1. **Small Sample Sizes for Quantum Kernels**:
   Evaluating a quantum kernel requires $O(N^2)$ circuit executions ($44,850$ circuits for $N=300$). While classical models can easily train on millions of rows, quantum kernel algorithms on current simulators and hardware are constrained to small training subsets ($N=300$).
2. **PCA Information Loss**:
   Reducing 218 features down to 4 or 6 principal components captures $39.6\% - 46.9\%$ of total variance, discarding fine-grained transactional details (such as IP addresses and device fingerprints). Classical XGBoost on all 218 features achieves higher PR-AUC ($\approx 0.49$) than any model restricted to 4-6 features.
3. **Random Stratified vs. Temporal Split**:
   In commercial production fraud detection, transactions are strictly ordered by time (training on past weeks, testing on future weeks). We used a stratified random split to benchmark quantum algorithmic properties under controlled statistical conditions across 5 random seeds, but production systems must account for concept drift over time.
4. **Tiny Hardware Test Set**:
   Due to quantum cloud queue limits and free-tier shot budgets, the real hardware slice was restricted to $N_{\text{train}}=20, N_{\text{test}}=10$. Therefore, the hardware benchmark's primary scientific value lies in **kernel matrix fidelity and Pearson correlation recovery**, rather than high-confidence classification test statistics.
5. **No Universal Quantum Advantage**:
   We do NOT claim quantum advantage. Under fair, apples-to-apples conditions on the exact same 4-6 features, the quantum kernel matches classical RBF-SVM and gains an edge only under severe class imbalance ($5\%$ fraud).

---

## 4. 10 Questions an Evaluator Might Ask (with Short Answers)

### Question 1: What exactly is a quantum kernel matrix, and how is it computed?
**Answer**: A quantum kernel matrix $K$ is an $N \times N$ Gram matrix where each entry $K_{ij} = \kappa(x_i, x_j) = |\langle 0^{\otimes n} | U^\dagger(x_j) U(x_i) | 0^{\otimes n} \rangle|^2$ represents the fidelity (overlap) between the quantum states encoding data points $x_i$ and $x_j$. In simulation, it is computed via statevector inner products; on quantum hardware, it is computed by running the circuit $U(x_i)$ followed by $U^\dagger(x_j)$ and measuring the frequency of the all-zeros bitstring $|0\dots 0\rangle$.

### Question 2: Why did `CustomAngleRingMap` outperform IBM's standard `ZZFeatureMap`?
**Answer**: On continuous tabular PCA fraud features, `ZZFeatureMap` introduces non-linear pairwise phase products $(\pi - x_i)(\pi - x_j)$ and deeper circuits (depth 66-143 with 10-38 entangling gates), which create severe oscillatory phase cancellation and lower Kernel-Target Alignment ($\sim 0.52 - 0.59$). In contrast, `CustomAngleRingMap` uses clean $R_Y(\theta)$ angle encoding with a circular $CZ$ ring, producing higher alignment ($KTA \approx 0.65 - 0.68$), shallow depth (5-41), and superior PR-AUC ($0.3675$ vs $0.2668$).

### Question 3: How does your benchmark prevent data leakage between train and test sets?
**Answer**: All preprocessing steps—including missing value column dropping thresholds, numeric median imputation, categorical frequency dictionaries, standard scaling, and PCA projection axes—are fitted strictly on the training partition. Test samples are transformed using only training statistics, and test PCA coordinates are clipped to the training $[0, \pi]$ bounds. Furthermore, SMOTE and undersampling are applied exclusively to training data.

### Question 4: Why is PR-AUC preferred over ROC-AUC and Accuracy for fraud detection?
**Answer**: Fraud is rare ($3.5\%$). A dummy model predicting legitimate for every transaction achieves $96.5\%$ accuracy with $0\%$ recall, exposing the accuracy paradox. ROC-AUC is distorted by the overwhelming number of True Negatives. PR-AUC evaluates the exact trade-off between Precision and Recall for the positive minority class, making it the gold standard for imbalanced fraud benchmarks.

### Question 5: What is "kernel concentration," and did you observe it in your experiments?
**Answer**: Kernel concentration is the tendency of quantum kernel entries to concentrate exponentially around a constant value ($K_{ij} \to 1/2^n$) as the number of qubits and circuit depth increase, causing the off-diagonal variance to collapse toward zero. We observed this in Figure 5: as qubits increased from 4 to 6 under noisy simulation, off-diagonal variance retention fell from $75.4\%$ to $60.1\%$, illustrating early concentration.

### Question 6: What error mitigation techniques did you use on the hardware pipeline?
**Answer**: We utilized Qiskit Runtime SamplerV2's built-in error mitigation:
1. **Dynamical Decoupling**: Inserting periodic `XX` spin-echo pulses during idle delays to suppress $T_2$ phase decoherence.
2. **Pauli Twirling**: Twirling two-qubit gates and measurements to convert coherent systematic errors into stochastic Pauli noise.
Together, these improved Pearson correlation with the exact kernel from $0.9954$ to $0.9965$.

### Question 7: Is your Positive Semi-Definite (PSD) projection considered quantum error mitigation?
**Answer**: No. We explicitly document that PSD projection is **classical mathematical post-processing**. It takes an estimated noisy matrix and projects it onto the closest valid correlation matrix with unit diagonal via spectral eigenvalue clipping. It does not alter quantum states or reduce physical hardware noise.

### Question 8: How did you ensure a strictly fair comparison between classical and quantum models?
**Answer**: We evaluated classical models in two separate regimes. Regime A tested full features on the large dataset as an upper-bound reference. Regime B restricted classical RBF-SVM, Random Forest, and XGBoost to the **exact same 4-6 PCA features** and the **exact same 5 stratified quantum subsets** ($N_{\text{train}}=300, N_{\text{test}}=200, 10\%$ fraud) with 3-fold cross-validation tuning on training data only.

### Question 9: Which imbalance handling strategy worked best, and why?
**Answer**: **Cost-sensitive class weighting (`class_weights='balanced'`)** proved superior to both SMOTE and undersampling. It boosted fraud recall from $4\%$ to $62\%$ and achieved the highest F1-score ($0.3685$) and PR-AUC ($0.3674$). SMOTE had lower PR-AUC ($0.3291$) because linear Euclidean interpolation in input space does not respect the curved Riemannian geometry of the quantum Hilbert space.

### Question 10: Did this project demonstrate "Quantum Advantage"?
**Answer**: No. On tabular financial data, classical XGBoost on the full feature space achieves PR-AUC $\approx 0.495$, easily outperforming quantum models constrained to 4-6 features. What we demonstrated is **low-dimensional parity**: on identical 4-6 features, quantum kernel learning matches classical RBF-SVM and achieves a slight advantage ($+2.2\%$ PR-AUC) under severe class imbalance ($5\%$ fraud), providing an empirical baseline for future fault-tolerant systems.
