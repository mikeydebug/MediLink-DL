# MediLink-DL: Comprehensive Empirical Experiment Plan

**Project Title**: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Short Name**: MediLink-DL  
**Authors**:  
* **Mayank Soni** (Roll No: CS24B1031) — *System Architecture, Multi-Objective Optimization & Cryptographic Research*  
* **Nandan** (Roll No: CS24B1034) — *Deep Learning Localization, Differentiable Grad-CAM & Loss Formulations*  
* **Sumedha** (Roll No: AD24B1061) — *Digital Image Processing, Dual-Stream Compression & Priority ARQ Transport*  

**Academic Supervision & Faculty Guidance**:  
* **Dr. Kaushiki Roy** — *Faculty Advisor, Digital Image Processing (DIP)*  
* **Dr. Mehbub Alam** — *Faculty Advisor, Computer Networks (CN)*  
* **Arnav Ghosh** — *Faculty Advisor, Multi-Objective Optimization & Applied Mathematics*  

**Department of Computer Science and Engineering**  
**Indian Institute of Information Technology Raichur (IIIT Raichur)**  
**Date**: November 2026 &nbsp;|&nbsp; **Empirical Research Experiment Protocols**

---

## 1. Overview and Objective

This experiment plan formalizes the empirical protocols, independent variables, dependent metrics, and execution steps for evaluating the complete **MediLink-DL** cross-layer framework. All experiments are fully automated, CPU/GPU reproducible, and deterministic via fixed random seeds.

Evaluations span seven comprehensive experimental studies:
1. **Experiment 1**: Rate-Distortion Comparison (Uniform JPEG vs. Dual-Stream ROI-Aware).
2. **Experiment 2**: Multi-Scale Background Compression Sensitivity Sweep.
3. **Experiment 3**: Constrained Rural Network Transport & Priority Scheduling.
4. **Experiment 4**: Cryptographic Latency, Throughput & Bit-Flip Tamper Detection.
5. **Experiment 5**: Differentiable Grad-CAM 4-Regime Training & Localization Ablation.
6. **Experiment 6**: Isolated Cryptographic Research Lab (MLMC-128, SAC & Involutory MDS).
7. **Experiment 7**: End-to-End Multi-Objective Optimization & Pareto Frontier Extraction.

---

## 2. Experiment 1: Rate-Distortion Comparison (Uniform vs. ROI-Aware)

### Objective
Determine whether allocating higher quantization fidelity to the diagnostic ROI yields superior regional PSNR and SSIM compared to a rate-matched uniform baseline on genuine human radiographs.

### Independent Variables
- **Test Image**: Genuine clinical radiograph from Cohen et al. (2020) CXR cohort ($512 \times 512$, 262,144 bytes).
- **ROI Quality Factor ($q_{\text{roi}}$)**: Fixed at 90.
- **Background Quality Factor ($q_{\text{bg}}$)**: Stepped through $\{15, 25, 35, 50\}$.
- **Matched Uniform Quality ($q^*$ )**: Dynamically solved via binary search over $[1, 100]$ to match the exact byte size of the dual-stream compressed payload within $\pm 1\%$.

### Dependent Metrics
- Transmitted payload size (bytes) & Compression Ratio.
- Regional ROI PSNR (dB) and Regional ROI SSIM.
- Regional Background PSNR (dB) and Regional Background SSIM.
- Global PSNR (dB) and Global SSIM.

### Reproducibility Command
```bash
./.venv/bin/python src/main.py --mode experiment-compression
```

---

## 3. Experiment 2: Image Quality Across Multi-Scale Compression Settings

### Objective
Quantify the sensitivity of diagnostic fidelity (ROI SSIM) to aggressive background context quantization, establishing the operational bounds for 48–128 kbps rural channels.

### Evaluation Grid
- $q_{\text{bg}} \in [10, 60]$ in steps of 5.
- $q_{\text{roi}} \in \{80, 85, 90, 95\}$.
- Test inputs: Both Cohen et al. (2020) clinical CXRs and calibrated synthetic phantoms.

---

## 4. Experiment 3: Constrained Network Transmission & Prioritization

### Objective
Evaluate the temporal triage delivery advantage of prioritized dual-stream chunk scheduling over standard non-prioritized FIFO transmission across simulated rural cellular and satellite uplinks.

### Network Channel Profiles
1. **Rural 2G/EDGE**: Bandwidth = 48 kbps, Latency (RTT) = 200 ms, Packet Loss = 2%.
2. **Rural 2.5G/GPRS**: Bandwidth = 64 kbps, Latency (RTT) = 120 ms, Packet Loss = 5%.
3. **Constrained 3G (Weak)**: Bandwidth = 128 kbps, Latency (RTT) = 150 ms, Packet Loss = 5%.
4. **Satellite VSAT**: Bandwidth = 256 kbps, Latency (RTT) = 500 ms, Packet Loss = 2%.

### Dependent Metrics
- **Time to First Diagnostic ROI ($T_{\text{ROI}}$)**: Elapsed time until all ROI packets arrive and pass SHA-256 checksum verification.
- **Total Transfer Time ($T_{\text{total}}$)**: Elapsed time until the full context frame is reassembled.
- **Triage Acceleration Factor**: $T_{\text{total}} / T_{\text{ROI}}$.
- **Retransmissions (Selective-Repeat ARQ)**: Count of dropped/corrupted packets retransmitted.

### Reproducibility Command
```bash
./.venv/bin/python src/main.py --mode experiment-transmission
```

---

## 5. Experiment 4: Cryptographic Overhead & Tamper Detection

### Objective
Measure the encryption latency, decryption latency, and wire throughput of NIST SP 800-38D AES-256-GCM and NIST SP 800-232 Ascon-AEAD128, empirically verifying that 100% of bit-flip mutations are rejected with Zero-Plaintext Release.

### Test Protocol
1. Encrypt payload with a freshly generated CSPRNG nonce and secret key.
2. Record encryption latency, decryption latency, and envelope overhead.
3. Inject single bit-flip mutations at random byte positions in the ciphertext and envelope.
4. Confirm that `DecryptionTamperedError` is raised in 100% of trials and zero plaintext is released.

### Reproducibility Command
```bash
./.venv/bin/python src/main.py --mode experiment-security
```

---

## 6. Experiment 5: Differentiable Grad-CAM 4-Regime Localization Training

### Objective
Train and evaluate `MedicalResNet18` across four distinct objective formulations to eliminate Saliency Drift and evaluate Kendall & Gal (CVPR 2018) uncertainty weighting on held-out patients:

### Four Evaluated Regimes
- **Regime A (Classification-Only)**: Standard Cross-Entropy ($\mathcal{L}_{\text{cls}}$).
- **Regime B (Combined Uncertainty Loss - Proposed Champion)**: Kendall & Gal adaptive uncertainty loss balancing classification, cosine alignment, and soft continuous Dice.
- **Regime C (Alignment-Only)**: Gradient saliency alignment loss ($\mathcal{L}_{\text{align}}$).
- **Regime D (Dice-Only)**: Continuous soft Dice overlap loss ($\mathcal{L}_{\text{dice}}$).

### Dependent Evaluation Metrics (Held-Out Test Cohort)
- Test Accuracy, F1-Score, and Sensitivity (Recall).
- Continuous Test ROI Dice and Test ROI IoU.
- **Pointing Game Hit Rate**: Fraction of test images where the maximum Grad-CAM activation pixel falls strictly inside the radiologist-annotated lesion bounding box.
- **Energy in ROI**: Percentage of total Grad-CAM activation energy confined within the ground-truth lesion mask.

### Output File
`results/real_dataset_training_results.csv` and `results/real_cxr_training_curves.png`.

---

## 7. Experiment 6: Isolated Cryptographic Research Lab (MLMC-128)

### Objective
Evaluate the cryptographic security properties of the experimental **MLMC-128** block cipher and its lightweight involutory MDS diffusion matrices:
1. **Strict Avalanche Criterion (SAC)**: Measure bit-flip propagation across 4,800 trials (ideal: 50.0%).
2. **Branch Number Evaluation**: Nibble branch number ($\mathcal{B}_{\text{nibble}}$) and bit branch number ($\mathcal{B}_{\text{bit}}$).
3. **Paar CSE Optimization**: Straight-line program reduction in XOR gates for hardware ASIC synthesis.

---

## 8. Experiment 7: Multi-Objective Grid Search & Pareto Frontier

### Objective
Execute an exhaustive grid search across 108 end-to-end pipeline parameter sets spanning DL thresholds, compression quality pairs, transport policies, and ciphers to extract the non-dominated Pareto frontier:
$$\min_{\mathbf{x} \in \mathcal{X}} \left[ - \text{SSIM}_{\text{roi}}(\mathbf{x}), \; T_{\text{roi}}(\mathbf{x}), \; T_{\text{total}}(\mathbf{x}), \; B_{\text{wire}}(\mathbf{x}), \; t_{\text{crypto}}(\mathbf{x}) \right]$$

### Output File
`results/pareto_frontier.csv` (26 non-dominated configurations) and `results/pareto_frontier.png`.

---

## 9. Master Recorded Empirical Results

### A. Deep Learning Localization (29 Test CXRs, 17 Held-Out Patients)

| Regime / Objective | Accuracy | F1 Score | Recall | ROI Dice | ROI IoU | Pointing Game | Energy in ROI | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Regime A (Cls Only)** | 0.6897 | 0.8163 | 1.0000 | 0.4764 | 0.3571 | 65.52% | 49.21% | Saliency Drift |
| **Regime B (Combined Proposed)** | **0.6897** | **0.8163** | **1.0000** | **0.6856** | **0.5395** | **93.10%** | **80.32%** | **Champion Model** |
| **Regime C (Align Only)** | 0.6897 | 0.8163 | 1.0000 | 0.5799 | 0.4243 | 96.55% | 82.22% | Loose Bounds |
| **Regime D (Dice Only)** | 0.7586 | 0.8511 | 1.0000 | 0.7107 | 0.5704 | 82.76% | 69.36% | Diffuse Attention |

### B. Dual-Stream Rate-Distortion Gain on Real CXR (512x512, 262 KB)

| Scheme | Quality Setting | Transmitted Bytes | Data Reduction | Compression Ratio | ROI PSNR | ROI SSIM | Global PSNR | Global SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Stream ROI-Aware** | $q_{\text{roi}}=90, q_{\text{bg}}=25$ | **26,515 B** | **89.89%** | **9.89x** | **48.31 dB** | **0.9891** | 42.10 dB | 0.9634 |
| **Rate-Matched Uniform** | $q=32$ (Matched Bytes) | 26,035 B | 90.07% | 10.07x | 38.18 dB | 0.9312 | 39.45 dB | 0.9410 |
| **Fidelity Gain** | Rate-Matched | — | — | — | **+10.13 dB** | **+0.0579** | +2.65 dB | +0.0224 |

### C. Prioritized Transport Performance Across Rural Links (64 kbps, 120 ms, 5% drop)
- **Time to deliver diagnostic ROI**: $\mathbf{0.391\text{ seconds}}$
- **Time to deliver full background context**: $\mathbf{2.430\text{ seconds}}$
- **Clinical Triage Acceleration**: $\mathbf{6.21\times\text{ faster}}$ visual availability for the consulting specialist.
