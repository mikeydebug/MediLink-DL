# MediLink-DL: Consolidated Experimental Results & Performance Benchmarks

**Document**: Master Experimental Results, Numerical Data Tables, and Comparative Analysis  
**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
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
**Date**: November 2026  

---

## 1. Deep Learning Localization on Real Clinical CXRs

Evaluated on the held-out test cohort of 29 clinical radiographs across 17 unique patients (zero patient leakage) from the COVID-19 Chest X-ray Dataset Initiative (Cohen et al., 2020).

### Table 1: Four-Regime Training Comparison (`results/real_dataset_training_results.csv`)

| Training Regime / Objective | Test Acc | Test F1 | Test Recall | Continuous ROI Dice | Binarized IoU ($\tau=0.35$) | Pointing Game Hit Rate | Energy in ROI | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Regime A: Cls-Only ($\mathcal{L}_{\text{cls}}$)** | 0.6897 | 0.8163 | 1.0000 | 0.4764 | 0.3571 | 65.52% | 49.21% | Saliency Drift |
| **Regime B: Combined (Proposed)** | **0.6897** | **0.8163** | **1.0000** | **0.6856** | **0.5395** | **93.10%** | **80.32%** | **Operational Champion** |
| **Regime C: Align-Only ($\mathcal{L}_{\text{align}}$)** | 0.6897 | 0.8163 | 1.0000 | 0.5799 | 0.4243 | **96.55%** | **82.22%** | Loose Boundaries |
| **Regime D: Dice-Only ($\mathcal{L}_{\text{dice}}$)** | **0.7586** | **0.8511** | 1.0000 | **0.7107** | **0.5704** | 82.76% | 69.36% | Diffuse Attention |

> **Key Architectural Finding**: While Regime D achieves higher continuous Dice (0.7107), its energy inside the ROI drops to 69.36% and Pointing Game accuracy drops to 82.76%. **Regime B was chosen as the operational champion** because it achieves **93.10% Pointing Game Hit Rate** and **80.32% Energy Confinement**, ensuring gradient attention is locked directly onto pulmonary consolidations while maintaining **100% Sensitivity (Zero False Negatives)** on acute viral lesions.

---

## 2. Rate-Distortion & Compression Performance on Real CXRs

### Table 2: Dual-Stream vs. Matched Uniform JPEG on Real COVID-19 Radiograph

- Input Image: Clinical chest radiograph `results/real_cxr_gradcam_comparison.png` (512 $\times$ 512, 262,144 bytes uncompressed).
- Settings: $q_{\text{roi}}=90, q_{\text{bg}}=25$ vs. Rate-Matched Uniform $q=32$.

| Coding Scheme | Settings | Transmitted Bytes | Compression Ratio | ROI PSNR | ROI SSIM | Global PSNR | Global SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Stream ROI-Aware** | $q_{\text{roi}}=90, q_{\text{bg}}=25$ | **26,515 B** | **9.89x (89.89% drop)** | **48.31 dB** | **0.9891** | 42.10 dB | 0.9634 |
| **Rate-Matched Uniform JPEG**| $q=32$ (Matched Bytes) | 26,035 B | 10.07x | 38.18 dB | 0.9312 | 39.45 dB | 0.9410 |
| **High-Quality Uniform JPEG** | $q=90$ (High Quality) | 98,412 B | 2.66x | 48.31 dB | 0.9897 | 49.11 dB | 0.9882 |

> **Diagnostic Quality Gain**: At equal transmission budgets (~26.5 KB), MediLink-DL delivers a **+10.13 dB gain in ROI PSNR** and **+0.0579 in ROI SSIM** over uniform JPEG.

### Table 3: Multi-Rate Background Sensitivity Sweep (`results/experiment_compression.csv`)

| $q_{\text{bg}}$ | $q_{\text{roi}}$ | Uniform $q^*$ | Dual-Stream (KB) | Uniform (KB) | ROI PSNR (dB) | Uniform ROI PSNR | ROI SSIM | Uniform ROI SSIM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 15 | 90 | 22 | 7.29 | 7.26 | **38.08** | 33.86 | **0.9275** | 0.8237 |
| 25 | 90 | 32 | 9.61 | 9.40 | **38.08** | 34.58 | **0.9293** | 0.8440 |
| 35 | 90 | 41 | 11.93 | 12.17 | **38.08** | 34.82 | **0.9311** | 0.8516 |
| 45 | 90 | 49 | 14.92 | 14.95 | **38.08** | 35.08 | **0.9317** | 0.8600 |
| 60 | 90 | 64 | 20.25 | 20.40 | **38.08** | 35.53 | **0.9324** | 0.8744 |
| 75 | 90 | 76 | 29.58 | 28.79 | **38.08** | 36.00 | **0.9339** | 0.8887 |

---

## 3. Prioritized Telemedicine Transport & Latency Benchmarks

Evaluated over discrete-event channel models with Selective-Repeat ARQ retransmission (`results/experiment_transmission.csv`).

### Table 4: Latency Comparison Across Rural Channel Profiles

| Channel Profile | Bandwidth | Latency (RTT) | Packet Loss | Time-to-First-ROI (Priority) | Time-to-First-ROI (FIFO) | Triage Speedup | Total Delivery Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rural 2G/EDGE** | 48 kbps | 200 ms | 2.0% | **0.592 s** | 3.592 s | **6.07x** | 3.916 s |
| **Rural 2.5G/GPRS** | 64 kbps | 120 ms | 5.0% | **0.391 s** | 2.430 s | **6.21x** | 2.608 s |
| **Rural 3G (Weak)** | 128 kbps | 150 ms | 5.0% | **0.372 s** | 2.097 s | **5.63x** | 2.294 s |
| **Satellite VSAT** | 256 kbps | 500 ms | 2.0% | **1.036 s** | 5.299 s | **5.11x** | 5.822 s |
| **Lossy Wireless** | 64 kbps | 120 ms | 10.0% | **0.384 s** | 2.887 s | **7.51x** | 3.100 s |

> **Key Finding**: Priority scheduling consistently accelerates acute triage availability by **5.11x to 7.51x** over FIFO transport, delivering diagnostic lesion packets in under 0.6 seconds even on severe 48 kbps 2G connections.

---

## 4. Cryptographic Security, Latency & Tamper Audit

Evaluated across 100 independent random encryption runs and single-bit ciphertext mutations (`results/experiment_security.csv`).

### Table 5: AEAD Security and Throughput Benchmarks

| Metric | AES-256-GCM | AES-128-GCM | Ascon-AEAD128 |
| :--- | :---: | :---: | :---: |
| **Key Length** | 256 bits | 128 bits | 128 bits |
| **Nonce Length** | 96 bits | 96 bits | 128 bits |
| **Tag Length** | 128 bits | 128 bits | 128 bits |
| **Standard Specification** | NIST SP 800-38D | NIST SP 800-38D | NIST SP 800-232 |
| **Tamper Rejection Rate** | **100/100 (100.0%)** | **100/100 (100.0%)** | **100/100 (100.0%)** |
| **Mean Encryption Latency** | **0.0022 ms (2.2 $\mu$s)**| **0.0020 ms** | 0.4120 ms (412 $\mu$s) |
| **Mean Decryption Latency** | **0.0024 ms (2.4 $\mu$s)**| **0.0021 ms** | 0.4180 ms (418 $\mu$s) |
| **Framing Wire Overhead** | 28 bytes | 28 bytes | 32 bytes |
| **Nonce Collision Rate** | 0.0% (0/10,000) | 0.0% (0/10,000) | 0.0% (0/10,000) |
| **Zero-Plaintext Release**| **Enforced** | **Enforced** | **Enforced** |

---

## 5. Cryptographic Diffusion Architecture Benchmarks

All properties independently audited and verified across all $2^{16}-1 = 65,535$ non-zero vectors and all 69 square submatrices.

### Table 6: Master Diffusion Layer Comparison

| Diffusion Layer | Matrix Family | Finite Field | Involutory? ($M^2=I$) | $\mathcal{B}_{\text{nibble}}$ | $\mathcal{B}_{\text{bit}}$ | Direct XORs | Paar CSE XORs | Analytical Area (GE)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PROPOSED-OIN-MDS`**| Hadamard | $\mathbb{F}_{2^4} / (x^4+x+1)$ | **Yes** | **5 (MDS)** | **8 (Maximal)** | 64 | **48** | ~96.0 GE |
| **`ABC-MDS-OPT`** | Alternating Block-Circulant | $\mathbb{F}_{2^4} / (x^4+x+1)$ | **Yes** | **5 (MDS)** | 6 | 64 | **49** | ~98.0 GE |
| **`ABC-MDS-MAXBIT`** | Alternating Block-Circulant | $\mathbb{F}_{2^4} / (x^4+x+1)$ | **Yes** | **5 (MDS)** | **7** | 94 | 58 | ~116.0 GE |
| **`TBOI-16`** | Binary Orthogonal Involution| $\mathbb{F}_2$ | **Yes** | 4 (Near-MDS) | 6 | 128 | 60 | ~120.0 GE |

---

## 6. Multi-Objective Optimization: 26 Non-Dominated Pareto Configurations

Evaluated across 108 end-to-end grid search configurations (`results/pareto_frontier.csv`), the multi-objective optimization extracted **26 non-dominated Pareto solutions** establishing the global performance boundary between diagnostic SSIM, wire payload bytes, triage latency, and cryptographic CPU cost.
