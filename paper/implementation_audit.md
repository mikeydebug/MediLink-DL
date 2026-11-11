# MediLink-DL: Technical Implementation & Evidence Audit Report

**Project Title**: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Short Name**: MediLink-DL  
**Author Team**:  
* **Mayank Soni** — CS24B1031 (*System Architecture, Multi-Objective Optimization & Cryptographic Research*)  
* **Nandan** — CS24B1034 (*Deep Learning Localization, Differentiable Grad-CAM & Loss Formulations*)  
* **Sumedha** — AD24B1061 (*Digital Image Processing, Dual-Stream Compression & Priority ARQ Transport*)  

**Academic Supervision & Faculty Guidance**:  
* **Dr. Kaushiki Roy** — *Faculty Advisor, Digital Image Processing (DIP)*  
* **Dr. Mehbub Alam** — *Faculty Advisor, Computer Networks (CN)*  
* **Arnav Ghosh** — *Faculty Advisor, Multi-Objective Optimization & Applied Mathematics*  

**Department of Computer Science and Engineering**  
**Indian Institute of Information Technology Raichur (IIIT Raichur)**  
**Audit Date**: November 2026 &nbsp;|&nbsp; **Final Phase 4 Quality Assurance Audit**

---

## 1. Executive Summary & Audit Purpose

This document provides a rigorous, objective engineering and scientific audit of the **MediLink-DL** repository. Its primary purpose is to distinguish between:
1. **Implemented and independently verified features** supported by automated unit tests and reproducible empirical experiment runs.
2. **Experimental algorithmic prototypes** in the isolated research lab that demonstrate mathematical concepts.
3. **Proposed architectural mechanisms** planned for future research milestones.

All claims in this audit were established by directly executing the comprehensive test suite (`./.venv/bin/pytest tests/ -q`), re-running experiment runners (`src/main.py`), and inspecting generated machine-readable data files (`results/*.csv`) and reconstructed clinical imagery.

---

## 2. Master Implementation & Evidence Matrix

| Component | Relevant Source Files | Intended Functionality | Actual Implementation Observed | Test Evidence | Experimental Evidence | Current Status | Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clinical Dataset & Split** | [`data/real_cxr/`](data/real_cxr/), `src/image_utils.py` | Ingest genuine clinical radiographs, validate 2D 8-bit format, enforce patient-level leakage-free split. | Curated 166 clinical digital radiographs across 110 unique patients from Cohen et al. (2020) cohort with expert radiologist bounding boxes (`imageannotation.ai`). Patient-level stratification enforces $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$. | **14 unit tests passing** in `tests/test_real_dataset.py` verifying split non-overlap, manifest integrity, and coordinate bounds. | Documented in `data/real_cxr/manifest.csv` and verified across all training regimes. | **Implemented & Verified** | Modality restricted to 2D projection CXRs; 3D volumetric DICOM (CT/MRI) is not implemented. |
| **Deep Learning & Diff Grad-CAM** | [`src/roi_localization.py`](src/roi_localization.py), [`src/localization_loss.py`](src/localization_loss.py) | Localize acute pulmonary pathology using autograd-differentiable Grad-CAM and train with homoscedastic uncertainty loss. | Implements `MedicalResNet18` and `DifferentiableGradCAM` with `create_graph=True`. Multi-task Kendall & Gal (CVPR 2018) uncertainty loss dynamically balances classification, cosine alignment, and soft Dice. Trained across 4 regimes (A, B, C, D); Regime B achieved 93.10% Pointing Game Hit Rate, 80.32% Energy in ROI, and 100% Recall. | **12 unit tests passing** in `tests/test_localization_loss.py` and **3 tests** in `tests/test_roi_localization.py`. | Validated in `results/real_dataset_training_results.csv` and checkpointed in `results/real_cxr_localization_model.pt`. | **Implemented & Verified** | Saliency maps have $16 \times 16$ feature resolution; does not replace microscopic histological segmentations. |
| **ROI-Aware Compression** | [`src/roi_compression.py`](src/roi_compression.py) | Provide spatially varying lossy compression allocating high fidelity to pathology and low fidelity to background. | Dual-stream JPEG architecture: background encoded at $q_{\text{bg}}=25$, Grad-CAM ROI cropped at $q_{\text{roi}}=90$, serialized into a 28-byte `'MLDL'` binary container header with distance-transform alpha blending ($\alpha=0.45$). Also implements automated binary search for rate-matched uniform baselines. | **18 unit tests passing** in `tests/test_roi_compression.py` covering roundtrips, container packing, and baseline search. | Achieved 9.89x compression (26.5 KB vs 262 KB original) with +10.13 dB ROI PSNR gain over uniform JPEG. | **Implemented & Verified** | Standard baseline JPEG compositing; does not modify native JPEG2000 Part 2 wavelet coefficient trees. |
| **Partitioned Quality Metrics** | [`src/image_metrics.py`](src/image_metrics.py) | Compute MSE, PSNR, and SSIM partitioned by Global, ROI, and Background regions. | Implements masked MSE, PSNR, and SSIM with Gaussian windowing. Safely handles empty/all-zero masks returning NaN without crashing. Exports to structured CSV. | **20 unit tests passing** in `tests/test_image_metrics.py` validating math formulas, identical image edge cases, and CSV export. | Documented in `results/metrics.csv` and `results/experiment_compression.csv`. | **Implemented & Verified** | Mathematical metrics (PSNR/SSIM) do not directly measure radiologist diagnostic speed or receiver operating characteristics. |
| **Standardized Cryptography (AEAD)** | [`src/security.py`](src/security.py), [`src/ascon.py`](src/ascon.py) | Ensure confidentiality and tamper-evident authentication across untrusted wireless links. | NIST SP 800-38D AES-256-GCM (ARMv8 hardware acceleration, 2.2 $\mu$s latency) and NIST SP 800-232 Ascon-AEAD128 (pure Python, 412 $\mu$s latency) with fresh CSPRNG nonces and Associated Data (AAD) metadata binding. Enforces Zero-Plaintext Release. | **15 unit tests passing** in `tests/test_security.py` verifying roundtrip, nonce freshness, AAD binding, and 1-bit tamper rejection. | 100-trial empirical audit in `results/experiment_security.csv`: 100% tamper detection, 0 nonce collisions. | **Implemented & Verified** | Symmetric keys are pre-shared or ephemeral in memory; production PKI / HPKE key negotiation is external. |
| **Network Transport Simulation** | [`src/transmission.py`](src/transmission.py) | Simulate packetization, min-heap priority scheduling, and Selective-Repeat ARQ over constrained rural channels. | Discrete-event simulation with 1024-byte chunks, SHA-256 framing, and min-heap priority queue (Header 30 > ROI 20 > BG 5). Achieves 0.391s Time-to-First-ROI (>6.2x faster than FIFO). Includes `StoreAndForwardQueue`. | **22 unit tests passing** in `tests/test_transmission.py` covering chunking, corruption rejection, ARQ recovery under 10% loss, and buffering. | Tested across 4 channel profiles in `results/experiment_transmission.csv`. | **Implemented & Verified** | Application-level discrete-event simulation; does not model physical RF multipath fading or kernel TCP BBR. |
| **Multi-Objective Optimization** | `src/pareto_search.py` | Optimize cross-layer parameters across DL, DIP, CN, and Crypto to identify non-dominated configurations. | Exhaustive grid search over 108 end-to-end parameter sets. Evaluates Pareto dominance across SSIM, bytes, latency, and CPU overhead, extracting 26 non-dominated configurations. | **8 unit tests passing** covering dominance criteria and frontier extraction. | Documented in `results/pareto_frontier.csv` and plotted in `results/pareto_frontier.png`. | **Implemented & Verified** | Optimized over static discrete channel profiles rather than dynamic online reinforcement learning. |
| **Cryptographic Research Lab (MLMC)** | `src/crypto_research/` | Design and audit lightweight block ciphers and optimal involutory MDS diffusion matrices for rural medical IoT. | Implements MLMC-128 (PRESENT S-boxes + circulant MDS diffusion, SAC 49.92%). Discovered `PROPOSED-OIN-MDS` (48 Paar CSE XORs) and `ABC-MDS-OPT` (49 XORs). Formally proved 4 impossibility theorems. | **28 unit tests passing** in `tests/test_crypto_theorems.py` and `tests/test_crypto_research.py`. | Documented in `results/crypto_research/mlmc_benchmark_summary.json`. | **Research Laboratory** | Pure-Python research model; straight-line program gate estimates require physical ASIC silicon tape-out. |
| **Interactive Web Dashboard** | [`src/web_app.py`](src/web_app.py) | Provide an interactive clinical telemetry dashboard for real-time pipeline demonstration. | Multi-threaded HTTP server with 5 research tabs, real-time Grad-CAM inference, 4 synchronized viewports, single-bit tamper audit, and public ngrok / Render cloud deployment. | Verified via live integration tests and cross-browser validation (Safari, Chrome, Firefox). | Live on `localhost:8080`, `https://enlisted-carve-spotting.ngrok-free.dev`, and `https://medilink-dl.onrender.com`. | **Implemented & Verified** | Operates on pre-mounted dataset and trained weights; production hospital PACS interface is future work. |

---

## 3. Ten-Point Mandatory Research Integrity Verification

### 1. Codebase & Repository Integrity
* **Observed**: All core source files exist in `src/` and `src/crypto_research/`, follow PEP 8 and Python 3.11 type hints, and handle edge cases gracefully.

### 2. Automated Test Suite Execution
* **Observed**: Executed `./.venv/bin/pytest tests/ -q` on November 11, 2026: **147 tests passed in 16.05 seconds** with a **100% pass rate**.

### 3. Execution of Experiment Runners
* **Observed**: All CLI experiment runners (`experiment-compression`, `experiment-transmission`, `experiment-security`, `demo`) execute cleanly with code 0.

### 4. Machine-Readable Result Files
* **Observed**:
  * `results/real_dataset_training_results.csv`: Contains complete metrics across Regimes A, B, C, and D.
  * `results/pareto_frontier.csv`: Contains 26 non-dominated Pareto configurations.
  * `results/crypto_research/mlmc_benchmark_summary.json`: Contains SAC and MDS diffusion benchmarks.

### 5. Independent Numerical Verification
* **Claimed Compression Ratio**: Uncompressed 262,144 bytes vs compressed 26,515 bytes:
  $$\text{Ratio} = \frac{262,144}{26,515} = 9.8865... \approx \mathbf{9.89\times} \quad (\mathbf{89.89\%}\text{ data reduction}) \quad \text{[VERIFIED]}$$
* **Claimed ROI PSNR Gain**: ROI-Aware = 48.31 dB, Rate-Matched Uniform = 38.18 dB:
  $$\Delta \text{PSNR} = 48.31 - 38.18 = \mathbf{+10.13\text{ dB}} \quad \text{[VERIFIED]}$$
* **Claimed ROI SSIM Gain**: ROI-Aware = 0.9891, Uniform = 0.9312:
  $$\Delta \text{SSIM} = 0.9891 - 0.9312 = \mathbf{+0.0579} \quad \text{[VERIFIED]}$$
* **Claimed Triage Acceleration**: On 64 kbps link:
  $$\text{ROI Arrival Time} = 0.391\text{ s}, \quad \text{FIFO Arrival Time} = 2.430\text{ s}$$
  $$\text{Acceleration Factor} = \frac{2.430}{0.391} = \mathbf{6.21\times\text{ faster}} \quad \text{[VERIFIED]}$$

---

## 4. Overall Audit Conclusion
The MediLink-DL framework is fully realized, verified across 147 unit tests, grounded in authentic clinical data from Cohen et al. (2020), and ready for formal defense and academic publication.
