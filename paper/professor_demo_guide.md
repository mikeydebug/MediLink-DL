# MediLink-DL: Professor Demonstration & Viva Defense Guide

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
**Date**: November 2026 &nbsp;|&nbsp; **Interactive Demonstration & Faculty Defense Manual**

---

## 1. Quick Environment Activation & Prerequisite Check

Open your terminal on macOS / Linux and navigate to the project directory:

```bash
cd /Users/mayanksoni/Desktop/MediLink-DL
source .venv/bin/activate
```

Verify your Python environment:
```bash
python3 --version
# Expected: Python 3.11.x
which pytest
# Expected: /Users/mayanksoni/Desktop/MediLink-DL/.venv/bin/pytest
```

---

## 2. Automated Test Suite Execution (Evidence of Code Robustness)

Before presenting the prototype, demonstrate that all core modules pass unit and integration testing:

```bash
./.venv/bin/pytest tests/ -q
```

* **Expected Output**: `147 passed in ~16.0s` (**100% pass rate**).
* **What this proves**:
  * `test_real_dataset.py` (14 tests): Proves patient-level partition isolation ($\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$), manifest integrity, and bounding box normalization across all 166 Cohen et al. CXRs.
  * `test_localization_loss.py` (12 tests): Proves second-order autograd graphs (`create_graph=True`), cosine alignment loss, soft continuous Dice loss, and Kendall & Gal homoscedastic uncertainty loss optimization.
  * `test_roi_compression.py` (18 tests): Proves dual-stream packing/unpacking, 28-byte `'MLDL'` binary container serialization, alpha blending, and uniform rate matching.
  * `test_image_metrics.py` (20 tests): Proves partitioned MSE, PSNR, and SSIM metrics over arbitrary binary masks with zero division errors.
  * `test_security.py` & `test_crypto_research.py` (25 tests): Proves AES-256-GCM, Ascon-AEAD128 (NIST SP 800-232), MLMC-128 SPN round functions, and 100% bit-flip tamper rejection with Zero-Plaintext Release.
  * `test_transmission.py` (22 tests): Proves priority min-heap chunk scheduling, Selective-Repeat ARQ under 2% to 10% packet drop, and store-and-forward buffering.
  * `test_crypto_theorems.py` (28 tests): Formally audits all 4 mathematical diffusion theorems across $\mathbb{F}_{2^4}$ and straight-line program XOR graphs.

---

## 3. Interactive Web Dashboard Demonstration (Live Presentation)

The most impactful way to demonstrate the system to the faculty panel is through the **Interactive Clinical Dashboard**:

### A. Launch Local Server & Tunnels
```bash
# Terminal 1: Launch multi-threaded web server
./.venv/bin/python3 src/web_app.py 8080

# Terminal 2 (Optional for remote panel): Launch ngrok tunnel
ngrok http 8080
```

### B. Live Telemetry Links
* **Local Web Interface**: `http://localhost:8080`
* **Live Public Ngrok Tunnel**: `https://enlisted-carve-spotting.ngrok-free.dev`
* **Cloud Deployment**: `https://medilink-dl.onrender.com`

### C. Live Interactive Workflow on Tab 1
1. **Select Patient Study**: In the dropdown, pick a genuine clinical case (e.g., `lancet-case2b.jpg (Pt 6 - COVID-19)`).
2. **Execute Pipeline**: Click **"Run Pipeline"**.
3. **Inspect the 4 Synchronized Viewports**:
   - **Viewport 1 (Input Radiograph)**: Genuine $512 \times 512$ clinical chest radiograph.
   - **Viewport 2 (Grad-CAM Heatmap)**: `MedicalResNet18` attention locked onto acute pulmonary opacities (>200,000 active pixels).
   - **Viewport 3 (Received Radiograph)**: Reconstructed radiograph transmitted in **26.5 KB** (9.89x compression).
   - **Viewport 4 (Residual Error Map)**: Amplified error map proving **zero noise inside the diagnostic lung fields**.
4. **Live Metric Telemetry**:
   - **Diagnostic ROI PSNR**: $\mathbf{48.31\text{ dB}}$ (vs 38.18 dB for rate-matched uniform JPEG, a **+10.13 dB gain**).
   - **Diagnostic ROI SSIM**: $\mathbf{0.9891}$ (vs 0.9312 for uniform).
   - **Bandwidth Reduction**: $\mathbf{89.89\%}$ data reduction.
   - **Time to First ROI**: $\mathbf{0.391\text{ s}}$ (**>6.2x faster than FIFO**).

### D. Single-Bit Cryptographic Tamper Audit (Live on Tab 3)
1. Navigate to **Tab 3 ("NIST SP 800-232 Cryptographic Lab")**.
2. Select **Ascon-AEAD128** or **AES-256-GCM**.
3. Click **"Run Tamper Audit"**.
4. Demonstrate that inverting a single bit in the 26.5 KB ciphertext triggers immediate tag rejection (`DecryptionTamperedError`) and raises Zero-Plaintext Release.

---

## 4. Execution Commands for Supported CLI Experiments

### A. End-to-End Pipeline Demonstration
```bash
./.venv/bin/python3 src/main.py --mode demo
```
* **Execution Time**: ~2 seconds.
* **Output**: Step-by-step console logs covering image ingestion, dual-stream compression, rate-matched uniform baseline finding, partitioned metrics, tamper audit, and simulated packet transmission.

### B. Rate-Distortion Parameter Sweep (Experiment 1 & 2)
```bash
./.venv/bin/python3 src/main.py --mode experiment-compression
```
* **Output**: Sweeps $q_{\text{bg}} \in [15, 75]$ at $q_{\text{roi}}=90$, solves for rate-matched uniform quality factors, and logs metrics to [`results/experiment_compression.csv`](results/experiment_compression.csv).

### C. Constrained Network Transmission Simulation (Experiment 3)
```bash
./.venv/bin/python3 src/main.py --mode experiment-transmission
```
* **Output**: Evaluates Priority vs. FIFO packet transmission across Rural 2G (48 kbps), 3G (128 kbps), and Satellite VSAT (256 kbps), logging to [`results/experiment_transmission.csv`](results/experiment_transmission.csv).

### D. Cryptographic Authenticity & Tamper Audit (Experiment 4)
```bash
./.venv/bin/python3 src/main.py --mode experiment-security
```
* **Output**: Executes 100 independent encryption, mutation, and decryption trials, logging to [`results/experiment_security.csv`](results/experiment_security.csv).

---

## 5. Output Artifacts Directory

All outputs and visual plots are saved in [`results/`](results):

| Filename | Type | Description |
| :--- | :--- | :--- |
| [`results/comparison.png`](results/comparison.png) | Image | 5-panel comparative visualization (Original, Mask, Uniform, ROI-Aware, Error Map). |
| [`results/real_cxr_gradcam_comparison.png`](results/real_cxr_gradcam_comparison.png) | Image | Real CXR Grad-CAM localization vs ground-truth radiologist masks. |
| [`results/real_cxr_training_curves.png`](results/real_cxr_training_curves.png) | Plot | Training curves across Regimes A, B, C, and D. |
| [`results/pareto_frontier.png`](results/pareto_frontier.png) | Plot | Multi-objective trade-off space across 108 configurations. |
| [`results/real_dataset_training_results.csv`](results/real_dataset_training_results.csv) | CSV | Complete performance table for all 4 regimes on the held-out test cohort. |
| [`results/pareto_frontier.csv`](results/pareto_frontier.csv) | CSV | 26 non-dominated Pareto configurations. |

---

## 6. Faculty-Specific Presentation Tips

### For Dr. Kaushiki Roy (Digital Image Processing)
* Focus on **Section 4 & 5**: Explain the dual-stream JPEG tiling, the 28-byte `'MLDL'` binary container header, and why partitioned PSNR/SSIM is clinically necessary over global metrics.
* Highlight the **+10.13 dB ROI PSNR gain** over rate-matched uniform baseline JPEG.

### For Dr. Mehbub Alam (Computer Networks)
* Focus on **Section 7**: Explain the application-layer chunking protocol (1024B chunks), SHA-256 integrity framing, and min-heap priority scheduling (Header 30 > ROI 20 > BG 5).
* Highlight the **0.391s Time-to-First-ROI** on 64 kbps rural channels (**>6.2x triage acceleration over FIFO**).

### For Arnav Ghosh (Multi-Objective Optimization)
* Focus on **Section 3.4 & 8**: Explain the Kendall & Gal CVPR 2018 homoscedastic uncertainty loss formulation ($\mathcal{L}_{\text{total}}(w, s_1, s_2)$) and the Pareto dominance criteria across the 108 evaluated configurations that yielded **26 non-dominated solutions**.
* Explain why **Regime B** was selected as the operational champion over Regime D.
