# MediLink-DL: Reproducibility & Execution Guide

**Document**: Step-by-Step Instructions for Environment Setup, Automated Verification, Benchmarking, and Paper Generation  
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
**Date**: November 2026 &nbsp;|&nbsp; **Reproducibility & Execution Protocol**

---

## 1. System Requirements & Environment Setup

### 1.1 Hardware & Operating System
- **Supported OS**: macOS (Apple Silicon / Intel), Linux (Ubuntu 20.04+, Debian, Fedora, Arch), Windows 10/11 (WSL2 recommended).
- **RAM**: Minimum 8 GB (16 GB recommended for PyTorch training).
- **Disk Space**: ~2 GB for virtual environment, models, and clinical datasets.

### 1.2 Virtual Environment Preparation
Ensure Python 3.11+ is installed. MediLink-DL includes a dedicated virtual environment in `.venv`.

To activate the environment:
```bash
# On macOS / Linux:
source .venv/bin/activate

# Alternatively, directly invoke the venv binary:
./.venv/bin/python3 --version
```

If initializing a fresh environment from scratch:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Automated Test Suite Execution

The repository contains **147 unit, property, and mathematical audit tests** across 12 test files.

### Run All Tests:
```bash
./.venv/bin/pytest tests/ -q
```
Expected output:
```
147 passed in ~16.0s (100% pass rate)
```

### Run Specialized Test Subsets:
```bash
# 1. Clinical Dataset Stratification & Patient-Leakage Zero Audit:
./.venv/bin/pytest tests/test_real_dataset.py -v

# 2. Differentiable Grad-CAM & Kendall Uncertainty Loss:
./.venv/bin/pytest tests/test_localization_loss.py -v

# 3. Dual-Stream ROI Compression & Container:
./.venv/bin/pytest tests/test_roi_compression.py -v

# 4. Authenticated Ciphers (AES-256-GCM / Ascon-AEAD128) & Tamper Rejection:
./.venv/bin/pytest tests/test_security.py -v

# 5. Non-Hadamard Diffusion & Mathematical Impossibility Theorems:
./.venv/bin/pytest tests/test_crypto_theorems.py -v
```

---

## 3. End-to-End Pipeline Execution

The master CLI `src/main.py` provides 12 distinct execution modes:

### 3.1 Clinical CXR Pipeline Demo (Real COVID-19 Radiograph + Grad-CAM)
Runs deep learning inference on a real radiograph, extracts diagnostic thoracic ROI via Grad-CAM, executes dual-stream compression, encrypts with AES-256-GCM, simulates prioritized transport, and exports comparative visualizations:
```bash
./.venv/bin/python3 src/main.py --mode demo
```
- **Outputs generated**:
  - `results/comparison.png` (5-panel qualitative visualization)
  - `results/metrics.csv` (partitioned PSNR, SSIM, and compression statistics)

### 3.2 Rate-Distortion & Quality Sweep
Sweeps background quality factors $q_{\text{bg}} \in [15, 75]$ and compares against matched uniform baselines:
```bash
./.venv/bin/python3 src/main.py --mode experiment-compression
```
- **Outputs generated**: `results/experiment_compression.csv`, `results/experiment_compression.png`

### 3.3 Telemedicine Network Simulation
Simulates packet transmission across 2G, 3G, VSAT, and lossy wireless channels under Priority vs. FIFO scheduling:
```bash
./.venv/bin/python3 src/main.py --mode experiment-transmission
```

### 3.4 Authenticated Security & Tamper Mutation Audit
Executes 100 random bit-flip mutation trials:
```bash
./.venv/bin/python3 src/main.py --mode experiment-security
```

---

## 4. Cryptographic Research Laboratory Execution

### 4.1 Diffusion Layer Matrix Benchmarks
Benchmarks candidate diffusion families over $\mathbb{F}_{2^4}$:
```bash
./.venv/bin/python3 -c "from src.crypto_research.diffusion.non_hadamard import verify_abc_mds_opt; print('ABC-MDS-OPT:', verify_abc_mds_opt())"
```

### 4.2 MLMC-128 Benchmark Suite
Executes SPN round transformations, SAC evaluation, and throughput benchmarks:
```bash
./.venv/bin/python3 src/crypto_research/benchmark_mlmc.py
```

---

## 5. Interactive Telemedicine Web Dashboard

Launch the multi-threaded browser-based dashboard on port 8080:
```bash
./.venv/bin/python3 src/web_app.py 8080
```
- **Local Dashboard**: [http://localhost:8080](http://localhost:8080)
- **Live Ngrok Public Tunnel**: [https://enlisted-carve-spotting.ngrok-free.dev](https://enlisted-carve-spotting.ngrok-free.dev)
- **Cloud Deployment**: [https://medilink-dl.onrender.com](https://medilink-dl.onrender.com)

Interactive Tabs:
- **Tab 1: Live Telemedicine Pipeline**: Real-time `MedicalResNet18` Grad-CAM inference on 166 clinical CXRs, dual-stream JPEG compression, and priority transport telemetry.
- **Tab 2: Pareto Optimization & Regimes**: 26 non-dominated Pareto configurations.
- **Tab 3: NIST SP 800-232 Cryptographic Lab**: Single-bit tamper audit with Zero-Plaintext Release.
- **Tab 4: Research Novelty & Viva Defense**: Speaker roles and defense guides.
- **Tab 5: Cryptographic Research Lab**: MLMC-128 SPN round transformations and diffusion matrices.

---

## 6. Research Paper & PDF Compilation

Compile the publication-ready PDF research paper and technical report:
```bash
# 1. Compile IEEE-style 12-section research paper:
./.venv/bin/python3 paper/generate_pdf.py

# 2. Compile DIP Technical Report:
./.venv/bin/python3 docs/generate_dl_dip_pdf.py

# 3. Compile Progress Presentation Deck:
./.venv/bin/python3 paper/generate_pptx.py
```
Outputs generated:
- `paper/MediLink_DL_Research_Paper.pdf`
- `docs/MediLink_DL_DIP_Technical_Report.pdf`
- `paper/MediLink_DL_Progress_Presentation.pptx`
