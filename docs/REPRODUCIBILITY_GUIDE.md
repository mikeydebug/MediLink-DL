# MediLink-DL: Reproducibility & Execution Guide

**Document**: Step-by-Step Instructions for Environment Setup, Automated Verification, Benchmarking, and Paper Generation  
**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Authors**: Mayank Soni (CS24B1031), Nandan (CS24B1034), Sumedha (AD24B1061)  
**Affiliation**: Department of Computer Science & Engineering, Indian Institute of Information Technology Raichur  
**Date**: October 31, 2026  

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

The repository contains 144 unit, property, and mathematical audit tests across 12 test files.

### Run All Tests:
```bash
./.venv/bin/pytest tests/ -v
```
Expected output:
```
============================= 144 passed in 9.37s ==============================
```

### Run Specialized Test Subsets:
```bash
# 1. Independent Zero-Trust Mathematical Audit (All 69 Minors & Branch Numbers):
./.venv/bin/pytest tests/test_phase2_1_audit.py -v

# 2. Authenticated Ciphers & Tamper Rejection:
./.venv/bin/pytest tests/test_aead_ciphers.py -v

# 3. Non-Hadamard Diffusion & Impossibility Theorems:
./.venv/bin/pytest tests/test_non_hadamard.py -v

# 4. Clinical Dataset Stratification & Patient-Leakage Prevention:
./.venv/bin/pytest tests/test_real_dataset.py -v
```

---

## 3. End-to-End Pipeline Execution

The master CLI `src/main.py` provides 12 distinct execution modes:

### 3.1 Clinical CXR Pipeline Demo (Real COVID-19 Radiograph + Grad-CAM)
Runs deep learning inference on a real radiograph, extracts diagnostic thoracic ROI via Grad-CAM, executes dual-stream compression, encrypts with AES-256-GCM, simulates prioritized transport, and exports comparative visualizations:
```bash
./.venv/bin/python3 src/main.py --mode demo --roi-method gradcam
```
- **Outputs generated**:
  - `results/comparison.png` (5-panel qualitative visualization)
  - `results/metrics.csv` (partitioned PSNR, SSIM, and compression statistics)

### 3.2 Synthetic Phantom Pipeline Demo (Standardized Calibrated Benchmark)
Runs the identical pipeline over the calibrated synthetic radiograph phantom:
```bash
./.venv/bin/python3 src/main.py --mode demo --roi-method synthetic
```

### 3.3 Rate-Distortion & Quality Sweep
Sweeps background quality factors $q_{\text{bg}} \in [15, 75]$ and compares against matched uniform baselines:
```bash
./.venv/bin/python3 src/main.py --mode experiment-compression
```
- **Outputs generated**: `results/experiment_compression.csv`, `results/experiment_compression.png`

### 3.4 Telemedicine Network Simulation
Simulates packet transmission across 2G, 3G, VSAT, and lossy wireless channels under Priority vs. FIFO scheduling:
```bash
./.venv/bin/python3 src/main.py --mode experiment-transmission
```
- **Outputs generated**: `results/experiment_transmission.csv`, `results/experiment_transmission.png`

### 3.5 Cryptographic Tamper Mutation Audit
Executes 100 random cipher mutation trials to audit tamper detection:
```bash
./.venv/bin/python3 src/main.py --mode experiment-security
```
- **Outputs generated**: `results/experiment_security.csv`

### 3.6 Multi-Objective Pareto Frontier Optimization
Sweeps 108 end-to-end system configurations across DL, DIP, CN, and Crypto to extract non-dominated operating points:
```bash
./.venv/bin/python3 src/main.py --mode pareto-sweep
```
- **Outputs generated**: `results/pareto_frontier.png`, `results/pareto_frontier.csv`

---

## 4. Cryptographic Research Laboratory CLI

### 4.1 Diffusion Layer Matrix Benchmarks
Benchmarks all 10 candidate diffusion families over $\mathbb{F}_{2^4}$:
```bash
./.venv/bin/python3 src/main.py --mode benchmark-crypto
```

### 4.2 Non-Hadamard Diffusion Search & Verification
Executes the Phase 2 Alternating Block-Circulant and Tensor search:
```bash
./.venv/bin/python3 src/main.py --mode search-non-hadamard
```

### 4.3 Phase 2.1 Formal Audit Verification
Directly runs the automated audit of `ABC-MDS-OPT`, `ABC-MDS-MAXBIT`, `TBOI-16`, and impossibility theorems:
```bash
./.venv/bin/python3 -c "
from src.crypto_research.diffusion.non_hadamard import verify_abc_mds_opt, verify_abc_mds_maxbit, verify_tboi_16, verify_impossibility_theorems;
print('ABC-MDS-OPT:', verify_abc_mds_opt());
print('ABC-MDS-MAXBIT:', verify_abc_mds_maxbit());
print('TBOI-16:', verify_tboi_16());
print('Theorems:', verify_impossibility_theorems());
"
```

---

## 5. Interactive Telemedicine Web Dashboard

Launch the browser-based dashboard on port 8080:
```bash
./.venv/bin/python3 src/web_app.py --port 8080
```
Open [http://localhost:8080](http://localhost:8080) to interact with:
- **Tab 1: Clinical CXR Pipeline**: Interactive slider controls for $\tau$, $q_{\text{roi}}$, $q_{\text{bg}}$, cipher selection, and channel bandwidth.
- **Tab 2: Rate-Distortion & Network Sweeps**: Interactive charts and raw data tables.
- **Tab 3: Cryptography Lab**: AEAD benchmarks and tamper-evidence demonstrations.
- **Tab 4: MLMC Cryptanalysis**: Differential DDT, linear LAT, and diffusion matrices.

---

## 6. Research Paper & PDF Compilation

Compile the publication-ready PDF research paper with embedded high-resolution figures:
```bash
./.venv/bin/python3 paper/generate_pdf.py
```
- Output compiled to: `paper/MediLink_DL_Research_Paper.pdf`

