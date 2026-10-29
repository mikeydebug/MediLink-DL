# MediLink-DL

**A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine**

---

## 1. Project Overview

**MediLink-DL** is an academic research prototype designed to address the critical challenges of transmitting medical chest X-ray images over constrained, lossy, and bandwidth-limited communication channels in rural and remote telemedicine clinics.

The framework provides an end-to-end integrated cross-layer pipeline:
1. **Clinical Medical Dataset & Leakage-Free Stratification**: Built on the open-access COVID-19 Chest X-ray Dataset (Cohen et al., 2020) with verified COCO-format lung bounding-box annotations from `imageannotation.ai`. Enforces strict **patient-level stratification** (zero patient overlap between train, val, and test splits).
2. **Deep Learning Diagnostic Localization**:
   - ResNet-18 classifier trained on genuine clinical chest radiographs.
   - Differentiable Grad-CAM formulation enabling end-to-end backpropagation through Grad-CAM saliency maps.
   - Comprehensive 4-regime comparison: Classification-Only baseline vs. Composite, Alignment-Only, and Soft Continuous Dice localization losses.
3. **Bandwidth-Adaptive ROI-Aware Compression**:
   - Preserves high diagnostic fidelity ($q_{\text{roi}} \in [80, 95]$) in pathological lung zones while aggressively quantizing non-critical background structures ($q_{\text{bg}} \in [15, 25]$).
   - Achieves ~10x compression ratios and 90% bandwidth reduction while preserving ROI PSNR $> 48$ dB and SSIM $> 0.98$.
4. **End-to-End Standardized Authenticated Encryption (AEAD)**:
   - Hardware-accelerated NIST SP 800-38D AES-256-GCM and AES-128-GCM (GHASH MAC over $GF(2^{128})$).
   - Constant-time reference implementation of NIST SP 800-232 Ascon-AEAD128 lightweight cipher.
   - Fresh nonces per chunk, strict tag authentication, and bit-level tamper rejection.
5. **Computer Networks Transmission Simulation**:
   - Stop-and-Wait ARQ simulation over lossy links with priority min-heap scheduling for ROI packets vs FIFO baseline.
   - Delivers triage-ready diagnostic regions in $< 0.4$ seconds over rural 64 kbps links.
6. **Cross-Layer Multi-Objective Optimization**:
   - Evaluates 108 end-to-end configurations across DL, DIP, CN, and Security layers to extract the empirical non-dominated Pareto frontier.

> **Academic & Clinical Notice**: This software is an engineering research prototype developed for academic evaluation. It has not been approved for clinical diagnostic care or patient management.

---

## 2. Directory Structure

```text
MediLink-DL/
├── data/
│   ├── README.md               # Dataset documentation and patient split rules
│   ├── samples/                # Sample diagnostic radiographs for testing
│   └── real_cxr/               # Genuine chest X-rays (images excluded from git)
│       ├── annotations.json    # Original COCO lung bounding box annotations
│       ├── manifest.csv        # Validated image metadata, patient IDs, and splits
│       ├── splits.json         # Deterministic train/val/test patient partitions
│       └── images/             # Radiograph JPEG/PNG files (~25 MB total)
├── src/
│   ├── dataset_real.py         # Real dataset download, validation, & PyTorch loaders
│   ├── train_real_models.py    # ResNet-18 training & evaluation across 4 regimes
│   ├── localization_loss.py    # Differentiable Grad-CAM & continuous Dice/Alignment loss
│   ├── train_localization.py   # Synthetic model training & Grad-CAM ablation pipeline
│   ├── roi_localization.py     # Inference Grad-CAM & dynamic checkpoint loader
│   ├── roi_compression.py      # Dual-stream ROI-aware JPEG compression
│   ├── image_metrics.py       # Partitioned PSNR/SSIM calculation (ROI vs background)
│   ├── security.py             # Unified AEADCipher suite (AES-GCM & Ascon-AEAD128)
│   ├── ascon.py                # NIST SP 800-232 Ascon-AEAD128 implementation
│   ├── benchmark_crypto.py     # Cryptographic microbenchmark suite
│   ├── transmission.py         # Network simulation, priority scheduling & reassembly
│   ├── optimization.py         # Cross-layer multi-objective Pareto optimization
│   └── main.py                 # Unified CLI entry point
├── tests/                      # 74 automated unit and integration tests (100% passing)
├── results/                    # Experimental CSVs, Pareto plots, and trained models
└── paper/                      # Academic research paper, presentation slides, & methodology
```

---

## 3. Installation & Setup (macOS / Linux)

### Step 1: Open Workspace & Activate Virtual Environment
```bash
cd /Users/mayanksoni/Desktop/MediLink-DL
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. CLI Execution & Reproducibility Guide

The unified CLI (`src/main.py`) provides commands for all research modules:

### 4.1. Dataset Preparation & Validation
```bash
# Download and prepare the real COVID-19 Chest X-ray dataset
python3 src/main.py --mode prepare-dataset

# Validate patient partition isolation (guarantees zero patient leakage)
python3 src/main.py --mode validate-dataset
```

### 4.2. Deep Learning Model Training
```bash
# Train ResNet-18 on real chest X-rays across 4 regimes (Apple Silicon MPS / CUDA)
python3 src/main.py --mode train-real-models

# Train synthetic demonstration models
python3 src/main.py --mode train-localization
```

### 4.3. End-to-End Telemedicine Pipeline Demonstration
```bash
# Run full pipeline with trained Grad-CAM and AES-256-GCM on a real radiograph
python3 src/main.py --mode demo --roi-method gradcam --crypto AES-256-GCM

# Run full pipeline with Ascon-AEAD128 lightweight authenticated cipher
python3 src/main.py --mode demo --roi-method gradcam --crypto ASCON-128

# Run on a custom medical image
python3 src/main.py --mode demo --image data/real_cxr/images/<sample>.jpeg --roi-method gradcam
```

### 4.4. Experimental Benchmarks & Cross-Layer Optimization
```bash
# Microbenchmark AEAD ciphers across chunk sizes (256 B to 4096 B)
python3 src/main.py --mode benchmark-crypto

# Run cross-layer multi-objective optimization (108 configurations -> Pareto frontier)
python3 src/main.py --mode optimize

# Run compression rate-distortion experiments
python3 src/main.py --mode experiment-compression

# Run transmission loss and recovery experiments
python3 src/main.py --mode experiment-transmission

# Run cryptographic tamper detection experiments
python3 src/main.py --mode experiment-security
```

---

## 5. Automated Testing

Run the full automated test suite (74 tests covering real data loading, autograd gradient flow, compression, AEAD ciphers, transmission reassembly, and optimization):

```bash
pytest tests/ -v
```

All 74 tests execute deterministically and pass in $< 2.0$ seconds.

---

## 6. Key Measured Research Findings

| Experimental Regime | Test Accuracy | Test F1 | Test ROI Dice | Pointing Game Hit Rate | Energy in Anatomical ROI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime A: Classification-Only** | 0.6897 | 0.8163 | 0.4764 | 65.5% | 49.2% |
| **Regime B: Combined Loss** | 0.6897 | 0.8163 | 0.6856 | **93.1%** | 80.3% |
| **Regime C: Alignment-Only** | 0.6897 | 0.8163 | 0.5799 | **96.5%** | **82.2%** |
| **Regime D: Dice-Only (Proposed)** | **0.7586** | **0.8511** | **0.7107** | 82.8% | 69.4% |

**Cross-Layer Highlights**:
- **Diagnostic Triage Speedup**: Priority scheduling delivers the diagnostic ROI in **0.328s** vs **1.046s** under FIFO scheduling (>2.5x triage latency reduction).
- **Compression Efficiency**: 9.89x payload compression with ROI PSNR of 48.31 dB and ROI SSIM of 0.9891.
- **Security Overhead**: AES-256-GCM processes 26 KB chunks in $< 0.2$ ms with standard 28-byte overhead, while rejecting 100% of single-bit tampering attempts.
