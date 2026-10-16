# MediLink-DL

**A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine**

---

## 1. Project Overview

MediLink-DL is an academic research prototype designed to address the challenges of transmitting medical chest X-ray images over constrained, lossy, and bandwidth-limited communication channels common in rural and remote telemedicine clinics.

The framework integrates:
1. **Diagnostic ROI Localization**: Identifying diagnostically critical regions (e.g., lung pathologies) via model-based Grad-CAM saliency or clinical bounding masks.
2. **Bandwidth-Adaptive ROI-Aware Compression**: Allocating high reconstruction fidelity to diagnostically sensitive regions (ROI) while aggressively compressing non-critical anatomical backgrounds to reduce overall payload size.
3. **End-to-End Authenticated Encryption**: Protecting patient confidentiality and image integrity against tampering using AES-256-GCM.
4. **Resilient Network Transmission Simulation**: Simulating application-level chunking, packet sequencing, selective retransmission, latency, and packet loss with a store-and-forward fallback mechanism.
5. **Comprehensive Quantitative Evaluation**: Measuring Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) partitioned by ROI vs. background, alongside transmission throughput and latency metrics.

> **Academic & Clinical Disclaimer**: This software is an engineering research prototype developed for experimental evaluation. Reconstructed imagery has not been cleared for clinical diagnostic use. Do not use this software for clinical triage or diagnostic decision-making without formal radiological validation.

---

## 2. System Requirements

- **Operating System**: macOS (tested on Apple Silicon arm64 / Darwin 27.x) or Linux (x86_64 / aarch64).
- **Python**: Version 3.10 or 3.11 recommended.
- **Hardware**: Compatible with standard CPU execution (GPU/MPS acceleration is optional).

---

## 3. Installation Guide (macOS)

### Step 1: Clone or Open Workspace
```bash
cd /Users/mayanksoni/Desktop/MediLink-DL
```

### Step 2: Initialize Virtual Environment
Using Python 3.11 (e.g. from Homebrew):
```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
```
*(Or with system `python3 -m venv .venv && source .venv/bin/activate`)*

### Step 3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 4. Quickstart & Demonstration

Run the automated end-to-end baseline pipeline:

```bash
python3 src/main.py --mode demo
```

This command will:
1. Generate or load a verified grayscale chest X-ray test image.
2. Define or compute an ROI mask.
3. Perform uniform compression and ROI-aware dual-quality compression at comparable bit budgets.
4. Evaluate global, ROI, and background PSNR and SSIM.
5. Encrypt the compressed payload using AES-256-GCM, demonstrate tamper detection, and decrypt the authenticated payload.
6. Export visual comparison figures to `results/comparison.png` and quantitative metrics to `results/metrics.csv`.

---

## 5. Running Experiments

Execute the experimental evaluation suite:

```bash
# Experiment 1 & 2: Rate-distortion comparison across quality levels
python3 src/main.py --mode experiment-compression

# Experiment 3: Transmission simulation under packet loss and bandwidth constraints
python3 src/main.py --mode experiment-transmission

# Experiment 4: Cryptographic overhead and tamper verification
python3 src/main.py --mode experiment-security
```

---

## 6. Running Automated Tests

Run the full pytest test suite:

```bash
pytest tests/ -v
```

---

## 7. Project Architecture

```
MediLink-DL/
├── data/                  # Sample images and synthetic generation scripts
├── paper/                 # Research paper drafts and methodology notes
│   ├── research_outline.md
│   ├── related_work.md
│   ├── methodology.md
│   └── experiment_plan.md
├── results/               # Experimental plots, metrics CSVs, reconstructed images
├── src/                   # Core Python modules
│   ├── image_utils.py     # Image I/O, format checks, synthetic CXR generator
│   ├── roi_compression.py # ROI-aware dual-quality coding & uniform baseline
│   ├── image_metrics.py   # PSNR, SSIM, ROI/background partitioned metrics
│   ├── security.py        # Authenticated encryption (AES-256-GCM)
│   ├── transmission.py    # Network simulation, packet chunking, retransmissions
│   ├── roi_localization.py# Deep learning & Grad-CAM saliency extraction
│   └── main.py            # CLI entry point and experiment runner
├── tests/                 # Automated unit and integration test suite
├── requirements.txt       # Frozen dependencies
└── README.md              # Project documentation
```
