# MediLink-DL: Scientific Limitations & Research Disclosure

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
**Date**: November 2026 &nbsp;|&nbsp; **Scientific Disclosures & Boundary Conditions**

---

## 1. Clinical Dataset Scope and Sample Size

The empirical validation of MediLink-DL utilizes the open-access **COVID-19 Chest X-ray Dataset Initiative** (Cohen et al., 2020) with reference lung and pathology bounding boxes from `imageannotation.ai`:
- **Sample Distribution**: The active verified cohort comprises **166 clinical digital radiographs** across **110 unique patients** (77 training, 16 validation, and 17 test patients).
- **Patient-Level Stratification**: Strict patient-level separation was enforced to ensure zero patient leakage between training and evaluation splits ($\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$).
- **Sample Scale Limitation**: While sufficient for reproducible academic capstone proof-of-concept and demonstrating statistically significant localization gains (+43.9% continuous Dice improvement and 93.10% Pointing Game Hit Rate), broader generalization across multi-center hospital cohorts requires prospective scaling to massive clinical archives (e.g., RSNA Pneumonia Detection, VinDr-CXR, CheXpert, or MIMIC-CXR) comprising tens of thousands of radiographs across heterogeneous imaging hardware.
- **Non-Clinical Research Disclaimer**: MediLink-DL is an engineering research prototype. Reconstructed radiographs and Grad-CAM attention masks are not certified by statutory medical bodies (e.g., US FDA 510(k), EU CE-MDR, or Indian CDSCO) and must not be used for autonomous patient triage or clinical diagnostic management without human radiological over-read.

---

## 2. Deep Learning Architecture and Saliency Bounds

The deep learning pipeline implements **`MedicalResNet18`** trained end-to-end with second-order autograd **`DifferentiableGradCAM`**:
- **Coarse Feature Granularity**: Grad-CAM saliency is calculated at the spatial resolution of the final convolutional block (`backbone.layer4[-1]`, $16 \times 16$ feature grid for $512 \times 512$ input tensors). Consequently, upsampled heatmaps reflect regional pulmonary attention rather than pixel-level histological ground truth.
- **Bounding Box Approximation**: Reference target regions are derived by rasterizing COCO-format $[x, y, w, h]$ bounding boxes into binary masks. While bounding boxes effectively capture the extent of pulmonary opacities and consolidations, they include marginal non-pathological lung parenchyma within the rectangular bounding envelope.
- **Threshold Dependency**: Binarization of continuous heatmaps into discrete transmission ROI masks relies on empirical cutoffs ($\tau \in [0.25, 0.45]$). An overly aggressive threshold risks truncating subtle infiltrative margins, whereas an excessively relaxed threshold expands the ROI footprint and increases wireless transmission bandwidth.

---

## 3. Cryptographic Implementation Boundaries

MediLink-DL evaluates both **NIST SP 800-38D AES-GCM** (128-bit and 256-bit) and **NIST SP 800-232 Ascon-AEAD128**:
- **Pure-Python Ascon Throughput**: Ascon-AEAD128 was developed in pure Python (`src/ascon.py`) to provide an auditable, dependency-free reference implementation compliant with NIST SP 800-232. Because it runs in the interpreted Python VM rather than utilizing dedicated hardware cryptographic instructions (such as ARMv8 Crypto Extensions used by AES-GCM via OpenSSL), its execution latency (~412 $\mu$s per chunk) is not representative of optimized assembly or hardware-accelerated Ascon ASICs/FPGAs (~40 $\mu$s).
- **Key Distribution Assumption**: Pre-shared authenticated encryption keys are assumed to be securely negotiated out-of-band or via an external public-key exchange mechanism (e.g., ECDH over Curve25519).

---

## 4. Digital Image Processing and Network Model

- **Boundary Ringing Artifacts**: The dual-stream compression pipeline encodes ROI foreground and background separately at distinct quality factors ($q_{\text{roi}}=90, q_{\text{bg}}=25$) before alpha-compositing them at the receiver. Sharp spatial transitions between high-quality and aggressively quantized regions can introduce subtle high-frequency ringing along mask borders. While our 4-pixel distance-transform alpha blending ($\alpha=0.45$) substantially mitigates boundary seams, future iterations should incorporate multi-scale wavelet or feathering transitions.
- **Simulated vs. Real-World Wireless Channels**: Computer network evaluations were conducted using a verified discrete-event simulation over modeled rural channels (48–128 kbps, 120–200 ms latency, 2–10% loss). While the simulator accurately models serialization delay, packet drop recovery, Selective-Repeat ARQ, and min-heap priority scheduling, real-world cellular links (2G/GPRS/EDGE/3G) exhibit dynamic time-varying fading, multi-path Doppler shifts, and intermittent connection drops.

---

## 5. Cryptographic Research Laboratory: MLMC-128 Boundaries

In `src/crypto_research/`, we evaluated the experimental **MLMC-128** block cipher:
- **Research Status**: MLMC-128 is an academic research candidate designed to investigate the mathematical interaction between PRESENT S-boxes and optimal involutory MDS diffusion layers. It is **NOT** a finalized, cryptanalyzed industry standard and must not be deployed in production medical environments in lieu of NIST-standardized AES-256-GCM or Ascon-AEAD128.
- **Silicon Gate Estimates**: Gate Equivalent (GE) figures (e.g. ~96.0 GE for `PROPOSED-OIN-MDS`) represent analytical estimates derived from Paar Common Subexpression Elimination (CSE) straight-line programs counting 2-input XORs (~2.0 GE each). Physical ASIC silicon tape-out (e.g. TSMC 28nm) is required to account for parasitics, interconnect routing congestion, and clock tree distribution.

---

## 6. Multi-Objective Optimization Boundaries

- **Static vs Dynamic Channels**: The extracted 26-point Pareto frontier is optimized across static discrete channel profiles (48 kbps, 64 kbps, 128 kbps). Future work involves implementing online reinforcement learning or model-predictive control (MPC) to adaptively modulate $q_{\text{roi}}$ and $q_{\text{bg}}$ in real-time as instantaneous wireless signal-to-noise ratio (SNR) fluctuates.
