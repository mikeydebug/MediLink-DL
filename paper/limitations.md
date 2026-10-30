# Limitations and Scientific Disclosure

**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Authors**: Mayank Soni (CS24B1031), Nandan (CS24B1034), Sumedha (AD24B1061)  
**Affiliation**: Indian Institute of Information Technology, Raichur (IIIT Raichur)  

---

## 1. Clinical Dataset Scope and Sample Size
The empirical validation of MediLink-DL utilizes the open-access **COVID-19 Chest X-ray Dataset Initiative** (Cohen et al., 2020) with reference lung and pathology bounding boxes from `imageannotation.ai`:
- **Sample Distribution**: The active verified cohort comprises 166 clinical digital radiographs across 110 unique patients (77 training, 16 validation, and 17 test patients).
- **Patient-Level Stratification**: Strict patient-level separation was enforced to ensure zero patient leakage between training and evaluation splits ($\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset$).
- **Sample Scale Limitation**: While sufficient for reproducible academic capstone proof-of-concept and demonstrating statistically significant localization gains (+49.2% continuous Dice improvement), broader generalization across multi-center hospital cohorts requires prospective scaling to massive clinical archives (e.g., RSNA Pneumonia Detection, VinDr-CXR, CheXpert, or MIMIC-CXR) comprising tens of thousands of radiographs.
- **Non-Clinical Research Disclaimer**: MediLink-DL is an engineering research prototype. Reconstructed radiographs and Grad-CAM attention masks are not certified by the FDA, CE, or CDSCO and must not be used for autonomous patient triage or clinical diagnostic management without human radiological over-read.

---

## 2. Deep Learning Architecture and Saliency Bounds
The deep learning pipeline implements **`MedicalResNet18`** trained end-to-end with second-order autograd **`DifferentiableGradCAM`**:
- **Coarse Feature Granularity**: Grad-CAM saliency is calculated at the spatial resolution of the final convolutional block (`backbone.layer4[-1]`, $7 \times 7$ feature grid for $224 \times 224$ input tensors). Consequently, upsampled heatmaps reflect regional pulmonary attention rather than pixel-level histological ground truth.
- **Bounding Box Approximation**: Reference target regions are derived by rasterizing COCO-format $[x, y, w, h]$ bounding boxes into binary masks. While bounding boxes effectively capture the extent of pulmonary opacities and consolidations, they include marginal non-pathological lung parenchyma within the rectangular bounding envelope.
- **Threshold Dependency**: Binarization of continuous heatmaps into discrete transmission ROI masks relies on empirical cutoffs ($\tau \in [0.25, 0.45]$). An overly aggressive threshold risks truncating subtle infiltrative margins, whereas an excessively relaxed threshold expands the ROI footprint and increases wireless transmission bandwidth.

---

## 3. Cryptographic Implementation Boundaries
MediLink-DL evaluates both **NIST SP 800-38D AES-GCM** (128-bit and 256-bit) and **NIST SP 800-232 Ascon-AEAD128**:
- **Pure-Python Ascon Throughput**: Ascon-AEAD128 was developed in pure Python (`src/ascon.py`) to provide an auditable, dependency-free reference implementation compliant with NIST SP 800-232. Because it runs in the interpreted Python VM rather than utilizing dedicated hardware cryptographic instructions (such as ARMv8 Crypto Extensions used by AES-GCM via OpenSSL), its execution latency (~1.3 ms per chunk) is not representative of optimized assembly or hardware-accelerated Ascon ASICs/FPGAs.
- **Key Distribution Assumption**: Pre-shared authenticated encryption keys are assumed to be securely negotiated out-of-band or via an external public-key exchange mechanism (e.g., ECDH over Curve25519).

---

## 4. Digital Image Processing and Network Model
- **Boundary Ringing Artifacts**: The dual-stream compression pipeline encodes ROI foreground and background separately at distinct quality factors ($q_{\text{roi}}=90, q_{\text{bg}}=25$) before alpha-compositing them at the receiver. Sharp spatial transitions between high-quality and aggressively quantized regions can introduce subtle high-frequency ringing along mask borders. Future iterations should incorporate multi-scale wavelet or feathering transitions.
- **Simulated vs. Real-World Wireless Channels**: Computer network evaluations were conducted using a verified Stop-and-Wait ARQ simulation over modeled rural channels (64 kbps, 120 ms latency, 5% loss). While the simulator accurately models serialization delay, packet drop recovery, and min-heap priority scheduling, real-world cellular links (2G/GPRS/EDGE/3G) exhibit dynamic time-varying fading, multi-path Doppler shifts, and intermittent connection drops.
