# Limitations and Scientific Disclosure

**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Authors**: Mayank Soni (CS24B1031), Nandan (CS24B1034), Sumedha (AD24B1061)  
**Affiliation**: Indian Institute of Information Technology, Raichur (IIIT Raichur)  

---

## 1. Data Integrity and Non-Clinical Disclaimer
The empirical experiments in this repository utilize **mathematically synthesized chest radiograph phantoms** rather than patient-derived clinical radiographs. While the phantom incorporates anatomical landmarks (rib cage attenuation, bilateral radiolucent lung fields, mediastinal cardiac shadows, vascular arborization) and simulated focal consolidations with known geometric ground truth:
- **No Diagnostic Certification**: The synthetic images do not possess the biological variability, micro-texture, or clinical subtlety of real human pulmonary pathology.
- **Watermark Enforcement**: All generated synthetic phantoms are permanently stamped with a disclaimer watermark along the lower margin to prevent accidental clinical misuse.
- **Clinical Translation Path**: Formal diagnostic efficacy can only be established through prospective clinical trials on authentic 12-bit/16-bit DICOM radiographs from certified clinical archives (e.g., NIH ChestX-ray14, MIMIC-CXR, CheXpert) evaluated by board-certified radiologists.

---

## 2. Deep Learning Architecture and Saliency Scope
The convolutional model implemented in `src/roi_localization.py` (`SimpleCXRClassifier`) is a lightweight, 3-block convolutional neural network designed specifically for CPU-compatible testing of higher-order gradient backpropagation:
- **Proof-of-Concept Status**: The demonstration model demonstrates that differentiable localization loss ($L_{\text{dice}}$) directly penalizes off-target Grad-CAM saliency and guides attention toward focal lesions. It must not be claimed as an alternative to clinical diagnostic architectures (e.g., CheXNet / DenseNet-121).
- **Grad-CAM Theoretical Bounds**: Grad-CAM calculates coarse, class-discriminative feature importance derived from the spatial resolution of the final convolutional layer. It generates continuous saliency heatmaps, not histological ground-truth lesion boundaries.
- **Threshold Sensitivity**: The binarization of continuous heatmaps into discrete ROI masks is governed by empirical thresholds (e.g., $\tau \in [0.25, 0.50]$). In clinical practice, overly aggressive thresholding risks truncating ill-defined infiltrative margins, whereas overly relaxed thresholds diminish compression savings.

---

## 3. Cryptographic Implementation and Hardware Realities
Our evaluation benchmarks standard AES-GCM (NIST SP 800-38D) against the newly standardized Ascon-AEAD128 (NIST SP 800-232 Final):
- **Implementation Divergence**: The AES-GCM baseline utilizes the `cryptography` Python package backed by OpenSSL C-bindings, which leverage native CPU hardware instructions (Intel AES-NI and ARMv8 Cryptographic Extensions).
- **Software Ascon Performance**: The Ascon-AEAD128 engine is implemented in pure, constant-time Python without C extensions or hardware-specific instruction support. Consequently, its software latency on general-purpose CPUs (~1.3 ms per 1024-byte chunk) is substantially higher than hardware-accelerated AES (~2.3 $\mu$s).
- **Target Deployment Context**: Ascon's standardized advantages—low silicon area (~2,500 gate equivalents), low power consumption, and side-channel resistance—are realized on ultra-constrained 8-bit/16-bit microcontrollers and custom ASIC/FPGA edge nodes in rural clinics, rather than high-performance 64-bit desktop processors.

---

## 4. Network Channel Modeling Scope
The transmission module (`src/transmission.py`) models application-layer network dynamics:
- **Discrete-Event Simulation**: The channel simulates queuing delays, propagation latency, bandwidth throttling, stochastic packet loss, and Stop-and-Wait / ARQ retransmissions.
- **Unmodeled Lower Layers**: The simulation does not model physical-layer radio frequency (RF) phenomena (e.g., multipath Rayleigh fading, Doppler shift, terrestrial microwave antenna alignment) or full kernel-level TCP congestion state machines (e.g., BBR, CUBIC window collapse).
- **Controlled Scope**: While representative of rural cellular (2G/3G) and geostationary VSAT links, real-world telemedicine deployment requires field validation across live wireless mesh topologies and satellite terminals.

---

## 5. Compression Codec Baseline
- **Dual-Stream JPEG vs. JPEG 2000 Wavelets**: The compression framework implements an operational dual-stream JPEG baseline (combining an aggressive context stream with a high-fidelity diagnostic patch in a custom `MLDL` binary container). This provides universal compatibility with existing picture archiving and communication systems (PACS) and web browsers without requiring specialized JPEG 2000 Part 2 MAXSHIFT wavelet decoders.
- **Boundary Discontinuity**: While the high-quality ROI patch is positioned over the background at pixel-perfect coordinates, slight block boundary differences may be visible along the bounding box perimeter at very aggressive background quality settings ($q_{\text{bg}} < 15$).
