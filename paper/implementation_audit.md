# MediLink-DL: Technical Implementation and Evidence Audit

**Project Title**: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Short Name**: MediLink-DL  
**Author Team**:  
* Mayank Soni — CS24B1031  
* Nandan — CS24B1034  
* Sumedha — AD24B1061  
**Audit Date**: October 16, 2026  
**Auditor / Mentor Evaluation**: Independent Technical Verification & Code Inspection  

---

## 1. Executive Summary & Audit Purpose

This document provides a rigorous, objective engineering and scientific audit of the **MediLink-DL** repository. Its primary purpose is to distinguish between:
1. **Implemented and independently verified features** supported by automated unit tests and reproducible empirical experiment runs.
2. **Experimental algorithmic prototypes** that demonstrate mathematical concepts but are not yet integrated into the end-to-end clinical workflow.
3. **Proposed architectural mechanisms** planned for future research milestones.

All claims in this audit were established by directly executing the test suite (`pytest tests/`), re-running experiment runners (`src/main.py`), and inspecting generated machine-readable data files (`results/*.csv`) and reconstructed imagery.

---

## 2. Core Implementation and Evidence Matrix

| Component | Relevant Source Files | Intended Functionality | Actual Implementation Observed | Test Evidence | Experimental Evidence | Current Status | Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image I/O & Validation** | [`src/image_utils.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/image_utils.py) | Ingest grayscale chest radiographs (CXRs), validate 2D 8-bit format and minimum dimensions, handle corrupted files gracefully, and supply a standardized test phantom. | Implements `load_grayscale_image`, `validate_image`, and `generate_synthetic_cxr_phantom`. Enforces 2D uint8 format. Generates an anatomically stylized $512 \times 512$ synthetic CXR phantom with rib cage, radiolucent lung cavities, heart shadow, vascularity, and focal pathology with an indelible banner: `[SYNTHETIC PHANTOM - NOT A REAL MEDICAL X-RAY]`. | **9 unit tests passing** in [`tests/test_image_utils.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_image_utils.py) covering valid/invalid shapes, corruption handling, and phantom generation. | Verified across all four experiment runs and the CLI demonstration. | **Implemented and verified** | Tested exclusively with synthetic phantom data; real DICOM parsing (12/16-bit HDR, photometric interpretation, rescale slope/intercept) is not implemented. |
| **ROI-Aware Compression** | [`src/roi_compression.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/roi_compression.py) | Provide spatially varying lossy compression that preserves high diagnostic fidelity within a Region of Interest (ROI) while aggressively compressing background anatomy. | Dual-stream JPEG approximation: background compressed at $q_{\text{bg}}$, bounding box of ROI cropped and compressed at $q_{\text{roi}}$, packaged into a custom binary stream with 26-byte `MLDL` header. Also implements `find_comparable_uniform_quality` via linear search over $q \in [1, 95]$ to match payload sizes within minimal byte delta. | **6 unit tests passing** in [`tests/test_roi_compression.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_roi_compression.py) covering JPEG roundtrips, dual-stream packing, empty masks, and matched uniform quality finding. | Evaluated across 6 rate points in [`results/experiment_compression.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_compression.csv). Achieved 26.64x compression (9.84 KB vs 262.14 KB original). | **Implemented and verified** | Standard JPEG (ITU-T T.81) does not natively support spatially varying DCT quantization matrices. This is an operational dual-stream compositing baseline, not native JPEG2000 Part 2 MAXSHIFT wavelet coefficient scaling. |
| **Partitioned Quality Metrics** | [`src/image_metrics.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/image_metrics.py) | Compute Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) partitioned by Global, ROI, and Background regions. | Implements `calculate_mse`, `calculate_psnr`, `calculate_ssim`, and `evaluate_quality_metrics`. Uses masked averaging over spatial error maps. Handles empty/all-zero masks gracefully by returning `NaN` without dividing by zero. Exports to structured CSV. | **11 unit tests passing** in [`tests/test_image_metrics.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_image_metrics.py) validating mathematical formulas, identical images (`inf` PSNR, `1.0` SSIM), shape errors, and CSV output. | Documented in [`results/metrics.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/metrics.csv) and [`results/experiment_compression.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_compression.csv). Verified: ROI PSNR = 38.08 dB vs 34.58 dB for Uniform (+3.50 dB gain). | **Implemented and verified** | Mathematical metrics (PSNR/SSIM) do not directly measure clinical diagnostic sensitivity, lesion boundary conspicuity, or radiologist reader performance. |
| **Cryptographic Security** | [`src/security.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/security.py) | Ensure confidentiality and tamper-evident authentication of medical payloads during transit over unsecure public channels. | Authenticated encryption with associated data (AEAD) via AES-256-GCM using `cryptography`. Generates a fresh 96-bit (12-byte) CSPRNG nonce per encryption. Returns `nonce + ciphertext + 16-byte tag`. `demonstrate_tamper_detection` verifies that 1-bit mutation triggers rejection. | **6 unit tests passing** in [`tests/test_security.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_security.py) verifying roundtrip, nonce freshness, associated data binding, and bit-flip rejection. | 100-trial empirical audit in [`results/experiment_security.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_security.csv): 100% tamper detection, 0 nonce collisions, 0.005 ms encryption overhead. | **Implemented and verified** | Symmetric keys are ephemeral and generated in memory. Production public-key infrastructure (PKI), Hybrid Public-Key Encryption (HPKE, RFC 9180), and mTLS channel termination are not implemented. |
| **Network Transport Simulation** | [`src/transmission.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/transmission.py) | Simulate transmission across low-bandwidth, high-latency, and lossy rural wireless links with packet sequencing, checksum verification, priority queuing, and retransmissions. | Discrete-event simulation with packet chunking (default 1024 bytes), SHA-256 checksums, and Stop-and-Wait / ARQ retransmission. Priority mode queues diagnostic ROI chunks ahead of background anatomy. Includes `StoreAndForwardQueue` for intermittent disconnections. | **6 unit tests passing** in [`tests/test_transmission.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_transmission.py) covering chunking, reassembly, corruption rejection, ARQ recovery under 10% loss, and store-and-forward. | Tested across 4 channel profiles in [`results/experiment_transmission.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_transmission.csv). Priority mode achieved 5.1x–7.5x faster diagnostic ROI delivery vs FIFO. | **Implemented and verified** | Application-level discrete-event simulation. Does not simulate physical-layer RF fading, Doppler shift, multipath interference, or Linux kernel-level TCP congestion control (BBR/Cubic). |
| **Deep Learning & Grad-CAM** | [`src/roi_localization.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/roi_localization.py) | Automatically identify pathological regions in radiographs using weakly supervised deep-learning saliency maps. | Implements `GradCAM` forward/backward gradient hooks on conv layers, Global Average Pooling of gradients, ReLU filtering, bilinear resizing, min-max normalization, thresholding into binary masks, and pseudo-color overlay rendering. Includes a 2-class CNN demonstration backbone (`SimpleCXRClassifier`). | **3 unit tests passing** in [`tests/test_roi_localization.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/tests/test_roi_localization.py) verifying hook mechanics, heatmap normalization, thresholding, and overlay rendering. | Verified as an isolated algorithmic module on synthetic inputs. Pretrained medical weights (e.g. CheXNet) are **NOT** downloaded or wired to the end-to-end CLI. | **Experimental prototype** | Standalone CNN is for algorithmic verification only; no verified clinical weights are mounted. Grad-CAM heatmaps are coarse and correlate with model activations, not clinical ground truth. |
| **Integrated CLI Demonstration** | [`src/main.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/main.py) | Provide a single command to execute the end-to-end pipeline and individual experiment runners. | Implements `--mode demo`, `--mode experiment-compression`, `--mode experiment-transmission`, and `--mode experiment-security`. Generates multi-panel figure [`results/comparison.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/comparison.png) and experiment plots. | All CLI commands execute with code 0. Validated through end-to-end integration runs. | Outputs saved in [`results/`](file:///Users/mayanksoni/Desktop/MediLink-DL/results). | **Implemented and verified** | Defaults to synthetic phantom when no user image is passed; Grad-CAM localization is not yet wired as the default mask provider in `main.py`. |

---

## 3. Mandatory Ten-Point Research Integrity Verification

### 1. Codebase Inspection
* **Observed**: All core source files (`src/image_utils.py`, `src/roi_compression.py`, `src/image_metrics.py`, `src/security.py`, `src/transmission.py`, `src/roi_localization.py`, `src/main.py`) exist, are cleanly written, follow PEP 8 and Python 3.11 type hints, and handle boundary conditions gracefully.

### 2. Automated Test Suite Execution
* **Observed**: Re-running `pytest tests/ -v` on October 16, 2026 executed **41 tests** with a **100% pass rate** in 1.47 seconds.
* **Test Breakdown**:
  * `test_image_metrics.py`: 11 tests
  * `test_image_utils.py`: 9 tests
  * `test_roi_compression.py`: 6 tests
  * `test_roi_localization.py`: 3 tests
  * `test_security.py`: 6 tests
  * `test_transmission.py`: 6 tests

### 3. Execution of Experiment Runners
* **Observed**: All three experiment subcommands (`experiment-compression`, `experiment-transmission`, `experiment-security`) and the end-to-end `demo` subcommand executed successfully without errors, generating corresponding CSV tables and matplotlib PNG figures.

### 4. Inspection of Result Files
* **Observed**:
  * [`results/metrics.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/metrics.csv): Contains baseline rate-matched comparison between ROI-Aware and Uniform coding.
  * [`results/experiment_compression.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_compression.csv): Contains 6 rate points ($q_{\text{bg}} \in [15, 75]$).
  * [`results/experiment_transmission.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_transmission.csv): Contains transmission latencies across 4 simulated rural network profiles.
  * [`results/experiment_security.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_security.csv): Contains 100-trial mutation audit metrics.

### 5. Independent Recalculation of Numerical Claims
* **Claimed Compression Ratio**: Uncompressed $512 \times 512 \times 1 = 262,144$ bytes. Compressed payload = 9,842 bytes.
  $$\text{Ratio} = \frac{262,144}{9,842} = 26.6352... \approx 26.64\times \quad \text{[VERIFIED]}$$
  $$\text{Size Reduction} = \left(1 - \frac{9,842}{262,144}\right) \times 100\% = 96.245...\% \approx 96.25\% \quad \text{[VERIFIED]}$$
* **Claimed Rate Match**: Matched uniform baseline at $q=32$ is 9,622 bytes. Delta = $-220$ bytes ($-2.24\%$) [VERIFIED].
* **Claimed ROI PSNR Gain**: ROI-Aware = 38.08 dB, Uniform = 34.58 dB. Delta = $+3.50$ dB [VERIFIED].
* **Claimed ROI SSIM Gain**: ROI-Aware = 0.9293, Uniform = 0.8440. Delta = $+0.0853$ [VERIFIED].
* **Claimed Transmission Acceleration**: On Rural 2G/EDGE (48 kbps, 200 ms RTT, 2% loss):
  $$\text{ROI Arrival Time (Priority)} = 0.592\text{ s}, \quad \text{ROI Arrival Time (FIFO)} = 3.592\text{ s}$$
  $$\text{Acceleration Factor} = \frac{3.592}{0.592} = 6.067... \approx 6.07\times \quad \text{[VERIFIED]}$$

### 6. Component Maturity Breakdown
* **Implemented and Verified**:
  * Image validation and grayscale conversion.
  * Synthetic anatomical CXR phantom generator.
  * Dual-stream JPEG differential compression with binary serialization.
  * Rate-matched uniform JPEG search.
  * Partitioned MSE, PSNR, and SSIM evaluation.
  * AES-256-GCM encryption with per-message nonces and tamper verification.
  * Discrete-event network simulation with packet chunking, checksums, ARQ, and priority queueing.
  * Store-and-forward buffer for intermittent links.
* **Experimental Prototype**:
  * Grad-CAM gradient hook extraction and thresholding on a test CNN backbone.
* **Proposed / Future Work**:
  * Direct DICOM PACS integration (12-bit / 16-bit ingestion).
  * Mounting verified CheXNet / DenseNet-121 model weights trained on NIH ChestX-ray14.
  * End-to-end wiring of Grad-CAM heatmaps directly into the live `main.py` compression pipeline.
  * Production PKI / HPKE key exchange.
  * Testing over physical rural software-defined radios or `tc/netem` Linux testbeds.

### 7. ROI Mask Source Verification
* **Finding**: The baseline demo and experiments use the **ground-truth lesion mask generated by the synthetic phantom generator** (or a rectangular geometric mask for custom images). It does **NOT** use a pretrained medical neural network mask during baseline evaluation.

### 8. Grad-CAM Connection Verification
* **Finding**: The Grad-CAM module (`src/roi_localization.py`) is implemented and unit-tested using a standalone lightweight demonstration CNN (`SimpleCXRClassifier`). It is **NOT** connected to an external medical chest X-ray model, nor is it connected to the live compression loop in `src/main.py`. This separation was intentional to allow CPU-only, offline execution without multi-gigabyte downloads.

### 9. Dataset Authenticity Verification
* **Finding**: **Zero real patient images were used.** All experimental data in `results/` was generated exclusively using the procedurally synthesized anatomical phantom (`generate_synthetic_cxr_phantom`). The phantom contains an indelible disclaimer banner stamped into the pixel grid: `[SYNTHETIC PHANTOM - NOT A REAL MEDICAL X-RAY]`.

### 10. Pipeline Integration Verification
* **Finding**: The compression, encryption, and network transmission modules are genuinely connected into an operational pipeline in `src/main.py`:
  1. The image is compressed into a serialized binary payload (`MLDL` stream).
  2. The serialized payload is encrypted via AES-256-GCM.
  3. The encrypted payload is packetized into chunked packets.
  4. The packets are transmitted through the simulated lossy channel with priority queueing.
  5. The receiver reassembles the stream, decrypts the ciphertext, unpacks the dual streams, and reconstructs the composite radiograph.

---

## 4. Summary for Academic Faculty Presentation

When presenting this work to a faculty supervisor or examination committee:
1. **Be upfront about synthetic data**: Explicitly state that all current results are evaluated on an anatomically stylized phantom with a simulated lesion to establish engineering benchmarks before seeking IRB approval for patient radiographs.
2. **Clarify the compression algorithm**: Explain that standard JPEG does not support regional quantization tables, so our prototype uses an operational dual-stream compositing baseline rather than full JPEG2000 Part 2 MAXSHIFT.
3. **Highlight the transmission finding**: The primary novelty is not merely compressing the image, but **scheduling diagnostic bytes ahead of context bytes**, allowing a remote specialist to begin triage in **0.38 to 0.59 seconds** over a 2G/satellite link rather than waiting 3.5 to 5.8 seconds for the full image.
4. **Distinguish Grad-CAM from segmentation**: Emphasize that Grad-CAM produces coarse saliency heatmaps that reflect feature activations, which require thresholding to serve as an engineering ROI.
