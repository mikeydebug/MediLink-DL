# MediLink-DL: Professor Demonstration & Viva Defense Guide

**Project Title**: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Short Name**: MediLink-DL  
**Team Members**:  
* Mayank Soni — Roll No: CS24B1031  
* Nandan — Roll No: CS24B1034  
* Sumedha — Roll No: AD24B1061  
**Target Audience**: Academic Review Committee, Faculty Supervisor, Viva Voce Examiners  

---

## 1. Quick Environment Activation & Prerequisite Check

Open your terminal on macOS and navigate to the project directory:

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
pytest tests/ -v
```

* **Expected Output**: `41 passed in ~1.2s`.
* **What this proves**:
  * 11 tests in `test_image_metrics.py`: Proves MSE, PSNR, and SSIM formulas and masked empty-region edge cases are mathematically sound.
  * 9 tests in `test_image_utils.py`: Proves validation, dimension bounds, corrupt file rejection, and phantom generation.
  * 6 tests in `test_roi_compression.py`: Proves dual-stream packing/unpacking and uniform rate-matching.
  * 6 tests in `test_security.py`: Proves AES-256-GCM authenticated encryption, fresh nonces, and 1-bit tamper detection.
  * 6 tests in `test_transmission.py`: Proves packet chunking, SHA-256 checksums, ARQ packet recovery under 10% loss, and store-and-forward buffering.
  * 3 tests in `test_roi_localization.py`: Proves Grad-CAM hooks, heatmap normalization, and thresholding.

---

## 3. Execution Commands for Supported Demos & Experiments

### A. End-to-End Pipeline Demonstration
```bash
python3 src/main.py --mode demo
```
* **Execution Time**: ~2 seconds.
* **Output Displayed**: Step-by-step console logs covering image generation, dual-stream compression, rate-matched uniform baseline finding, partitioned metrics table, tamper mutation test, and simulated packet transmission.

### B. Experiment 1 & 2: Rate-Distortion Parameter Sweep
```bash
python3 src/main.py --mode experiment-compression
```
* **What it runs**: Sweeps background quality $q_{\text{bg}} \in [15, 75]$ at fixed $q_{\text{roi}}=90$, solves for rate-matched uniform quality factors, and logs metrics to CSV.

### C. Experiment 3: Constrained Network Transmission Simulation
```bash
python3 src/main.py --mode experiment-transmission
```
* **What it runs**: Evaluates Priority vs. FIFO packet transmission across Rural 2G (48 kbps), Rural 3G (128 kbps), Satellite VSAT (256 kbps), and Lossy Wireless (10% drop).

### D. Experiment 4: Cryptographic Authenticity & Tamper Audit
```bash
python3 src/main.py --mode experiment-security
```
* **What it runs**: Executes 100 independent encryption, mutation, and decryption trials with fresh nonces, confirming 100% rejection of bit-flip attacks.

---

## 4. Expected Output Artifacts

All outputs are saved in [`results/`](file:///Users/mayanksoni/Desktop/MediLink-DL/results):

| Filename | Type | Description |
| :--- | :--- | :--- |
| [`results/comparison.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/comparison.png) | Image (2.1 MB) | 5-panel comparative visualization (Original, Mask, Uniform, ROI-Aware, Difference Map). |
| [`results/experiment_compression.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_compression.png) | Plot (160 KB) | Rate-distortion curves showing ROI PSNR and SSIM vs. Payload Size. |
| [`results/experiment_transmission.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_transmission.png) | Plot (66 KB) | Bar chart comparing Time-to-First-ROI vs. Total Transfer Time across network profiles. |
| [`results/metrics.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/metrics.csv) | CSV (482 B) | Exact numerical baseline measurements (Global, ROI, BG PSNR and SSIM). |
| [`results/experiment_compression.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_compression.csv) | CSV (629 B) | Multi-point rate-distortion table across 6 background quality factors. |
| [`results/experiment_transmission.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_transmission.csv) | CSV (440 B) | Time to first ROI, total transfer time, and effective throughput table. |
| [`results/experiment_security.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_security.csv) | CSV (122 B) | 100-trial tamper rejection audit metrics and encryption latency. |

To open the figures directly on macOS:
```bash
open results/comparison.png
open results/experiment_compression.png
open results/experiment_transmission.png
```

---

## 5. Detailed Explanation of Result Figures

### Figure 1: `results/comparison.png` (5-Panel Visual Comparison)
1. **Panel 1 (Original Image)**: Reference $512 \times 512$ 8-bit grayscale radiograph phantom showing anatomical structures and the indelible bottom disclaimer watermark.
2. **Panel 2 (Diagnostic ROI Mask)**: Binary mask isolating the simulated focal consolidation lesion ($3,409$ pixels; $1.30\%$ of total image area).
3. **Panel 3 (Uniform Baseline)**: Standard uniform JPEG at matched size ($q=32$, 9.6 KB). Demonstrates visible blocking artifacts and blurring across the lesion.
4. **Panel 4 (ROI-Aware Dual-Stream)**: Dual-stream reconstruction ($q_{\text{roi}}=90, q_{\text{bg}}=25$, 9.8 KB). The lesion retains crisp edges and internal texture while background anatomy is compressed.
5. **Panel 5 (Absolute Residual Map)**: Pixel-wise difference $|I_{\text{orig}} - \hat{I}_{\text{roi}}|$ amplified with `magma` colormap. Shows virtually zero error within the diagnostic ROI circle, with errors concentrated in background tissue.

### Figure 2: `results/experiment_compression.png` (Rate-Distortion Curves)
* **Left Subplot (ROI PSNR vs Payload)**: The green curve (ROI-Aware) remains flat at **38.08 dB** even as payload drops from 29.6 KB to 7.3 KB. The red dashed line (Uniform Baseline) degrades steadily from 36.0 dB down to 33.86 dB.
* **Right Subplot (ROI SSIM vs Payload)**: The ROI-Aware structural similarity remains at $\sim 0.93$ while uniform SSIM collapses to 0.82.

### Figure 3: `results/experiment_transmission.png` (Delivery Acceleration)
* Compares the green bars (Time to First Diagnostic ROI) against blue bars (Total Transfer Time).
* Shows that on Rural 2G (48 kbps), the specialist can begin reviewing the lesion in **0.59 seconds**, rather than waiting **3.92 seconds** for the complete image.

---

## 6. Three-Minute Professor Demonstration Script

> **Minute 1: The Telemedicine Problem & Our Core Hypothesis**  
> *"Good morning, Professor. In rural clinics, medical radiographs must be transmitted over 48 to 128 kbps cellular or satellite links. Transmitting raw radiographs takes several minutes, but aggressive uniform compression ruins subtle pathologies. Our project, MediLink-DL, investigates whether non-uniform dual-stream compression combined with priority network transmission can accelerate diagnostic triage while preserving image integrity. Let me show you our automated test suite."*  
> *(Run: `pytest tests/ -v`)*  
> *"All 41 tests pass in under 1.5 seconds, proving our mathematical, cryptographic, and networking modules are bug-free."*

> **Minute 2: The Working Prototype & Visual Results**  
> *"Now I will run our end-to-end prototype pipeline."*  
> *(Run: `python3 src/main.py --mode demo`)*  
> *"Notice the console output: We achieved a 26.6x compression ratio, reducing a 256 KB image to 9.8 KB. Crucially, we do not compare against an arbitrary quality level; our algorithm solved for an exactly rate-matched uniform baseline at q=32 (9.6 KB). Looking at the metrics table: in the diagnostic lesion, our ROI PSNR is 38.08 dB compared to 34.58 dB for uniform—a +3.50 dB gain. Let me open the visual comparison figure."*  
> *(Run: `open results/comparison.png`)*  
> *"As you can see in Panel 5, the residual error inside the lesion circle is virtually zero, while background tissue absorbs the lossy compression."*

> **Minute 3: Priority Transport & Cryptographic Defense**  
> *"Next, we simulated transmission across rural wireless profiles."*  
> *(Run: `open results/experiment_transmission.png`)*  
> *"In a standard FIFO network, the doctor waits 3.59 seconds for the lesion bytes to arrive. Under our prioritized chunking scheduler, the diagnostic ROI packets arrive in just 0.59 seconds—a 6.07x acceleration in clinical triage. Finally, our payload is protected with AES-256-GCM: in 100 random bit-flip trials, 100% of tampered packets were detected and rejected with less than 0.005 ms encryption overhead. All results are documented in our research paper draft and implementation audit."*

---

## 7. Ten Likely Viva Voce Questions & Model Answers

### Q1: Why didn't you use standard JPEG2000 Part 2 MAXSHIFT for ROI coding?
**Answer**: *"JPEG2000 Part 2 MAXSHIFT scales wavelet transform coefficients above the background bitplanes. While theoretically elegant, JPEG2000 is computationally demanding, has a large memory footprint, and lacks native decoding support in standard web browsers and rural clinical DICOM viewports. Our dual-stream JPEG approach provides an operational, universally deployable baseline using standard DCT codecs."*

### Q2: Did you train a deep learning model on real hospital chest X-rays?
**Answer**: *"No, Professor. To maintain academic and clinical integrity, we did not use real patient X-rays without formal ethics clearance, nor did we claim an ImageNet model is medically validated. We implemented and unit-tested the Grad-CAM saliency hooks on a test CNN architecture, while conducting our baseline compression experiments using the ground-truth mask of an anatomically stylized synthetic phantom with an indelible disclaimer watermark."*

### Q3: How do you guarantee that the uniform comparison is fair?
**Answer**: *"Many compression papers compare ROI coding against arbitrary uniform quality factors (like q=50). We implemented an automated parameter sweep (`find_comparable_uniform_quality`) that searches all integer quality factors q in [1, 95] to match the compressed byte size within -2.24% of our ROI payload. This guarantees that both approaches use an essentially identical byte budget over the channel."*

### Q4: If the ROI quality improves by +3.50 dB, what happens to the background?
**Answer**: *"Because total bitrate is conserved, bits allocated to the ROI are borrowed from the background context. However, because the diagnostic lesion occupies only 1.30% of the image area, the background suffers a negligible penalty of only -0.55 dB in PSNR and -0.0096 in SSIM. The contextual anatomy remains completely recognizable for spatial orientation."*

### Q5: How does your network simulation differ from standard TCP?
**Answer**: *"Standard TCP enforces in-order byte stream delivery: if packet 2 (background) is dropped, packet 5 (diagnostic ROI) is blocked at the receiver (Head-of-Line blocking). Our application-layer transport chunks the payload into independently sequenced datagrams, queues ROI packets ahead of background packets in a min-heap, and uses selective ARQ. This allows the receiver to reconstruct and display the diagnostic lesion before the background even finishes transferring."*

### Q6: Why did you use AES-256-GCM instead of AES-CBC or AES-CTR?
**Answer**: *"AES-CBC and AES-CTR provide confidentiality, but they do NOT provide integrity or authenticity. In telemedicine, an attacker could manipulate ciphertext bits to alter diagnostic features without detection. AES-GCM is an Authenticated Encryption with Associated Data (AEAD) standard that produces a 128-bit authentication tag, guaranteeing that any single-bit tampering causes decryption to strictly fail."*

### Q7: Why do you need a fresh nonce for every AES-GCM encryption?
**Answer**: *"In Galois/Counter Mode, reusing a nonce with the same key destroys the authenticity guarantee and allows an attacker to recover the GHASH authentication key and forge arbitrary ciphertexts. We generate a fresh 96-bit cryptographically secure random nonce (`os.urandom(12)`) for every single transmission."*

### Q8: What are the limitations of PSNR and SSIM in medical imaging?
**Answer**: *"PSNR is based on pixel-wise mean squared error, which penalizes uniform luminance shifts identically to structural blurring. SSIM improves upon this by evaluating local contrast and structural covariance. However, neither metric understands radiological significance: a 5% error in a subtle pleura line could mean a missed pneumothorax, while a 5% error in abdominal air has zero clinical consequence. That is why partitioned ROI evaluation is essential."*

### Q9: Could an attacker forge an ROI mask and cause the system to compress a real tumor aggressively?
**Answer**: *"Yes, that is a genuine architectural risk if ROI extraction is unauthenticated. If an automated AI model fails to detect a secondary lesion or an attacker injects a false mask, the unflagged lesion will be aggressively compressed at q_bg=25. In our discussion, we recommend that background quality never drop below q_bg=20 and that critical triage retains raw uncompressed fallback capability."*

### Q10: What are the immediate next steps to make this a full research publication?
**Answer**: *"Three concrete milestones: First, obtain ethical clearance to ingest clinical DICOM datasets like NIH ChestX-ray14. Second, mount verified CheXNet DenseNet-121 weights to feed live Grad-CAM heatmaps into the compression loop. Third, conduct a multi-reader ROC observer study with practicing radiologists to measure diagnostic accuracy."*

---

## 8. Summary of Known Limitations to Disclose Upfront

1. **Synthetic Phantom Data**: All experimental data is based on a synthetic mathematical phantom.
2. **Dual-Stream JPEG vs. JPEG2000**: Operational compositing approximation, not native wavelet MAXSHIFT.
3. **Application-Level Simulation**: Does not simulate physical RF multipath fading or kernel TCP dynamics.
4. **Grad-CAM Resolution**: Saliency heatmaps are coarse and do not delineate pixel-exact lesion boundaries.
5. **Key Management**: Uses ephemeral in-memory symmetric keys; production requires PKI/HPKE.

---

## 9. Troubleshooting & Runtime Edge Cases

* **Issue: `ModuleNotFoundError: No module named 'src'`**
  * *Cause*: Running python without project root in `PYTHONPATH`.
  * *Fix*: `src/main.py` already includes dynamic `sys.path.insert(0, PROJECT_ROOT)`. Always execute commands from `/Users/mayanksoni/Desktop/MediLink-DL`.
* **Issue: Matplotlib font cache warning**
  * *Observation*: `findfont: Failed to find font weight heavy for DejaVu Sans`.
  * *Resolution*: Benign warning. The script automatically falls back to font weight 700 without crashing.
* **Issue: Re-running experiments overwrites CSVs**
  * *Behavior*: CSV files in `results/` are overwritten with fresh deterministic values. Backups can be made by copying `results/` to `results_backup/`.
