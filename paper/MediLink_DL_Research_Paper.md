# A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine

**Project Working Name**: MediLink-DL  
**Authors**:  
* **Mayank Soni** (Roll No: CS24B1031)  
* **Nandan** (Roll No: CS24B1034)  
* **Sumedha** (Roll No: AD24B1061)  

**Academic Context**: Undergraduate Capstone Research / Digital Image Processing & Computer Networks  
**Date**: October 2026  

---

## Abstract

Rural primary healthcare clinics in resource-constrained regions often lack on-site radiologist specialists and depend on fragile, low-bandwidth communication uplinks (such as 2G/EDGE, weak 3G, or high-latency satellite connections operating at 48–128 kbps). Transmitting uncompressed high-resolution digital radiographs (typically 10–35 MB per study) is impractical over these links. Furthermore, standard uniform lossy compression introduces high-frequency blurring and quantization blocking across the entire image raster, which degrades diagnostically critical pathological zones. 

This paper introduces **MediLink-DL**, an integrated, bandwidth-adaptive, and cryptographically secure framework tailored for medical radiograph transmission. MediLink-DL incorporates: (i) an operational dual-stream Region-of-Interest (ROI) compression architecture that allocates high quantization fidelity ($q_{\text{roi}}=90$) to diagnostic zones while aggressively compressing background context ($q_{\text{bg}}=25$); (ii) an authenticated encryption envelope utilizing AES-256-GCM with fresh 96-bit nonces to protect confidentiality and integrity; (iii) an application-layer prioritized transmission scheduler that delivers diagnostic ROI packets before anatomical background packets over lossy links; and (iv) a weakly supervised Grad-CAM saliency module designed to guide spatial ROI selection.

To uphold academic integrity without unauthorized access to clinical PACS repositories, the prototype was experimentally validated on a calibrated $512 \times 512$ synthetic chest radiograph phantom featuring an indelible watermark. Across empirical evaluations, MediLink-DL achieved a **26.64x compression ratio** (reducing the 262.14 KB uncompressed payload to 9.84 KB). When benchmarked against an exactly rate-matched uniform baseline ($q=32$, 9.62 KB), MediLink-DL improved diagnostic **ROI PSNR by +3.50 dB** (38.08 dB vs. 34.58 dB) and **ROI SSIM by +0.0853** (0.9293 vs. 0.8440). Under simulated rural network conditions (48–128 kbps, 2–10% packet loss), prioritized scheduling reduced the time required for a consulting radiologist to receive the diagnostic lesion by **5.1x to 7.5x** compared to standard FIFO transport. In a 100-trial mutation audit, AES-256-GCM rejected 100% of bit-flip tampering attacks with negligible cryptographic latency ($<0.005$ ms). We discuss engineering trade-offs, Grad-CAM resolution limits, and the operational differences between dual-stream JPEG compositing and standardized JPEG2000 Part 2 MAXSHIFT.

**Keywords**: Medical Image Compression, Telemedicine, Region-of-Interest Coding, Priority Packet Transmission, Authenticated Encryption, Grad-CAM Saliency.

---

## 1. Introduction

Digital radiography (CR/DR) is the primary frontline diagnostic modality for detecting respiratory infections, pulmonary consolidations, tuberculosis, and pneumothoraces [1]. In urban tertiary hospitals, gigabit fiber and on-site Picture Archiving and Communication Systems (PACS) enable instantaneous transmission and review of uncompressed 12-bit to 16-bit DICOM studies. In contrast, rural primary healthcare centers (PHCs) and mobile health outreach units in developing countries operate under severe infrastructural limitations:
1. **Severe Bandwidth Throttling**: Wireless connectivity is frequently constrained to 2G/EDGE (32–64 kbps), congested rural 3G (100–256 kbps), or high-latency geostationary satellite uplinks (RTT > 500 ms).
2. **Channel Instability**: Wireless propagation over rural terrain suffers from high packet drop rates (2% to 10%) and intermittent signal fading.
3. **Eavesdropping and Tampering Risks**: Radiographs transmitted over public cellular networks are susceptible to interception and active packet modification, violating medical privacy mandates (e.g., HIPAA Security Rule, EU GDPR).

Transmitting uncompressed radiographs under such conditions causes transmission delays spanning several minutes, rendering acute triage impossible. Conversely, applying aggressive uniform lossy compression (e.g., uniform JPEG or H.264 intra-coding) introduces irreversible blocking artifacts across the entire visual field, blurring fine diagnostic details such as subtle pleura lines or micro-nodules.

To reconcile these competing demands, this paper develops **MediLink-DL**, a bandwidth-adaptive framework combining deep-learning-guided regional compression, authenticated encryption, and prioritized transport.

---

## 2. Problem Statement

Standard lossy image compression codecs treat every pixel in an image grid with equal mathematical importance. In a chest radiograph, however, the diagnostic utility is spatially concentrated: a $30 \times 30$ pixel pulmonary nodule may dictate life-saving clinical decisions, whereas large peripheral areas (e.g., sub-diaphragmatic abdomen, mediastinal bulk, surrounding air) serve primarily as anatomical context. 

Applying aggressive uniform compression to reduce payload size to $\le 10$ KB severely degrades the pathological region. Conversely, preserving high quality uniformly across a $512 \times 512$ or $1024 \times 1024$ raster inflates transmission time over rural links. The core research challenge is:

> *How can an image transmission pipeline minimize wire payload size and channel delay while isolating and preserving the diagnostic fidelity of critical pathological regions, ensuring tamper-evident integrity, and prioritizing the delivery of diagnostic bytes to a remote specialist?*

---

## 3. Research Objectives

The measurable research objectives of this investigation are:
1. **Design and Implement a Dual-Stream ROI Compression Baseline**: Develop an operational, CPU-executable dual-stream codec that partitions diagnostic zones from contextual background.
2. **Enforce Fair Rate-Matched Evaluation**: Construct an automated parameter sweep to benchmark ROI-aware compression against uniform compression at approximately matched byte sizes ($\pm 5\%$).
3. **Quantify Partitioned Distortion**: Measure Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) partitioned by Global, ROI, and Background areas.
4. **Simulate Application-Layer Prioritized Transport**: Implement packet chunking, SHA-256 framing, selective retransmission (ARQ), and priority queuing to measure time-to-first-ROI delivery under varied channel constraints.
5. **Verify Cryptographic Authenticity**: Deploy AES-256-GCM AEAD encryption and empirically audit tamper detection under bit-inversion mutations.
6. **Investigate Weakly Supervised Saliency Localization**: Implement and verify a Grad-CAM module for model-based spatial focus extraction.

---

## 4. Research Questions and Hypotheses

* **Research Question 1 (RQ1)**: Can non-uniform dual-stream bit allocation yield a measurable increase in regional ROI PSNR and SSIM compared to a uniform JPEG baseline at identical byte budgets?
  * *Hypothesis 1 (H1)*: At payload sizes $\le 15$ KB, allocating $q_{\text{roi}}=90$ to the diagnostic bounding box will improve ROI PSNR by $\ge 3.0$ dB over the uniform baseline without catastrophic background degradation.
* **Research Question 2 (RQ2)**: Does priority-based application-layer packet scheduling significantly accelerate clinical triage availability over lossy, low-bandwidth links?
  * *Hypothesis 2 (H2)*: Prioritizing ROI packets ahead of background context packets will reduce time-to-first-ROI delivery by $\ge 60\%$ across simulated 48–128 kbps channels compared to standard FIFO transmission.
* **Research Question 3 (RQ3)**: Does authenticated envelope encryption introduce prohibitive computational latency on CPU-constrained edge hardware?
  * *Hypothesis 3 (H3)*: AES-256-GCM encryption will impose $<1$ ms of per-frame latency and $<50$ bytes of framing overhead while achieving 100% detection of corrupted ciphertexts.

---

## 5. Related Work

### 5.1 Region-of-Interest (ROI) Image Coding
The necessity of preserving local diagnostic detail in compressed medical imagery led to the standardization of Region-of-Interest coding in JPEG2000 Part 1 and Part 2 (ISO/IEC 15444-1/2) [2]. The seminal MAXSHIFT method, introduced by Christopoulos, Skodras, and Ebrahimi [2], scales the Discrete Wavelet Transform (DWT) coefficients of the ROI above the highest background coefficient bitplane. During entropy coding, ROI bitplanes are transmitted first, enabling progressive, lossless-fidelity reconstruction of the diagnostic zone. 

*Distinction from Present Work*: While JPEG2000 is mathematically elegant, its adoption in rural telemedicine has been hindered by computational complexity, significant memory footprints, and lack of native browser/viewer decoder support. MediLink-DL implements an operational dual-stream JPEG compositing approach that achieves comparable regional quality differentiation using universal baseline JPEG tools, but does not claim equivalence to native wavelet bitplane shifting.

### 5.2 Image Quality Assessment
Image distortion evaluation has evolved from mathematical error measures to perceptual metrics. Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) evaluate pixel-wise Euclidean distance but fail to account for human visual perception [3]. Wang et al. [3] introduced the Structural Similarity Index (SSIM), which models structural luminance, contrast, and cross-correlation covariance. In medical imaging, however, global SSIM can be deceptive: an image with severe pathological blurring in a small $20 \times 20$ lesion can still exhibit a high global SSIM of $>0.90$. MediLink-DL addresses this by partitioning both PSNR and SSIM into explicit ROI and background subsets.

### 5.3 Deep Learning Saliency in Chest Radiography
The introduction of large-scale open chest radiograph datasets, such as NIH ChestX-ray14 [4] and CheXpert [5], catalyzed convolutional neural network (CNN) diagnostic models. Rajpurkar et al. [6] demonstrated radiologist-level pneumonia detection (CheXNet) using a 121-layer DenseNet. 

To visualize model attention without pixel-level ground truth, Selvaraju et al. [7] formulated Gradient-weighted Class Activation Mapping (Grad-CAM). Grad-CAM computes the gradient of a target class score with respect to convolutional feature maps to produce coarse localization heatmaps. In MediLink-DL, Grad-CAM is investigated as a spatial mask provider; we emphasize that Grad-CAM reflects model feature correlations, not clinical ground truth.

### 5.4 Telemedicine Transport and Cryptographic Standards
Medical imagery transmitted across untrusted cellular networks must adhere to statutory security standards (e.g., HIPAA 45 CFR § 164.312). Unauthenticated block ciphers (e.g., AES in CBC mode) are vulnerable to padding oracle attacks and ciphertext bit-flipping. The National Institute of Standards and Technology (NIST SP 800-38D) [8] specifies Galois/Counter Mode (AES-GCM), an authenticated encryption with associated data (AEAD) standard that guarantees confidentiality and tamper-evident authentication.

---

## 6. System Architecture

The MediLink-DL framework comprises five interconnected pipeline stages, illustrated below:

```
+---------------------------------------------------------------------------------------+
|                                    MediLink-DL                                        |
+---------------------------------------------------------------------------------------+
|  [Stage 1: Ingestion]                                                                 |
|   Grayscale Radiograph (512x512) -> Range/Dim Validation -> Diagnostic ROI Mask       |
|                                                                                       |
|  [Stage 2: Compression]                                                               |
|   +--> High-Quality ROI Patch (q_roi=90) --------+                                    |
|   +--> Lossy Context Background (q_bg=25) -------+--> Binary Serialization ('MLDL')   |
|                                                                                       |
|  [Stage 3: Security]                                                                  |
|   Serialized Payload -> AES-256-GCM AEAD (Fresh 96-bit Nonce) -> Encrypted Envelope   |
|                                                                                       |
|  [Stage 4: Transport]                                                                 |
|   Envelope -> Packet Framing & SHA-256 -> Priority Queue (Header > ROI > Background)  |
|   -> Simulated Lossy Channel (48-128 kbps, 2-10% Loss) -> ARQ Retransmission          |
|                                                                                       |
|  [Stage 5: Receiver & Audit]                                                          |
|   Reassembly -> AEAD Tag Verification -> Stream Decoding -> ROI Composite Recon      |
|   -> Partitioned Metrics Export (Global, ROI, BG PSNR/SSIM -> metrics.csv)           |
+---------------------------------------------------------------------------------------+
```

*(A complete publication-grade vector architecture diagram is provided in [`paper/figures/system_architecture.svg`](file:///Users/mayanksoni/Desktop/MediLink-DL/paper/figures/system_architecture.svg).)*

---

## 7. Methodology

### 7.1 Image Ingestion and Synthetic Phantom Specification
The input is an 8-bit single-channel grayscale raster $I \in [0, 255]^{H \times W}$. In strict compliance with research ethics, no unverified patient data was utilized. Instead, a standardized anatomical phantom ($512 \times 512$ pixels) was procedurally synthesized using `image_utils.generate_synthetic_cxr_phantom`:
* **Thoracic Cavity**: Elliptical boundary attenuation gradient ($a=0.82, b=0.88$).
* **Bilateral Lung Cavities**: Low-attenuation elliptical zones offset horizontally by $\pm 0.36$.
* **Mediastinal Column**: Gaussian-profile central spine shadow ($\sigma=0.12$).
* **Cardiac Silhouette**: Left-predominant radiopaque ellipsoid ($a=0.26, b=0.28$).
* **Rib Attenuation**: Sinusoidal horizontal banding ($\text{freq}=14.0$).
* **Simulated Pathology (ROI)**: A focal consolidation lesion injected at $(y=317, x=194)$ with radius $r=33$ pixels ($3,409$ active pixels; $1.30\%$ of total image area).
* **Indelible Watermark**: A permanent high-contrast label stamped across rows $[H-24, H]$: `[SYNTHETIC PHANTOM - NOT A REAL MEDICAL X-RAY]`.

### 7.2 Dual-Stream ROI-Aware Differential Compression
Standard baseline JPEG (ITU-T T.81) enforces uniform $8 \times 8$ Discrete Cosine Transform (DCT) quantization tables across the entire image raster. To achieve regional quality differentiation, MediLink-DL implements a dual-stream encoding protocol:
1. **Background Full-Frame Stream**:
   $$\mathcal{B}_{\text{bg}} = \text{JPEG}_{\text{enc}}(I, q_{\text{bg}}), \quad q_{\text{bg}} \in [10, 30]$$
2. **Localized ROI Patch Stream**: Given binary mask $M \in \{0, 255\}^{H \times W}$, extract the minimal bounding box $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$. The cropped sub-image $I_{\text{crop}} = I[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$ is encoded at diagnostic fidelity:
   $$\mathcal{B}_{\text{roi}} = \text{JPEG}_{\text{enc}}(I_{\text{crop}}, q_{\text{roi}}), \quad q_{\text{roi}} \in [85, 95]$$
3. **Binary Serialization**: The payload is serialized into a packed binary stream with a 26-byte big-endian header:
   ```
   [4B Magic: 'MLDL'] [2B Ver] [2B H] [2B W] [1B q_roi] [1B q_bg]
   [2B Top] [2B Left] [2B Bottom] [2B Right] [4B bg_len] [4B roi_len]
   [N bytes bg_stream] [M bytes roi_stream]
   ```
4. **Reconstruction**: The receiver decompresses $\mathcal{B}_{\text{bg}}$ into $\hat{I}_{\text{bg}}$, decompresses $\mathcal{B}_{\text{roi}}$ into $\hat{I}_{\text{roi}}$, and blends the high-quality patch over the background frame:
   $$\hat{I}_{\text{recon}}[y_{\min}:y_{\max}, x_{\min}:x_{\max}] = \hat{I}_{\text{roi}}$$

### 7.3 Rate-Matched Uniform Baseline Formulation
To avoid biased comparisons against arbitrary uniform compression levels, MediLink-DL performs an automated linear sweep over $q \in [1, 95]$ to find the exact uniform quality factor $q^*$ that minimizes size difference:
$$q^* = \arg\min_{q \in [1, 95]} \left| \text{len}(\text{JPEG}_{\text{enc}}(I, q)) - \text{len}(\mathcal{P}_{\text{roi}}) \right|$$
This guarantees that both compression paradigms are evaluated at matched byte budgets.

### 7.4 Partitioned Quality Metrics
Let $I$ be the reference image and $\hat{I}$ be the reconstructed image.
1. **Partitioned Mean Squared Error (MSE)**:
   $$\text{MSE}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (I(i,j) - \hat{I}(i,j))^2$$
   where $\Omega \in \{\Omega_{\text{global}}, \Omega_{\text{roi}}, \Omega_{\text{bg}}\}$.
2. **Partitioned PSNR**:
   $$\text{PSNR}_{\Omega} = 10 \log_{10} \left( \frac{255^2}{\text{MSE}_{\Omega}} \right) \quad (\text{dB})$$
3. **Partitioned SSIM**:
   Using sliding Gaussian windows ($\sigma=1.5, K_1=0.01, K_2=0.03$), the local SSIM map $S(i,j)$ is computed globally [3], and regional means are derived:
   $$\text{SSIM}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} S(i,j)$$

### 7.5 Authenticated Encryption Protocol
* **Cipher**: AES-256 in Galois/Counter Mode (AES-GCM, NIST SP 800-38D) [8].
* **Nonce Discipline**: Every message generates a fresh 96-bit (12-byte) cryptographically secure pseudorandom nonce ($N \leftarrow \text{CSPRNG}(12)$).
* **Wire Envelope**:
  $$\text{Envelope} = N \,\|\, \text{Ciphertext} \,\|\, \text{Tag}_{128}$$
  Total framing overhead is exactly 28 bytes (12-byte nonce + 16-byte authentication tag).

### 7.6 Prioritized Application-Layer Network Simulation
The transmission module models constrained wireless channels via discrete-event simulation:
* **Framing**: Payloads are chunked into 1024-byte packets with sequence ID, total count, payload type (`HEADER`, `ROI`, `BG`), data, and SHA-256 checksum.
* **Priority Queue**: Chunks are scheduled in a priority min-heap:
  $$\text{Priority}(\text{Header}) = 30 > \text{Priority}(\text{ROI}) = 20 > \text{Priority}(\text{BG}) = 5$$
* **Channel Delays**: Per-packet delay is modeled as serialization delay plus one-way latency:
  $$\Delta t = \frac{(L + 64) \times 8}{R} + D$$
  where $L$ is chunk bytes, $R$ is channel bandwidth in bps, and $D$ is one-way latency.
* **Reliability**: Packet drops are simulated stochastically via $U(0, 1) < P_{\text{loss}}$, triggering Stop-and-Wait / ARQ retransmission with timeout $2D$.
* **Store-and-Forward**: A local buffer retains packets during link outages, draining upon reconnection.

### 7.7 Deep Learning and Grad-CAM Localization Module
The standalone `roi_localization.py` module implements Grad-CAM [7] using forward and full-backward PyTorch hooks:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
The resulting heatmap is bilinearly interpolated to $(H, W)$, min-max normalized to $[0.0, 1.0]$, and thresholded at $\tau=0.50$ to produce a binary candidate ROI mask.

---

## 8. Experimental Setup

* **Computing Environment**: macOS Darwin 27.2.0, Apple Silicon (arm64).
* **Software Stack**: Python 3.11.14, NumPy 1.26.4, Pillow 12.3.0, OpenCV 4.11.0, Matplotlib 3.11.2, scikit-image 0.26.0, Cryptography 50.0.2, PyTorch 2.14.1, Torchvision 0.29.1, PyTest 9.1.1.
* **Input Image**: $512 \times 512$ 8-bit grayscale synthetic CXR phantom ($262,144$ bytes).
* **Test Matrix**:
  * Fixed random seed: $42$ (phantom generation and channel simulation).
  * Rate-Distortion Sweep: $q_{\text{roi}}=90$; $q_{\text{bg}} \in \{15, 25, 35, 45, 60, 75\}$.
  * Network Channel Profiles:
    1. *Rural 2G/EDGE*: $R=48\text{ kbps}, \text{RTT}=200\text{ ms}, P_{\text{loss}}=2\%$.
    2. *Rural 3G (Weak)*: $R=128\text{ kbps}, \text{RTT}=150\text{ ms}, P_{\text{loss}}=5\%$.
    3. *Satellite VSAT*: $R=256\text{ kbps}, \text{RTT}=500\text{ ms}, P_{\text{loss}}=2\%$.
    4. *Lossy Wireless*: $R=64\text{ kbps}, \text{RTT}=120\text{ ms}, P_{\text{loss}}=10\%$.
  * Security Trials: 100 independent encryption, mutation, and authentication runs.

---

## 9. Results

All numerical results reported below are directly derived from the machine-readable output files in [`results/`](file:///Users/mayanksoni/Desktop/MediLink-DL/results) and were independently verified.

### 9.1 Baseline Rate-Distortion Comparison
Table 1 summarizes the baseline performance of MediLink-DL against the matched uniform JPEG baseline on the $512 \times 512$ radiograph phantom.

**Table 1: Baseline Rate-Distortion Metrics ([`results/metrics.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/metrics.csv))**

| Metric | ROI-Aware Coding ($q_{\text{roi}}=90, q_{\text{bg}}=25$) | Uniform Baseline ($q=32$) | Delta / Gain |
| :--- | :--- | :--- | :--- |
| **Uncompressed Size** | 262,144 bytes (256.0 KB) | 262,144 bytes (256.0 KB) | — |
| **Compressed Payload** | **9,842 bytes (9.61 KB)** | 9,622 bytes (9.40 KB) | $+220$ bytes ($+2.24\%$) |
| **Compression Ratio** | **26.64x** | 27.24x | — |
| **Size Reduction** | **96.25%** | 96.33% | — |
| **Global PSNR** | 32.51 dB | 33.04 dB | $-0.53$ dB |
| **Global SSIM** | 0.8049 | 0.8133 | $-0.0084$ |
| **ROI PSNR (Pathology)** | **38.08 dB** | **34.58 dB** | **+3.50 dB** |
| **ROI SSIM (Pathology)** | **0.9293** | **0.8440** | **+0.0853** |
| **Background PSNR** | 32.47 dB | 33.02 dB | $-0.55$ dB |
| **Background SSIM** | 0.8035 | 0.8131 | $-0.0096$ |

*(The complete 5-panel comparative visualization is saved at [`results/comparison.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/comparison.png).)*

### 9.2 Rate-Distortion Sensitivity Across Quality Settings
Table 2 details the multi-point rate-distortion sweep evaluating how background compression affects overall payload and diagnostic quality.

**Table 2: Multi-Rate Evaluation ([`results/experiment_compression.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_compression.csv))**

| $q_{\text{bg}}$ | $q_{\text{roi}}$ | $q_{\text{uniform}}$ | ROI Size (KB) | Uniform Size (KB) | ROI PSNR (dB) | Uniform ROI PSNR (dB) | ROI SSIM | Uniform ROI SSIM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 15 | 90 | 22 | 7.29 | 7.26 | **38.08** | 33.86 | **0.9275** | 0.8237 |
| 25 | 90 | 32 | 9.61 | 9.40 | **38.08** | 34.58 | **0.9293** | 0.8440 |
| 35 | 90 | 41 | 11.93 | 12.17 | **38.08** | 34.82 | **0.9311** | 0.8516 |
| 45 | 90 | 49 | 14.92 | 14.95 | **38.08** | 35.08 | **0.9317** | 0.8600 |
| 60 | 90 | 64 | 20.25 | 20.40 | **38.08** | 35.53 | **0.9324** | 0.8744 |
| 75 | 90 | 76 | 29.58 | 28.79 | **38.08** | 36.00 | **0.9339** | 0.8887 |

*(Visual rate-distortion curves are plotted in [`results/experiment_compression.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_compression.png).)*

### 9.3 Prioritized Transmission Simulation
Table 3 compares transmission latencies under priority-based scheduling versus standard FIFO baseline across simulated rural channels.

**Table 3: Simulated Transmission Across Rural Channels ([`results/experiment_transmission.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_transmission.csv))**

| Channel Profile | Bandwidth (kbps) | RTT (ms) | Loss Rate | Time to First ROI (Priority) | Time to First ROI (FIFO) | Total Transfer Time | Triage Acceleration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rural 2G/EDGE** | 48 | 200 | 2% | **0.592 s** | 3.592 s | 3.916 s | **6.07x** |
| **Rural 3G (Weak)** | 128 | 150 | 5% | **0.372 s** | 2.097 s | 2.294 s | **5.63x** |
| **Satellite VSAT** | 256 | 500 | 2% | **1.036 s** | 5.299 s | 5.822 s | **5.11x** |
| **Lossy Wireless** | 64 | 120 | 10% | **0.384 s** | 2.887 s | 3.100 s | **7.51x** |

*(The comparative transmission bar chart is saved in [`results/experiment_transmission.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_transmission.png).)*

### 9.4 Cryptographic Verification and Tamper Audit
The empirical results from 100 independent encryption/decryption trials recorded in [`results/experiment_security.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/experiment_security.csv) confirm:
* **Total Mutation Trials**: 100.
* **Tamper Rejection Rate**: **100/100 (100.0%)**. Every 1-bit mutation in the ciphertext triggered `cryptography.exceptions.InvalidTag`.
* **Nonce Freshness**: 100 unique 96-bit nonces generated (0% collision rate).
* **Computational Latency**: Mean encryption time was **0.0047 ms**; mean decryption time was **0.0021 ms**.
* **Framing Overhead**: Exactly 28 bytes per study.

---

## 10. Discussion

### 10.1 Regional vs. Global Distortion Trade-Offs
The empirical findings substantiate Hypothesis 1 (H1). At matched payload sizes (~9.6 KB), MediLink-DL produces an ROI PSNR of 38.08 dB, outperforming the uniform baseline by **+3.50 dB**. Similarly, ROI SSIM is improved from 0.8440 to 0.9293. 

Importantly, this gain incurs a minor penalty of only $-0.53$ dB in global PSNR and $-0.55$ dB in background PSNR. Because the diagnostic ROI comprises only $1.30\%$ of the image raster, shifting bits from the background into the ROI delivers substantial local diagnostic preservation with negligible overall visual impact. Furthermore, as shown in Table 2, even when background compression is intensified to $q_{\text{bg}}=15$ (reducing payload to 7.29 KB), the diagnostic ROI maintains an invariant PSNR of 38.08 dB, whereas the uniform baseline degrades to 33.86 dB.

### 10.2 Triage Acceleration vs. Total Bandwidth
A common misconception in telemedicine transmission is that compression alone resolves transmission bottlenecks. As shown in Table 3, total transfer time over a 48 kbps 2G link is 3.916 seconds. Under standard FIFO transmission, the consulting radiologist must wait 3.592 seconds before the lesion bytes arrive. 

Under MediLink-DL’s priority scheduling, however, the header and ROI packets are scheduled at the head of the queue, arriving at the receiver in **0.592 seconds** (**6.07x faster**). On a lossy wireless channel (10% loss), priority delivery arrives in **0.384 seconds** compared to 2.887 seconds (**7.51x faster**). This acceleration enables immediate preliminary triage of acute findings while the background anatomical context continues to load progressively.

### 10.3 Cryptographic Feasibility
The experimental overhead of AES-256-GCM ($<0.005$ ms) proves that modern cryptographic authentication introduces zero perceptible bottleneck on telemedicine client workstations. The 28-byte framing overhead constitutes less than $0.29\%$ of the 9.8 KB compressed payload.

---

## 11. Limitations

To maintain rigorous academic transparency, we highlight the following engineering and clinical limitations:
1. **Synthetic Data Evaluation**: All experiments were conducted on a synthetic mathematical phantom. While anatomical proportions, attenuation gradients, and lesions were simulated, the phantom lacks real clinical radiographic noise, DICOM 12/16-bit dynamic range, rib calcifications, and patient positioning variability.
2. **Algorithmic JPEG Approximation vs. Standardized JPEG2000**: MediLink-DL uses an operational dual-stream JPEG compositing baseline. It does not implement standardized JPEG2000 Part 2 MAXSHIFT wavelet coefficient bitplane shifting, nor HEVC tiled coding.
3. **Application-Level Network Modeling**: Channel simulation uses discrete-event queuing and stochastic packet drops. It does not capture physical-layer multipath fading, Doppler shifts, or Linux kernel TCP congestion control algorithms (Cubic/BBR).
4. **Grad-CAM Saliency Resolution**: The Grad-CAM module operates on coarse convolutional feature maps ($16 \times 16$ or $32 \times 32$), producing smooth heatmaps that do not correspond to precise anatomical lesion boundaries. Grad-CAM identifies model-correlated features, not radiological ground truth.
5. **Key Management Scope**: The prototype generates ephemeral symmetric keys in memory. Production deployment requires Public-Key Infrastructure (PKI) or Hybrid Public-Key Encryption (HPKE, RFC 9180).

---

## 12. Conclusion and Future Work

This paper presented **MediLink-DL**, a unified framework integrating ROI-aware dual-stream compression, authenticated encryption, and prioritized network transmission for rural telemedicine. Evaluated on a standardized $512 \times 512$ synthetic radiograph phantom, the framework demonstrated a **26.64x compression ratio**, a **+3.50 dB gain in diagnostic ROI PSNR**, a **+0.0853 gain in ROI SSIM**, and a **5.1x to 7.5x acceleration in time-to-first-ROI delivery** across simulated rural 2G/3G/satellite channels.

**Future Work** includes:
1. Ingesting clinical DICOM datasets (e.g., NIH ChestX-ray14) under institutional ethics approval.
2. Integrating verified pretrained CheXNet DenseNet-121 weights into the live compression pipeline.
3. Evaluating transmission over physical software-defined radio (SDR) testbeds and Linux `tc/netem` emulators.
4. Conducting multi-reader Receiver Operating Characteristic (ROC) observer studies with clinical radiologists to assess diagnostic equivalency.

---

## References

[1] World Health Organization, *Telemedicine: Opportunities and Developments in Member States*, Global Observatory for eHealth Series, vol. 2, Geneva: WHO Press, 2010.

[2] C. Christopoulos, A. Skodras, and T. Ebrahimi, "The JPEG2000 still image coding system: an overview," *IEEE Transactions on Consumer Electronics*, vol. 46, no. 4, pp. 1103–1127, Nov. 2000, doi: [10.1109/30.920468](https://doi.org/10.1109/30.920468).

[3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, Apr. 2004, doi: [10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861).

[4] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Honolulu, HI, USA, Jul. 2017, pp. 2097–2106.

[5] J. Irvin, P. Rajpurkar, M. Ko, Y. Yu, S. Ciurea-Ilcus, C. Chute, H. Marklund, B. Hepworth, A. Ng, et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in *Proc. AAAI Conf. Hum. Comput. Crowdsourcing*, vol. 33, no. 1, pp. 590–597, Jul. 2019.

[6] P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding, A. Bagul, C. Langlotz, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," *arXiv preprint arXiv:1711.05225*, 2017.

[7] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedaldi, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, Oct. 2017, pp. 618–626.

[8] M. Dworkin, "Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode (GCM) and GMAC," *NIST Special Publication 800-38D*, National Institute of Standards and Technology, Gaithersburg, MD, Nov. 2007, doi: [10.6028/NIST.SP.800-38D](https://doi.org/10.6028/NIST.SP.800-38D).
