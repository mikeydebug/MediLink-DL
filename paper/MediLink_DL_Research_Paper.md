# MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine

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
**Date**: November 2026 &nbsp;|&nbsp; **Academic Capstone Research Paper**  

---

## Abstract

Rural primary healthcare centers (PHCs) and mobile medical clinics in developing geographies lack on-site specialist radiologists and rely on fragile, bandwidth-throttled communication uplinks (2G/EDGE, weak 3G, or geostationary satellite uplinks operating at 48–128 kbps). Transmitting uncompressed high-resolution digital radiographs (typically 10–35 MB per study) is intractable over these links, while standard uniform lossy compression introduces high-frequency blurring and blocking artifacts that degrade subtle pathological lesions.

This paper presents **MediLink-DL**, an integrated cross-layer framework engineered to resolve the trilemma of **diagnostic fidelity, transmission latency, and cryptographic security**. The system synthesizes five core scientific and engineering contributions:
1. **Clinical Data Curation & Leakage-Free Stratification**: Evaluated on 166 real clinical chest radiographs across 110 patients from the Cohen et al. (2020) cohort with ground-truth bilateral lung and lesion annotations, strictly partitioned at the patient level ($\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$) to prevent data leakage.
2. **Deep Learning Diagnostic Localization & Uncertainty Weighting**: Features a customized single-channel `MedicalResNet18` paired with **Differentiable Grad-CAM** (`create_graph=True`). Multi-task loss balancing is resolved via Kendall & Gal (CVPR 2018) homoscedastic uncertainty optimization, dynamically weighting classification and continuous spatial alignment without heuristic tuning. The operational model (Regime B) achieves a **93.10% Pointing Game Hit Rate**, **80.32% Energy in ROI**, and **100% Sensitivity (Recall)** on held-out COVID-19/SARS test cohorts.
3. **Dual-Stream ROI-Aware Differential Compression**: Implements an operational dual-stream coding architecture that quantizes the critical diagnostic lung patch at high quality ($q_{\text{roi}}=90$) while aggressively compressing background context ($q_{\text{bg}}=25$). Transmitted inside a 28-byte binary container header (`'MLDL'`), it achieves an **89.89% data reduction** (**9.89x compression ratio**, dropping wire footprint from 262 KB to 26.5 KB) while sustaining an **ROI PSNR > 48.3 dB** and **ROI SSIM > 0.989**, offering a **+10.13 dB gain** over rate-matched uniform baseline JPEG at the lesion site.
4. **Standardized & Lightweight Authenticated Encryption (AEAD)**: Employs NIST SP 800-38D AES-256-GCM and NIST SP 800-232 Ascon-AEAD128 with unique CSPRNG nonces and Associated Data (AAD) metadata binding, strictly enforcing **Zero-Plaintext Release** with **100% bit-flip tamper detection**. An isolated cryptographic testbed introduces **MLMC-128**, a research block cipher achieving a **49.92% Strict Avalanche Criterion (SAC)** across 4,800 trials and optimal circulant MDS diffusion ($\mathcal{B}=5$).
5. **Prioritized Transport & Cross-Layer Optimization**: Uses a Min-Heap priority scheduler to transmit diagnostic ROI packets ahead of background context, accelerating clinical triage availability to **0.391 seconds** (**>6.2x faster than FIFO**). A multi-objective grid search across 108 end-to-end parameter sets identifies **26 non-dominated Pareto configurations**.

**Keywords**: Telemedicine, Digital Image Processing, Differentiable Grad-CAM, Kendall Uncertainty Weighting, Patient-Level Stratification, Authenticated Encryption (AEAD), NIST SP 800-232, Priority Packet Scheduling, Pareto Optimization.

---

## 1. Introduction & Clinical Motivation

Digital radiography (CR/DR) constitutes the primary diagnostic frontline for detecting acute pulmonary consolidations, pneumothoraces, tuberculosis, and viral pneumonias [1]. In urban tertiary hospitals, gigabit fiber and on-site Picture Archiving and Communication Systems (PACS) enable instantaneous transmission of uncompressed 12-bit to 16-bit DICOM studies. In contrast, rural primary healthcare centers (PHCs) in resource-constrained regions operate under severe infrastructural limitations:
1. **Severe Bandwidth Throttling**: Wireless links are restricted to 2G/EDGE (32–64 kbps), weak 3G (100–256 kbps), or high-latency satellite connections (RTT > 500 ms).
2. **High Packet Loss & Channel Jitter**: Physical transmission over rural cellular links experiences 2% to 10% packet drop rates and intermittent signal fading.
3. **Eavesdropping and Tampering Risks**: Radiographs transmitted over untrusted public channels require cryptographic integrity to prevent active packet modification and comply with statutory healthcare privacy mandates (e.g., HIPAA Security Rule, EU GDPR).

Transmitting uncompressed radiographs (10–35 MB) across a 64 kbps link requires 20 to 75 minutes per patient, making acute triage impossible. Conversely, standard uniform lossy compression (such as JPEG or consumer messaging applications like WhatsApp) introduces irreversible DCT blocking artifacts and high-frequency blurring across subtle opacity boundaries. Furthermore, consumer messaging platforms strip DICOM headers, inject uncontrolled quantization noise, and lack end-to-end HIPAA-compliant cryptographic provenance.

MediLink-DL addresses this challenge through an integrated, cross-layer architecture uniting:
- **Digital Image Processing (Supervised by Dr. Kaushiki Roy)**: Continuous differentiable spatial saliency extraction, dual-stream JPEG tiling, and partitioned structural fidelity metrics.
- **Computer Networks (Supervised by Dr. Mehbub Alam)**: Application-layer priority min-heap chunk scheduling, selective-repeat ARQ simulation, and 2G/EDGE channel modeling.
- **Multi-Objective Optimization (Supervised by Arnav Ghosh)**: Kendall & Gal homoscedastic uncertainty loss balancing and end-to-end Pareto frontier optimization across 108 operational configurations.

---

## 2. Clinical Dataset Curation & Leakage-Free Stratification

### 2.1 Clinical Dataset Specifications
To validate MediLink-DL on authentic radiological data rather than synthetic phantoms, we curated a cohort from the open-access **COVID-19 Chest X-ray Dataset Initiative** (Cohen et al., 2020) [2]. The dataset contains genuine clinical radiographs (PA/AP view) associated with acute respiratory distress, COVID-19, and SARS. Bilateral thoracic cage and focal consolidation bounding boxes were verified through expert radiologist segmentations provided by `imageannotation.ai`.

The verified cohort comprises 166 clinical radiographs across 110 unique patients:
- Image Dimensions: Resampled to standardized $512 \times 512$ single-channel 8-bit grayscale rasters.
- Class Distribution: Binary classification between COVID-19/SARS positive opacities and bacterial/normal control radiographs.

### 2.2 Mathematical Leakage-Free Patient Stratification Protocol
In published medical machine learning literature, random image-level train/test partitioning frequently leads to severe **data leakage**: multiple radiographs from the same patient appear in both the training set and the test set. Because deep convolutional networks easily memorize patient-specific anatomical idiosyncrasies (e.g., rib shape, cardiac silhouette, clavicle implants), random splitting artificially inflates test accuracy.

To establish authentic generalizability, MediLink-DL enforces **strict patient-level stratification**:
- **Training Partition ($\mathcal{P}_{\text{train}}$)**: 77 patients (111 radiographs)
- **Validation Partition ($\mathcal{P}_{\text{val}}$)**: 16 patients (26 radiographs)
- **Test Partition ($\mathcal{P}_{\text{test}}$)**: 17 patients (29 radiographs)

The partition non-overlap criterion is formally enforced:
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$

This constraint is programmatically verified in the test harness (`tests/test_real_dataset.py::test_patient_split_leakage_zero`).

---

## 3. Deep Learning Localization & Uncertainty Weighting

### 3.1 Network Architecture
We adapted the standard `ResNet-18` backbone into `MedicalResNet18` for digital radiography:
- **Grayscale Adaptation**: The initial convolutional layer (`conv1`) is modified from 3 input channels to 1 single-channel input ($7 \times 7$ kernel, stride 2, padding 3).
- **Feature Extraction**: Four residual convolutional blocks (`layer1` through `layer4`) extract high-level semantic representations, yielding an activation map $A \in \mathbb{R}^{512 \times 7 \times 7}$ at `layer4[-1]`.
- **Classification Head**: Global average pooling maps the 512-channel activation to a fully connected linear layer producing binary disease logits $\hat{y} \in \mathbb{R}^2$.

### 3.2 Differentiable Grad-CAM Formulation
Standard Grad-CAM [3] is a detached post-hoc interpretability technique where activations and gradients are evaluated with gradient tracking disabled. In MediLink-DL, `DifferentiableGradCAM` preserves the backward computational graph (`create_graph=True`), allowing second-order gradients to backpropagate into convolutional feature maps during training:

1. **Neuron Importance Weights**: Let $y^c$ denote the classification score for target class $c$ (COVID-19 positive). The gradient with respect to feature map $A^k$ of layer $L$ is spatially pooled:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. **Continuous Heatmap Assembly**:
   $$H_{\text{CAM}}^c = \operatorname{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right)$$
3. **Bilinear Upsampling & Normalization**: $H_{\text{CAM}}^c$ is bilinearly interpolated to the native image resolution $(H, W) = (512, 512)$ and normalized:
   $$M_{\text{CAM}} = \frac{H_{\text{CAM}}^c - \min(H_{\text{CAM}}^c)}{\max(H_{\text{CAM}}^c) - \min(H_{\text{CAM}}^c) + \epsilon}$$

### 3.3 Multi-Task Loss Formulation
To eliminate **Saliency Drift** (where the CNN correctly classifies the image but focuses on non-pathological artifacts such as clavicles or text markers), the training loss combines classification and anatomical localization penalties:
$$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{CE}}(y, \hat{y})$$
$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum M_{\text{CAM}} M_{\text{GT}} + \epsilon}{\sum M_{\text{CAM}}^2 + \sum M_{\text{GT}}^2 + \epsilon}$$
$$\mathcal{L}_{\text{align}} = 1 - \frac{\langle \operatorname{vec}(M_{\text{CAM}}), \operatorname{vec}(M_{\text{GT}}) \rangle}{\|\operatorname{vec}(M_{\text{CAM}})\|_2 \|\operatorname{vec}(M_{\text{GT}})\|_2}$$
$$\mathcal{L}_{\text{loc}} = (1 - \beta)\mathcal{L}_{\text{align}} + \beta \mathcal{L}_{\text{dice}}$$

### 3.4 Kendall & Gal (CVPR 2018) Homoscedastic Uncertainty Optimization
Manually setting fixed hyperparameters ($\lambda_{\text{loc}}, \beta$) leads to suboptimal local minima or gradient domination. Following Kendall & Gal (CVPR 2018) [4], we implemented `AdaptiveLocalizationAwareLoss`, which treats multi-task balancing as maximum likelihood inference over homoscedastic task uncertainties:

$$\mathcal{L}_{\text{total}}(w, \sigma_1, \sigma_2) = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{cls}}(w) + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{loc}}(w) + \log \sigma_1 + \log \sigma_2$$

To enforce numerical stability and avoid negative variances, we parameterize $s_1 = \log \sigma_1^2$ and $s_2 = \log \sigma_2^2$:
$$\mathcal{L}_{\text{total}}(w, s_1, s_2) = \frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}}(w) + \frac{1}{2}\exp(-s_2)\mathcal{L}_{\text{loc}}(w) + \frac{1}{2}s_1 + \frac{1}{2}s_2$$

Both $s_1$ and $s_2$ are registered as unconstrained learnable parameters optimized concurrently with network weights $w$ via AdamW, automatically discovering optimal task weighting during backpropagation.

---

## 4. Digital Image Processing: Dual-Stream ROI-Aware Compression

### 4.1 Dual-Stream Encoding Architecture
Standard lossy compression (ITU-T T.81 / JPEG) applies uniform $8 \times 8$ Discrete Cosine Transform (DCT) quantization across the entire image raster. When bandwidth is constrained to rural limits, aggressive uniform compression degrades high-frequency trabecular lung textures.

MediLink-DL employs an operational dual-stream decomposition:
1. **Background Full-Frame Stream**: The contextual full radiograph $I \in \mathbb{R}^{512 \times 512}$ is compressed at low quality ($q_{\text{bg}}=25$):
   $$\mathcal{S}_{\text{bg}} = \text{JPEG}_{\text{enc}}(I, q_{\text{bg}})$$
2. **Diagnostic ROI Patch Stream**: The continuous Grad-CAM heatmap $M_{\text{CAM}}$ is thresholded at cutoff $\tau=0.35$ to generate binary mask $M_{\text{bin}} = (M_{\text{CAM}} \ge \tau)$. The minimal bounding box enclosing all active pixels $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ is cropped to yield sub-image $I_{\text{crop}}$. This diagnostic patch is encoded at high fidelity ($q_{\text{roi}}=90$):
   $$\mathcal{S}_{\text{roi}} = \text{JPEG}_{\text{enc}}(I_{\text{crop}}, q_{\text{roi}})$$

### 4.2 28-Byte Binary Container Header (`'MLDL'`)
The dual streams are packaged into a self-describing, big-endian binary container:
```text
+--------+--------+--------+--------+--------+--------+--------+--------+
|  'M'   |  'L'   |  'D'   |  'L'   |  Ver (uint16)   |  H (uint16)     |
+--------+--------+--------+--------+--------+--------+--------+--------+
|  W (uint16)     | q_roi  |  q_bg  |  Y_min (uint16) |  X_min (uint16) |
+--------+--------+--------+--------+--------+--------+--------+--------+
|  Y_max (uint16) |  X_max (uint16) |  ROI Stream Length (uint32)     |
+--------+--------+--------+--------+--------+--------+--------+--------+
|  Background Stream Length (uint32)|  [ROI Bytes ...] [BG Bytes ...] |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

### 4.3 Receiver Reassembly & Alpha Blending
At the receiver, both streams are decoded independently. The high-quality ROI patch is composited onto the background canvas using distance-transform alpha blending ($\alpha=0.45$) across a 4-pixel perimeter, preventing tiling boundary seams.

### 4.4 Automated Rate-Matched Baseline Search
To guarantee a mathematically rigorous evaluation against standard JPEG, we implement an automated binary search that identifies the exact uniform quality setting $q^*$ matching the dual-stream byte budget:
$$q^* = \arg\min_{q \in [1, 100]} \left| \operatorname{len}(\text{JPEG}(I, q)) - \operatorname{len}(\mathcal{P}_{\text{dual}}) \right|$$

---

## 5. Partitioned Radiological Distortion Metrics

Global image metrics (e.g., global PSNR and SSIM) fail to reflect clinical utility: a high global score can be achieved by perfectly reconstructing the dark background while severely blurring a small focal lesion. We therefore compute partitioned metrics over mutually exclusive regions:

Let $\Omega_{\text{roi}} = \{(i,j) \mid M_{\text{bin}}(i,j) = 1\}$ and $\Omega_{\text{bg}} = \{(i,j) \mid M_{\text{bin}}(i,j) = 0\}$:

1. **Partitioned Mean Squared Error (MSE)**:
   $$\text{MSE}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (I(i,j) - \hat{I}(i,j))^2$$
2. **Partitioned Peak Signal-to-Noise Ratio (PSNR)**:
   $$\text{PSNR}_{\Omega} = 10 \log_{10}\left( \frac{255^2}{\text{MSE}_{\Omega}} \right) \quad (\text{dB})$$
3. **Partitioned Structural Similarity Index (SSIM)**:
   $$\text{SSIM}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} S(i,j)$$
   where $S(i,j)$ is the local SSIM map computed using an $11 \times 11$ circular Gaussian filter ($\sigma=1.5$).

---

## 6. Authenticated Encryption Layer (AEAD) & Security Verification

### 6.1 Cryptographic Protocol Specifications
To guarantee confidentiality and tamper-evident authentication across untrusted cellular networks, MediLink-DL enforces Authenticated Encryption with Associated Data (AEAD):

1. **Industry Standard: AES-256-GCM (NIST SP 800-38D)** [5]:
   - Key: 256-bit CSPRNG key.
   - Nonce: 96-bit unique counter nonce ($N \leftarrow \text{CSPRNG}(12)$).
   - Authentication: 128-bit GHASH authenticator over $\mathbb{F}_{2^{128}}$.
2. **IoT & Edge Standard: Ascon-AEAD128 (NIST SP 800-232)** [6]:
   - Ratified in 2024 as the NIST Lightweight Cryptography Standard.
   - Operates over a 320-bit permutation state with 128-bit key, 128-bit nonce, and 128-bit tag.
3. **Research Prototype: MLMC-128 (MediLink Mini Cipher)**:
   - 128-bit block cipher utilizing PRESENT S-boxes and a circulant MDS diffusion matrix ($\mathcal{B}=5$).
   - Operated in Counter (CTR) mode with GHASH authentication, enabling out-of-order block decryption over lossy channels.

### 6.2 Associated Data (AAD) Binding & Zero-Plaintext Release
To prevent cross-patient injection, replay, and swapping attacks, hospital metadata is bound into the authentication tag as Associated Data:
$$\text{AAD} = \text{StudyUID} \,\|\, \text{PatientID} \,\|\, \text{Timestamp} \,\|\, \text{HeaderChecksum}$$

**Zero-Plaintext Release**: The receiver executes constant-time tag verification prior to releasing decoded buffers. If a single bit in the ciphertext or envelope is inverted, decryption raises an unrecoverable exception and purges memory, ensuring corrupted telemetry never reaches the clinician.

---

## 7. Prioritized Computer Networks Transport & Scheduling

### 7.1 Packet Framing Protocol
Telemetry streams are fragmented into fixed chunks ($L \in \{512, 1024, 2048\}$ bytes). Each packet header encodes:
`[4B Magic: 'MLPK'] [2B SeqID] [2B TotalChunks] [1B PriorityTier] [2B PayloadLen] [32B SHA-256 Checksum]`.

### 7.2 Priority Min-Heap Scheduling
Chunks are enqueued into an application-layer priority min-heap:
$$\text{Priority}(\text{Header}) = 30 > \text{Priority}(\text{ROI Chunks}) = 20 > \text{Priority}(\text{Background Chunks}) = 5$$

Under standard FIFO scheduling, diagnostic ROI packets are delayed until preceding background context chunks are delivered. With priority scheduling, the complete diagnostic ROI arrives in the first burst of transmissions.

### 7.3 Channel Model & Selective-Repeat ARQ Simulation
The transmission simulator implements discrete-event cellular modeling:
$$\Delta t_{\text{chunk}} = \frac{(L + 64) \times 8}{R} + \frac{\text{RTT}}{2} + \xi_{\text{jitter}}$$
where $R$ is link bandwidth (48–128 kbps), $\text{RTT}$ is round-trip propagation delay (120–500 ms), and $\xi$ is Gaussian channel jitter. Packet drops (simulated at 2% to 10%) trigger Selective-Repeat ARQ retransmissions.

---

## 8. Multi-Objective Optimization & Pareto Analysis

A grid search was executed across 108 end-to-end pipeline configurations:
- Grad-CAM Threshold: $\tau \in \{0.25, 0.35, 0.50\}$
- ROI Quality: $q_{\text{roi}} \in \{85, 90, 95\}$
- Background Quality: $q_{\text{bg}} \in \{15, 25, 40\}$
- Cryptographic Cipher: AES-256-GCM vs. Ascon-AEAD128
- Transport Policy: Priority Scheduling vs. FIFO
- Uplink Bandwidth: 48 kbps (2G), 64 kbps (GPRS), 128 kbps (3G)

### 8.1 Pareto Dominance Formulation
Configuration $A$ dominates configuration $B$ ($A \succ B$) if and only if:
$$\forall m \in \{\text{SSIM}_{\text{roi}}, -\text{Bytes}, -T_{\text{roi}}, -\text{CPU}_{\text{crypto}}\}, \quad m(A) \ge m(B)$$
$$\exists m \in \{\text{SSIM}_{\text{roi}}, -\text{Bytes}, -T_{\text{roi}}, -\text{CPU}_{\text{crypto}}\}, \quad m(A) > m(B)$$

The optimization extracted **26 non-dominated Pareto configurations**, establishing the global upper bound of telemedicine transmission efficiency.

---

## 9. Empirical Results & Performance Evaluation

### 9.1 Deep Learning Localization Evaluation (Held-Out Test Cohort)
Evaluated on 29 test radiographs from 17 held-out patients (`results/real_dataset_training_results.csv`):

| Regime / Objective Mode | Test Accuracy | Test F1 | Test Recall | Test ROI Dice | Test ROI IoU | Pointing Game Hit Rate | Energy in ROI | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime A: Classification-Only** | 0.6897 | 0.8163 | 1.0000 | 0.4764 | 0.3571 | 65.52% | 49.21% | 27.05 s |
| **Regime B: Combined Loss (Proposed)** | **0.6897** | **0.8163** | **1.0000** | **0.6856** | **0.5395** | **93.10%** | **80.32%** | **28.03 s** |
| **Regime C: Alignment-Only** | 0.6897 | 0.8163 | 1.0000 | 0.5799 | 0.4243 | **96.55%** | **82.22%** | 27.24 s |
| **Regime D: Dice-Only** | **0.7586** | **0.8511** | **1.0000** | **0.7107** | **0.5704** | 82.76% | 69.36% | 26.88 s |

> **Key Architectural Finding**: While Regime D achieves higher continuous Dice (0.7107), its energy inside the ROI drops to 69.36% and Pointing Game accuracy drops to 82.76%. **Regime B was chosen as the operational champion** because it achieves **93.10% Pointing Game Hit Rate** and **80.32% Energy Confinement**, ensuring gradient attention is locked directly onto pulmonary consolidations while maintaining **100% Sensitivity (Zero False Negatives)** on viral lesions.

### 9.2 Rate-Distortion & Compression Performance on Real CXRs
Evaluated on genuine human chest radiograph (512x512, 262,144 bytes uncompressed):

| Compression Scheme | Parameter Setting | Transmitted Bytes | Data Reduction | Compression Ratio | ROI PSNR | ROI SSIM | Global PSNR | Global SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Stream ROI-Aware** | $q_{\text{roi}}=90, q_{\text{bg}}=25$ | **26,515 B** | **89.89%** | **9.89x** | **48.31 dB** | **0.9891** | 42.10 dB | 0.9634 |
| **Rate-Matched Uniform JPEG**| $q=32$ (Matched Bytes) | 26,035 B | 90.07% | 10.07x | 38.18 dB | 0.9312 | 39.45 dB | 0.9410 |
| **High-Quality Uniform JPEG** | $q=90$ (High Quality) | 98,412 B | 62.46% | 2.66x | 48.31 dB | 0.9897 | 49.11 dB | 0.9882 |

> **Diagnostic Quality Gain**: At equal transmission budgets (~26.5 KB), MediLink-DL delivers a **+10.13 dB gain in ROI PSNR** and **+0.0579 in ROI SSIM** over uniform JPEG.

### 9.3 Cryptographic Latency & Throughput Benchmark
Evaluated across 1024-byte payload chunks on an Apple Silicon M-series host:

| Cryptographic Cipher | Architecture / Standard | Encrypt Latency ($\mu$s) | Decrypt Latency ($\mu$s) | Wire Throughput | Bit-Flip Tamper Detection |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AES-256-GCM** | NIST SP 800-38D (ARMv8 Crypto) | **2.2 $\mu$s** | **2.4 $\mu$s** | **465.4 MB/s** | **100.0% Detected** |
| **Ascon-AEAD128** | NIST SP 800-232 (IoT Standard) | **412.0 $\mu$s** | **418.0 $\mu$s** | **2.48 MB/s** | **100.0% Detected** |
| **MLMC-128 (Raw Block)**| Research Prototype (Pure Python) | **2,140.0 $\mu$s** | **2,190.0 $\mu$s** | **0.47 MB/s** | **N/A (Block Only)** |
| **MLMC-128 (AEAD Exp)** | Research Candidate (CTR + GHASH) | **7,889.0 $\mu$s** | **7,920.0 $\mu$s** | **0.13 MB/s** | **100.0% Detected** |

> **Channel Budget Reality**: On a 64 kbps rural wireless channel, transmitting 1024 bytes over the air takes $128,000\ \mu\text{s}$ (128 ms). Even unoptimized pure-Python MLMC-128 encryption overhead (7.8 ms) represents only **6.1% of physical transmission latency**, while production AES-256-GCM takes <0.002% of the channel airtime.

### 9.4 Prioritized Transmission Across Rural Channels

| Channel Link Profile | Bandwidth | Latency (RTT) | Loss Rate | Time-to-First-ROI (Priority) | Time-to-First-ROI (FIFO) | Triage Acceleration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rural 2G/EDGE** | 48 kbps | 200 ms | 2.0% | **0.592 s** | 3.592 s | **6.07x** |
| **Rural 2.5G/GPRS** | 64 kbps | 120 ms | 5.0% | **0.391 s** | 2.430 s | **6.21x** |
| **Rural 3G (Weak)** | 128 kbps | 150 ms | 5.0% | **0.372 s** | 2.097 s | **5.63x** |
| **Satellite VSAT** | 256 kbps | 500 ms | 2.0% | **1.036 s** | 5.299 s | **5.11x** |

---

## 10. Cryptographic Research Laboratory: Involutory Diffusion & Impossibility Proofs

In our isolated research laboratory (`src/crypto_research/`), we investigated lightweight linear diffusion matrices ($y = M x$) for next-generation rural medical microcontrollers:

### 10.1 Novel Matrix Architectures
1. **`PROPOSED-OIN-MDS`**: Optimal parameter Hadamard instantiation $\operatorname{Had}(1, 8, 13, 5)$ over $\mathbb{F}_{2^4}/(x^4+x+1)$:
   - Involutory: $M^2 = I$ verified.
   - Branch Numbers: $\mathcal{B}_{\text{nibble}} = 5$ (Optimal MDS), $\mathcal{B}_{\text{bit}} = 8$ (Maximal).
   - Implementation: 64 direct XORs $\to$ **48 Paar CSE XORs** (~96.0 GE).
2. **`ABC-MDS-OPT`**: Alternating Block-Circulant matrix with parameters $(0, 1, 2, 3)$:
   - Branch Numbers: $\mathcal{B}_{\text{nibble}} = 5$ (Optimal MDS), $\mathcal{B}_{\text{bit}} = 6$.
   - Complexity: 64 direct XORs $\to$ **49 Paar CSE XORs** (~98.0 GE).
3. **`TBOI-16`**: Tensor-Product Binary Orthogonal Involution ($K_4 \otimes K_4$) over $\mathbb{F}_2$:
   - Branch Numbers: $\mathcal{B}_{\text{nibble}} = 4$ (Near-MDS), $\mathcal{B}_{\text{bit}} = 6$.
   - Complexity: 128 direct XORs $\to$ **60 Paar CSE XORs** (~120.0 GE).

### 10.2 Formal Impossibility Theorems
1. **Theorem 1 (Hankel Non-Existence)**: *No $4 \times 4$ Hankel matrix over $\mathbb{F}_{2^m}$ can be simultaneously MDS and involutory.*
2. **Proposition 2 (Cauchy Involution Collapse)**: *Symmetric Cauchy matrices over $\mathbb{F}_{2^4}$ satisfying self-inversion degenerate into singular or non-MDS structures.*
3. **Theorem 3 (Galois Matrix Symmetry Obstruction)**: *For elements $\alpha, \beta \in \mathbb{F}_{2^4}$, the binary representation matrices satisfy $L_\alpha = L_\beta^T$ if and only if $\alpha = \beta \in \{0, 1\}$.*
4. **Theorem 4 (Tensor Bound)**: *The Kronecker product $A \otimes B$ of two invertible $2 \times 2$ matrices cannot achieve optimal MDS branch number $\mathcal{B}=5$; it is strictly bounded by $\mathcal{B} \le 4$.*

---

## 11. Scientific Discussion, Clinical Limitations & Future Work

### 11.1 Clinical & Regulatory Limitations
1. **Cohort Scale**: Evaluated on 166 radiographs from 110 patients. Formal regulatory clearance (e.g., US FDA 510(k), EU CE-MDR, CDSCO) requires prospective multi-center validation on 20,000+ studies across diverse radiological equipment.
2. **Planar 2D Modality**: The current framework targets 2D projection chest radiographs. Extending the pipeline to 3D volumetric modalities (CT, MRI) requires 3D convolutional attention and volumetric wavelet codecs.
3. **Absence of Observer Studies**: While numerical metrics (PSNR > 48 dB, SSIM > 0.989) demonstrate high mathematical fidelity, formal Multi-Reader Multi-Case (MRMC) receiver operating characteristic (ROC) trials with board-certified radiologists are required to confirm zero diagnostic non-inferiority.

### 11.2 Engineering & Silicon Implementation Roadmap
1. **Physical ASIC Tape-Out**: Reported Gate Equivalent (GE) figures represent analytical estimates based on straight-line program XOR graphs. Future work involves RTL synthesis using TSMC 28nm standard-cell libraries.
2. **Learned Neural Codecs**: Exploring lightweight quantized neural autoencoders to replace JPEG DCT quantization.

---

## 12. Acknowledgements

The authors express their deepest gratitude to our faculty advisors at the **Indian Institute of Information Technology Raichur (IIIT Raichur)** for their invaluable guidance, domain expertise, and rigorous mentorship throughout the design and execution of this project:
- **Dr. Kaushiki Roy** (Assistant Professor, CSE) for expert mentorship in Digital Image Processing, spatial frequency analysis, and radiological fidelity evaluation.
- **Dr. Mehbub Alam** (Assistant Professor, CSE) for guidance in Computer Networks, transport layer protocol design, and wireless channel simulation.
- **Arnav Ghosh** for rigorous insights in multi-objective Pareto optimization, loss formulation mathematics, and algebraic diffusion structures.

We also thank the open-access medical imaging community and Cohen et al. (2020) for curating and releasing the COVID-19 Chest X-ray Dataset Initiative.

---

## References

1. World Health Organization, *Telemedicine: Opportunities and Developments in Member States*, Global Observatory for eHealth Series, vol. 2, Geneva: WHO Press, 2010.
2. J. P. Cohen, P. Morrison, and L. Dao, "COVID-19 Image Data Collection," *arXiv preprint arXiv:2003.11597*, 2020.
3. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.
4. A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 7482–7491.
5. M. Dworkin, "Recommendation for block cipher modes of operation: Galois/Counter Mode (GCM) and GMAC," *NIST Special Publication 800-38D*, National Institute of Standards and Technology, 2007.
6. C. Dobraunig, M. Eichlseder, F. Mendel, and M. Schläffer, "Ascon v1.2: Lightweight Authenticated Encryption and Hashing," *NIST Special Publication 800-232*, 2024.
7. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004.
8. P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," *arXiv preprint arXiv:1711.05225*, 2017.
9. M. Sajadieh, M. Wei, and E. Pasalic, "On construction of involutory MDS matrices," *IEEE Transactions on Information Theory*, vol. 58, no. 9, pp. 6157–6163, 2012.
10. C. Beierle, A. Canteaut, and G. Leander, "Lightweight invertible diffusion layers," *IACR Transactions on Symmetric Cryptology*, vol. 2018, no. 3, pp. 1–34, 2018.
11. V. Cauchois and P. Loidreau, "On the number of involutory MDS matrices over finite fields," *Designs, Codes and Cryptography*, vol. 87, no. 2, pp. 527–542, 2019.
12. D. Han, X. Wang, and B. Preneel, "Constructing lightweight MDS and near-MDS matrices," in *Information Security and Cryptology (Inscrypt 2016)*, LNCS vol. 10148, Springer, 2016, pp. 112–129.
