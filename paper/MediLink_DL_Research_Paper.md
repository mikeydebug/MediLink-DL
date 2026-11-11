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

Sending a chest X-ray from a rural health center in Chhattisgarh to a specialist radiologist 300 kilometers away should take seconds. On a 2G connection running at 48 kbps, it takes over an hour. That is not a network engineering statistic — it is a triage decision delayed long enough that the patient may deteriorate in the waiting room. We started MediLink-DL because we kept coming back to that number and could not find a satisfying answer to it in the existing literature.

The gap is not really about compression ratios or encryption algorithms in isolation. It is about the fact that existing approaches treat the medical image as a uniform blob of pixels, without asking which part of that image actually matters for the clinical decision being made right now. A chest radiograph in a COVID-19 triage context carries its entire diagnostic weight in a few square centimeters of lung parenchyma showing consolidation. The ribs, the cardiac silhouette, the dark background — none of that is what a remote radiologist needs to see in the first half-second. We built a pipeline that knows this.

The system we describe here starts with a modified ResNet-18 we call `MedicalResNet18`, trained on 166 real chest X-rays from 110 patients — split at the patient level to prevent the data leakage that inflates accuracy numbers in many published pipelines. The model uses a differentiable version of Grad-CAM (with `create_graph=True` to preserve second-order gradients during training) so that localization is a training objective, not a post-hoc visualization. We also implemented the Kendall & Gal (CVPR 2018) uncertainty-based loss weighting, which turned out to solve a real practical headache we had with manually tuning the balance between classification and localization objectives. The best of our four training configurations — Regime B — achieves 93.10% Pointing Game accuracy, 80.32% energy inside the radiologist-annotated ROI, and 100% recall on the test set.

Once the model gives us the ROI bounding box, the compression strategy becomes straightforward: compress the diagnostic patch at JPEG quality 90, the background at quality 25, and package everything into a 28-byte custom binary container with a `'MLDL'` magic header. The result is 26,515 bytes for a 512×512 radiograph that starts as 262 KB — a 9.89× compression ratio. The ROI comes out at 48.31 dB PSNR and 0.9891 SSIM. At the same byte budget, uniform JPEG only manages 38.18 dB and 0.9312 SSIM on the ROI — a 10.13 dB gap that corresponds to visibly sharper lung textures.

For security, we implemented and benchmarked AES-256-GCM (2.2 µs, 465 MB/s, 100% tamper detection) alongside Ascon-AEAD128, ratified by NIST in SP 800-232 as the new lightweight cryptography standard, and an in-house research cipher we built to explore involutory MDS diffusion matrices. On the networking side, priority scheduling cuts time-to-first-ROI from 2.430 seconds to 0.391 seconds on a 64 kbps rural uplink — a 6.21× triage acceleration without sending a single extra byte. A grid search over 108 pipeline configurations identified 26 non-dominated Pareto-optimal operating points. All 147 tests pass. The live dashboard is at [https://medilink-dl.onrender.com](https://medilink-dl.onrender.com).

**Keywords**: Telemedicine, Digital Image Processing, Differentiable Grad-CAM, Kendall Uncertainty Weighting, Patient-Level Stratification, Authenticated Encryption (AEAD), NIST SP 800-232, Priority Packet Scheduling, Pareto Optimization.

---

## 1. Introduction & Clinical Motivation

Digital radiography is often the first and sometimes only imaging modality available at primary health centers (PHCs) in India. It provides the front line of detection for pulmonary consolidations, pneumothorax, tuberculosis, and viral pneumonias [1]. In a well-resourced urban hospital with gigabit fiber and an on-site PACS server, a radiograph reaches a specialist within seconds of acquisition. But the moment you step outside that infrastructure, everything falls apart in ways that are individually manageable but collectively paralyzing.

The first problem is bandwidth. Rural PHCs in states like Rajasthan, Chhattisgarh, and Jharkhand typically operate on wireless uplinks throttled to 2G/EDGE speeds — 32 to 64 kbps on a good day, with occasional 3G (100–256 kbps) when the weather cooperates and the tower is not congested. A 512×512 uncompressed chest X-ray is roughly 262 KB. A full DICOM study can run anywhere from 10 to 35 MB. At 64 kbps, that is 33 seconds to 73 minutes per study. Any triage decision requiring remote specialist review is essentially impossible under those latencies. The obvious workaround — just compress the image heavily — brings in the second problem.

Standard JPEG compression applies the same DCT quantization across every 8×8 pixel block in the image, regardless of whether that block is a lung lesion or an empty background corner. When you push the quality down far enough to fit into the bandwidth budget, the blocks that matter most — the ones containing subtle opacity boundaries and early consolidation — get blurred out first, because that high-frequency detail is exactly what aggressive JPEG quantization discards. We also checked what happens in practice when radiographs are shared over consumer apps: WhatsApp strips DICOM metadata, applies uncontrolled re-compression, and has no authenticated encryption whatsoever. That leads directly to the third problem.

Transmitting patient radiographs over public cellular channels without cryptographic protection is not a theoretical concern — it is a legal and clinical one. India's emerging health data privacy framework, along with HIPAA and GDPR compliance for institutions with international partnerships, all require that patient imaging data in transit be both confidential and tamper-proof. Tamper detection matters especially in this context: an attacker who can flip bits in a radiograph in transit without the receiver knowing has the ability to cause diagnostic errors downstream, and the receiver has no way to know the image was modified.

We started building MediLink-DL because we wanted to address all three of these problems as a connected system rather than three independent engineering tasks. The design is organized around three research areas that we worked on with our faculty advisors: image processing and compression (Dr. Kaushiki Roy), network transport (Dr. Mehbub Alam), and multi-objective optimization and the mathematical side of the loss formulations (Arnav Ghosh). What we ended up with is a tightly integrated pipeline where the deep learning model's output directly drives the compression strategy, and the compression output directly determines what gets transmitted first. The rest of this paper describes how that pipeline works and what we measured.

---

## 2. Clinical Dataset Curation & Leakage-Free Stratification

### 2.1 Clinical Dataset Specifications

We wanted to work with real clinical radiographs rather than synthetic phantoms, so we built our cohort from the **COVID-19 Chest X-ray Dataset Initiative** (Cohen et al., 2020) [2]. This dataset contains genuine PA/AP view radiographs of patients with COVID-19, SARS, and acute respiratory distress, alongside bacterial and normal control studies. Ground-truth bilateral thoracic and focal consolidation bounding boxes were taken from expert radiologist annotations provided by `imageannotation.ai`.

After filtering, our verified cohort consists of 166 clinical radiographs across 110 unique patients, all resampled to standardized $512 \times 512$ single-channel 8-bit grayscale. The task is binary: COVID-19/SARS-positive versus bacterial/normal.

### 2.2 Mathematical Leakage-Free Patient Stratification Protocol

This was one of the first things we got wrong, and then fixed. The subtle problem with splitting radiographs randomly across train, validation, and test sets is that the same patient often ends up in multiple splits — not the same scan, but different scans of the same chest. Deep CNNs pick up patient-specific anatomy with surprising reliability: rib curvature, cardiac silhouette shape, the position and density of implant markers, even the characteristic noise pattern of a particular X-ray machine. If a patient's first scan is in the training set and their second is in the test set, the model sees that anatomy twice and test accuracy inflates artificially. The model learns to recognize the patient, not the pathology. This problem is well-documented in the medical imaging literature, but it still appears quietly in many published pipelines.

We enforced strict patient-level stratification from the start. Let $\mathcal{U} = \{p_1, p_2, \ldots, p_{110}\}$ be the set of unique patients. The partition is:

- **Training Partition ($\mathcal{P}_{\text{train}}$)**: 77 patients (111 radiographs)
- **Validation Partition ($\mathcal{P}_{\text{val}}$)**: 16 patients (26 radiographs)
- **Test Partition ($\mathcal{P}_{\text{test}}$)**: 17 patients (29 radiographs)

The non-overlap condition is:

$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$

This constraint is verified programmatically in the test harness (`tests/test_real_dataset.py::test_patient_split_leakage_zero`). We checked it on every experimental run. We did not want to discover a leakage bug after spending training compute on a corrupt split.

---

## 3. Deep Learning Localization & Uncertainty Weighting

### 3.1 Network Architecture

We adapted the standard ResNet-18 backbone into `MedicalResNet18` specifically for single-channel grayscale radiographs. The changes are minimal but necessary: the first convolutional layer (`conv1`) is modified from 3 input channels to 1, keeping the $7 \times 7$ kernel, stride 2, and padding 3. The four residual convolutional blocks (`layer1` through `layer4`) remain intact, producing an activation map $A \in \mathbb{R}^{512 \times 7 \times 7}$ at `layer4[-1]`. Global average pooling reduces this to a 512-dimensional vector, and a linear classification head outputs binary disease logits $\hat{y} \in \mathbb{R}^2$.

We chose ResNet-18 over heavier architectures deliberately. The pipeline runs inference on PHC-grade hardware, and we wanted a model that could be realistically deployed without a GPU. The 18-layer version is aggressive enough to pick up pulmonary features but light enough that the full end-to-end system stays practical on constrained edge devices.

### 3.2 Differentiable Grad-CAM Formulation

We needed the saliency map to actually affect training, not just visualize after the fact. That distinction is what separates our approach from standard Grad-CAM [3].

Standard Grad-CAM detaches the backward graph — it is a post-hoc visualization tool, not a training signal. We noticed early in our experiments that if we just used vanilla Grad-CAM overlays as a diagnostic check, the model would sometimes classify correctly while focusing on completely wrong regions: text annotations burned into the image margin, medical equipment edges, or the clavicle instead of the lung parenchyma. We call this saliency drift. It is a genuine clinical problem because a system that localizes poorly provides no useful guidance to a remote nurse about where to look on the reconstructed image.

The fix is `DifferentiableGradCAM` with `create_graph=True`, which preserves the backward computational graph so that second-order gradients can flow through the CAM computation during training. The formulation:

1. **Neuron Importance Weights**: Let $y^c$ be the classification score for target class $c$. The gradient with respect to feature map $A^k$ of layer $L$ is spatially pooled:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. **Heatmap Assembly**:
   $$H_{\text{CAM}}^c = \operatorname{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right)$$
3. **Bilinear Upsampling & Normalization**: $H_{\text{CAM}}^c$ is bilinearly interpolated to $(H, W) = (512, 512)$ and normalized:
   $$M_{\text{CAM}} = \frac{H_{\text{CAM}}^c - \min(H_{\text{CAM}}^c)}{\max(H_{\text{CAM}}^c) - \min(H_{\text{CAM}}^c) + \epsilon}$$

### 3.3 Multi-Task Loss Formulation

To train the model to simultaneously classify and localize, we combined a cross-entropy classification term with spatial alignment penalties using ground-truth bounding box masks:

$$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{CE}}(y, \hat{y})$$
$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum M_{\text{CAM}} M_{\text{GT}} + \epsilon}{\sum M_{\text{CAM}}^2 + \sum M_{\text{GT}}^2 + \epsilon}$$
$$\mathcal{L}_{\text{align}} = 1 - \frac{\langle \operatorname{vec}(M_{\text{CAM}}), \operatorname{vec}(M_{\text{GT}}) \rangle}{\|\operatorname{vec}(M_{\text{CAM}})\|_2 \|\operatorname{vec}(M_{\text{GT}})\|_2}$$
$$\mathcal{L}_{\text{loc}} = (1 - \beta)\mathcal{L}_{\text{align}} + \beta \mathcal{L}_{\text{dice}}$$

The alignment loss pushes the CAM direction toward the ground-truth mask in vector space; the Dice loss penalizes spatial overlap failure. Together they form the localization objective.

### 3.4 Kendall & Gal (CVPR 2018) Homoscedastic Uncertainty Optimization

The hardest part of multi-task training is not writing the loss — it is finding the right $\lambda$ to weight the two losses against each other. We spent an afternoon manually sweeping hyperparameters and found the results were fragile and inconsistent. Either classification dominated and localization stayed poor, or localization gradients overwhelmed the classification signal. Neither extreme produced a model that was actually useful for the downstream ROI crop.

We implemented the Kendall & Gal (CVPR 2018) [4] approach, which treats task weighting as maximum likelihood inference over homoscedastic (task-level, input-independent) noise parameters. The intuition is elegant: instead of manually deciding how much to weight localization against classification, let the network learn both tasks and their relative uncertainties simultaneously. The combined loss becomes:

$$\mathcal{L}_{\text{total}}(w, \sigma_1, \sigma_2) = \frac{1}{2\sigma_1^2} \mathcal{L}_{\text{cls}}(w) + \frac{1}{2\sigma_2^2} \mathcal{L}_{\text{loc}}(w) + \log \sigma_1 + \log \sigma_2$$

We reparameterize as $s_1 = \log \sigma_1^2$ and $s_2 = \log \sigma_2^2$ for numerical stability:

$$\mathcal{L}_{\text{total}}(w, s_1, s_2) = \frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}}(w) + \frac{1}{2}\exp(-s_2)\mathcal{L}_{\text{loc}}(w) + \frac{1}{2}s_1 + \frac{1}{2}s_2$$

Both $s_1$ and $s_2$ are unconstrained learnable parameters optimized together with the network weights via AdamW. The optimizer naturally finds the right balance over training without us imposing it. We implemented this as `AdaptiveLocalizationAwareLoss`, and it noticeably stabilized training curves compared to the fixed-weight baselines.

---

## 4. Digital Image Processing: Dual-Stream ROI-Aware Compression

### 4.1 Dual-Stream Encoding Architecture

The core insight is straightforward: JPEG treats every 8×8 block of pixels identically. It has no idea which part of the image is a lung lesion and which is empty background. We do. So we stop pretending that uniform compression is the right tool here.

We decompose the radiograph into two streams based on what the deep learning model tells us about diagnostic importance. The background — patient table, chest wall periphery, image borders — is contextually useful but not acutely diagnostic. The lung parenchyma where consolidations appear is where the clinical decision lives. Standard JPEG ignores this distinction and applies a uniform DCT quantization matrix everywhere, which means aggressive compression degrades the parts that matter most.

1. **Background Full-Frame Stream**: The full radiograph $I \in \mathbb{R}^{512 \times 512}$ is JPEG-compressed at low quality:
   $$\mathcal{S}_{\text{bg}} = \text{JPEG}_{\text{enc}}(I, q_{\text{bg}})$$
   with $q_{\text{bg}} = 25$.

2. **Diagnostic ROI Patch Stream**: The Grad-CAM heatmap $M_{\text{CAM}}$ is thresholded at $\tau = 0.35$ to yield a binary mask $M_{\text{bin}} = (M_{\text{CAM}} \ge \tau)$. We find the tight bounding box $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ of active pixels and crop the diagnostic patch $I_{\text{crop}}$. This patch is encoded at high quality:
   $$\mathcal{S}_{\text{roi}} = \text{JPEG}_{\text{enc}}(I_{\text{crop}}, q_{\text{roi}})$$
   with $q_{\text{roi}} = 90$.

### 4.2 28-Byte Binary Container Header (`'MLDL'`)

Both streams are packaged into a self-describing big-endian binary container with a fixed 28-byte header:

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

The magic bytes serve dual purpose: quick format validation and a recognizable marker for any network proxy or logging middleware that needs to identify MediLink-DL traffic.

### 4.3 Receiver Reassembly & Alpha Blending

At the receiver, both streams decode independently. The high-quality ROI patch composites onto the background canvas using distance-transform alpha blending ($\alpha = 0.45$) across a 4-pixel perimeter, which smooths the boundary seam so the reconstructed image does not show a visible tile edge.

### 4.4 Automated Rate-Matched Baseline Search

To compare fairly against standard JPEG, we did not want to pick a competitor quality setting by hand — that would bias the comparison in our favour. We implemented a binary search that finds the exact uniform quality $q^*$ matching our dual-stream byte budget:

$$q^* = \arg\min_{q \in [1, 100]} \left| \operatorname{len}(\text{JPEG}(I, q)) - \operatorname{len}(\mathcal{P}_{\text{dual}}) \right|$$

This gives us a genuinely rate-matched baseline, so the compression quality numbers are comparable at equal transmission cost.

---

## 5. Partitioned Radiological Distortion Metrics

One thing we learned from the image processing literature fairly early is that global PSNR and SSIM scores can be misleading for medical images. A system could score well globally by perfectly reconstructing the dark background while blurring the lung lesion entirely. Clinically, that is the worst possible trade-off — the background does not change whether the patient gets diagnosed.

We compute all metrics partitioned over the ROI and background separately. Let $\Omega_{\text{roi}} = \{(i,j) \mid M_{\text{bin}}(i,j) = 1\}$ and $\Omega_{\text{bg}} = \{(i,j) \mid M_{\text{bin}}(i,j) = 0\}$:

1. **Partitioned MSE**:
   $$\text{MSE}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (I(i,j) - \hat{I}(i,j))^2$$

2. **Partitioned PSNR**:
   $$\text{PSNR}_{\Omega} = 10 \log_{10}\left( \frac{255^2}{\text{MSE}_{\Omega}} \right) \quad (\text{dB})$$

3. **Partitioned SSIM** [7]:
   $$\text{SSIM}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} S(i,j)$$
   where $S(i,j)$ is the local SSIM map computed using an $11 \times 11$ circular Gaussian filter ($\sigma = 1.5$).

Reporting both partitioned and global metrics gives a complete picture: the ROI numbers tell you about diagnostic quality, the global numbers tell you about perceptual reconstruction of the full image.

---

## 6. Authenticated Encryption Layer (AEAD) & Security Verification

### 6.1 Cryptographic Protocol Specifications

Sending compressed patient radiographs over a rural cellular uplink without encryption is not acceptable, and not just for legal reasons. Unprotected medical telemetry is vulnerable to active packet tampering — an attacker can modify image data in transit without the receiver knowing. That is a clinical safety problem, not just a privacy one. We built the cryptographic layer around authenticated encryption with associated data (AEAD), which provides both confidentiality and tamper detection in a single pass.

We implemented and benchmarked three schemes:

1. **AES-256-GCM (NIST SP 800-38D)** [5]: The industry standard. 256-bit CSPRNG key, 96-bit unique counter nonce ($N \leftarrow \text{CSPRNG}(12)$), 128-bit GHASH authenticator over $\mathbb{F}_{2^{128}}$.

2. **Ascon-AEAD128 (NIST SP 800-232)** [6]: Ratified in 2024 as the NIST Lightweight Cryptography Standard, specifically designed for constrained IoT and edge environments. 320-bit permutation state, 128-bit key, 128-bit nonce, 128-bit tag.

3. **MLMC-128 (MediLink Mini Cipher)**: Our in-house research prototype — a 128-bit block cipher built from PRESENT S-boxes and a circulant MDS diffusion matrix ($\mathcal{B} = 5$), operated in CTR mode with GHASH authentication. This is an academic exploration, not a production recommendation.

### 6.2 Associated Data (AAD) Binding & Zero-Plaintext Release

To prevent replay and cross-patient injection attacks, hospital metadata binds into the authentication tag as associated data:

$$\text{AAD} = \text{StudyUID} \,\|\, \text{PatientID} \,\|\, \text{Timestamp} \,\|\, \text{HeaderChecksum}$$

The receiver performs constant-time tag verification before releasing any decoded image buffer. If even a single bit in the ciphertext or envelope is flipped, decryption raises an unrecoverable exception and purges memory — no partial plaintext is ever exposed to the application layer. We verified 100% bit-flip tamper detection across all three AEAD implementations in the test suite.

---

## 7. Prioritized Computer Networks Transport & Scheduling

### 7.1 Packet Framing Protocol

We fragment the encrypted dual-stream container into fixed-size chunks with $L \in \{512, 1024, 2048\}$ bytes per chunk. Each packet carries a structured header:

`[4B Magic: 'MLPK'] [2B SeqID] [2B TotalChunks] [1B PriorityTier] [2B PayloadLen] [32B SHA-256 Checksum]`

The SHA-256 per-chunk checksum is separate from the AEAD tag — it allows the transport layer to quickly identify corrupted chunks and trigger selective-repeat ARQ without waiting for full decryption.

### 7.2 Priority Min-Heap Scheduling

This is where the network piece connects back to the DL output. Under standard FIFO scheduling, packets transmit in the order they are enqueued — background chunks typically go out first because the full background image is encoded first. That means the diagnostically important ROI arrives last, which is exactly backwards for triage. A nurse waiting for guidance on where to look at a radiograph should not have to wait for all the background pixels to arrive before the lung patch appears.

We use an application-layer priority min-heap with three tiers:

$$\text{Priority}(\text{Header}) = 30 > \text{Priority}(\text{ROI Chunks}) = 20 > \text{Priority}(\text{Background Chunks}) = 5$$

The header transmits first (carrying bounding box coordinates), so the receiver knows where the ROI will land. ROI chunks follow immediately. Background fills in afterward. The receiving clinician sees the high-quality diagnostic region appear in the first burst of packets, well before the full image has transmitted.

### 7.3 Channel Model & Selective-Repeat ARQ Simulation

We modeled the rural cellular channel in a discrete-event simulator:

$$\Delta t_{\text{chunk}} = \frac{(L + 64) \times 8}{R} + \frac{\text{RTT}}{2} + \xi_{\text{jitter}}$$

where $R$ is link bandwidth (48–128 kbps), RTT is round-trip propagation delay (120–500 ms), and $\xi$ is Gaussian channel jitter. We simulated packet drop rates between 2% and 10%, with dropped packets triggering selective-repeat ARQ retransmissions. The simulator is straightforward but parameterized carefully to match the channel conditions documented in the rural connectivity literature.

---

## 8. Multi-Objective Optimization & Pareto Analysis

The pipeline has many configurable parameters: CAM threshold, ROI quality, background quality, cipher choice, scheduler type, uplink bandwidth. No single combination is universally optimal — different rural deployments have different bandwidth budgets, hardware constraints, and latency tolerances. Rather than picking one operating point and declaring victory, we ran a grid search across 108 end-to-end configurations:

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

Across the full 108-configuration search, **26 non-dominated Pareto configurations** emerged. These represent the global efficiency frontier — no other configuration in the search space beats any of these 26 on all four objectives simultaneously. A deployment engineer choosing between latency and image quality can find the right operating point within this frontier rather than making arbitrary guesses about which settings to use.

---

## 9. Empirical Results & Performance Evaluation

### 9.1 Deep Learning Localization Evaluation (Held-Out Test Cohort)

We evaluated all training regimes on the 29 test radiographs from our 17 held-out patients (`results/real_dataset_training_results.csv`). The four regimes correspond to different loss configurations: classification-only, the combined multi-task loss (Regime B), alignment-only, and Dice-only.

| Regime / Objective Mode | Test Accuracy | Test F1 | Test Recall | Test ROI Dice | Test ROI IoU | Pointing Game Hit Rate | Energy in ROI | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regime A: Classification-Only** | 0.6897 | 0.8163 | 1.0000 | 0.4764 | 0.3571 | 65.52% | 49.21% | 27.05 s |
| **Regime B: Combined Loss (Proposed)** | **0.6897** | **0.8163** | **1.0000** | **0.6856** | **0.5395** | **93.10%** | **80.32%** | **28.03 s** |
| **Regime C: Alignment-Only** | 0.6897 | 0.8163 | 1.0000 | 0.5799 | 0.4243 | **96.55%** | **82.22%** | 27.24 s |
| **Regime D: Dice-Only** | **0.7586** | **0.8511** | **1.0000** | **0.7107** | **0.5704** | 82.76% | 69.36% | 26.88 s |

The result that surprised us most was the Regime D vs. Regime B comparison. Regime D achieves the best continuous Dice score (0.7107) and even slightly better accuracy. But its Energy in ROI drops to 69.36% and its Pointing Game falls to 82.76%. In practice, what that means is that the model focuses its saliency mass on a broad region that overlaps the ground truth but bleeds well outside the consolidation boundary. For ROI-guided compression, that is a problem: a loose bounding box inflates the ROI patch size and drives up transmission cost without benefit.

Regime B's tighter spatial confinement — 93.10% Pointing Game, 80.32% energy inside the ROI — means the extracted bounding box is a much more precise crop. We chose it as the operational configuration because precision matters downstream in the compression and transmission pipeline. All test regimes maintain 100% sensitivity (zero false negatives on COVID-19/SARS cases), which is the hard constraint for a triage tool.

### 9.2 Rate-Distortion & Compression Performance on Real CXRs

Evaluated on a genuine 512×512 chest radiograph (262,144 bytes uncompressed):

| Compression Scheme | Parameter Setting | Transmitted Bytes | Data Reduction | Compression Ratio | ROI PSNR | ROI SSIM | Global PSNR | Global SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Stream ROI-Aware** | $q_{\text{roi}}=90, q_{\text{bg}}=25$ | **26,515 B** | **89.89%** | **9.89x** | **48.31 dB** | **0.9891** | 42.10 dB | 0.9634 |
| **Rate-Matched Uniform JPEG** | $q=32$ (Matched Bytes) | 26,035 B | 90.07% | 10.07x | 38.18 dB | 0.9312 | 39.45 dB | 0.9410 |
| **High-Quality Uniform JPEG** | $q=90$ (High Quality) | 98,412 B | 62.46% | 2.66x | 48.31 dB | 0.9897 | 49.11 dB | 0.9882 |

At the same wire budget (~26.5 KB), our dual-stream approach delivers **+10.13 dB gain in ROI PSNR** and **+0.0579 in ROI SSIM** over uniform JPEG. The high-quality uniform JPEG baseline shows what is achievable at $q=90$ — essentially identical ROI quality to our system — but it takes 98.4 KB to get there, which is 3.7× more data over the wire. On a 64 kbps rural link, that difference is about 10 extra seconds of transmission time per image. Per patient, every day, that adds up.

### 9.3 Cryptographic Latency & Throughput Benchmark

Evaluated across 1024-byte payload chunks on an Apple Silicon M-series host:

| Cryptographic Cipher | Architecture / Standard | Encrypt Latency ($\mu$s) | Decrypt Latency ($\mu$s) | Wire Throughput | Bit-Flip Tamper Detection |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AES-256-GCM** | NIST SP 800-38D (ARMv8 Crypto) | **2.2 $\mu$s** | **2.4 $\mu$s** | **465.4 MB/s** | **100.0% Detected** |
| **Ascon-AEAD128** | NIST SP 800-232 (IoT Standard) | **412.0 $\mu$s** | **418.0 $\mu$s** | **2.48 MB/s** | **100.0% Detected** |
| **MLMC-128 (Raw Block)** | Research Prototype (Pure Python) | **2,140.0 $\mu$s** | **2,190.0 $\mu$s** | **0.47 MB/s** | **N/A (Block Only)** |
| **MLMC-128 (AEAD Exp)** | Research Candidate (CTR + GHASH) | **7,889.0 $\mu$s** | **7,920.0 $\mu$s** | **0.13 MB/s** | **100.0% Detected** |

An important sanity check on these numbers: on a 64 kbps rural wireless channel, transmitting 1024 bytes over the air takes 128,000 µs (128 ms). Even the slowest of our AEAD implementations — the pure-Python MLMC-128 AEAD at 7.8 ms — contributes only 6.1% of total latency. Production AES-256-GCM at 2.2 µs is essentially free relative to channel airtime. The bottleneck in rural telemedicine is the wireless channel, not cryptography, and that matters for how you prioritize engineering effort.

The 412 µs Ascon latency might look concerning against AES's 2.2 µs, but Ascon targets microcontrollers without hardware AES acceleration — the realistic comparison is Ascon on a constrained edge device versus AES-256-GCM on the same constrained device, where the advantage narrows considerably.

### 9.4 Prioritized Transmission Across Rural Channels

| Channel Link Profile | Bandwidth | Latency (RTT) | Loss Rate | Time-to-First-ROI (Priority) | Time-to-First-ROI (FIFO) | Triage Acceleration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rural 2G/EDGE** | 48 kbps | 200 ms | 2.0% | **0.592 s** | 3.592 s | **6.07x** |
| **Rural 2.5G/GPRS** | 64 kbps | 120 ms | 5.0% | **0.391 s** | 2.430 s | **6.21x** |
| **Rural 3G (Weak)** | 128 kbps | 150 ms | 5.0% | **0.372 s** | 2.097 s | **5.63x** |
| **Satellite VSAT** | 256 kbps | 500 ms | 2.0% | **1.036 s** | 5.299 s | **5.11x** |

The 6.21× triage acceleration on the 64 kbps GPRS link (0.391 s vs. 2.430 s time-to-first-ROI) is the number we keep coming back to. The improvement is not from sending less data — priority scheduling transmits the same total bytes as FIFO. It is purely from reordering delivery so that the diagnostically important content arrives first. A remote specialist who needs to make a triage call can see the lung ROI in under half a second rather than waiting over two seconds for background context to clear first. At scale, across a PHC seeing dozens of patients a day on a shared uplink, that reorders into real clinical time savings.

---

## 10. Cryptographic Research Laboratory: Involutory Diffusion & Impossibility Proofs

Alongside the applied pipeline work, we ran a parallel investigation into lightweight linear diffusion matrices for the MLMC-128 block cipher (`src/crypto_research/`). The motivation was finding matrix structures that are simultaneously MDS (maximizing algebraic branch number) and involutory ($M^2 = I$), which eliminates the need for a separate inverse matrix during decryption and halves hardware implementation cost — important for the microcontroller deployments we have in mind.

### 10.1 Matrix Architectures We Investigated

1. **`PROPOSED-OIN-MDS`**: Optimal-parameter Hadamard instantiation $\operatorname{Had}(1, 8, 13, 5)$ over $\mathbb{F}_{2^4}/(x^4+x+1)$:
   - Involutory: $M^2 = I$ verified.
   - Branch Numbers: $\mathcal{B}_{\text{nibble}} = 5$ (Optimal MDS), $\mathcal{B}_{\text{bit}} = 8$ (Maximal).
   - Implementation: 64 direct XORs → **48 Paar CSE XORs** (~96.0 GE).

2. **`ABC-MDS-OPT`**: Alternating Block-Circulant matrix with parameters $(0, 1, 2, 3)$:
   - Branch Numbers: $\mathcal{B}_{\text{nibble}} = 5$ (Optimal MDS), $\mathcal{B}_{\text{bit}} = 6$.
   - Complexity: 64 direct XORs → **49 Paar CSE XORs** (~98.0 GE).

3. **`TBOI-16`**: Tensor-Product Binary Orthogonal Involution ($K_4 \otimes K_4$) over $\mathbb{F}_2$:
   - Branch Numbers: $\mathcal{B}_{\text{nibble}} = 4$ (Near-MDS), $\mathcal{B}_{\text{bit}} = 6$.
   - Complexity: 128 direct XORs → **60 Paar CSE XORs** (~120.0 GE).

The 49.92% SAC across 4,800 trials for MLMC-128 confirms near-ideal avalanche behavior — essentially indistinguishable from the theoretical ideal of 50%.

### 10.2 Formal Impossibility Results

Part of what made this research exercise interesting was proving what cannot work, not just what does. We derived four impossibility results:

1. **Theorem 1 (Hankel Non-Existence)**: *No $4 \times 4$ Hankel matrix over $\mathbb{F}_{2^m}$ can be simultaneously MDS and involutory.*

2. **Proposition 2 (Cauchy Involution Collapse)**: *Symmetric Cauchy matrices over $\mathbb{F}_{2^4}$ satisfying self-inversion degenerate into singular or non-MDS structures.*

3. **Theorem 3 (Galois Matrix Symmetry Obstruction)**: *For elements $\alpha, \beta \in \mathbb{F}_{2^4}$, the binary representation matrices satisfy $L_\alpha = L_\beta^T$ if and only if $\alpha = \beta \in \{0, 1\}$.*

4. **Theorem 4 (Tensor Bound)**: *The Kronecker product $A \otimes B$ of two invertible $2 \times 2$ matrices cannot achieve optimal MDS branch number $\mathcal{B}=5$; it is strictly bounded by $\mathcal{B} \le 4$.*

These results helped us stop exploring unpromising directions (Hankel and symmetric Cauchy families) and focus on Hadamard-type constructions, which is where `PROPOSED-OIN-MDS` came from [9, 10, 11, 12].

---

## 11. Scientific Discussion, Clinical Limitations & Future Work

We want to be honest about where this work stands. The results are encouraging and we think the system design is sound, but there is a meaningful gap between an academic prototype and something that could responsibly be deployed in a clinical setting.

The most immediate limitation is cohort size. 166 radiographs across 110 patients is enough to demonstrate the concept and produce meaningful experimental results. But it is nowhere near sufficient for regulatory clearance — formal evaluation under a framework like FDA 510(k), EU CE-MDR, or India's CDSCO would require prospective multi-center validation across 20,000+ studies, diverse radiological hardware, and multiple disease classes beyond COVID-19 and SARS. Our test accuracy numbers (around 69–75% depending on regime) also reflect the difficulty of generalizing on this small a dataset. We are not claiming clinical-grade classification performance.

The 2D chest radiograph is a relatively forgiving modality for this kind of pipeline. Extending the approach to 3D CT or MRI studies would require substantially different architectural choices: 3D convolutional attention mechanisms, volumetric saliency maps, and codecs that handle volumetric data efficiently. The current dual-stream approach does not translate directly to volumetric data.

We also have no observer study data. ROI PSNR of 48.31 dB and SSIM of 0.9891 are strong by computational metrics standards. But whether the reconstructed images are diagnostically non-inferior to uncompressed originals requires a Multi-Reader Multi-Case (MRMC) trial with board-certified radiologists. That is future work we would genuinely like to see done — the numbers look good, but human readers are the actual ground truth here.

On the engineering side, the GE figures for `PROPOSED-OIN-MDS` (~96 GE) are analytical estimates derived from XOR gate counts in the straight-line program. Actual silicon area after ASIC synthesis with TSMC 28nm standard-cell libraries, with routing and timing closure, will differ. The path from "48 Paar CSE XORs" to a taped-out chip has many steps we have not taken yet.

One direction we are curious about is replacing the JPEG codec entirely with a lightweight learned neural autoencoder. The DCT-based approach works well but fixes the compression trade-off at design time. A trained codec could potentially adapt the rate-distortion curve to the specific pathology being transmitted. That said, any neural codec running on PHC-grade embedded hardware would need aggressive quantization and probably custom silicon support, which circles back to the ASIC roadmap.

---

## 12. Acknowledgements

We owe a lot to our faculty advisors at IIIT Raichur. **Dr. Kaushiki Roy** guided us through the image processing side — the partitioned metric design, spatial frequency analysis, and the nuances of radiological fidelity evaluation. She also pushed us to think more carefully about why global metrics can mislead in medical imaging contexts, which shaped how we report results throughout this paper. **Dr. Mehbub Alam** kept us grounded on the networking reality: what rural cellular channels actually look like, how ARQ protocols behave under sustained loss, and how to build a channel simulator that reflects real conditions rather than optimistic assumptions. More than once, he pointed out that we were benchmarking against an idealized channel that does not exist in Rajasthan. **Arnav Ghosh** worked through the multi-objective optimization formulation with us and pushed us on the mathematical rigor of the diffusion matrix proofs — especially the impossibility results, which became much cleaner after those conversations.

We also thank Cohen et al. (2020) and the broader medical imaging community for maintaining the COVID-19 Chest X-ray Dataset Initiative as an open-access resource. Without real clinical data, this project would have been a simulation exercise in a vacuum.

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
