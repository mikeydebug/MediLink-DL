# MediLink-DL: Experimental Methodology & Scientific Evaluation Protocols

**Document**: Comprehensive Experimental Design, Dataset Stratification, and Evaluation Protocols  
**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
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
**Date**: November 2026 &nbsp;|&nbsp; **Scientific Research Methodology**

---

## 1. Research Questions & Hypotheses

| ID | Research Question | Formal Hypothesis | Success Criterion |
| :--- | :--- | :--- | :--- |
| **RQ1** | Can non-uniform dual-stream bit allocation yield a measurable increase in regional ROI PSNR and SSIM compared to a uniform JPEG baseline at identical byte budgets? | Allocating high quality ($q_{\text{roi}}=90$) to the diagnostic zone while aggressively compressing background context ($q_{\text{bg}}=25$) improves diagnostic ROI PSNR by $\ge 3.0$ dB and ROI SSIM by $\ge 0.05$ at rate-matched byte budgets ($\pm 5\%$). | $\Delta \text{PSNR}_{\text{roi}} \ge 3.0\text{ dB}$, $\Delta \text{SSIM}_{\text{roi}} \ge 0.05$ (Achieved: **+10.13 dB gain**) |
| **RQ2** | Does application-layer priority packet scheduling significantly accelerate clinical triage availability over lossy rural wireless links? | Prioritizing ROI packets ahead of anatomical background packets reduces time-to-first-ROI delivery by $\ge 60\%$ across simulated 48–128 kbps wireless channels compared to standard FIFO transport. | $\text{Acceleration} \ge 2.5\text{x}$ (Achieved: **>6.2x acceleration**) |
| **RQ3** | Does authenticated envelope encryption (AEAD) introduce prohibitive computational latency on CPU-constrained edge hardware? | Modern AEAD encryption (AES-256-GCM / Ascon-AEAD128) imposes $<1.0\text{ ms}$ processing overhead per image with 100% detection of corrupted ciphertexts under bit-flip attacks. | Latency $<1.0\text{ ms}$, Tamper detection = 100% (Achieved: **2.2 $\mu$s AES, 100% detection**) |
| **RQ4** | Can non-Hadamard involutory MDS diffusion matrices achieve branch number $\mathcal{B}=5$ with lower implementation complexity than known Hadamard baselines? | Alternating Block-Circulant structures can achieve optimal branch number $\mathcal{B}_{\text{nibble}}=5$ and competitive bit branch number while satisfying $M^2=I$ over $\mathbb{F}_{2^4}$. | $\mathcal{B}_{\text{nibble}}=5$, $M^2=I$, Paar XOR $\le 50$ (Achieved: **48 & 49 XORs**) |

---

## 2. Clinical Dataset Curation & Patient-Level Leakage Prevention

### 2.1 Clinical Dataset Source
- **Origin**: Open-access **COVID-19 Chest X-ray Dataset Initiative** (Cohen et al., 2020), combined with expert thoracic annotations provided by `imageannotation.ai`.
- **Modality**: Clinical Digital Radiography (CR/DR) and Computed Radiography chest X-rays in anterior-posterior (AP) and posterior-anterior (PA) projections.
- **Annotations**: Ground-truth bounding boxes designating bilateral pulmonary fields and focal opacities / consolidations.
- **Total Validated Cohort**: 166 clinical digital radiographs across 110 unique patients.

### 2.2 Patient-Level Stratification Protocol
In medical deep learning literature, random image-level splitting causes severe data leakage: multiple radiographs from the same patient appear in both training and evaluation subsets, artificially inflating generalization accuracy. MediLink-DL enforces **strict patient-level stratification**:
- **Unique Patients**: 110
- **Partition Ratios**: ~70% Training / ~15% Validation / ~15% Test
- **Partition Counts**:
  - **Training Set**: 77 patients (111 radiographs)
  - **Validation Set**: 16 patients (26 radiographs)
  - **Test Set**: 17 patients (29 radiographs)
- **Mathematical Non-Overlap Verification**:
  $$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$
  Verified programmatically by `tests/test_real_dataset.py::test_patient_split_leakage_zero`.

---

## 3. Deep Learning Training Protocols & Experimental Regimes

### 3.1 Neural Network Backbones
1. `MedicalResNet18`: Modified ResNet-18 with initial convolution adapted from 3 channels to 1 single-channel grayscale raster ($7 \times 7$, stride 2, padding 3); adaptive average pooling; linear classifier with 2 output logits (COVID-19 vs. Normal).
2. `MedicalDenseNet121`: Modified 121-layer DenseNet for architectural comparative validation.

### 3.2 Four Experimental Training Regimes
Models were trained for 30 epochs on Apple Silicon (MPS) using AdamW optimizer ($\text{lr} = 10^{-4}$, weight decay $= 10^{-5}$, batch size $= 8$):
- **Regime A (Classification-Only Baseline)**:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}}(y, \hat{y})$$
  Evaluates standard post-hoc Grad-CAM where convolutional weights receive no localization feedback.
- **Regime B (Combined Uncertainty Loss - Proposed Champion)**:
  $$\mathcal{L}_{\text{total}}(w, s_1, s_2) = \frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}}(w) + \frac{1}{2}\exp(-s_2)\mathcal{L}_{\text{loc}}(w) + \frac{1}{2}s_1 + \frac{1}{2}s_2$$
  Dynamically balances classification and continuous spatial alignment using Kendall & Gal (CVPR 2018) homoscedastic uncertainty optimization. **Operational champion model achieving 93.10% Pointing Game Hit Rate, 80.32% Energy in ROI, and 100% Sensitivity.**
- **Regime C (Alignment-Only)**:
  $$\mathcal{L} = \mathcal{L}_{\text{align}} = 1 - \frac{\langle \operatorname{vec}(M_{\text{CAM}}), \operatorname{vec}(M_{\text{GT}}) \rangle}{\|\operatorname{vec}(M_{\text{CAM}})\|_2 \|\operatorname{vec}(M_{\text{GT}})\|_2}$$
  Focuses the model on spatial gradient alignment.
- **Regime D (Dice-Only)**:
  $$\mathcal{L} = \mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum M_{\text{CAM}} M_{\text{GT}} + \epsilon}{\sum M_{\text{CAM}}^2 + \sum M_{\text{GT}}^2 + \epsilon}$$
  Directly maximizes soft continuous Dice overlap between model attention and ground truth.

### 3.3 Localization Evaluation Metrics
- **Continuous ROI Dice**: Soft overlap between continuous Grad-CAM heatmap $A \in [0, 1]$ and binary ground-truth mask $M \in \{0, 1\}$.
- **Binarized IoU ($\tau=0.35$)**: Intersection-over-Union after thresholding $A$ at $\tau=0.35$.
- **Pointing Game Hit Rate**: Evaluates whether the maximum intensity pixel $\arg\max_{(i,j)} A(i,j)$ falls strictly within the ground-truth thoracic mask.
- **Energy-in-ROI**: Proportion of total saliency energy contained inside the ground truth: $\frac{\sum_{(i,j) \in \text{ROI}} A(i,j)}{\sum_{i,j} A(i,j)}$.

---

## 4. Rate-Distortion & Image Quality Metrics

### 4.1 Automated Fair Rate-Matching Search
To ensure rigorous comparison against uniform JPEG compression, MediLink-DL conducts an automated binary search over uniform quality factors $q \in [1, 95]$:
$$q^* = \arg\min_{q \in [1, 95]} \left| \text{len}(\text{JPEG}(I, q)) - \text{len}(\mathcal{P}_{\text{roi\_dual}}) \right|$$
This ensures the uniform baseline uses essentially identical byte budgets ($\pm 2\%$).

### 4.2 Partitioned Image Metrics
Given reference image $I$ and reconstructed image $\hat{I}$:
1. **Partitioned MSE**:
   $$\text{MSE}_\Omega = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (I(i,j) - \hat{I}(i,j))^2, \quad \Omega \in \{\Omega_{\text{global}}, \Omega_{\text{roi}}, \Omega_{\text{bg}}\}$$
2. **Partitioned PSNR**:
   $$\text{PSNR}_\Omega = 10 \log_{10} \left( \frac{255^2}{\text{MSE}_\Omega} \right) \quad (\text{dB})$$
3. **Partitioned SSIM**:
   $$\text{SSIM}_\Omega = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} S(i,j)$$
   where $S(i,j)$ is computed via Gaussian windowing ($\sigma=1.5, 11 \times 11$).

---

## 5. Network Channel Simulation Profiles

The discrete-event channel simulator models four realistic rural telemedicine deployment scenarios:

| Channel Profile Name | Bandwidth ($R$) | RTT Latency | Loss Rate ($P_{\text{loss}}$) | Real-World Operational Context |
| :--- | :---: | :---: | :---: | :--- |
| **Rural 2G/EDGE** | 48 kbps | 200 ms | 2.0% | Remote Primary Health Centers connected via legacy 2G GSM cellular networks. |
| **Rural 2.5G/GPRS** | 64 kbps | 120 ms | 5.0% | Typical rural wireless uplinks in developing geographies. |
| **Rural 3G (Weak)** | 128 kbps | 150 ms | 5.0% | Fringes of mobile broadband coverage with fading signal-to-noise ratio. |
| **Satellite VSAT** | 256 kbps | 500 ms | 2.0% | Geostationary satellite uplinks deployed in mountainous or island health posts. |
