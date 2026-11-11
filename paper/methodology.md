# MediLink-DL: Scientific & Engineering Methodology

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
**Date**: November 2026 &nbsp;|&nbsp; **Technical Methodology Document**

---

## 1. System Architecture Overview

The MediLink-DL framework is structured into five sequential, reproducible modules resolving the clinical telemedicine trilemma of **diagnostic fidelity, transmission latency, and cryptographic security**:

1. **Clinical Data Curation & Localization**: Single-channel grayscale preprocessing, leakage-free patient-level cohort stratification, and Differentiable Grad-CAM diagnostic localization.
2. **Dual-Stream ROI-Aware Differential Compression**: Spatially varying quantization allocating high fidelity ($q_{\text{roi}}=90$) to pulmonary consolidations and aggressive lossy compression ($q_{\text{bg}}=25$) to background context, packaged into a self-describing 28-byte `'MLDL'` binary container.
3. **Authenticated Encryption (AEAD)**: NIST SP 800-38D AES-256-GCM and NIST SP 800-232 Ascon-AEAD128 with unique CSPRNG nonces, Associated Data (AAD) binding, and Zero-Plaintext Release. An experimental lab evaluates the custom **MLMC-128** research block cipher.
4. **Prioritized Computer Networks Transport**: Application-layer min-heap priority scheduling, packet framing, and Selective-Repeat ARQ simulation over constrained 2G/EDGE, 3G, and satellite channels.
5. **Multi-Objective Cross-Layer Optimization & Metrics**: End-to-end Pareto frontier extraction across 108 pipeline configurations alongside partitioned radiological distortion evaluation (Global, ROI, and Background PSNR/SSIM).

```text
+-----------------------+      +---------------------------+
| Clinical Radiograph I | ---> | Differentiable Grad-CAM M |
| (Cohen et al. CXR)    |      | (MedicalResNet18 - Reg B) |
+-----------------------+      +---------------------------+
            |                                |
            +---------------+----------------+
                            v
            +-------------------------------+
            | Dual-Stream ROI Compression   |
            | (q_roi = 90, q_bg = 25)       |
            | Packaged into 28-Byte 'MLDL'  |
            +-------------------------------+
                            v
            +-------------------------------+
            | NIST AEAD Cryptography        |
            | (AES-256-GCM / Ascon-AEAD128) |
            | Fresh CSPRNG Nonce + AAD Tag  |
            +-------------------------------+
                            v
            +-------------------------------+
            | Priority Transport Scheduling |
            | (HDR: 30 > ROI: 20 > BG: 5)   |
            | Selective-Repeat ARQ over 2G  |
            +-------------------------------+
                            v
            +-------------------------------+
            | Reassembly & Partitioned      |
            | Distortion (PSNR / SSIM)      |
            | Triage Acceleration (>6.2x)   |
            +-------------------------------+
```

---

## 2. Clinical Dataset Curation & Leakage-Free Stratification

### 2.1 Dataset Ingestion
- **Pixel Grid**: Standardized $512 \times 512$ single-channel 8-bit grayscale rasters $I \in [0, 255]^{H \times W}$.
- **Clinical Cohort**: Evaluated on genuine clinical chest radiographs from the open-access **COVID-19 Chest X-ray Dataset Initiative** (Cohen et al., 2020). The active verified cohort comprises **166 clinical digital radiographs** across **110 unique patients** presenting with acute respiratory distress, viral pneumonia (COVID-19/SARS), and control conditions.
- **Radiological Ground Truth**: Bilateral thoracic cages and focal pulmonary opacities/consolidations are annotated with expert radiologist bounding boxes (`imageannotation.ai`), rasterized into binary masks $M_{\text{GT}} \in \{0, 1\}^{H \times W}$.

### 2.2 Mathematical Leakage-Free Patient-Level Stratification
Random image-level splitting introduces severe data leakage because deep convolutional networks memorize patient-specific rib contours and thoracic geometry. MediLink-DL enforces strict patient-level partition isolation:
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \quad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \quad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$

- **Training Partition ($\mathcal{P}_{\text{train}}$)**: 77 patients (111 radiographs)
- **Validation Partition ($\mathcal{P}_{\text{val}}$)**: 16 patients (26 radiographs)
- **Test Partition ($\mathcal{P}_{\text{test}}$)**: 17 patients (29 radiographs)

---

## 3. Deep Learning Diagnostic Localization & Uncertainty Weighting

### 3.1 `MedicalResNet18` Backbone
The standard `ResNet-18` architecture is customized for digital radiography:
- Single-channel grayscale input layer (`conv1`: $7 \times 7$ kernel, stride 2, padding 3).
- High-level semantic feature representations are tapped at `layer4[-1]`, yielding feature activation tensor $A \in \mathbb{R}^{B \times 512 \times H_f \times W_f}$ ($H_f = W_f = 16$ for $512 \times 512$ inputs).
- Fully connected linear layer maps pooled features to binary classification logits $\hat{y} \in \mathbb{R}^2$.

### 3.2 Differentiable Grad-CAM (`create_graph=True`)
Unlike standard post-hoc Grad-CAM where gradients are detached, `DifferentiableGradCAM` preserves autograd computation graphs, allowing second-order backpropagation directly into convolutional weights:
1. **Neuron Importance Weights**:
   $$\alpha_k^c = \frac{1}{H_f W_f} \sum_{i=1}^{H_f} \sum_{j=1}^{W_f} \frac{\partial \hat{y}^c}{\partial A_{k,i,j}}$$
2. **Continuous Saliency Assembly**:
   $$H_{\text{CAM}}^c = \operatorname{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right)$$
3. **Bilinear Upsampling & Min-Max Normalization**:
   $$M_{\text{CAM}} = \frac{H_{\text{CAM}}^c - \min(H_{\text{CAM}}^c)}{\max(H_{\text{CAM}}^c) - \min(H_{\text{CAM}}^c) + \epsilon}$$

### 3.3 Multi-Task Localization-Aware Loss Formulation
To eliminate **Saliency Drift** (where the model classifies accurately while focusing on text labels or clavicles), training combines categorical classification and continuous spatial alignment:
$$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{CE}}(y, \hat{y})$$
$$\mathcal{L}_{\text{align}} = 1 - \frac{\langle \operatorname{vec}(M_{\text{CAM}}), \operatorname{vec}(M_{\text{GT}}) \rangle}{\|\operatorname{vec}(M_{\text{CAM}})\|_2 \|\operatorname{vec}(M_{\text{GT}})\|_2}$$
$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum M_{\text{CAM}} M_{\text{GT}} + \epsilon}{\sum M_{\text{CAM}}^2 + \sum M_{\text{GT}}^2 + \epsilon}$$
$$\mathcal{L}_{\text{loc}} = (1 - \beta)\mathcal{L}_{\text{align}} + \beta \mathcal{L}_{\text{dice}}$$

### 3.4 Kendall & Gal (CVPR 2018) Homoscedastic Uncertainty Optimization
Manually setting fixed hyperparameters ($\lambda_{\text{loc}}, \beta$) leads to gradient imbalance. MediLink-DL formulates loss balancing as maximum likelihood inference over homoscedastic task uncertainties:
$$\mathcal{L}_{\text{total}}(w, s_1, s_2) = \frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}}(w) + \frac{1}{2}\exp(-s_2)\mathcal{L}_{\text{loc}}(w) + \frac{1}{2}s_1 + \frac{1}{2}s_2$$
where $s_1 = \log \sigma_1^2$ and $s_2 = \log \sigma_2^2$ are unconstrained learnable parameters optimized concurrently with network weights $w$ via AdamW.

### 3.5 Four Operational Training Regimes
We conduct systematic ablations across 4 objective regimes:
- **Regime A (Classification-Only)**: $\mathcal{L} = \mathcal{L}_{\text{cls}}$ (Baseline CNN).
- **Regime B (Combined Uncertainty Loss - Proposed Champion)**: Kendall multi-task optimization.
- **Regime C (Alignment-Only)**: Focuses purely on cosine gradient alignment.
- **Regime D (Dice-Only)**: Focuses purely on soft continuous Dice overlap.

---

## 4. Digital Image Processing: Dual-Stream ROI-Aware Compression

### 4.1 Dual-Stream Decomposition
Baseline JPEG (ITU-T T.81) quantizes all $8 \times 8$ DCT blocks uniformly. MediLink-DL applies spatial decomposition:
1. **Background Full-Frame Stream**: The contextual radiograph $I \in \mathbb{R}^{512 \times 512}$ is encoded at low quality ($q_{\text{bg}}=25$):
   $$\mathcal{S}_{\text{bg}} = \text{JPEG}_{\text{enc}}(I, q_{\text{bg}})$$
2. **Diagnostic ROI Patch Stream**: The continuous heatmap $M_{\text{CAM}}$ is thresholded at cutoff $\tau=0.35$ to derive binary mask $M_{\text{bin}} = (M_{\text{CAM}} \ge \tau)$. The minimal enclosing bounding box $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ is cropped to yield sub-image $I_{\text{crop}}$, encoded at high diagnostic fidelity ($q_{\text{roi}}=90$):
   $$\mathcal{S}_{\text{roi}} = \text{JPEG}_{\text{enc}}(I_{\text{crop}}, q_{\text{roi}})$$

### 4.2 28-Byte Binary Container Header (`'MLDL'`)
The dual streams are serialized into a self-describing, big-endian binary container:
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
To guarantee a mathematically rigorous evaluation against standard JPEG, we implement an automated binary search identifying the uniform quality setting $q^*$ matching the dual-stream byte budget:
$$q^* = \arg\min_{q \in [1, 100]} \left| \operatorname{len}(\text{JPEG}(I, q)) - \operatorname{len}(\mathcal{P}_{\text{dual}}) \right|$$

---

## 5. Partitioned Radiological Distortion Metrics

Global image metrics fail to capture clinical utility: high global scores can occur while subtle focal lesions are obliterated by quantization noise. We compute partitioned metrics:

Let $\Omega_{\text{roi}} = \{(i,j) \mid M_{\text{bin}}(i,j) = 1\}$ and $\Omega_{\text{bg}} = \{(i,j) \mid M_{\text{bin}}(i,j) = 0\}$:

1. **Partitioned Mean Squared Error (MSE)**:
   $$\text{MSE}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (I(i,j) - \hat{I}(i,j))^2$$
2. **Partitioned Peak Signal-to-Noise Ratio (PSNR)**:
   $$\text{PSNR}_{\Omega} = 10 \log_{10}\left( \frac{255^2}{\text{MSE}_{\Omega}} \right) \quad (\text{dB})$$
3. **Partitioned Structural Similarity Index (SSIM)**:
   $$\text{SSIM}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} S(i,j)$$
   where $S(i,j)$ is computed via Gaussian windowing ($\sigma=1.5, 11 \times 11$).

---

## 6. Authenticated Encryption Layer (AEAD) & Security Protocols

### 6.1 Cryptographic Suite
1. **AES-256-GCM (NIST SP 800-38D Baseline)**: 256-bit key, 96-bit CSPRNG counter nonce, 128-bit GHASH tag.
2. **Ascon-AEAD128 (NIST SP 800-232 Lightweight Standard)**: 128-bit key, 128-bit nonce, 128-bit tag operating over a 320-bit permutation state.
3. **MLMC-128 Research Prototype**: 128-bit block cipher utilizing PRESENT S-boxes and a circulant MDS diffusion matrix ($\mathcal{B}=5$).

### 6.2 Associated Data (AAD) Binding & Zero-Plaintext Release
Hospital metadata is authenticated as Associated Data:
$$\text{AAD} = \text{StudyUID} \,\|\, \text{PatientID} \,\|\, \text{Timestamp} \,\|\, \text{HeaderChecksum}$$

**Zero-Plaintext Release**: The receiver validates the authentication tag in constant time prior to buffer release. If any single bit is tampered with, decryption raises `DecryptionTamperedError` and purges memory.

---

## 7. Prioritized Computer Networks Transport & Scheduling

### 7.1 Packet Framing Protocol
Payloads are partitioned into fixed chunks ($L \in \{512, 1024, 2048\}$ bytes). Header framing:
`[4B 'MLPK'] [2B SeqID] [2B TotalChunks] [1B PriorityTier] [2B PayloadLen] [32B SHA-256 Checksum]`.

### 7.2 Priority Min-Heap Scheduling
Chunks are enqueued into an application-layer priority min-heap:
$$\text{Priority}(\text{Header}) = 30 > \text{Priority}(\text{ROI Chunks}) = 20 > \text{Priority}(\text{Background Chunks}) = 5$$

Under FIFO, ROI chunks wait for preceding background context. Under priority scheduling, the complete diagnostic ROI arrives in the first burst of transmissions.

### 7.3 Channel Model & Selective-Repeat ARQ Simulation
$$\Delta t_{\text{chunk}} = \frac{(L + 64) \times 8}{R} + \frac{\text{RTT}}{2} + \xi_{\text{jitter}}$$
Packet drops (simulated at 2% to 10%) trigger Selective-Repeat ARQ retransmissions.

---

## 8. Multi-Objective Cross-Layer Optimization & Pareto Frontier

We formulate cross-layer parameter selection across DL, DIP, CN, and Cryptography as a constrained multi-objective optimization:
$$\min_{\mathbf{x} \in \mathcal{X}} \left[ - \text{SSIM}_{\text{roi}}(\mathbf{x}), \; T_{\text{roi}}(\mathbf{x}), \; T_{\text{total}}(\mathbf{x}), \; B_{\text{wire}}(\mathbf{x}), \; t_{\text{crypto}}(\mathbf{x}) \right]$$
subject to:
$$\text{AuthVerified}(\mathbf{x}) = \text{True}, \quad \text{ReconSuccess}(\mathbf{x}) = \text{True}$$
where decision vector $\mathbf{x} = (\text{ROI\_Threshold}, q_{\text{roi}}, q_{\text{bg}}, L_{\text{chunk}}, \text{Priority}, \text{Cipher})$.

A configuration $\mathbf{x}_A$ dominates $\mathbf{x}_B$ ($\mathbf{x}_A \succ \mathbf{x}_B$) if it is no worse across all objectives and strictly superior in at least one. An exhaustive grid search across 108 end-to-end parameter sets identified **26 non-dominated Pareto configurations**.

---

## 9. Cryptographic Research Laboratory: Involutory MDS Diffusion

In `src/crypto_research/`, we investigated lightweight linear diffusion layers ($y = M x$) for next-generation rural medical microcontrollers:
- **`PROPOSED-OIN-MDS`**: Optimal parameter Hadamard instantiation $\operatorname{Had}(1, 8, 13, 5)$ over $\mathbb{F}_{2^4}/(x^4+x+1)$ with optimal nibble branch number $\mathcal{B}=5$ and 48 Paar CSE XORs (~96 GE).
- **`ABC-MDS-OPT`**: Alternating Block-Circulant matrix with parameters $(0, 1, 2, 3)$, $\mathcal{B}=5$, and 49 Paar CSE XORs (~98 GE).
- **Formal Impossibility Proofs**: Formally proved that no $4 \times 4$ Hankel matrix can be simultaneously MDS and involutory over $\mathbb{F}_{2^m}$, and established the Kronecker product branch bound $\mathcal{B} \le 4$.
