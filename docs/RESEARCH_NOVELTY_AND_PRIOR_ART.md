# MediLink-DL: Research Novelty, Prior-Art Taxonomy, and Differentiation Analysis

**Document**: Formal Novelty Assessment and Literature Positioning  
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
**Date**: November 2026 &nbsp;|&nbsp; **Novelty & Literature Positioning Document**

---

## 1. Executive Summary & Research Positioning

This document establishes the formal cryptographic, image processing, and telemedicine engineering novelty of the contributions embodied in `MediLink-DL`. Academic rigor requires clear differentiation between:
1. **Established standards and well-known theoretical constructions** (e.g., NIST AEAD standards, baseline JPEG DCT compression, standard post-hoc Grad-CAM).
2. **Novel parameter instantiations and heuristic optimizations within known families** (e.g., optimal involutory Hadamard matrices over $\mathbb{F}_{2^4}$).
3. **Genuinely differentiated algebraic structures, multi-objective optimization, and cross-layer pipeline integrations** (e.g., Alternating Block-Circulant MDS search, Differentiable Grad-CAM second-order autograd loss with Kendall homoscedastic uncertainty optimization, and cross-layer Pareto optimization).

To avoid overstated claims, all components are evaluated against a standardized **6-Tier Cryptographic & Engineering Novelty Taxonomy**.

---

## 2. Standardized 6-Tier Novelty Taxonomy

| Tier | Category Name | Definition | MediLink-DL Assignment |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Direct Reproduction** | Straightforward reimplementation of an existing algorithm or published structure without algorithmic alteration. | NIST AES-256-GCM (`src/security.py`), PRESENT S-box (`src/crypto_research/sbox.py`). |
| **Tier 2** | **Parameter Variation / Optimal Instantiation** | Exhaustive search or optimization over an established mathematical parametric family yielding optimal implementation coefficients. | `PROPOSED-OIN-MDS` = $\operatorname{Had}(1, 8, 13, 5)$ over $\mathbb{F}_{2^4}$ (`src/crypto_research/diffusion/matrix_generator.py`). |
| **Tier 3** | **Structural Extension / Novel Construction within Existing Framework** | Adaptation of established mathematical techniques to novel constraints or hybrid architectures with proven security bounds. | `ABC-MDS-OPT` and `ABC-MDS-MAXBIT` Alternating Block-Circulant involutory MDS transformations; Differentiable Grad-CAM with Kendall & Gal (CVPR 2018) uncertainty weighting. |
| **Tier 4** | **New Structural Paradigm** | Introduction of a previously undocumented algebraic structure or hybrid topology providing non-trivial trade-offs. | `TBOI-16` Tensor-Product Binary Orthogonal Involution achieving exact binary self-inversion without finite field multipliers (Near-MDS, $\mathcal{B}_{\text{bit}}=6, \mathcal{B}_{\text{nibble}}=4$). |
| **Tier 5** | **Novel Mathematical Class** | Discovery of an entirely new algebraic ring or group operation previously unused in symmetric cryptography. | *Not claimed by this project* (maintains strict intellectual integrity). |
| **Tier 6** | **Breakthrough Cryptographic Primitive** | A newly discovered cryptanalytic principle or foundational hardness assumption. | *Not claimed by this project*. |

---

## 3. Detailed Component-by-Component Prior-Art and Novelty Evaluation

### 3.1 Differentiable Grad-CAM vs. Standard Post-Hoc Saliency
- **Prior Art**:
  - *Selvaraju et al. (ICCV 2017)*: Introduced standard Grad-CAM for post-hoc interpretability. Standard implementations call `.backward()` with detached activations, producing a static visualization after model convergence.
  - *CheXNet (Rajpurkar et al., 2017)*: Applied standard DenseNet-121 architectures for classification on NIH ChestX-ray14 without spatial backpropagation.
- **MediLink-DL Contribution (Tier 3)**:
  - `src/localization_loss.py` introduces `DifferentiableGradCAM`, maintaining the autograd graph (`create_graph=True`).
  - Formulates a Kendall & Gal (CVPR 2018) homoscedastic uncertainty loss dynamically weighting classification and continuous localization:
    $$\mathcal{L}_{\text{total}}(w, s_1, s_2) = \frac{1}{2}\exp(-s_1)\mathcal{L}_{\text{cls}}(w) + \frac{1}{2}\exp(-s_2)\mathcal{L}_{\text{loc}}(w) + \frac{1}{2}s_1 + \frac{1}{2}s_2$$
  - Backpropagates second-order gradients into convolutional weights, achieving a **93.10% Pointing Game Hit Rate**, **80.32% Energy in ROI**, and **100% Sensitivity** on held-out COVID-19 clinical radiographs.

### 3.2 Dual-Stream ROI Compression vs. JPEG2000 MAXSHIFT
- **Prior Art**:
  - *ISO/IEC 15444-1 / 15444-2 (JPEG2000 Part 1/2)*: Standardized MAXSHIFT Region-of-Interest coding by scaling Wavelet transform coefficients.
  - *Limitations in Telemedicine Practice*: JPEG2000 decoders suffer from high computational memory footprint and lack universal hardware acceleration in rural clinics.
- **MediLink-DL Contribution (Tier 2/3)**:
  - Implements an operational dual-stream binary container (`MLDL` 28-byte header) combining full-frame contextual JPEG ($q_{\text{bg}}=25$) and cropped diagnostic bounding box ($q_{\text{roi}}=90$) with alpha blending ($\alpha=0.45$).
  - Implements automated fair rate-matching ($q^*$ search) ensuring evaluation against an exactly size-equivalent uniform JPEG baseline.
  - Delivers a **9.89x compression ratio** (89.89% wire reduction) while sustaining **ROI PSNR > 48.3 dB** and **ROI SSIM > 0.989**, offering a **+10.13 dB gain** over rate-matched uniform JPEG.

### 3.3 Prioritized Application-Layer Scheduling vs. Standard FIFO Transport
- **Prior Art**:
  - Standard PACS DICOM transmission uses single-stream TCP or HTTP transport where packets are transmitted strictly FIFO.
- **MediLink-DL Contribution (Tier 3)**:
  - Implements packet-level prioritization in a min-heap queue:
    $$\text{Priority}(\text{Header}) = 30 > \text{Priority}(\text{ROI}) = 20 > \text{Priority}(\text{BG}) = 5$$
  - Evaluated under realistic rural channel models (48 kbps 2G, 64 kbps GPRS, 128 kbps 3G, VSAT satellite with 500ms RTT, and lossy wireless with 10% packet drop).
  - Delivers diagnostic ROI packets in **0.391 seconds** on real radiographs, achieving a **>6.2x acceleration** in time-to-first-ROI delivery compared to standard FIFO transport.

---

## 4. Cryptographic Diffusion Architecture Novelty Matrix

| Transformation Identifier | Algebraic Structure | Finite Field / Ring | Direct XORs | Paar CSE XORs | $\mathcal{B}_{\text{nibble}}$ | $\mathcal{B}_{\text{bit}}$ | Involutory? | Prior Art Reference | Novelty Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **AES MixColumns** | Circulant | $\mathbb{F}_{2^8}$ | 108 | 92 | 5 (MDS) | — | No ($M^{-1} \ne M$) | Daemen & Rijmen (2002) | Prior Art Baseline |
| **PRESENT Permutation** | Bit Permutation | $\mathbb{F}_2^{64}$ | 0 | 0 | 2 | 2 | No | Bogdanov et al. (2007) | Prior Art Baseline |
| **GIFT Permutation** | Bit Permutation | $\mathbb{F}_2^{64}$ | 0 | 0 | 2 | 2 | No | Banik et al. (2017) | Prior Art Baseline |
| **Anubis Diffusion** | Circulant MDS | $\mathbb{F}_{2^8}$ | — | — | 5 (MDS) | — | Yes ($M^2=I$) | Barreto & Rijmen (2000) | Prior Art Baseline |
| **Khafre / Had(1,2,x,y)** | Hadamard | $\mathbb{F}_{2^4}$ | — | — | 5 (MDS) | 4–6 | Yes | Sajadieh et al. (2012) | Prior Art Family |
| **PROPOSED-OIN-MDS** | Hadamard $\operatorname{Had}(1,8,13,5)$ | $\mathbb{F}_{2^4} / (x^4+x+1)$ | 64 | **48** | **5 (MDS)** | **8 (Maximal)** | **Yes ($M^2=I$)** | Optimal parameter instantiation | **Tier 2: Optimal Instantiation** |
| **ABC-MDS-OPT** | Alternating Block-Circulant | $\mathbb{F}_{2^4} / (x^4+x+1)$ | 64 | **49** | **5 (MDS)** | 6 | **Yes ($M^2=I$)** | Non-Hadamard parameter search | **Tier 3: Novel Construction** |
| **ABC-MDS-MAXBIT** | Alternating Block-Circulant | $\mathbb{F}_{2^4} / (x^4+x+1)$ | 94 | 58 | **5 (MDS)** | **7** | **Yes ($M^2=I$)** | Non-Hadamard parameter search | **Tier 3: Novel Construction** |
| **TBOI-16** | Kronecker Tensor $K_4 \otimes K_4$ | $\mathbb{F}_2$ | 128 | 60 | 4 (Near-MDS) | 6 | **Yes ($M^2=I$)** | Direct binary involution search | **Tier 4: New Structural Paradigm** |

---

## 5. Formal Impossibility Theorems and Structural Bounds

1. **Theorem 1 (Hankel Non-Existence)**: *No $4 \times 4$ Hankel matrix over $\mathbb{F}_{2^m}$ can be simultaneously MDS and involutory.*
2. **Proposition 2 (Cauchy Involution Collapse)**: *Symmetric Cauchy matrices over $\mathbb{F}_{2^4}$ satisfying self-inversion degenerate into singular or non-MDS structures.*
3. **Theorem 3 (Galois Matrix Symmetry Obstruction)**: *For elements $\alpha, \beta \in \mathbb{F}_{2^4}$, the binary representation matrices satisfy $L_\alpha = L_\beta^T$ if and only if $\alpha = \beta \in \{0, 1\}$.*
4. **Theorem 4 (Tensor Bound)**: *The Kronecker product $A \otimes B$ of two invertible $2 \times 2$ matrices cannot achieve optimal MDS branch number $\mathcal{B}=5$; it is strictly bounded by $\mathcal{B} \le 4$.*
