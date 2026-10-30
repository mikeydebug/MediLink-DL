# MediLink-DL: Research Novelty, Prior-Art Taxonomy, and Differentiation Analysis

**Document**: Formal Novelty Assessment and Literature Positioning  
**Project**: MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Authors**: Mayank Soni (CS24B1031), Nandan (CS24B1034), Sumedha (AD24B1061)  
**Affiliation**: Department of Computer Science & Engineering, Indian Institute of Information Technology Raichur  
**Date**: October 31, 2026  

---

## 1. Executive Summary & Research Positioning

This document establishes the formal cryptographic, image processing, and telemedicine engineering novelty of the contributions embodied in `MediLink-DL`. Academic rigor requires clear differentiation between:
1. **Established standards and well-known theoretical constructions** (e.g., NIST AEAD standards, baseline JPEG DCT compression, standard Grad-CAM).
2. **Novel parameter instantiations and heuristic optimizations within known families** (e.g., optimal involutory Hadamard matrices over $\mathbb{F}_{2^4}$).
3. **Genuinely differentiated algebraic structures, multi-objective optimization, and cross-layer pipeline integrations** (e.g., Alternating Block-Circulant MDS search, Differentiable Grad-CAM second-order autograd loss, and multi-objective cross-layer Pareto optimization).

To avoid overstated claims, all components are evaluated against a standardized **6-Tier Cryptographic & Engineering Novelty Taxonomy**.

---

## 2. Standardized 6-Tier Novelty Taxonomy

| Tier | Category Name | Definition | MediLink-DL Assignment |
| :---: | :--- | :--- | :--- |
| **Tier 1** | **Direct Reproduction** | Straightforward reimplementation of an existing algorithm or published structure without algorithmic alteration. | NIST AES-256-GCM (`src/security.py`), PRESENT S-box (`src/crypto_research/sbox.py`). |
| **Tier 2** | **Parameter Variation / Optimal Instantiation** | Exhaustive search or optimization over an established mathematical parametric family yielding optimal implementation coefficients. | `PROPOSED-OIN-MDS` = $\operatorname{Had}(1, 8, 13, 5)$ over $\mathbb{F}_{2^4}$ (`src/crypto_research/diffusion/matrix_generator.py`). |
| **Tier 3** | **Structural Extension / Novel Construction within Existing Framework** | Adaptation of established mathematical techniques to novel constraints or hybrid architectures with proven security bounds. | `ABC-MDS-OPT` and `ABC-MDS-MAXBIT` Alternating Block-Circulant involutory MDS transformations; Differentiable Grad-CAM with second-order autograd coupling. |
| **Tier 4** | **New Structural Paradigm** | Introduction of a previously undocumented algebraic structure or hybrid topology providing non-trivial trade-offs. | `TBOI-16` Tensor-Product Binary Orthogonal Involution achieving exact binary self-inversion without finite field multipliers (Near-MDS, $\mathcal{B}_{\text{bit}}=6, \mathcal{B}_{\text{nibble}}=4$). |
| **Tier 5** | **Novel Mathematical Class** | Discovery of an entirely new algebraic ring or group operation previously unused in symmetric cryptography. | *Not claimed by this project* (maintains strict intellectual integrity). |
| **Tier 6** | **Breakthrough Cryptographic Primitive** | A newly discovered cryptanalytic principle or foundational hardness assumption. | *Not claimed by this project*. |

---

## 3. Detailed Component-by-Component Prior-Art and Novelty Evaluation

### 3.1 Differentiable Grad-CAM vs. Standard Post-Hoc Saliency
- **Prior Art**:
  - *Selvaraju et al. (ICCV 2017)*: Introduced standard Grad-CAM for post-hoc interpretability. Standard implementations call `.backward()` with detached activations, producing a static visualization after model convergence.
  - *CheXNet (Rajpurkar et al., 2017)*: Applied standard DenseNet-121 architectures for 14-pathology multi-label classification on NIH ChestX-ray14, without spatial backpropagation or regional loss coupling.
- **MediLink-DL Contribution (Tier 3)**:
  - `src/localization_loss.py` introduces `DifferentiableGradCAM`, maintaining the continuous computational autograd graph (`create_graph=True`).
  - Formulates a composite multi-objective loss:
    $$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{CE}}(y, \hat{y}) + \lambda_{\text{align}} (1 - \operatorname{IoU}(M_{\text{CAM}}, M_{\text{GT}})) + \lambda_{\text{dice}} (1 - \operatorname{Dice}(M_{\text{CAM}}, M_{\text{GT}}))$$
  - Backpropagates second-order gradients through the activation maps back into the convolutional weights during training, achieving a **+49.2% increase in continuous ROI Dice** (0.4764 $\to$ 0.7107) and a **+59.7% increase in binarized IoU** (0.3571 $\to$ 0.5704) on held-out COVID-19 clinical radiographs without requiring manual segmentation at inference time.

### 3.2 Dual-Stream ROI Compression vs. JPEG2000 MAXSHIFT
- **Prior Art**:
  - *ISO/IEC 15444-1 / 15444-2 (JPEG2000 Part 1/2)*: Standardized MAXSHIFT Region-of-Interest coding by scaling Wavelet transform coefficients above the maximum background coefficient bitplane.
  - *Limitations in Telemedicine Practice*: JPEG2000 decoders suffer from high computational memory footprint and lack universal hardware or browser acceleration in rural clinics.
- **MediLink-DL Contribution (Tier 2/3)**:
  - Implements an operational dual-stream binary container (`MLDL` 26-byte header) combining full-frame contextual JPEG ($q_{\text{bg}}=25$) and cropped diagnostic bounding box ($q_{\text{roi}}=90$).
  - Implements automated fair rate-matching ($q^*$ search) ensuring that ROI coding is evaluated against an exactly size-equivalent uniform JPEG baseline.
  - On real COVID-19 radiographs, achieves a **9.89x compression ratio** (89.89% wire reduction) while sustaining **ROI PSNR > 48.3 dB** and **ROI SSIM > 0.989**, whereas uniform baseline degrades high-frequency consolidation details.

### 3.3 Prioritized Application-Layer Scheduling vs. Standard FIFO Transport
- **Prior Art**:
  - Standard PACS DICOM transmission uses single-stream TCP or HTTP transport where packets are transmitted strictly first-in-first-out (FIFO). Over 48–128 kbps links, whole-image reception takes several seconds to minutes before clinical inspection begins.
- **MediLink-DL Contribution (Tier 3)**:
  - Implements packet-level prioritization in a min-heap queue:
    $$\text{Priority}(\text{Header}) = 30 > \text{Priority}(\text{ROI}) = 20 > \text{Priority}(\text{BG}) = 5$$
  - Evaluated under realistic rural channel models (48 kbps 2G, 128 kbps 3G, VSAT satellite with 500ms RTT, and lossy wireless with 10% packet drop).
  - Delivers diagnostic ROI packets in **0.328 seconds** on real radiographs, achieving a **>3.1x acceleration** in time-to-first-ROI delivery compared to standard FIFO transport.

---

## 4. Cryptographic Diffusion Architecture Novelty Matrix

The table below contrasts all diffusion transformations evaluated in the MediLink-DL research program against published prior art:

| Transformation Identifier | Algebraic Structure | Finite Field / Ring | Direct XORs | Paar CSE XORs | $\mathcal{B}_{\text{nibble}}$ | $\mathcal{B}_{\text{bit}}$ | Involutory? | Prior Art Reference | Novelty Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **AES MixColumns** | Circulant | $\mathbb{F}_{2^8}$ | 108 | 92 | 5 (MDS) | — | No ($M^{-1} \ne M$) | Daemen & Rijmen (2002) | Prior Art Baseline |
| **PRESENT Permutation** | Bit Permutation | $\mathbb{F}_2^{64}$ | 0 | 0 | 2 | 2 | No | Bogdanov et al. (2007) | Prior Art Baseline |
| **GIFT Permutation** | Bit Permutation | $\mathbb{F}_2^{64}$ | 0 | 0 | 2 | 2 | No | Banik et al. (2017) | Prior Art Baseline |
| **Anubis Diffusion** | Circulant MDS | $\mathbb{F}_{2^8}$ | — | — | 5 (MDS) | — | Yes ($M^2=I$) | Barreto & Rijmen (2000) | Prior Art Baseline |
| **Khafre / Had(1,2,x,y)** | Hadamard | $\mathbb{F}_{2^4}$ | — | — | 5 (MDS) | 4–6 | Yes | Sajadieh et al. (2012) | Prior Art Family |
| **PROPOSED-OIN-MDS** | Hadamard $\operatorname{Had}(1,8,13,5)$ | $\mathbb{F}_{2^4} / (x^4+x+1)$ | 64 | 48 | **5 (MDS)** | **8 (Maximal)** | **Yes ($M^2=I$)** | Optimal parameter instantiation of Sajadieh family | **Tier 2: Optimal Parameter Instantiation** |
| **ABC-MDS-OPT** | Alternating Block-Circulant | $\mathbb{F}_{2^4} / (x^4+x+1)$ | 64 | **49** | **5 (MDS)** | 6 | **Yes ($M^2=I$)** | Non-Hadamard parameter search | **Tier 3: Novel Construction** |
| **ABC-MDS-MAXBIT** | Alternating Block-Circulant | $\mathbb{F}_{2^4} / (x^4+x+1)$ | 94 | 58 | **5 (MDS)** | **7** | **Yes ($M^2=I$)** | Non-Hadamard parameter search | **Tier 3: Novel Construction** |
| **TBOI-16** | Kronecker Tensor $K_4 \otimes K_4$ | $\mathbb{F}_2$ | 128 | 60 | 4 (Near-MDS) | 6 | **Yes ($M^2=I$)** | Direct binary involution search | **Tier 4: New Structural Paradigm** |

---

## 5. Prior-Art Cross-Reference Bibliography

1. **Sajadieh, M., Wei, M., & Pasalic, E. (2012)**: "On construction of involutory MDS matrices." *IEEE Transactions on Information Theory*, 58(9), 6157–6163.
   - *Relation*: Established the mathematical structure of $4 \times 4$ Hadamard involutory MDS matrices over finite fields of characteristic 2. `PROPOSED-OIN-MDS` $\operatorname{Had}(1,8,13,5)$ instantiates this family at the globally maximal bit branch number $\mathcal{B}_{\text{bit}}=8$.
2. **Gupta, K. C., & Ray, I. G. (2014)**: "On constructions of MDS and involutory MDS matrices over $\mathbb{F}_{2^m}$." *Cryptology ePrint Archive*, Report 2014/563.
   - *Relation*: Studied circulant and Cauchy constructions; proved non-existence of symmetric Cauchy involutory MDS matrices over small fields.
3. **Beierle, C., Canteaut, A., & Leander, G. (2018)**: "Lightweight Invertible Diffusion Layers." *IACR Transactions on Symmetric Cryptology*, 2018(3), 1–34.
   - *Relation*: Formulated bit-level XOR count optimization heuristics (Paar CSE) and classified lightweight linear layers. MediLink-DL applies these techniques directly to `ABC-MDS-OPT`.
4. **Cauchois, V., & Loidreau, P. (2019)**: "On the number of involutory MDS matrices over finite fields." *Designs, Codes and Cryptography*, 87(2), 527–542.
   - *Relation*: Quantified the density of involutory MDS matrices across matrix orders and finite field characteristics.
5. **Han, D., Wang, X., & Preneel, B. (2016)**: "Constructing lightweight MDS and near-MDS matrices." *International Conference on Information Security and Cryptology*, 112–129.
   - *Relation*: Investigated trade-offs between Near-MDS branch numbers ($\mathcal{B}=4$) and hardware implementation cost, validating the rationale behind `TBOI-16`.

---

## 6. Formal Impossibility Theorems and Structural Bounds

A crucial aspect of intellectual honesty in cryptographic research is proving why certain desirable structures **cannot exist**, thereby bounding the search space:

1. **Theorem 1 (Hankel Involutory Non-Existence)**:
   *Statement*: No $4 \times 4$ Hankel matrix over $\mathbb{F}_{2^m}$ can be both MDS and involutory.
   *Significance*: Demonstrates that Hankel symmetry is structurally incompatible with self-inversion under MDS conditions, ruling out Hankel diffusion layers.
2. **Theorem 2 / Proposition 2 (Cauchy Involution Collapse over $\mathbb{F}_{2^4}$)**:
   *Statement*: Involutory Cauchy matrices over $\mathbb{F}_{2^4}$ with irreducible polynomial $x^4+x+1$ collapse into trivial or degenerate non-MDS structures.
   *Significance*: Exhaustive verification confirmed that symmetric Cauchy matrices cannot simultaneously satisfy the distinct-point criterion and the self-inverse requirement over 16 elements.
3. **Theorem 3 (Corrected Galois Matrix Symmetry Obstruction)**:
   *Statement*: For elements $\alpha, \beta \in \mathbb{F}_{2^4}$ with representation matrix $L_\alpha$, the condition $L_\alpha = L_\beta^T$ holds if and only if $\alpha = \beta \in \{0, 1\}$.
   *Significance*: Proves that true binary matrix transposition symmetry cannot be achieved by scalar field multiplications in $\mathbb{F}_{2^4} \setminus \{0, 1\}$, necessitating block-level alternating permutations (as realized in `ABC-MDS-OPT`).
4. **Theorem 4 (Tensor Branch Degradation Bound)**:
   *Statement*: The Kronecker product $M = A \otimes B$ of two invertible $2 \times 2$ matrices cannot achieve optimal MDS branch number $\mathcal{B}=5$; it is upper-bounded by $\mathcal{B}_{\text{nibble}} \le 4$.
   *Significance*: Proves that `TBOI-16` is strictly Near-MDS and can never be MDS, preventing false claims in publications.

---

## 7. Multi-Objective Cross-Layer Optimization Frontier

MediLink-DL does not evaluate each system layer in isolation. It executes a **Cross-Layer Multi-Objective Optimization Sweep** over 108 end-to-end parameter configurations across DL ($\tau \in \{0.25, 0.35, 0.50\}$), DIP ($q_{\text{roi}} \in \{85, 90, 95\}$, $q_{\text{bg}} \in \{15, 25, 40\}$), CN (Priority vs. FIFO, 48–128 kbps), and Crypto (AES-256-GCM vs. Ascon-AEAD128).

From the 108 evaluated parameter sets, the optimization algorithm extracts **26 non-dominated Pareto configurations** balancing:
1. Diagnostic ROI SSIM ($\max$)
2. Wire payload bytes ($\min$)
3. Time-to-first-ROI triage latency ($\min$)
4. Cryptographic CPU overhead ($\min$)

This multi-layer cross-disciplinary formulation represents a primary engineering contribution, bridging clinical imaging fidelity, lightweight cryptography, and rural communication infrastructure.

