# MediLink-DL: Master Research Outline

**Project Title**: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Short Name**: MediLink-DL  
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
**Date**: November 2026 &nbsp;|&nbsp; **Master Research Framework Outline**

---

## 1. Central Research Question

> **How can diagnostic fidelity in digital medical chest radiographs be preserved under severe bandwidth and channel loss constraints (typical of rural telemedicine environments) through deep-learning-guided regional prioritization, while simultaneously ensuring cryptographic confidentiality, tamper-evident payload integrity, and optimal multi-objective cross-layer transport?**

---

## 2. Research Problem & Telemedicine Context

- **Infrastructure Asymmetry**: Rural primary healthcare centers (PHCs) frequently lack on-site specialist radiologists and rely on constrained wireless uplinks (2G/EDGE, low-bandwidth 3G, or satellite links with 32–128 kbps throughput, propagation latency of 120–500 ms, and intermittent packet loss rates of 2–10%).
- **Diagnostic Bottleneck**: High-resolution digital radiography (CR/DR) generates uncompressed files of 10–35 MB per study. Standard uniform lossy compression (such as uniform JPEG or H.264 intra-coding) introduces high-frequency blurring and blocking artifacts uniformly across the entire image raster, which degrades subtle pathological features (such as faint ground-glass opacities, consolidations, or pneumothorax pleura lines).
- **Security & Regulatory Compliance**: Medical data transmission over public wireless networks is vulnerable to eavesdropping and active packet modification. Statutory regulatory frameworks (HIPAA Security Rule, EU GDPR) mandate authenticated confidentiality and Zero-Plaintext Release.

---

## 3. Investigated Hypothesis & Research Gap

- **Research Gap**: Existing literature either evaluates deep-learning saliency localization (e.g. Grad-CAM) in offline diagnostic pipelines or evaluates regional image coding on synthetic test shapes without network-adaptive transport prioritization. Conversely, computer network transmission protocols rarely account for non-uniform medical semantics to schedule critical diagnostic bytes ahead of contextual anatomy.
- **Hypothesis**: By coupling model-derived or radiologist-verified Region of Interest (ROI) boundaries with dual-stream differential compression, standardized Authenticated Encryption (AEAD), and application-layer priority transport scheduling, a transmission system can:
  1. Reduce the time required to deliver diagnostic-grade visual information by **>60%** (>6.2x acceleration) compared to standard FIFO baselines.
  2. Achieve an **89.9% data reduction** while delivering a **+10.13 dB gain in ROI PSNR** over rate-matched uniform baselines.
  3. Enforce **100% bit-flip tamper rejection** with zero plaintext release.

---

## 4. Key Scientific and Engineering Contributions

1. **Clinical Data Curation & Leakage-Free Stratification**:
   - Evaluated on 166 clinical digital radiographs across 110 unique patients from the Cohen et al. (2020) cohort.
   - Enforces strict patient-level partition isolation ($\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset$) preventing convolutional memory leakage.
2. **Differentiable Grad-CAM & Kendall Uncertainty Optimization**:
   - `MedicalResNet18` with autograd `create_graph=True` backpropagates through continuous saliency heatmaps.
   - Multi-task loss balancing via Kendall & Gal (CVPR 2018) homoscedastic uncertainty optimization eliminates manual heuristic tuning.
   - 4-regime ablation identifies Regime B as operational champion (93.10% Pointing Game Hit Rate, 80.32% Energy in ROI, 100% Recall).
3. **Dual-Stream ROI-Aware Differential Compression**:
   - Spatially varying quantization allocating high quality ($q_{\text{roi}}=90$) to pathology and low quality ($q_{\text{bg}}=25$) to background.
   - Packaged into a self-describing 28-byte `'MLDL'` binary container header with distance-transform alpha blending ($\alpha=0.45$).
   - Yields an **89.89% data reduction** (9.89x compression) and **+10.13 dB ROI PSNR gain** over rate-matched uniform baseline JPEG.
4. **Standardized Cryptography (AEAD) & Zero-Plaintext Release**:
   - Evaluates NIST SP 800-38D AES-256-GCM and NIST SP 800-232 Ascon-AEAD128 with unique CSPRNG nonces and Associated Data (AAD) metadata binding.
   - Isolated research lab evaluates **MLMC-128** block cipher (SAC 49.92%, MDS branch number $\mathcal{B}=5$, 48 Paar CSE XORs) and establishes 4 formal impossibility theorems.
5. **Prioritized Transport & Selective-Repeat ARQ**:
   - Application-layer chunking (1024B) with SHA-256 framing and min-heap priority queue (Header 30 > ROI 20 > BG 5).
   - Delivers first diagnostic ROI in **0.391 seconds** (**>6.2x faster triage than FIFO**).
6. **Multi-Objective Cross-Layer Pareto Optimization**:
   - Exhaustive grid search over 108 pipeline configurations extracts **26 non-dominated Pareto solutions**.
