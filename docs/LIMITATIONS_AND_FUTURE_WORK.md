# MediLink-DL: Scientific Limitations, Theoretical Constraints, and Future Work

**Document**: Transparent Analysis of Methodological Limitations and Future Roadmap  
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
**Date**: November 2026 &nbsp;|&nbsp; **Scientific Limitations & Future Roadmap**

---

## 1. Clinical and Radiological Scope Limitations

1. **Active Dataset Cohort Scale**:
   - The clinical validation is conducted on 166 chest radiographs across 110 patients from the open-access COVID-19 Chest X-ray Dataset Initiative (Cohen et al., 2020). While patient-level leakage was strictly eliminated, a production medical AI deployment requires validation across tens of thousands of studies spanning diverse demographics, imaging manufacturers, and pathology co-morbidities (e.g., NIH ChestX-ray14, CheXpert, or MIMIC-CXR).
2. **2D Projectional Radiography Scope**:
   - MediLink-DL is formulated for planar projectional chest radiographs (CR/DR). It does not natively handle multi-slice volumetric modalities such as Computed Tomography (CT), Magnetic Resonance Imaging (MRI), or Positron Emission Tomography (PET), which require 3D spatial wavelet or volumetric neural compression.
3. **Absence of Prospective Clinical Reader Trials**:
   - Diagnostic fidelity is evaluated quantitatively via partitioned PSNR and SSIM. Although these metrics are widely accepted in engineering literature, regulatory clearance (e.g., US FDA 510(k), EU CE-MDR, CDSCO) necessitates prospective multi-reader multi-case (MRMC) Receiver Operating Characteristic (ROC) studies conducted by board-certified radiologists.
4. **Regulatory and Clinical Disclaimer**:
   - MediLink-DL is an academic research prototype developed as an undergraduate capstone project. It is **not FDA or CE cleared** and must not be used for primary diagnostic decisions or live patient management without human expert oversight.

---

## 2. Digital Image Processing Limitations

1. **Dual-Stream JPEG Compositing vs. Native Wavelet Shifting**:
   - MediLink-DL implements an operational dual-stream binary container (`MLDL` 28-byte header) that serializes a background context JPEG and an ROI crop JPEG. While this approach is universally compatible with standard JPEG decoders and fast on CPUs, it is an engineering approximation of true progressive Region-of-Interest coding. It does not implement standardized JPEG2000 Part 2 MAXSHIFT (ISO/IEC 15444-2), which operates directly within the Discrete Wavelet Transform (DWT) coefficient bitplanes without boundary compositing.
2. **Boundary Discontinuity Artifacts**:
   - In rare cases where background compression is extremely aggressive ($q_{\text{bg}} \le 15$) and the ROI bounding box intersects prominent anatomical edges, subtle boundary tiling differences may appear at the transition between the high-fidelity crop and the lossy background. Our 4-pixel distance-transform alpha blending ($\alpha=0.45$) substantially mitigates this, but wavelet feathering remains future work.
3. **Spatial Mask Resolution**:
   - Grad-CAM heatmaps are derived from coarse convolutional activations ($16 \times 16$), producing smooth regional focus maps. They do not delineate precise pixel-level lesion boundaries (unlike full segmentation models such as U-Net).

---

## 3. Computer Networks Simulation Limitations

1. **Discrete-Event Simulation vs. Kernel-Level TCP**:
   - The transmission module uses discrete-event simulation with Selective-Repeat ARQ and stochastic drop rates. It does not model physical-layer radio propagation effects (e.g., Rayleigh/Rician multipath fading, Doppler frequency shift in mobile ambulances) or operating system kernel TCP congestion control algorithms (such as TCP CUBIC or BBR).
2. **Application-Layer vs. Transport-Layer Prioritization**:
   - Prioritization is currently enforced at the application layer using a min-heap chunk queue. In real-world IP networks, application-layer priority headers require mapping into network-layer Differentiated Services Code Point (DSCP) or IP Precedence fields to guarantee QoS across intermediate ISP routers.

---

## 4. Cryptographic Research & Implementation Limitations

1. **Research Candidate Status of MLMC-128**:
   - MLMC-128 is an academic research prototype designed to explore the algebraic interplay between PRESENT S-boxes and optimal involutory MDS diffusion layers over $\mathbb{F}_{2^4}$. **It must not be used in production in lieu of NIST-standardized AES-256-GCM or Ascon-AEAD128.**
2. **Standard-Cell Analytical Gate Estimates vs. Physical ASIC Synthesis**:
   - All Gate Equivalent (GE) numbers reported for diffusion transformations (e.g., 96.0 GE for `PROPOSED-OIN-MDS`, 98.0 GE for `ABC-MDS-OPT`) are **analytical standard-cell model estimates** based on straight-line program XOR counts (2.0 GE per 2-input XOR gate). Physical ASIC tape-out with wire routing, parasitic capacitance, and dynamic power dissipation was not executed.

---

## 5. Future Research Roadmap

1. **Clinical Dataset Expansion**: Integrate automated ingestion of the full NIH ChestX-ray14 dataset (112,120 radiographs, 30,805 patients) and MIMIC-CXR.
2. **Direct Wavelet & Learned Neural Compression**: Extend the dual-stream concept to modern neural codecs (e.g., Learned Image Compression with Variational Autoencoders).
3. **Hardware Testbed & Physical SDR Transmission**: Validate transmission latencies over physical Software-Defined Radio (SDR) testbeds (USRP B210) running open-source 4G/5G stacks (srsRAN).
4. **Physical ASIC / FPGA Cryptographic Synthesis**: Translate `ABC-MDS-OPT` and `PROPOSED-OIN-MDS` into synthesizable Verilog/VHDL for TSMC 28nm standard-cell place-and-route.
5. **Multi-Reader Clinical Radiologist Observer Study**: Conduct a double-blind diagnostic observer study with clinical radiologists to verify that dual-stream compression preserves diagnostic sensitivity and specificity.
