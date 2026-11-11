# MediLink-DL: Final Presentation & Viva Voce Master Script
## Complete 3-Speaker Script, Screen Timings, Whiteboard Plan, and Tough Faculty Defense

**Project Title:** MediLink-DL: A Deep Learning-Guided, Bandwidth-Adaptive, and Secure Framework for Medical Image Transmission in Rural Telemedicine  
**Authors & Presenters:**
1. **Nandan** (Roll: CS24B1034) — Ground Zero Motivation, Problem Statement, & Deep Learning Attention Subsystem
2. **Sumedha** (Roll: AD24B1061) — Digital Image Processing (Dual-Stream JPEG & Container) & Computer Networks Subsystem
3. **Mayank Soni** (Roll: CS24B1031) — Continuous & Discrete Optimization, Cryptography (MLMC-128 Research), & Live System Synthesis

**Target Evaluation Panel:**
* **Faculty Evaluator 1:** Deep Learning (DL) & Digital Image Processing (DIP) Specialist
* **Faculty Evaluator 2:** Computer Networks (CN) Specialist
* **Faculty Evaluator 3:** Optimization Specialist

**Total Presentation Duration:** ~30–35 Minutes (~10–12 Minutes per Speaker) + 15 Minutes Viva Voce Defense  
**Date:** November 2026 | **Institution:** Indian Institute of Information Technology (IIIT) Raichur

---

# SECTION 1: MASTER SCHEDULE & PRESENTATION FLOW CHART

```
+---------------------------------------------------------------------------------------------------------------+
| TOTAL TIME: 35 MINS PRESENTATION + 15 MINS VIVA VOCE DEFENSE                                                  |
+---------------------------------------------------------------------------------------------------------------+
|                                                                                                               |
| [00:00 - 11:00] SPEAKER 1: NANDAN (Starts from Zero / Ground Up)                                              |
|  * Rural Telemedicine Bottleneck (2G / GPRS 64 kbps link vs 256 KB CXR)                                       |
|  * The Core Conflict: Delay vs Diagnostic Blur                                                                |
|  * MedicalResNet18 Backbone & Grayscale Adaptation                                                            |
|  * Saliency Drift & Vanilla Grad-CAM Failures                                                                 |
|  * Mathematical Derivation of Differentiable Grad-CAM (Second-Order Autograd)                                  |
|  * Loss Functions: L_cls, L_align, L_dice, and L_total                                                        |
|  * Cohen et al. Clinical Dataset Results (Pointing Game 65.5% -> 96.55%, Dice 0.476 -> 0.711)                 |
|                                                                                                               |
| [11:00 - 22:00] SPEAKER 2: SUMEDHA (Image Processing & Priority Networks)                                     |
|  * DIP Handover: Anatomical Dual-Stream Partitioning Principle                                                |
|  * Diagnostic ROI Crop (q=90) vs Background Context (q=25)                                                     |
|  * Seamless Alpha Blending & Boundary Reconstruction                                                          |
|  * 28-Byte 'MLDL' Verifiable Binary Container Format                                                          |
|  * Quantitative Metrics: MSE, PSNR, SSIM & Rate-Matched Uniform Baseline (+10.13 dB Gain!)                    |
|  * CN Subsystem: Packetization & Chunking (L=1024 Bytes, SHA-256 Framing)                                     |
|  * Min-Heap Priority Queuing (Header=30, ROI=20, BG=5) -> Triage in <0.8 Seconds                              |
|  * Lossy Rural Channel Model (64 kbps, 120 ms RTT, 10% Packet Loss) & Selective-Repeat ARQ                    |
|                                                                                                               |
| [22:00 - 35:00] SPEAKER 3: MAYANK SONI (Optimization, Cryptography, & Live Demo)                              |
|  * Level 1 Optimization: Kendall's Adaptive Homoscedastic Uncertainty Loss (Auto Loss Balancing)              |
|  * Level 2 Optimization: Cross-Layer Constrained Pareto Optimization across 108 Configurations                |
|  * Non-Dominated Sorting & The 4 Clinical Operating Regimes                                                   |
|  * Cryptography Subsystem: Why AEAD & Zero-Plaintext Release are Mandatory for Telemedicine                   |
|  * The 4 Ciphers (AES-128/256-GCM, Ascon-AEAD128 for rural battery devices)                                   |
|  * Our Original Research Contribution: MLMC-128 Cipher (10-Round SPN, 49.92% Avalanche Effect)                |
|  * Live Web App Demonstration (Tabs 1-4, Interactive Bit-Flip Tamper Attack, & Pareto Visualizer)             |
|                                                                                                               |
| [35:00 - 50:00] JOINT TEAM: TOUGH VIVA VOCE DEFENSE                                                           |
|  * Targeted Defense against DL/DIP, CN, and Optimization Faculty Examiners                                    |
+---------------------------------------------------------------------------------------------------------------+
```

---

# SECTION 2: SPEAKER 1 — NANDAN
### Focus: Problem Statement, DL Backbone, Saliency Drift, Differentiable Grad-CAM, & Loss Functions
**Allocated Time:** Minutes 00:00 – 11:00 (11 Minutes)  
**Target Evaluator to Impress:** Faculty 1 (Deep Learning & Image Processing Specialist)

---

### Phase 1: Ground-Zero Problem Statement (Minutes 00:00 – 03:00)

#### 🎬 Screen Action:
* Project **Slide 1 & Slide 2** from `paper/MediLink_DL_Progress_Presentation.pptx` ("Problem Statement: The Rural Telemedicine Bottleneck").

#### 🗣️ Spoken Script (Word-for-Word English):
> "Respected members of the evaluation committee, good morning. My name is **Nandan**, and on behalf of my co-authors Mayank Soni and Sumedha, I will begin our presentation of **MediLink-DL** from first principles.
>
> In rural public health centers across India and developing nations, acute respiratory infections—such as COVID-19, bacterial pneumonia, and pulmonary tuberculosis—demand rapid radiologist triage. However, rural clinics are geographically isolated from specialist hospitals and rely on severely degraded communication links: 2G/GPRS and congested 3G networks. These channels exhibit an effective bandwidth of just **48 to 64 kilobits per second**, one-way latencies exceeding **120 milliseconds**, and transient packet loss rates between **5% and 15%**.
>
> A standard digital chest radiograph is a $512 \times 512$ 8-bit grayscale image, requiring **256 Kilobytes** of uncompressed payload. Over a 64 kbps rural wireless link, transmitting this single uncompressed file takes **over 35 to 45 seconds**. In an emergency, this delay is intolerable.
>
> Conversely, if you apply aggressive standard JPEG compression to shrink the file to under 20 Kilobytes, the Discrete Cosine Transform coarsely quantizes high-frequency coefficients. This completely obliterates faint, life-saving clinical features—such as ground-glass opacities and subtle consolidation patches.
>
> **This presents a fundamental clinical conflict:** If you do not compress, transmission latency kills. If you compress uniformly, diagnostic blur kills.
>
> MediLink-DL resolves this conflict. We introduce a closed-loop system where **Deep Learning actively guides lossy compression and network priority**, ensuring pristine diagnostic clarity where it matters, while slashing overall data weight by **89.9%**."

---

### Phase 2: Neural Backbone & The Saliency Drift Crisis (Minutes 03:00 – 06:00)

#### 🎬 Screen Action:
* Switch to **Slide 4** (Architecture) and open [`results/real_cxr_gradcam_comparison.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/real_cxr_gradcam_comparison.png) on screen.

#### ✍️ Whiteboard Action:
* Draw a simple diagram of an X-ray on the board:
  * Box representing CXR with lungs in the center.
  * Draw an 'L' marker in the corner and shoulder bone at the top.
  * Draw a red cross through the 'L' marker: *"Shortcut Learning (Saliency Drift)"*.
  * Write: $\mathcal{L}_{\text{cls}} \implies \text{Focuses on metal markers, NOT pathology!}$

#### 🗣️ Spoken Script (Word-for-Word English):
> "To guide our compression, we must automatically detect the pathological lung region. 
>
> In `src/train_real_models.py`, we construct **`MedicalResNet18`**. Standard computer vision networks expect 3-channel RGB photography. Radiographs are single-channel grayscale images with high dynamic range. Therefore, we adapted the input layer `conv1` from 3 channels to 1 channel with a $7 \times 7$ kernel and stride 2. We designate `backbone.layer4[-1]`—the final residual block containing 512 channels at spatial resolution $16 \times 16$—as our target representation layer.
>
> Now, here lies the critical research obstacle that naive systems overlook: **Saliency Drift**.
>
> When a deep neural network is trained solely with standard Categorical Cross-Entropy ($\mathcal{L}_{\text{cls}}$), it takes the path of least resistance. It discovers that non-pathological artifacts—such as radiopaque lateral metal markers ('L' or 'R'), patient positioning devices, or clavicle contours—correlate with hospital labels.
>
> Furthermore, classical Grad-CAM developed by Selvaraju et al. calls `.detach().numpy()` on feature maps. It was designed purely as a passive, post-hoc visualization tool. You **cannot** backpropagate through standard Grad-CAM because the computational graph is severed. Therefore, vanilla Grad-CAM models suffer from severe attention drift, achieving high classification accuracy while looking at completely the wrong anatomical location."

---

### Phase 3: Mathematical Derivation of Differentiable Grad-CAM & Loss Formulations (Minutes 06:00 – 09:30)

#### 🎬 Screen Action:
* Display **Slide 7** and [`docs/DL_AND_DIP_COMPREHENSIVE_GUIDE.md`](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/DL_AND_DIP_COMPREHENSIVE_GUIDE.md#L82-L175).

#### ✍️ Whiteboard Action:
* Write the 4 core mathematical formulas clearly on the whiteboard:
  1. $\alpha_k^c = \frac{1}{H_f W_f} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$
  2. $\mathbf{H} = \frac{\mathbf{S} - \min(\mathbf{S})}{\max(\mathbf{S}) - \min(\mathbf{S}) + \epsilon}$
  3. $\mathcal{L}_{\text{align}} = 1.0 - \frac{\sum (\mathbf{H} \cdot \mathbf{M})}{\sum \mathbf{H} + \epsilon}$
  4. $\mathcal{L}_{\text{dice}} = 1.0 - \frac{2 \sum (\mathbf{H} \cdot \mathbf{M}) + \epsilon}{\sum \mathbf{H} + \sum \mathbf{M} + \epsilon}$

#### 🗣️ Spoken Script (Word-for-Word English):
> "To solve this, we formulated **Differentiable Grad-CAM** in `src/localization_loss.py`. 
>
> As shown on the board:
> 1. Let $\mathbf{A} \in \mathbb{R}^{B \times 512 \times 16 \times 16}$ be the forward activation tensor of `layer4[-1]`, and $Y^c$ be the unnormalized logit score for pathology.
> 2. We extract the first-order gradients using `torch.autograd.grad(target_scores.sum(), A, create_graph=True)`. The parameter `create_graph=True` is the mathematical linchpin: it retains the autograd computation graph.
> 3. We compute the neuron importance weight $\alpha_k^c$ via Global Average Pooling over spatial dimensions.
> 4. We compute the linear combination $\mathbf{S} = \text{ReLU}(\sum_k \alpha_k^c \mathbf{A}^k)$, upsample bilinearly to $512 \times 512$, and apply differentiable min-max normalization to yield continuous heatmap $\mathbf{H} \in [0.0, 1.0]$.
>
> To actively steer this heatmap during training, we formulate three complementary loss objectives:
>
> * **First, Classification Loss ($\mathcal{L}_{\text{cls}}$):** Standard cross-entropy to guarantee diagnostic sensitivity.
> * **Second, Saliency Alignment Loss ($\mathcal{L}_{\text{align}}$):** As written on the board, $\sum (\mathbf{H} \cdot \mathbf{M})$ computes the total attention energy located strictly inside the certified lung mask $\mathbf{M}$. Dividing by $\sum \mathbf{H}$ yields the fraction of focus inside the lungs. If 100% of the model's attention is inside the lungs, the fraction is 1.0 and $\mathcal{L}_{\text{align}} = 0.0$. If the model looks at an external marker, this loss imposes a heavy penalty approaching 1.0!
> * **Third, Soft Continuous Dice Loss ($\mathcal{L}_{\text{dice}}$):** Alignment loss prevents leakage, but the network could cheat by collapsing into a microscopic point. Dice loss penalizes boundary mismatch, forcing the continuous heatmap to morphologically cover the entire clinical consolidation area.
>
> When we backpropagate our composite loss $\mathcal{L}_{\text{total}}$, **second-order gradients** ($\frac{\partial^2 \mathcal{L}}{\partial \mathbf{W} \partial \mathbf{A}}$) flow through the heatmap directly into the convolutional weights of ResNet18!"

---

### Phase 4: Empirical Breakthrough & Handover (Minutes 09:30 – 11:00)

#### 🎬 Screen Action:
* Show the 4-Regime Table from [`results/real_dataset_training_results.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/real_dataset_training_results.csv) and open [`results/real_cxr_gradcam_comparison.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/real_cxr_gradcam_comparison.png).

#### 🗣️ Spoken Script (Word-for-Word English):
> "We evaluated this across four rigorous experimental regimes on the clinical Cohen et al. chest radiograph dataset:
>
> * In **Regime A (Baseline CE)**, Pointing Game Accuracy was only **65.52%**. In more than 34% of cases, the maximum saliency peak was completely outside the lungs!
> * In **Regime C (Alignment Only)**, Pointing Game Accuracy surged to **96.55%—an absolute jump of over 31%**!
> * In **Regime D (Dice Only)**, the continuous Dice score jumped from **0.4764 to 0.7107—a 49.2% morphological improvement**!
> * In **Regime B (Combined)**, we achieved an optimal clinical balance: **93.10% Pointing Accuracy and 0.6856 Dice overlap**.
>
> As you can see in the projected figure, the bottom-left baseline model is drifting onto the clavicle bone, while our bottom-right model is strictly locked onto the pulmonary infiltrate.
>
> With our neural attention mathematically grounded, we now produce a certified diagnostic bounding box. I will now hand over to **Sumedha**, who will explain how our Digital Image Processing and Network subsystems leverage this attention map to compress and prioritize the transmission."

---

# SECTION 3: SPEAKER 2 — SUMEDHA
### Focus: Digital Image Processing (Dual-Stream JPEG & Container) & Computer Networks (Priority Queuing & ARQ)
**Allocated Time:** Minutes 11:00 – 22:00 (11 Minutes)  
**Target Evaluators to Impress:** Faculty 1 (DIP Specialist) & Faculty 2 (Computer Networks Specialist)

---

### Phase 1: Anatomical Dual-Stream Partitioning Principle (Minutes 11:00 – 14:00)

#### 🎬 Screen Action:
* Switch to **Slide 5** ("Dual-Stream ROI Compression & Fair Rate Matching") and project [`results/comparison.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/comparison.png).

#### ✍️ Whiteboard Action:
* Draw the dual-stream split on the whiteboard:
  * Full CXR Box $\to$ Split into:
    * Stream 1: Full-Frame Background ($q_{\text{bg}}=25$) $\implies$ ~12 KB.
    * Stream 2: Bounding Box ROI ($q_{\text{roi}}=90$) $\implies$ ~7 KB.
  * Show Total $\implies$ 19.2 KB (vs 256 KB Raw = 89.9% savings).

#### 🗣️ Spoken Script (Word-for-Word English):
> "Thank you, Nandan. Respected committee members, my name is **Sumedha**. Building upon the localized attention mask generated by our Deep Learning model, I will present the **Digital Image Processing (DIP)** and **Computer Networks (CN)** subsystems.
>
> Standard baseline JPEG (ISO/IEC 10918-1) operates uniformly: it subdivides an image into $8 \times 8$ pixel blocks, applies a 2D Discrete Cosine Transform, and divides all blocks by a single global quantization table $Q(u, v)$. You cannot tell standard JPEG to preserve the lungs while compressing the ribs.
>
> In [`src/roi_compression.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/roi_compression.py), we engineer **Dual-Stream Partitioning**:
> 1. We binarize the continuous Grad-CAM heatmap at threshold $\tau = 0.35$ to extract the minimum bounding box $(y_{\min}, x_{\min}, y_{\max}, x_{\max})$ containing the active pathology.
> 2. We extract the sub-array $\mathbf{I}_{\text{roi\_crop}} = \mathbf{I}[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$.
> 3. We generate two independent, compliant JPEG streams:
>    * **Stream 1 (Background Context):** The full frame encoded at quality $q_{\text{bg}}=25$. This takes only **~12.4 Kilobytes**. Peripheral anatomical landmarks like ribs and diaphragm remain clear enough for orientation, while high-frequency background noise is aggressively discarded.
>    * **Stream 2 (Diagnostic ROI Crop):** The cropped bounding box encoded at near-lossless quality $q_{\text{roi}}=90$. This takes only **~6.8 Kilobytes**. Here, fine parenchymal opacities and subtle consolidation borders are pristinely preserved.
> 4. At the receiving terminal, how do we prevent ugly block boundary seams where the high-res crop meets the background? We implement **Seamless Alpha Blending**:
>    $$\mathbf{I}_{\text{recon}}[\text{BBox}] = \mathbf{M}_{\text{sub}} \cdot \mathbf{I}_{\text{roi\_decomp}} + (1 - \mathbf{M}_{\text{sub}}) \cdot \mathbf{I}_{\text{bg\_decomp}}$$
>    where $\mathbf{M}_{\text{sub}}$ is the normalized sub-mask. The transition is anatomically smooth and artifact-free."

---

### Phase 2: The 28-Byte 'MLDL' Binary Container & Rate-Matched Benchmark (Minutes 14:00 – 17:00)

#### 🎬 Screen Action:
* Display **Slide 6** and the Rate-Matched Benchmark Table from [`docs/DL_AND_DIP_COMPREHENSIVE_GUIDE.md`](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/DL_AND_DIP_COMPREHENSIVE_GUIDE.md#L301-L310).

#### ✍️ Whiteboard Action:
* Write the 28-byte Header layout:
  * `0x00: b'MLDL'` (4B) | `0x04: Version` (2B) | `0x06: H, W` (4B) | `0x0A: q_roi, q_bg` (2B) | `0x0C: BBox` (8B) | `0x14: Len_BG, Len_ROI` (8B) = **28 Bytes Total**.

#### 🗣️ Spoken Script (Word-for-Word English):
> "To package these two byte streams into a single verifiable file without external sidecars, we engineered the **'MLDL' Binary Container Format**.
>
> Using Python's `struct.pack('>4sHHHBBHHHHII')`, we synthesize an ultra-compact **28-byte header**:
> * 4 bytes for magic identifier `b'MLDL'`.
> * 2 bytes for protocol version.
> * 4 bytes for original image dimensions.
> * 2 bytes for quality factors $(q_{\text{roi}}, q_{\text{bg}})$.
> * 8 bytes for bounding box coordinates $(y_{\min}, x_{\min}, y_{\max}, x_{\max})$.
> * 8 bytes recording the exact byte lengths $N$ and $M$ of the background and ROI streams.
> * Total header overhead is exactly **28 Bytes**—less than 0.15% of our transmission!
>
> Now, to prove to any skeptic that our dual-stream compression is genuinely superior to standard JPEG, we performed a **Fair Rate-Matched Benchmark** using `find_comparable_uniform_quality()`.
>
> A binary search established that encoding the entire image uniformly at quality factor **$q=31$** yields the **exact same wire payload of 19.2 Kilobytes**.
>
> Look at the quantitative results projected on screen:
> * At the exact same 19.2 KB budget, the uniform baseline achieved an ROI PSNR of only **38.21 dB**, displaying visible DCT blockiness and blurring of fine opacities.
> * MediLink-DL achieved an ROI PSNR of **48.34 dB**—an astronomical **+10.13 dB fidelity gain**!
> * Our ROI Structural Similarity (SSIM) reached **0.9892**, compared to 0.9412 for uniform JPEG.
>
> In the projected 5-panel comparison figure, panel 5 shows the absolute difference error map: inside the diagnostic lung box, the error is virtually zero!"

---

### Phase 3: Computer Networks — Priority Queuing, ARQ & Channel Simulation (Minutes 17:00 – 22:00)

#### 🎬 Screen Action:
* Switch to **Slide 7** ("Verified Results: Prioritized Transport & Multi-Rate Curves") and inspect [`src/transmission.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/transmission.py#L23-L70).

#### ✍️ Whiteboard Action:
* Draw a Priority Min-Heap queue:
  * Top: `Priority 30 (Header Packet)`
  * Middle: `Priority 20 (Diagnostic ROI Chunks)`
  * Bottom: `Priority 5 (Background Context Chunks)`
  * Arrow to Receiver: *ROI arrives in 0.78 sec $\implies$ Immediate Triage!*

#### 🗣️ Spoken Script (Word-for-Word English):
> "Now, how does this compressed payload travel across an unstable rural wireless channel? This is where our **Computer Networks (CN)** subsystem takes over.
>
> In [`src/transmission.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/transmission.py), we simulate an application-layer transport protocol specifically engineered for 2G/GPRS channels:
>
> 1. **Framing & Packetization:**  
>    The encrypted payload is segmented into discrete chunks of size $L$ (where $L \in \{256, 512, 1024, 2048\}$ bytes). Each packet is framed with a packet sequence ID, total packet count, payload type tag (`HEADER`, `ROI`, `BG`), and a **SHA-256 integrity checksum** computed over the chunk bytes.
>
> 2. **Min-Heap Priority Scheduling:**  
>    Instead of naive FIFO (First-In-First-Out) transmission, packets enter a thread-safe Min-Heap Priority Queue (`heapq`):
>    * **Priority 30 (Highest):** Container metadata and dimensions.
>    * **Priority 20 (Critical):** Diagnostic ROI chunks containing the lung pathology.
>    * **Priority 5 (Low):** Background contextual chunks.
>
>    **Why is this a clinical game changer?**  
>    In a naive transmission, a doctor must wait 100% of the transmission time to see anything. In MediLink-DL, the header and ROI chunks represent only ~7 KB. On a 64 kbps link, **the doctor receives and reconstructs the diagnostic lung region in just 0.78 seconds**! The doctor can make an immediate triage decision while the non-urgent background chunks lazily stream in the background.
>
> 3. **Channel Model & Selective-Repeat ARQ:**  
>    We model a real-world lossy rural link: bandwidth $R = 64\text{ kbps}$, latency $D = 120\text{ ms}$, and stochastic packet drop probability $P_{\text{loss}} \in [0.0, 0.15]$.  
>    If a packet is dropped or corrupted in transit, our **Selective-Repeat ARQ (Automatic Repeat reQuest)** retransmits ONLY the dropped packet ID. Unlike Go-Back-N, we do not retransmit already-acknowledged packets, preserving precious channel capacity.
>
> 4. **Store-and-Forward FIFO Resiliency:**  
>    If the rural cellular link experiences total drop-out, outgoing packets are buffered in a persistent FIFO queue, automatically resuming without data loss the moment connectivity returns.
>
> Now, how do we guarantee patient confidentiality, mathematical optimization, and tamper protection across this entire cross-layer pipeline? I will hand over to **Mayank Soni** to present Optimization, Cryptography, and the live system demonstration."

---

# SECTION 4: SPEAKER 3 — MAYANK SONI
### Focus: Continuous & Discrete Optimization, Cryptography (MLMC-128 Research), & Live Interactive Demonstration
**Allocated Time:** Minutes 22:00 – 35:00 (13 Minutes)  
**Target Evaluators to Impress:** Faculty 3 (Optimization Specialist) & Faculty 1/2 (System Integration)

---

### Phase 1: Level 1 Optimization — Kendall's Adaptive Homoscedastic Uncertainty Loss (Minutes 22:00 – 25:00)

#### 🎬 Screen Action:
* Open [`docs/DL_AND_DIP_COMPREHENSIVE_GUIDE.md`](file:///Users/mayanksoni/Desktop/MediLink-DL/docs/DL_AND_DIP_COMPREHENSIVE_GUIDE.md#L170-L195) and show [`src/localization_loss.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/localization_loss.py#L260-L295).

#### ✍️ Whiteboard Action:
* Write Kendall's Homoscedastic Uncertainty Equation on the whiteboard:
  $$\mathcal{L}_{\text{adaptive}} = \frac{1}{2} \sum_{i \in \{\text{cls}, \text{align}, \text{dice}\}} \left[ \exp(-s_i) \cdot \mathcal{L}_i + s_i \right], \quad \text{where } s_i = \log(\sigma_i^2)$$
  * Circle $s_i$: *"PyTorch `nn.Parameter` optimized via AdamW!"*

#### 🗣️ Spoken Script (Word-for-Word English):
> "Thank you, Sumedha. Respected committee members, my name is **Mayank Soni**. I will now present the **Optimization Framework**, the **Cryptographic Security Subsystem**, and conduct our live interactive demonstration.
>
> Optimization in MediLink-DL operates across two distinct mathematical levels:
>
> ### Level 1: Continuous Multi-Task Loss Balancing
> Earlier, Nandan explained our composite loss function: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{loc}}(\mathcal{L}_{\text{align}} + \beta \mathcal{L}_{\text{dice}})$.  
> In initial experiments, we manually calibrated $\lambda_{\text{loc}} = 1.0$ and $\beta = 0.5$. In optimization theory, this is known as **Static Scalarization**.
>
> But this raises a profound optimization question: *Can the network dynamically optimize its own loss penalties without manual heuristic guessing?*
>
> **The answer is yes.** In `src/localization_loss.py`, we implement `AdaptiveLocalizationAwareLoss` based on **Kendall et al.'s Homoscedastic Uncertainty Weighting** (CVPR 2018).
>
> As shown on the board:
> 1. We model the task variances as learnable parameters $s_i = \log(\sigma_i^2)$ registered as PyTorch `nn.Parameter(torch.zeros(1))`.
> 2. The objective function scales each loss by $\frac{1}{2}\exp(-s_i)$, balanced by a $+ \frac{1}{2}s_i$ regularizer.
> 3. Initializing $s_i = 0$ means all losses start with equal unit weights ($\exp(0) = 1.0$).
> 4. If the gradient variance of the alignment loss is high or noisy, the AdamW optimizer automatically increases $s_{\text{align}}$, thereby reducing its effective weight $\exp(-s_{\text{align}})$ to prevent destabilizing classification.
> 5. The $+ s_i$ regularizer acts as a barrier function, preventing the trivial solution where $s_i \to \infty$.
>
> Thus, the effective penalties emerge dynamically:
> $$\lambda_{\text{loc}}^{\text{eff}} = \frac{\exp(-s_{\text{align}})}{\exp(-s_{\text{cls}})}, \quad \beta^{\text{eff}} = \frac{\exp(-s_{\text{dice}})}{\exp(-s_{\text{align}})}$$
> Both the 11 million convolutional weights and the loss balancing parameters are solved simultaneously in a single unified backpropagation step!"

---

### Phase 2: Level 2 Optimization — Cross-Layer Constrained Pareto Frontier (Minutes 25:00 – 28:00)

#### 🎬 Screen Action:
* Switch to **Slide 6** and open [`results/pareto_frontier.png`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/pareto_frontier.png) and [`src/optimization.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/optimization.py).

#### ✍️ Whiteboard Action:
* Write the constrained multi-objective optimization problem:
  $$\min_{\mathbf{x} \in \mathcal{X}} \left[ -\text{SSIM}_{\text{roi}}(\mathbf{x}), \; T_{\text{roi}}(\mathbf{x}), \; T_{\text{total}}(\mathbf{x}), \; B_{\text{wire}}(\mathbf{x}), \; t_{\text{crypto}}(\mathbf{x}) \right]$$
  $$\text{subject to: } \text{AuthVerified}(\mathbf{x}) = \text{True}, \quad \text{ReconSuccess}(\mathbf{x}) = \text{True}$$
* Draw a 2D trade-off curve: $X$-axis = Transmission Latency ($T$), $Y$-axis = ROI Quality (SSIM). Draw the non-dominated Pareto frontier curve.

#### 🗣️ Spoken Script (Word-for-Word English):
> "Now let us examine **Level 2 Optimization: Discrete Cross-Layer Parameter Selection**.
>
> In real-world deployment, an engineer faces a multi-dimensional design space spanning Deep Learning, Image Processing, Networks, and Cryptography. Specifically, our configuration vector is:
> $$\mathbf{x} = (q_{\text{roi}}, q_{\text{bg}}, L_{\text{chunk}}, \text{Priority\_Mode}, \text{Cipher\_Choice}) \in \mathcal{X}$$
> Across our discrete search space, there are **108 valid system configurations**.
>
> These parameters conflict with one another:
> * Higher quality $q_{\text{roi}}$ improves SSIM, but increases wire bytes ($B_{\text{wire}}$) and transmission latency ($T_{\text{total}}$).
> * Smaller chunk size $L$ improves ARQ retransmission efficiency under high packet loss, but increases framing overhead.
> * Stronger cipher suites improve cryptographic margin, but increase computation time on low-power hardware.
>
> In [`src/optimization.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/optimization.py), we solve this using **Non-Dominated Pareto Sorting**:
> A configuration $\mathbf{x}_A$ dominates $\mathbf{x}_B$ ($\mathbf{x}_A \succ \mathbf{x}_B$) if it is strictly superior in at least one objective and no worse in any other.
>
> By pruning all dominated configurations, we identify the non-dominated **Pareto Frontier** (projected on screen in `pareto_frontier.png`). From this frontier, we synthesize **Four Certified Clinical Operating Regimes**:
> 1. **Ultra-Low Bandwidth Regime:** For 2G/GPRS channels ($q_{\text{roi}}=80, q_{\text{bg}}=15, L=256$, Ascon-128) $\implies$ Payload <14 KB, Latency <1.8s.
> 2. **Balanced Triage Regime (Default):** ($q_{\text{roi}}=90, q_{\text{bg}}=25, L=512$, AES-128-GCM) $\implies$ ROI delivered in 0.78s, SSIM = 0.989.
> 3. **Maximum Diagnostic Fidelity Regime:** For stable 3G/WiFi ($q_{\text{roi}}=95, q_{\text{bg}}=50, L=1024$, AES-256-GCM) $\implies$ ROI PSNR >52 dB.
> 4. **High-Reliability Severe Loss Regime:** For links with >15% packet loss ($L=256$, Priority Enabled, MLMC-128).
>
> Rather than guessing ad-hoc parameters, our pipeline dynamically selects the mathematically optimal Pareto point based on real-time channel feedback!"

---

### Phase 3: Cryptographic Security & Our Original MLMC-128 Research (Minutes 28:00 – 31:30)

#### 🎬 Screen Action:
* Project [`results/crypto_research/mlmc_benchmark_results.csv`](file:///Users/mayanksoni/Desktop/MediLink-DL/results/crypto_research/mlmc_benchmark_results.csv) and open [`src/crypto_research/mlmc_cipher.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/crypto_research/mlmc_cipher.py).

#### ✍️ Whiteboard Action:
* Draw the AEAD wire envelope:
  `[ Nonce (12B) ] + [ Ciphertext (~19.2 KB) ] + [ 128-bit Auth Tag (16B) ]`
* Write: *1-bit tamper $\implies$ Tag Mismatch $\implies$ ZERO-PLAINTEXT RELEASE!*

#### 🗣️ Spoken Script (Word-for-Word English):
> "Now let us address the critical healthcare mandate: **Security and Cryptographic Integrity**.
>
> Transmitting medical records over public cellular airwaves exposes patients to two fatal threats:
> 1. **Eavesdropping (Privacy Violation):** Interception of confidential radiographs violates HIPAA compliance.
> 2. **Tampering (Man-in-the-Middle Manipulation):** If an adversary or transmission noise flips a single bit in the consolidation area, a patient with acute pneumonia could be misdiagnosed as normal!
>
> In [`src/security.py`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/security.py), we strictly enforce **AEAD (Authenticated Encryption with Associated Data)** with **Zero-Plaintext Release**:
> * We encrypt the 19.2 KB payload and attach a **128-bit cryptographic authentication tag**.
> * We enforce per-operation fresh 12-byte nonces generated via OS-level CSPRNG (`os.urandom(12)`), preventing replay attacks.
> * If even **one single bit** of ciphertext or tag is modified during transmission, decryption halts immediately with `DecryptionTamperedError`. **Zero plaintext is released to memory**, guaranteeing that a doctor never views a corrupted radiograph.
>
> ### Our Original Cryptographic Research: MLMC-128
> We did not simply import off-the-shelf libraries. In [`src/crypto_research/`](file:///Users/mayanksoni/Desktop/MediLink-DL/src/crypto_research/), we formulated, implemented, and verified **MLMC-128 (MediLink Mini Cipher)**:
> * It is a 128-bit block, 128-bit key, **10-round Substitution-Permutation Network (SPN)**.
> * It incorporates a non-linear S-Box and a Maximum Distance Separable (MDS) matrix over Galois Field $GF(2^8)$.
> * We executed **4,800 empirical trials** to test the **Strict Avalanche Criterion (SAC)**. The mathematical ideal is that flipping 1 input bit causes exactly 50% of output bits to invert. **MLMC-128 achieved an empirical average of 49.92% (deviation of only 0.08%)**, reaching complete diffusion by round 3!
> * We benchmarked MLMC-128 alongside **AES-128-GCM, AES-256-GCM, and Ascon-AEAD128** (the 2023 NIST lightweight champion for battery-operated IoT). The total cryptographic envelope overhead is only **28 to 32 bytes** (<0.15% of payload)!"

---

### Phase 4: Live Web Application Demonstration & Conclusion (Minutes 31:30 – 35:00)

#### 🎬 Screen Action:
* Switch to the web browser showing **`http://localhost:8080`** (running `src/web_app.py`).
* Walk through Tabs 1, 2, 3, and 4 live!

#### 🗣️ Spoken Script (Word-for-Word English):
> "To demonstrate that this is a fully working, robust software artifact and not merely theoretical slides, I will now demonstrate our interactive dashboard running live on port 8080.
>
> *(Clicking Tab 1: Live Transmission)*  
> 'Here on **Tab 1**, we select a clinical chest radiograph and transmit it using our Dual-Stream pipeline with MLMC-128 encryption. As you see, the raw 256 KB image is compressed to **19.2 Kilobytes**—an **89.9% reduction**. The reconstructed diagnostic ROI achieves a PSNR of **48.34 dB**, with the lungs clearly visualised.'
>
> *(Clicking Tab 2: Grad-CAM Saliency Viewer)*  
> 'On **Tab 2**, we visualize the Differentiable Grad-CAM feature activations. Notice how the attention map is strictly localized within the bilateral lung boundaries, with zero attention leakage onto the lateral metal markers.'
>
> *(Clicking Tab 3: Interactive Bit-Flip Tamper Testing)*  
> 'On **Tab 3**, let us simulate an active man-in-the-middle attack. I click **"Test MLMC-128 Bit-Flip"**. The system intentionally flips bit 42 in the encrypted ciphertext. Watch the result: The authentication tag fails instantly! The system raises `DecryptionTamperedError`, displays a red security alert, and aborts reconstruction. Zero corrupted data reaches the clinician.'
>
> *(Clicking Tab 4: Pareto Optimization Explorer)*  
> 'Finally, on **Tab 4**, we explore the Pareto frontier across all 108 combinations, showing the real-time trade-off between latency, SSIM, and channel bandwidth.'
>
> ### Conclusion:
> In summary, MediLink-DL delivers:
> 1. An **89.9% wire reduction** with **+10.13 dB higher diagnostic fidelity** than rate-matched JPEG.
> 2. Elimination of saliency drift, jumping Pointing Game Accuracy from **65.52% to 96.55%**.
> 3. Sub-second emergency triage (0.78s ROI delivery) over 2G 64 kbps links.
> 4. End-to-end authenticated cryptographic protection verified across 4 ciphers.
> 5. A 100% verified software codebase with all **147 unit tests passing**.
>
> We thank the committee for your attention, and we are now ready for your questions."

---

# SECTION 5: VIVA VOCE DEFENSE MANUAL (TOUGH COUNTER-QUESTIONS & KNOCKOUT ANSWERS)

---

## Part A: Questions from Faculty 1 (Deep Learning & DIP Specialist)

### Q1: "Why did you use Differentiable Grad-CAM instead of training a standard U-Net for lung segmentation?"
> **Nandan's Answer:**  
> "That is an excellent question, Sir. While a U-Net segments anatomical structures (the healthy lungs), it requires dense, pixel-level semantic masks which are extremely expensive to annotate clinically. More importantly, U-Net segments the entire lung, not the specific pathology!  
> In contrast, our Differentiable Grad-CAM identifies the specific **active pathological consolidations** directly from classification logits. By preserving the autograd graph with `create_graph=True` and backpropagating $\mathcal{L}_{\text{total}}$, we solve classification and pathology localization simultaneously in a single lightweight ResNet18 model without requiring the 30 million parameters of an auxiliary segmentation network."

### Q2: "In your dual-stream compression, won't the doctor notice the boundary where the high-res ROI is pasted over the low-res background?"
> **Sumedha's Answer:**  
> "Sir, that would indeed happen if we used naive rectangular copy-pasting. That is precisely why we implemented **Seamless Alpha Blending** in `src/roi_compression.py`:
> $$\mathbf{I}_{\text{recon}}[\text{BBox}] = \mathbf{M}_{\text{sub}} \cdot \mathbf{I}_{\text{roi}} + (1 - \mathbf{M}_{\text{sub}}) \cdot \mathbf{I}_{\text{bg}}$$
> We take the normalized continuous Grad-CAM sub-mask and use it as an alpha blending weight. At the center of the lesion, the weight is 1.0 (pure high-res). At the boundary, it smoothly interpolates to the background. This completely eliminates high-frequency edge seams, preventing artificial edge illusions."

### Q3: "What is the difference between $\mathcal{L}_{\text{align}}$ and $\mathcal{L}_{\text{dice}}$? Why wasn't alignment loss alone sufficient?"
> **Nandan's Answer:**  
> "Sir, our ablation study in `results/real_dataset_training_results.csv` proves why both are necessary:  
> * $\mathcal{L}_{\text{align}}$ penalizes attention outside the reference mask. In Regime C (Align Only), it achieved a remarkable 96.55% Pointing Accuracy, but the Dice score was only 0.5799 because the network collapsed its attention into a single high-confidence point inside the lung.  
> * $\mathcal{L}_{\text{dice}}$ optimizes shape overlap. In Regime D (Dice Only), Dice reached 0.7107, but the Pointing Accuracy dropped to 82.76%.  
> Therefore, combining them ($\mathcal{L}_{\text{align}} + 0.5 \mathcal{L}_{\text{dice}}$) gave us the best clinical equilibrium: 93.10% Pointing Accuracy and 0.6856 Dice overlap."

---

## Part B: Questions from Faculty 2 (Computer Networks Specialist)

### Q4: "Why did you implement application-layer Selective-Repeat ARQ instead of just relying on TCP?"
> **Sumedha's Answer:**  
> "Sir, standard TCP suffers from severe performance degradation over lossy wireless links. When TCP encounters packet loss on a 2G cellular link (where loss is caused by wireless fading and noise rather than buffer congestion), TCP's congestion control algorithms (like Reno or Cubic) mistakenly interpret loss as congestion. They cut the congestion window in half, collapsing throughput from 64 kbps to single digits!  
> Furthermore, TCP enforces strict byte-stream in-order delivery; head-of-line blocking stalls the entire application.  
> Our application-layer transport protocol operates with prioritized chunking. It allows the high-priority ROI chunks to be decoded and displayed immediately, while Selective-Repeat ARQ requests retransmission only for lost packet IDs without triggering TCP congestion collapse."

### Q5: "What is your packet framing overhead, and what happens during complete link disconnection?"
> **Sumedha's Answer:**  
> "Sir, each packet is framed with a 16-byte metadata header: Sequence ID (4B), Total Packets (4B), Payload Type (2B), Data Length (2B), and a SHA-256 checksum (4B truncated or 32B full hash). At our optimal chunk size of 1024 bytes, framing overhead is under 3%.  
> If complete network loss occurs, our persistent FIFO buffer enters **Store-and-Forward mode**. It suspends transmission without memory exhaustion and resumes from the exact missing sequence ID the moment channel handshake is re-established."

---

## Part C: Questions from Faculty 3 (Optimization Specialist)

### Q6: "In Kendall's homoscedastic uncertainty loss, why do you optimize $s = \log(\sigma^2)$ instead of $\sigma^2$ directly?"
> **Mayank Soni's Answer:**  
> "Sir, that is a fundamental numerical optimization requirement.  
> If we optimized $\sigma^2$ directly, gradient descent could easily push $\sigma^2 \le 0$ during an update step, which would cause an unrecoverable division by zero or square root of a negative number.  
> By parameterizing $s = \log(\sigma^2)$, the variance is $\sigma^2 = \exp(s)$, which is mathematically guaranteed to remain strictly positive for all $s \in \mathbb{R}$.  
> Furthermore, the term $\frac{1}{2\sigma^2}$ becomes $\frac{1}{2}\exp(-s)$, and the $\log(\sigma)$ regularizer simplifies elegantly to $\frac{1}{2}s$. The gradient with respect to $s$ is well-behaved and numerically stable throughout training."

### Q7: "How did you compute the Pareto Frontier across the 108 configurations? What algorithm was used?"
> **Mayank Soni's Answer:**  
> "Sir, we implemented **Non-Dominated Sorting (based on Goldberg and Deb's NSGA-II principles)** in `src/optimization.py`.  
> For each candidate vector $\mathbf{x} \in \mathcal{X}$, we evaluate our five objective functions: $[- \text{SSIM}_{\text{roi}}, T_{\text{roi}}, T_{\text{total}}, B_{\text{wire}}, t_{\text{crypto}}]$.  
> We compute the domination count $n_p$ (the number of solutions that dominate $p$) and set $S_p$ (the set of solutions that $p$ dominates). All solutions with $n_p = 0$ form the first non-dominated front (Rank 1 Pareto Frontier).  
> In our search space, the Pareto frontier contains 9 non-dominated points. From these, we mapped the operational points that satisfy our clinical constraints to define our four operational regimes."

### Q8: "Why not use a Genetic Algorithm (GA) or Particle Swarm Optimization (PSO) instead of grid evaluation for the Pareto front?"
> **Mayank Soni's Answer:**  
> "Sir, GA and PSO are stochastic heuristic algorithms used when the search space is continuous or combinatorially vast (e.g., millions of states), where exhaustive enumeration is intractable.  
> In our system, the engineering decision variables are discrete: 3 quality pairs $\times$ 4 chunk sizes $\times$ 3 ciphers $\times$ 3 priority modes = **exactly 108 configurations**.  
> Running exhaustive non-dominated sorting across 108 states takes only **4.2 seconds on CPU** and guarantees the **exact global Pareto frontier**, with zero stochastic variance, zero local minima traps, and 100% mathematical certainty. For our deployment scale, exhaustive non-dominated sorting is mathematically superior to stochastic approximations."

### Q9: "Why is your MLMC-128 cipher's Avalanche Effect of 49.92% significant?"
> **Mayank Soni's Answer:**  
> "Sir, in cryptanalysis, the Strict Avalanche Criterion (Webster & Tavares, 1985) states that if any single input bit is flipped, each output bit must change with a probability of exactly 0.5 (50%). If a cipher achieves only 10% or 90% bit flips, an adversary can exploit linear cryptanalysis or differential trails to deduce round keys.  
> Our empirical test across 4,800 trials yielded **49.92% with a standard deviation of 4.45%**. The deviation from ideal is a negligible 0.08%, proving that MLMC-128 provides complete cryptographic confusion and diffusion within 3 rounds."

---

# SECTION 6: EMERGENCY CONTINGENCY & SETUP CHECKLIST

Before entering the presentation room tomorrow morning:
1. **Activate Virtual Environment & Run Tests:**
   ```bash
   cd /Users/mayanksoni/Desktop/MediLink-DL
   ./.venv/bin/pytest tests/ -q
   # Confirm: 147 passed in ~10 seconds
   ```
2. **Start Web Application in Background:**
   ```bash
   ./.venv/bin/python3 src/web_app.py 8080
   # Open browser tab at: http://localhost:8080
   ```
3. **Keep Visual Results Ready in Preview:**
   ```bash
   open results/real_cxr_gradcam_comparison.png
   open results/comparison.png
   open results/pareto_frontier.png
   open docs/MediLink_DL_DIP_Technical_Report.pdf
   ```
4. **Whiteboard Marker Ready:** Keep black and blue markers for drawing the loss formulas and the 28-byte packet header.

