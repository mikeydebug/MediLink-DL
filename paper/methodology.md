# Methodology

## 1. System Architecture Overview
The MediLink-DL framework is structured into five sequential, reproducible modules:
1. **Diagnostic ROI Localization**: Derivation of spatial diagnostic focus masks.
2. **Dual-Stream ROI-Aware Compression**: Differential bit allocation between pathology (high quality) and background context (low quality).
3. **Cryptographic Protection**: AES-256-GCM envelope encryption with per-operation nonces.
4. **Prioritized Application-Layer Transport**: Packet chunking, priority-ordered transmission queueing, and Stop-and-Wait / Selective Repeat ARQ.
5. **Partitioned Distortion & Transmission Evaluation**: Global, ROI, and background PSNR and SSIM alongside channel latency and throughput measurements.

```
+--------------------+      +--------------------+
| Grayscale CXR (I)  | ---> | Diagnostic ROI (M) |
+--------------------+      +--------------------+
         |                           |
         +-------------+-------------+
                       v
         +----------------------------+
         | Dual-Stream ROI Compression|
         | (q_roi=90, q_bg=25)        |
         +----------------------------+
                       v
         +----------------------------+
         | AES-256-GCM Authentication |
         | (Fresh 96-bit Nonce)       |
         +----------------------------+
                       v
         +----------------------------+
         | Prioritized Packet Transmit|
         | (Header > ROI > Background)|
         +----------------------------+
                       v
         +----------------------------+
         | Receiver Reassembly &      |
         | Partitioned Metrics (PSNR) |
         +----------------------------+
```

---

## 2. Image Representation and Preprocessing
- **Pixel Grid**: 8-bit single-channel grayscale $I \in [0, 255]^{H \times W}$, natively standard in digital radiography viewports.
- **Reference Phantom Generation**: In the absence of local clinical PACS access, an anatomically stylized digital phantom ($512 \times 512$ pixels) simulates thoracic boundaries, radiolucent lung fields, cardiac shadows, vascularity, and a localized focal consolidation lesion centered at $(y_0, x_0)$ with radius $r$. An indelible disclaimer banner (`[SYNTHETIC PHANTOM - NOT A REAL MEDICAL X-RAY]`) is stamped onto pixel rows $[H-24, H]$.

---

## 3. Dual-Stream ROI-Aware Differential Compression
Standard baseline JPEG (ITU-T T.81) operates with uniform quantization tables across the entire image raster. To provide an operational, CPU-efficient, and universally decompressible prototype, we introduce a dual-stream payload:
1. **Background Full-Frame Stream**: The image $I$ is compressed with baseline DCT JPEG at aggressive lossy quality $q_{\text{bg}} \in [10, 30]$:
   $$\mathcal{B}_{\text{bg}} = \text{JPEG}_{\text{enc}}(I, q_{\text{bg}})$$
2. **Localized ROI Crop Stream**: Given binary mask $M \in \{0, 1\}^{H \times W}$, compute the minimal enclosing bounding box $\text{bbox} = (y_{\min}, x_{\min}, y_{\max}, x_{\max})$. The sub-image $I_{\text{roi}} = I[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$ is compressed at diagnostic quality $q_{\text{roi}} \in [85, 95]$:
   $$\mathcal{B}_{\text{roi}} = \text{JPEG}_{\text{enc}}(I_{\text{roi}}, q_{\text{roi}})$$
3. **Binary Packaging**: The streams are bundled with a 26-byte binary header (`MLDL` magic, dimensions, bounding box coordinates, and payload lengths).
4. **Client-Side Reconstruction**: The receiver decodes both streams and composites the high-fidelity crop into the context frame:
   $$I_{\text{recon}}[y_{\min}:y_{\max}, x_{\min}:x_{\max}] = \text{JPEG}_{\text{dec}}(\mathcal{B}_{\text{roi}})$$

---

## 4. Rate-Matched Uniform Baseline Construction
To satisfy strict experimental fairness, ROI-aware coding must not be compared to arbitrary uniform quality factors. We implement an iterative sweep to identify $q_{\text{uniform}} \in [1, 95]$ such that:
$$q^* = \arg\min_{q \in [1, 95]} \left| \text{len}(\text{JPEG}_{\text{enc}}(I, q)) - \text{len}(\mathcal{P}_{\text{roi}}) \right|$$
This guarantees that both reconstructed images utilize an essentially identical byte budget over the wire.

---

## 5. Image Quality Evaluation Metrics
Let $I$ be the reference image and $\hat{I}$ be the reconstructed image.
1. **Partitioned Mean Squared Error (MSE)**:
   $$\text{MSE}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (I(i,j) - \hat{I}(i,j))^2$$
   where $\Omega$ corresponds to:
   - $\Omega_{\text{global}} = \{ (i,j) : 0 \le i < H, 0 \le j < W \}$
   - $\Omega_{\text{roi}} = \{ (i,j) : M(i,j) = 1 \}$
   - $\Omega_{\text{bg}} = \{ (i,j) : M(i,j) = 0 \}$
2. **Partitioned PSNR**:
   $$\text{PSNR}_{\Omega} = 10 \log_{10} \left( \frac{255^2}{\text{MSE}_{\Omega}} \right)$$
3. **Partitioned SSIM**:
   Using Gaussian window filtering (Wang et al., 2004), the spatial SSIM map $S(i,j)$ is computed globally, and regional averages are derived:
   $$\text{SSIM}_{\Omega} = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} S(i,j)$$

---

## 6. Cryptographic Security Protocol
- **Cipher**: AES-256 in Galois/Counter Mode (AES-GCM, NIST SP 800-38D).
- **Nonce Discipline**: Every encryption invocation generates a cryptographically random 96-bit nonce ($N \leftarrow \text{CSPRNG}(12)$). Nonces are never reused.
- **Envelope Structure**:
  $$\text{Envelope} = N \,\|\, \text{AES-GCM-Enc}_{K}(N, \mathcal{P}, \text{AAD})$$
- **Tamper Detection**: Verification of the 128-bit authentication tag guarantees detection of unauthorized modification.

---

## 7. Priority-Based Network Transmission Simulation
- **Packetization**: Payloads are partitioned into $L$-byte chunks (default 1024 bytes) framed with a sequence ID, total packet count, payload type, and SHA-256 checksum.
- **Priority Scheduling**: Chunks are queued in a min-heap priority queue:
  - Header Chunks: Priority 30 (highest)
  - ROI Diagnostic Chunks: Priority 20
  - Background Context Chunks: Priority 5
- **Channel Model**:
  - Bandwidth: $R = 64\text{ kbps}$
  - One-way Latency: $D = 120\text{ ms}$
  - Packet Loss Probability: $P_{\text{loss}} \in [0.0, 0.15]$
- **Store-and-Forward**: A local persistent FIFO buffer handles upstream disconnections, holding packets until the link is restored.
