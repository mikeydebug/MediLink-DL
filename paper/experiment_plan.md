# Experiment Plan

## 1. Overview and Objective
This experiment plan details the protocols, independent variables, dependent metrics, and execution steps for evaluating the **MediLink-DL** framework. All experiments are fully automated, CPU-reproducible, and deterministic via fixed random seeds.

---

## 2. Experiment 1: Rate-Distortion Comparison (Uniform vs. ROI-Aware)

### Objective
Determine whether allocating higher quantization fidelity to the diagnostic ROI yields superior regional PSNR and SSIM compared to a rate-matched uniform JPEG baseline.

### Independent Variables
- **ROI Quality Factor ($q_{\text{roi}}$)**: Fixed at 90.
- **Background Quality Factor ($q_{\text{bg}}$)**: Stepped through $\{15, 25, 35, 50\}$.
- **Matched Uniform Quality ($q_{\text{uniform}}$)**: Dynamically solved via binary/linear search to match the byte size of each ROI-aware compressed payload within $\pm 5\%$.

### Dependent Metrics
- Payload size (bytes) & Compression Ratio.
- Global PSNR (dB) and Global SSIM.
- Regional ROI PSNR (dB) and Regional ROI SSIM.
- Regional Background PSNR (dB) and Regional Background SSIM.

### Reproducibility Command
```bash
./.venv/bin/python src/main.py --mode experiment-compression
```

---

## 3. Experiment 2: Image Quality Across Multi-Scale Compression Settings

### Objective
Quantify the sensitivity of diagnostic fidelity (ROI SSIM) to aggressive background compression, establishing the pareto-optimal operational regime for 32–128 kbps channels.

### Evaluation Grid
- $q_{\text{bg}} \in [10, 60]$ in steps of 5.
- $q_{\text{roi}} \in \{80, 85, 90, 95\}$.
- Test inputs: Synthetic CXR Phantom ($512 \times 512$).

---

## 4. Experiment 3: Constrained Network Transmission and Prioritization

### Objective
Evaluate the temporal delivery advantage of prioritized dual-stream transmission over non-prioritized FIFO transmission across simulated rural cellular/satellite links.

### Network Channel Profiles
1. **Low-Bandwidth 2G/EDGE**:
   - Bandwidth: 48 kbps
   - Latency: 200 ms
   - Packet Loss: 2%
2. **Constrained 3G / Remote Satellite**:
   - Bandwidth: 128 kbps
   - Latency: 450 ms
   - Packet Loss: 5%
3. **Severe Lossy Link**:
   - Bandwidth: 64 kbps
   - Latency: 120 ms
   - Packet Loss: 10%

### Dependent Metrics
- **Time to First Diagnostic ROI ($T_{\text{ROI}}$)**: Elapsed time until all ROI packets arrive and pass checksum verification at the receiver.
- **Total Transfer Time ($T_{\text{total}}$)**: Elapsed time until the complete context frame is reassembled.
- **Retransmissions (ARQ)**: Count of dropped/corrupted packets retransmitted.
- **Effective Throughput (kbps)**.

### Reproducibility Command
```bash
./.venv/bin/python src/main.py --mode experiment-transmission
```

---

## 5. Experiment 4: Cryptographic Overhead & Tamper Detection

### Objective
Measure the computational latency and storage overhead of AES-256-GCM authenticated envelope encryption, and empirically verify that 100% of bit-inversion attacks on the ciphertext are rejected by the authentication tag.

### Test Protocol
1. Encrypt payload with a freshly generated 96-bit nonce and AES-256 key.
2. Record encryption time, decryption time, and ciphertext overhead (28 bytes total: 12-byte nonce + 16-byte tag).
3. Inject bit-flip mutations at random byte positions in the ciphertext.
4. Verify that `cryptography.exceptions.InvalidTag` is raised on every corrupted envelope.

### Reproducibility Command
```bash
./.venv/bin/python src/main.py --mode experiment-security
```

---

## 6. Recorded Empirical Results
*(Baseline measurements recorded from initial prototype execution in `results/metrics.csv`)*

| Evaluation Mode | Payload (Bytes) | Compression Ratio | Global PSNR | ROI PSNR | ROI SSIM | Background PSNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ROI-Aware ($q_{\text{roi}}=90, q_{\text{bg}}=25$)** | 9,842 | **26.64x** | 32.51 dB | **38.08 dB** | **0.9293** | 32.47 dB |
| **Uniform Baseline ($q=32$)** | 9,622 | 27.24x | 33.04 dB | 34.58 dB | 0.8440 | 33.02 dB |
| **Empirical Advantage** | Rate-matched | — | -0.53 dB | **+3.50 dB** | **+0.0853** | -0.55 dB |

**Transmission Performance (64 kbps, 120 ms latency, 5% loss)**:
- Time to deliver diagnostic ROI: **0.384 seconds**.
- Time to deliver full background image: **2.608 seconds**.
- Telemedicine Diagnostic Acceleration: **6.79x faster triage** availability for the consulting radiologist compared to waiting for the full raster.
