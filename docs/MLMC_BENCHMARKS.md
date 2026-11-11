# MediLink-DL: Empirical Cryptographic Benchmark Report

**Benchmarking Suite**: `src/crypto_research/benchmark_mlmc.py`  
**Data File**: `results/crypto_research/mlmc_benchmark_results.csv`  
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
**Date**: November 2026 &nbsp;|&nbsp; **Empirical Cryptographic Benchmark Report**  
**Test Environment**: Darwin arm64 (Apple Silicon), Python 3.11.x, OpenSSL 3.x / Cryptography 50.0.2  

---

## 1. Benchmarking Methodology & Critical Disclosures

To ensure scientific honesty and reproducibility, this benchmark establishes the following ground rules:

1. **Implementation Disclosures**:
   - `AES-256-GCM` and `AES-128-GCM` utilize OpenSSL C bindings with **hardware-accelerated AES-NI / ARMv8 Crypto Extensions**.
   - `Ascon-AEAD128` is evaluated using its **pure-Python constant-time reference implementation** (`src/ascon.py`).
   - `MLMC-128 Raw` and `MLMC-GCM-Exp` are evaluated using **pure-Python research reference implementations** (`src/crypto_research/mlmc_cipher.py`).
   - Direct execution-time comparisons between Python and compiled C/hardware primitives reflect runtime overheads rather than theoretical gate-count complexity.
2. **Measurement Protocol**:
   - 100 trials per payload size preceded by 10 warm-up runs.
   - High-resolution timing via `time.perf_counter_ns()`.
   - Reported values are median execution latencies in microseconds ($\mu$s).
   - Throughput is calculated on total roundtrip processing: $\text{Throughput} = \frac{\text{Payload Bytes}}{(\text{Enc Median} + \text{Dec Median})}$.

---

## 2. Empirical Benchmark Measurements

### 2.1 Per-Chunk Processing Latency ($\mu$s) & Throughput (MB/s)

| Payload Size | Candidate | Type | Implementation | Enc ($\mu$s) | Dec ($\mu$s) | Throughput | Wire Overhead |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **256 B** | **AES-128-GCM** | Standard AEAD | C / HW Accelerated | **2.2** | **1.5** | **65.82 MB/s** | 28 B |
| | **AES-256-GCM** | Standard AEAD | C / HW Accelerated | **2.2** | **1.5** | **65.10 MB/s** | 28 B |
| | **Ascon-AEAD128** | Standard AEAD | Pure Python Ref | **414.7** | **413.0** | **0.29 MB/s** | 32 B |
| | **MLMC-128 Raw** | Exp. Block | Pure Python SPN | **1,686.8** | **1,678.2** | **0.07 MB/s** | 0 B |
| | **MLMC-GCM-Exp** | Exp. AEAD | Pure Python CTR+GHASH | **2,089.5** | **2,078.7** | **0.06 MB/s** | 28 B |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **512 B** | **AES-128-GCM** | Standard AEAD | C / HW Accelerated | **2.1** | **1.4** | **141.20 MB/s** | 28 B |
| | **AES-256-GCM** | Standard AEAD | C / HW Accelerated | **2.3** | **1.5** | **127.36 MB/s** | 28 B |
| | **Ascon-AEAD128** | Standard AEAD | Pure Python Ref | **709.2** | **708.9** | **0.34 MB/s** | 32 B |
| | **MLMC-128 Raw** | Exp. Block | Pure Python SPN | **3,358.0** | **3,348.7** | **0.07 MB/s** | 0 B |
| | **MLMC-GCM-Exp** | Exp. AEAD | Pure Python CTR+GHASH | **4,036.9** | **4,025.6** | **0.06 MB/s** | 28 B |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1024 B** | **AES-128-GCM** | Standard AEAD | C / HW Accelerated | **2.2** | **1.5** | **257.53 MB/s** | 28 B |
| | **AES-256-GCM** | Standard AEAD | C / HW Accelerated | **2.3** | **1.6** | **246.67 MB/s** | 28 B |
| | **Ascon-AEAD128** | Standard AEAD | Pure Python Ref | **1,337.6** | **1,336.3** | **0.37 MB/s** | 32 B |
| | **MLMC-128 Raw** | Exp. Block | Pure Python SPN | **6,720.4** | **6,688.8** | **0.07 MB/s** | 0 B |
| | **MLMC-GCM-Exp** | Exp. AEAD | Pure Python CTR+GHASH | **7,889.4** | **7,863.0** | **0.06 MB/s** | 28 B |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **2048 B** | **AES-128-GCM** | Standard AEAD | C / HW Accelerated | **2.5** | **1.7** | **468.71 MB/s** | 28 B |
| | **AES-256-GCM** | Standard AEAD | C / HW Accelerated | **2.5** | **1.8** | **450.65 MB/s** | 28 B |
| | **Ascon-AEAD128** | Standard AEAD | Pure Python Ref | **2,608.5** | **2,603.0** | **0.37 MB/s** | 32 B |
| | **MLMC-128 Raw** | Exp. Block | Pure Python SPN | **13,373.0** | **13,341.2** | **0.07 MB/s** | 0 B |
| | **MLMC-GCM-Exp** | Exp. AEAD | Pure Python CTR+GHASH | **15,564.3** | **15,512.4** | **0.06 MB/s** | 28 B |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **4096 B** | **AES-128-GCM** | Standard AEAD | C / HW Accelerated | **2.7** | **2.0** | **837.17 MB/s** | 28 B |
| | **AES-256-GCM** | Standard AEAD | C / HW Accelerated | **2.8** | **2.0** | **801.12 MB/s** | 28 B |
| | **Ascon-AEAD128** | Standard AEAD | Pure Python Ref | **5,150.0** | **5,143.4** | **0.38 MB/s** | 32 B |
| | **MLMC-128 Raw** | Exp. Block | Pure Python SPN | **26,838.8** | **26,745.8** | **0.07 MB/s** | 0 B |
| | **MLMC-GCM-Exp** | Exp. AEAD | Pure Python CTR+GHASH | **31,165.3** | **30,990.8** | **0.06 MB/s** | 28 B |

---

## 3. High-Level Cryptographic Architecture Comparison

| Dimension | NIST AES-GCM | NIST Ascon-AEAD128 | MLMC Candidate |
| :--- | :--- | :--- | :--- |
| **Standardization Status** | NIST SP 800-38D (Global Gold Standard) | NIST SP 800-232 (Lightweight Winner) | **Unstandardized Research Candidate** |
| **Permutation / Block** | 128-bit block (SPN) | 320-bit permutation state (Sponge) | 128-bit block (SPN) |
| **Key Size** | 128 or 256 bits | 128 bits | 128 bits |
| **Hardware Support** | Native AES-NI / ARMv8 Crypto Extensions | Optimized 64-bit word operations | Educational / Software emulation |
| **Tamper Detection** | Constant-time GHASH / GF(2^128) | Constant-time Sponge state squeeze | Constant-time GHASH / GF(2^128) |
| **Suitability for Patient Care** | **MANDATORY PRODUCTION USE** | **APPROVED LIGHTWEIGHT STANDARD** | **PROHIBITED (ACADEMIC ONLY)** |

---

## 4. Key Findings

1. **Hardware Acceleration Dominance**: Hardware-assisted AES-GCM processes 1024-byte chunks in ~2.2 $\mu$s (> 250 MB/s), outperforming all pure-software interpreters by two orders of magnitude.
2. **Ascon vs MLMC in Software**: Ascon's bit-slice 64-bit boolean bitwise operations achieve **$5.5\times$ higher throughput** than MLMC in pure Python, because Ascon avoids finite field polynomial matrix multiplications.
3. **Telemedicine Channel Perspective**: On a 64 kbps rural link (8 KB/s), transmitting a 1024-byte chunk takes **128,000 $\mu$s (128 ms)** over the wire. Cryptographic processing time (2 $\mu$s for AES-GCM, 1.3 ms for Ascon, 7.8 ms for MLMC) is negligible compared to physical radio propagation and ARQ latency.
