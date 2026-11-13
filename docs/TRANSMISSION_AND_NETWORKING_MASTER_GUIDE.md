# Telemedicine Transmission and Networking Master Guide
## Discrete-Event Channel Simulation, Priority Min-Heap Scheduling, 'MLPK' Packet Framing, and Selective-Repeat ARQ in MediLink-DL

**Authors & Engineering Team**:
* **Mayank Soni** (Roll No: CS24B1031) — *System Architecture, Multi-Objective Optimization & Cryptographic Research*
* **Nandan** (Roll No: CS24B1034) — *Deep Learning Localization, Differentiable Grad-CAM & Loss Formulations*
* **Sumedha** (Roll No: AD24B1061) — *Digital Image Processing, Dual-Stream Compression & Priority ARQ Transport*

**Faculty Advisor**:
* **Dr. Mehbub Alam** — *Faculty Advisor, Computer Networks (CN)*

**Department of Computer Science and Engineering**  
**Indian Institute of Information Technology Raichur (IIIT Raichur)**  
**Date**: November 2026 &nbsp;|&nbsp; **Technical Reference Manual**

---

## Table of Contents
1. [The Real-World Rural Networking Problem](#1-the-real-world-rural-networking-problem)
2. [Simulating Two Computers on One Machine (Discrete-Event Network Simulation)](#2-simulating-two-computers-on-one-machine-discrete-event-network-simulation)
3. [The 'MLPK' Application-Layer Packet Framing Protocol](#3-the-mlpk-application-layer-packet-framing-protocol)
4. [Priority Min-Heap Scheduling: Content-Aware vs. Content-Blind Transport](#4-priority-min-heap-scheduling-content-aware-vs-content-blind-transport)
5. [Channel Physics & Mathematical Equations (Delay, Jitter, and Loss)](#5-channel-physics--mathematical-equations-delay-jitter-and-loss)
6. [Error Control: Selective-Repeat ARQ Protocol](#6-error-control-selective-repeat-arq-protocol)
7. [Empirical Results Across 4 Wireless Channel Profiles](#7-empirical-results-across-4-wireless-channel-profiles)
8. [The Live Telemedicine Web Dashboard Architecture](#8-the-live-telemedicine-web-dashboard-architecture)
9. [Viva Defense Master Cheat Sheet (Networking Q&A)](#9-viva-defense-master-cheat-sheet-networking-qa)

---

## 1. The Real-World Rural Networking Problem

In primary healthcare centers (PHCs) across rural regions in India (such as Bastar in Chhattisgarh, the Thar desert belt in Rajasthan, or tribal blocks in Odisha), wireless communication links are constrained by geography and infrastructure:

* **Bandwidth Throttling**: Cellular towers in these regions frequently operate on legacy **2G/EDGE (32–64 kbps)** or weak, congested 3G (100–256 kbps). Under cloudy monsoon weather or power generator cuts, uplink speeds can degrade to sub-30 kbps.
* **Long Round-Trip Times (RTT)**: Multi-hop wireless uplinks and geostationary satellite backhauls introduce propagation delays between **120 ms and 500 ms**.
* **High Physical Packet Loss**: Fading radio-frequency (RF) signals and atmospheric interference produce packet drop rates between **2% and 10%**.

### Why Uncompressed Transmission Fails
* A single uncompressed $512 \times 512$ 8-bit digital radiograph is **262,144 bytes (262 KB)**.
* Full multi-view clinical DICOM studies range between **10 MB and 35 MB**.

Let us calculate the theoretical transmission airtime across a standard rural 64 kbps connection:
$$\Delta t_{\text{air}} = \frac{\text{Study Size (bits)}}{\text{Bandwidth (bps)}} = \frac{262,144 \times 8 \text{ bits}}{64,000 \text{ bps}} = \mathbf{32.76 \text{ seconds}}$$
For a full 20 MB DICOM study:
$$\Delta t_{\text{air}} = \frac{20 \times 10^6 \times 8}{64,000} = \mathbf{2,500 \text{ seconds}} \approx \mathbf{41.6 \text{ minutes}}$$

When a patient arrives in acute respiratory distress, waiting 40 minutes for an image to load at a specialist center is clinically unacceptable.

```
+-------------------------------------------------------------------------------------------------+
|                                 THE RURAL CHANNEL REALITY                                       |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   Rural PHC (Gaon)                                              District Hospital (Sheher)      |
|   [Patient in Distress]                                         [Specialist Radiologist]        |
|             |                                                               ^                   |
|             v                                                               |                   |
|     Digital X-Ray                                                   Needs diagnostic view       |
|    (262 KB - 35 MB)                                                 in < 1.0 second             |
|             |                                                               |                   |
|             +-------------[ 2G / EDGE WIRELESS UPLINK ]---------------------+                   |
|                           32 - 64 kbps, RTT = 120-500 ms                                        |
|                           Packet Loss Rate = 2% - 10%                                           |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Simulating Two Computers on One Machine (Discrete-Event Network Simulation)

A frequent question from reviewers and students is:
> *"Transmission is supposed to happen between two separate computers (a rural clinic and an urban hospital). How can we evaluate and test this on a single workstation?"*

### 2.1 The Academic Standard: Discrete-Event Network Simulation (DES)
In Computer Science and Computer Networks research (at institutions like MIT, Stanford, and IITs), network engineers do not physically travel to a remote village with two laptops to test every software iteration. Instead, they employ **Discrete-Event Simulation (DES)** or Linux network emulation (`tc` / `netem`).

In MediLink-DL (`src/transmission.py`), the software is designed with strict modular decoupling between the **Transmitter** and the **Receiver**:

```
+-------------------------------------------------------------------------------------------------+
|                             DISCRETE-EVENT NETWORK SIMULATOR                                    |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   [ TRANSMITTER MODULE ] (Simulates Rural Clinic Workstation)                                   |
|   - Reads radiograph raster                                                                     |
|   - Runs MedicalResNet18 localization                                                           |
|   - Compresses dual streams (q_roi=90, q_bg=25) -> 26,515 bytes                                 |
|   - Fragments payload into 1024-byte packets with 'MLPK' headers                                |
|   - Enqueues packets into a Priority Min-Heap                                                   |
|                        |                                                                        |
|                        v                                                                        |
|   +-------------------------------------------------------------------------+                   |
|   | CHANNEL SIMULATOR (Simulates Physical 2G/EDGE Wireless Uplink)          |                   |
|   | - Physical Serialization Delay: Delta t_tx = (Packet_Bytes * 8) / B     |                   |
|   | - Propagation Delay: Delta t_prop = RTT / 2                             |                   |
|   | - Channel Jitter: Gaussian perturbation xi                              |                   |
|   | - Packet Loss: Pseudo-random Bernoulli drops (2% to 10%)                |                   |
|   | - ARQ Simulation: Re-transmits dropped packets upon timeout             |                   |
|   +-------------------------------------------------------------------------+                   |
|                        |                                                                        |
|                        v                                                                        |
|   [ RECEIVER MODULE ] (Simulates Urban Hospital Workstation)                                    |
|   - Receives incoming chunks                                                                    |
|   - Validates 32-byte SHA-256 checksum per packet                                               |
|   - Measures exact timestamp when all ROI packets arrive (Time-to-First-ROI)                    |
|   - Executes constant-time AEAD decryption (Zero-Plaintext Release)                             |
|   - Reassembles background and ROI patch with alpha blending for doctor                         |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

Both modules execute in isolated namespaces. The `ChannelSimulator` acts as an artificial physical wire that injects exact, mathematically rigorous latency and packet drops.

### 2.2 Live Cloud Deployment Across the Internet
Beyond local discrete-event simulation, MediLink-DL is deployed live on cloud servers at:
**[https://medilink-dl.onrender.com](https://medilink-dl.onrender.com)**

In this production deployment:
* The cloud server hosts the pipeline backend.
* Any user accessing the website from a mobile phone or laptop over cellular data (Jio, Airtel, Vi) operates as the **Receiving Client**, streaming medical telemetry over the real, public internet.

---

## 3. The 'MLPK' Application-Layer Packet Framing Protocol

Standard raw TCP streams are byte-oriented: they provide a continuous stream without application-level boundaries. If a connection drops, recovering specific diagnostic chunks is difficult. 

To ensure verifiable packetization, we designed the **'MLPK' Framing Protocol** (`src/transmission.py`):

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       'M'     |      'L'      |      'P'      |      'K'      | Magic (4B)
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Packet ID (uint16)     |       Total Chunks (uint16)   | Chunk Index
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Priority (u8) |    Reserved   |       Payload Length (uint16) | Priority & Len
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                   SHA-256 Checksum (32 Bytes)                 | Checksum
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| [ Payload Bytes (up to 1024 Bytes) ... ]                      | Variable
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Packet Header Fields:
1. **Magic Bytes (`4B`)**: Fixed signature `0x4D4C504B` (`'MLPK'`) for frame synchronization.
2. **Packet ID (`2B`)**: Sequence index of the chunk ($0, 1, 2, \dots, N-1$).
3. **Total Chunks (`2B`)**: Total count of packets required to reconstruct the complete study.
4. **Priority Tier (`1B`)**: Application priority tier ($30 = \text{Header}, 20 = \text{ROI}, 5 = \text{Background}$).
5. **Payload Length (`2B`)**: Number of valid data bytes in this frame (typically 1024, or less on the final chunk).
6. **SHA-256 Checksum (`32B`)**: Cryptographic digest of the payload bytes for instant bit-flip detection prior to decryption.

---

## 4. Priority Min-Heap Scheduling: Content-Aware vs. Content-Blind Transport

### 4.1 The Flaw of Standard FIFO Scheduling
In traditional networking, transport protocols are **Content-Blind**: every byte is treated with equal importance. 

Consider a standard FIFO queue transmitting a 26.5 KB dual-stream radiograph:
* Chunks 1 to 17 contain the full-frame background context ($q_{\text{bg}}=25$), which consists mostly of black air around the ribcage and healthy soft tissue.
* Chunks 18 to 26 contain the high-priority diagnostic lung ROI ($q_{\text{roi}}=90$).

Under FIFO, the radiologist's workstation must wait through all 17 background chunks before the first byte of the diagnostic lesion arrives. If a packet drop occurs in chunk 4, the entire stream stalls while retransmissions take place, forcing the specialist to wait 2.5 to 3.5 seconds before seeing anything useful.

```
Standard FIFO Queue (Content-Blind):
[ Hdr ] -> [ BG_1 ] -> [ BG_2 ] -> ... -> [ BG_17 ] -> [ ROI_1 ] -> [ ROI_2 ] -> ...
                                                       ^
                                                       Doctor waits 2.43 seconds!
```

### 4.2 Priority Min-Heap Scheduling Formulation
MediLink-DL implements an application-layer **Priority Queue** using Python's `heapq` module (`src/transmission.py`):

$$\text{Priority}(\text{Header}) = 30 > \text{Priority}(\text{Diagnostic ROI Chunks}) = 20 > \text{Priority}(\text{Background Chunks}) = 5$$

In Python's `heapq` (which extracts the smallest number first), we push packets with negated priority values:
```python
pkt = Packet(
    priority=-priority_level,  # e.g., -30 for Header, -20 for ROI, -5 for BG
    packet_id=i,
    ...
)
heapq.heappush(queue, pkt)
```

```
Priority Min-Heap Queue (Content-Aware):
[ Hdr ] -> [ ROI_1 ] -> [ ROI_2 ] -> ... -> [ ROI_9 ] -> [ BG_1 ] -> [ BG_2 ] -> ...
           ^
           Doctor inspects sharp lesion in 0.391 seconds!
```

The header and all diagnostic lung packets are dispatched over the wire in the very first burst. The remote radiologist's viewer renders the lesion in under half a second, while the peripheral context trickles in progressively behind it.

---

## 5. Channel Physics & Mathematical Equations (Delay, Jitter, and Loss)

To ensure that simulation results match real-world physical networks, `ChannelSimulator` implements the exact mathematical laws of packet switching:

### 5.1 Serialization (Transmission) Delay ($\Delta t_{\text{tx}}$)
The time required to push packet bits onto the physical medium:
$$\Delta t_{\text{tx}} = \frac{(L + L_{\text{header}}) \times 8}{R_{\text{bandwidth}}}$$
* $L = 1024$ bytes (Payload).
* $L_{\text{header}} = 43$ bytes (MLPK overhead).
* $R_{\text{bandwidth}} = 64,000$ bps (2G link).

$$\Delta t_{\text{tx}} = \frac{(1024 + 43) \times 8}{64,000} = \frac{8536 \text{ bits}}{64,000 \text{ bps}} = \mathbf{0.1334 \text{ seconds}} = \mathbf{133.4 \text{ ms}}$$

### 5.2 Propagation Delay ($\Delta t_{\text{prop}}$)
The time for a signal to physically travel through cellular towers and backhauls:
$$\Delta t_{\text{prop}} = \frac{\text{RTT}}{2}$$
For a typical rural link with $\text{RTT} = 120 \text{ ms}$:
$$\Delta t_{\text{prop}} = \frac{0.120}{2} = \mathbf{0.060 \text{ seconds}} = \mathbf{60.0 \text{ ms}}$$

### 5.3 Channel Jitter ($\xi$)
Network traffic experiences unpredictable queue variations modeled as a zero-mean Gaussian random variable:
$$\xi \sim \mathcal{N}(0, \sigma_{\text{jitter}}^2), \quad \sigma_{\text{jitter}} = 0.015 \text{ seconds (15 ms)}$$

### 5.4 Total One-Way Packet Latency
$$\Delta t_{\text{packet}} = \Delta t_{\text{tx}} + \Delta t_{\text{prop}} + \xi = \frac{(L + 43) \times 8}{R} + \frac{\text{RTT}}{2} + \xi$$

On a 64 kbps link:
$$\Delta t_{\text{packet}} = 133.4\text{ ms} + 60.0\text{ ms} + \xi \approx \mathbf{193.4 \text{ ms per chunk}}$$

---

## 6. Error Control: Selective-Repeat ARQ Protocol

When wireless fading drops a packet, how should the system recover?

### 6.1 Why Go-Back-N and Standard TCP Fail
* In **Go-Back-N**, if packet 4 is dropped, the receiver discards packets 5, 6, and 7 even if they arrived intact, forcing the transmitter to re-send everything from packet 4 onward. On a 64 kbps link with 5% packet loss, this wastes precious bandwidth and creates catastrophic delays.
* In standard **TCP**, Head-of-Line (HoL) blocking prevents the application from reading any received bytes until missing earlier packets are retransmitted.

### 6.2 MediLink-DL Selective-Repeat ARQ Implementation
In `src/transmission.py`, we implement application-layer **Selective-Repeat Automatic Repeat reQuest (ARQ)**:

1. **Receiver Buffering**: The receiver accepts and stores packets that arrive out-of-order into a hashmap `received_packets[packet_id]`.
2. **Selective NACK**: Only missing sequence IDs are requested for retransmission:
   $$\text{Retransmit List} = \{ i \in [0, N-1] \mid i \notin \text{received\_packets} \}$$
3. **Checksum Verification**: Every received chunk is verified against its 32-byte SHA-256 header digest:
   ```python
   if compute_checksum(packet.data) != packet.checksum:
       return False  # Corrupted chunk dropped immediately
   ```
4. **Zero Waste**: Chunks that arrive successfully are never retransmitted, preserving the 2G channel capacity.

---

## 7. Empirical Results Across 4 Wireless Channel Profiles

We evaluated MediLink-DL on real patient radiograph payloads (26,515 bytes dual-stream vs. 26,035 bytes uniform baseline) across four real-world rural communication profiles (`results/experiment_transmission.csv`):

| Channel Link Profile | Bandwidth | Latency (RTT) | Loss Rate | Priority Time-to-ROI | Standard FIFO Time | Triage Acceleration Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rural 2G / EDGE** | 48 kbps | 200 ms | 2.0% | **0.592 s** | 3.592 s | **6.07x Faster** |
| **Rural 2.5G / GPRS** | 64 kbps | 120 ms | 5.0% | **0.391 s** | 2.430 s | **6.21x Faster** |
| **Rural 3G (Weak)** | 128 kbps | 150 ms | 5.0% | **0.372 s** | 2.097 s | **5.63x Faster** |
| **Satellite VSAT** | 256 kbps | 500 ms | 2.0% | **1.036 s** | 5.299 s | **5.11x Faster** |

```
+-------------------------------------------------------------------------------------------------+
|                        TIME-TO-FIRST-ROI ACROSS 64 KBPS RURAL CHANNEL                           |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   Priority Transport (Ours)  [0.391s] ===                                                       |
|                                                                                                 |
|   Standard FIFO Transport    [2.430s] ====================                                      |
|                                                                                                 |
|   Result: Priority Scheduling delivers the acute lesion 6.21x faster without                    |
|           transmitting a single additional byte!                                                |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---

## 8. The Live Telemedicine Web Dashboard Architecture

The web dashboard (`src/web_app.py`) provides a real-time visual interface for clinicians and faculty examiners.

```
+-------------------------------------------------------------------------------------------------+
|                                 WEB DASHBOARD ARCHITECTURE                                      |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|  [ LEFT PANEL: TRANSMITTER CONTROLS ]                                                           |
|  - Radiograph Selector: Choose clinical CXR from test cohort                                    |
|  - Grad-CAM Threshold Slider (tau): Controls lesion sensitivity [0.10 - 0.70]                   |
|  - Diagnostic Quality Slider (q_roi): Controls lesion fidelity [80 - 100]                        |
|  - Context Quality Slider (q_bg): Controls background compression [10 - 50]                     |
|  - Cryptographic Cipher Dropdown: AES-256-GCM vs NIST SP 800-232 Ascon-AEAD128                 |
|                                                                                                 |
|  [ CENTER: THREE-STAGE VISUAL TELEMETRY ]                                                       |
|  +--------------------+     +--------------------+     +--------------------+                   |
|  | 1. Input CXR       |     | 2. Grad-CAM        |     | 3. Received CXR    |                   |
|  | Original Grayscale | --> | Saliency Heatmap   | --> | Reassembled Image  |                   |
|  | 512 x 512 (262 KB) |     | 8,938 Active Px    |     | 47.1 KB Total Wire |                   |
|  +--------------------+     +--------------------+     +--------------------+                   |
|                                                                                                 |
|  [ TOP CARDS: REAL-TIME RADIOLOGICAL METRICS ]                                                  |
|  - Diagnostic ROI PSNR: 37.87 dB - 48.31 dB                                                     |
|  - Diagnostic ROI SSIM: 0.9891 (Pristine structural preservation)                               |
|  - Bandwidth Savings: 9.89x Compression (89.89% data reduction)                                 |
|  - Time-to-First-ROI: 0.391 seconds (6.21x triage acceleration)                                   |
|                                                                                                 |
|  [ BOTTOM BAR: WIRE TELEMETRY ALLOCATION ]                                                      |
|  [ HDR: 28B ] [ ROI Stream: 16.5 KB (35%) ] [ Background Stream: 30.6 KB (65%) ]                |
|                                                                                                 |
+-------------------------------------------------------------------------------------------------+
```

---
<!-- 
## 9. Viva Defense Master Cheat Sheet (Networking Q&A)

Use these concise, authoritative responses during viva defense:

### Q1: "How did you test network transmission between two computers when running on a single laptop?"
> **Answer**: *"Sir, we implemented a decoupled **Discrete-Event Network Simulation (DES)** in `src/transmission.py`. The Transmitter packages the dual streams into 1024-byte packets, the `ChannelSimulator` injects physical serialization delay $\Delta t_{\text{tx}} = \frac{\text{Bits}}{R}$, propagation delay $\frac{\text{RTT}}{2}$, and pseudo-random packet drops (2–10%), and the `Receiver` module validates checksums and measures time-to-first-ROI. Furthermore, our system is deployed live on cloud servers at `https://medilink-dl.onrender.com`, allowing multi-device client-server testing across the public cellular internet."*

### Q2: "What is the difference between Priority Min-Heap scheduling and standard FIFO?"
> **Answer**: *"Standard FIFO is content-blind: it transmits background context chunks (65% of the payload) before sending any diagnostic data. Our Priority Min-Heap scheduler assigns explicit priority tiers: Header (30) > ROI (20) > Background (5). Over a 64 kbps rural link, priority scheduling delivers the diagnostic lesion in **0.391 seconds**, compared to **2.430 seconds** for FIFO—achieving a **6.21× triage speedup** without sending an extra byte."*

### Q3: "Why did you use Selective-Repeat ARQ instead of TCP or Go-Back-N?"
> **Answer**: *"On low-bandwidth, high-latency links with 5% packet loss, Go-Back-N forces the transmitter to re-send all packets following a dropped chunk, wasting precious bandwidth. TCP introduces Head-of-Line blocking, stalling the application. Selective-Repeat ARQ buffers out-of-order packets and requests retransmissions only for missing sequence IDs, maximizing channel utilization."*

### Q4: "What is inside the 43-byte 'MLPK' packet header?"
> **Answer**: *"It contains 4 bytes magic (`'MLPK'`), 2 bytes sequence ID, 2 bytes total chunks count, 1 byte priority tier, 2 bytes payload length, and a 32-byte SHA-256 cryptographic checksum for independent packet-level integrity verification."*

---

*End of Networking Master Guide. Maintained under the MediLink-DL Project Repository at `docs/TRANSMISSION_AND_NETWORKING_MASTER_GUIDE.md`.*
 -->
