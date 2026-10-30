# MediLink Mini Cipher (MLMC): Known Limitations & Cryptographic Weaknesses

**Candidate**: MediLink Mini Cipher (MLMC-128)  
**Document Purpose**: Frank, unreserved disclosure of known theoretical, practical, and architectural limitations.  
**Classification**: Non-Production Academic Research Candidate  

---

## 1. Absolute Prohibition: Not for Clinical Patient Data

> [!CAUTION]
> **UNDER NO CIRCUMSTANCES SHOULD MLMC BE USED TO SECURE REAL PATIENT RADIOGRAPHS, ELECTRONIC HEALTH RECORDS (EHR), OR TELEMEDICINE TRANSMISSIONS IN PRODUCTION.**  
> MediLink-DL strictly enforces the use of NIST SP 800-38D (AES-GCM) or NIST SP 800-232 (Ascon-AEAD128) for all real clinical transmissions.

---

## 2. Technical Limitations & Attack Surfaces

### 2.1 Software Cache-Timing Side-Channel Vulnerability
- **Mechanism**: The pure-Python implementation of MLMC utilizes lookup tables for `SubBytes` (`MLMC_SBOX`), inverse substitution (`MLMC_INV_SBOX`), and Galois field multiplication (`exp` and `log` tables in $GF(2^8)$).
- **Vulnerability**: Table lookup memory addresses depend directly on secret internal state values:
  $$\text{Address} = \text{Base} + s_{r, c}$$
  On modern processors with hierarchical L1/L2 caches, cache hits and misses create subtle execution timing variations (Bernstein, 2005; Osvik et al., 2006). A co-located adversary on the telemedicine edge node could reconstruct the master key via cache-timing analysis.
- **Remediation Required for Production**: Implementation must be redesigned using bitsliced boolean circuits or constant-time arithmetic without memory table lookups.

### 2.2 Lack of Independent Cryptanalytic Scrutiny
- **Mechanism**: Modern ciphers (AES, Ascon, ChaCha20) gained trust through multi-year public cryptanalysis competitions (NIST AES competition 1997–2000, NIST Lightweight Cryptography 2019–2023).
- **Vulnerability**: MLMC has been evaluated only within the scope of this research sprint (DDT, LAT, SAC, NIST SP 800-22 statistical randomness). Subtle algebraic attacks (e.g., Gröbner basis reductions, higher-order differential cryptanalysis, invariant subspace attacks) have not been exhaustively proven absent.

### 2.3 Nonce-Misuse Vulnerability in CTR Mode
- **Mechanism**: Like standard GCM, the experimental `MLMCAEADExperimental` wrapper uses Counter (CTR) mode.
- **Catastrophic Failure on Nonce Reuse**: If the same 96-bit nonce is ever reused with the same key across two distinct chunks:
  $$C_1 \oplus C_2 = P_1 \oplus P_2$$
  Furthermore, two message-tag pairs with the same nonce allow an attacker to factor the GHASH polynomial over $GF(2^{128})$ and recover the hash subkey $H$, completely destroying message authenticity (Joux's Forbidden Attack).
- **Remediation**: SIV (Synthetic Initialization Vector) modes (e.g., AES-GCM-SIV, RFC 8452) or Ascon's sponge duplex construction should be preferred when nonces might collide.

### 2.4 Lack of Hardware Cryptographic Instructions
- **Mechanism**: AES benefits from dedicated silicon instructions on modern CPUs (Intel/AMD `AESENC`/`AESDEC`, ARMv8 `AESE`/`AESD`), which execute an entire round in a single CPU cycle with zero cache-timing leakage.
- **Impact**: MLMC must execute in software, incurring thousands of CPU cycles per block. In energy-constrained battery-operated rural edge devices, this translates directly to higher battery consumption.

### 2.5 Block Size sweet32 Consideration for MLMC-32 Profile
- The educational miniature profile `MLMC-32` has a 32-bit block size ($2^{32}$ states).
- **Birthday Bound**: Under the birthday paradox, block collisions occur after only $2^{16} = 65,536$ encrypted blocks. It is strictly suitable only for classroom differential trail visualization and must never encrypt streams exceeding 64 KB.

---

## 3. Comparison with Established Standards

| Security Property | AES-GCM (NIST SP 800-38D) | Ascon-AEAD128 (NIST SP 800-232) | MLMC Candidate |
| :--- | :--- | :--- | :--- |
| **Peer-Reviewed Track Record** | 25+ years (Global standard) | 5+ years (NIST Lightweight Winner) | **None (Academic prototype)** |
| **Constant-Time Hardware** | Native CPU instructions | 64-bit boolean bit-slice | Software table lookups (Vulnerable) |
| **Side-Channel Hardening** | Extensive literature & tooling | Native resistance by design | Unhardened |
| **Production Suitability** | **Fully Approved** | **Fully Approved** | **STRICTLY PROHIBITED** |

---

## 4. Conclusion & Pedagogical Value

The MediLink Mini Cipher successfully fulfills its academic objective:
1. It provides an intuitive, mathematically rigorous reference implementation of an SPN block cipher over $GF(2^8)$.
2. It allows students and researchers to empirically observe round-by-round diffusion, compute difference distribution tables (DDT), and construct linear approximation tables (LAT).
3. It clearly demonstrates why designing a secure block cipher requires not just passing unit tests, but years of side-channel hardening, constant-time engineering, and global peer review.
