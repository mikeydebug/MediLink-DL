# Known Limitations, Side-Channel Considerations, and Research Warnings

**Project**: MediLink-DL Experimental Cryptography Research Sprint  
**Target Cipher**: MediLink Mini Cipher (MLMC-G16 Educational Prototype)  
**Document**: Threat Model, Architectural Limitations, and Security Disclosures  
**Status**: Strict Academic Research Prototype — Non-Production Specification  

---

> [!CAUTION]
> **CRITICAL MEDICAL DATA NOTICE: NEVER USE FOR PRODUCTION HEALTHCARE DATA**  
> MLMC-G16 and its diffusion transformations are experimental educational research artifacts designed solely for academic evaluation, algorithmic exploration, and pedagogical comparison.  
> **Under no circumstances should MLMC-G16 be deployed in production, in clinical telemedicine workflows, or to protect real Protected Health Information (PHI) under HIPAA, GDPR, or related healthcare regulations.**  
> For actual clinical medical image transmissions in MediLink-DL, production-standard authenticated encryption (**AES-GCM-256** or **NIST SP 800-232 Ascon-128a**) MUST be used exclusively.

---

## 1. Architectural and Theoretical Limitations

### 1.1 Small Block Size ($n = 16$ bits)
The 16-bit block size ($2^{16} = 65,536$ distinct blocks) imposes severe, insurmountable information-theoretic constraints:
1. **Exhaustive Codebook Reconstruction**: An active eavesdropper observing $2^{16}$ chosen or known plaintext-ciphertext pairs can reconstruct the entire codebook for a fixed key, reducing decryption to an $O(1)$ memory lookup without learning the key.
2. **Birthday Paradox Collision Bound**: In any standard block cipher operating mode (such as CBC, CTR, or GCM), collisions in the internal state occur with probability $p \approx 0.5$ after observing only:
   $$q \approx 2^{n/2} = 2^{16/2} = 2^8 = 256 \text{ blocks (512 bytes)}$$
   Encrypting more than a few hundred bytes under the same key completely breaks confidentiality guarantees.

### 1.2 Small Key Size ($k = 16$ bits)
A 16-bit key provides a key space of size $2^{16} = 65,536$.
- On a standard modern commodity CPU (Apple M-series or Intel x86), an exhaustive brute-force search over the entire key space completes in **less than 5 milliseconds**.
- Therefore, MLMC-G16 has **zero operational security margin** against brute-force attacks. It exists purely as a scaled mathematical testbed to examine diffusion dynamics, branch numbers, and S-box bounds.

### 1.3 Scope of Diffusion Bounds
The wide-trail active S-box lower bounds ($\ge 35$ active S-boxes across 14 rounds) established in this work demonstrate resistance against **single differential characteristics** and **single linear trails**.
However, they do not automatically guarantee immunity against:
1. **Differential Hulls & Multiple Differentials**: Multiple distinct differential characteristics sharing the same input difference $\Delta_{\text{in}}$ and output difference $\Delta_{\text{out}}$ can combine constructively, amplifying the overall differential transition probability:
   $$DP(\Delta_{\text{in}} \to \Delta_{\text{out}}) = \sum_{\text{trails}} DP(\text{trail}) \gg 2^{-70}$$
2. **Linear Hulls**: Similarly, multiple linear approximations with the same input/output masks can cluster, increasing the overall correlation bias beyond individual trail bounds.
3. **Algebraic & Cube Attacks**: If the algebraic degree of the cipher grows slowly across rounds, higher-order differential and cube attacks can potentially bypass wide-trail bounds.
4. **Invariant Subspace & Nonlinear Invariant Attacks**: Symmetries in the round constants or matrix eigenvectors can permit invariant subspace distinguishers independent of branch numbers.

---

## 2. Side-Channel and Implementation Vulnerabilities

### 2.1 Software Cache-Timing Attacks
When implemented in software (Python, C, or assembly):
- If field multiplications over $\mathbb{F}_{2^4}$ or S-box evaluations are implemented using precomputed lookup tables (e.g., $16$-element or $256$-element lookup tables), table memory accesses are subject to **CPU cache-line timing leaks** (Bernstein 2005, Osvik et al. 2006).
- An attacker colocated on the same operating system or hypervisor can monitor cache hit/miss latencies to deduce secret-dependent memory indices, recovering round keys in milliseconds.
- **Mitigation Requirement**: Constant-time bit-sliced implementations or pure logical Boolean formulas (AND, OR, XOR) must be used if constant-time execution is required.

### 2.2 Hardware Power and Electromagnetic Analysis
In ASIC or FPGA hardware implementations:
- Unprotected CMOS implementations of the XOR network and PRESENT S-boxes exhibit data-dependent dynamic power consumption:
  $$P_{\text{dynamic}} \propto C_L \cdot V_{DD}^2 \cdot f \cdot \alpha$$
  where $\alpha$ is the switching activity.
- Correlation Power Analysis (CPA) and Differential Power Analysis (DPA) can easily extract the 16-bit key from fewer than 100 power traces.
- **Countermeasure Costs**: Applying 1st-order Threshold Implementations (TI) or Boolean masking requires at least 3 shared shares for quadratic S-box operations, increasing hardware area and power consumption by $300\%$ to $500\%$.

---

## 3. Comparison with Production Primitives

| Security Dimension | MLMC-G16 (Educational) | NIST SP 800-232 Ascon-128a | AES-128-GCM |
| :--- | :---: | :---: | :---: |
| **Block Size** | 16 bits (2 bytes) | 320 bits (State) | 128 bits (16 bytes) |
| **Key Size** | 16 bits | 128 bits | 128 bits |
| **Target Application** | Mathematical Research & Teaching | Lightweight IoT / Telemedicine | Production Enterprise / Web |
| **NIST Standardization** | None (Educational Candidate) | NIST Lightweight Winner (2023) | FIPS 197 Standard |
| **HIPAA PHI Compliance** | **NON-COMPLIANT (Prohibited)** | Compliant (Validated) | Compliant (Gold Standard) |
| **Collision Bound** | $2^8$ blocks (512 B) | $2^{64}$ data blocks | $2^{64}$ blocks ($2.5 \times 10^{11}$ GB) |
| **Brute-Force Complexity** | $2^{16}$ (~3 ms) | $2^{128}$ (~$10^{22}$ years) | $2^{128}$ (~$10^{22}$ years) |

---

## 4. Open Research Questions & Future Directions

1. **Dimensional Scaling to 64-bit and 128-bit Blocks**:
   - Investigate whether the Orthogonal Involutory Hadamard construction can be scaled to $8 \times 8$ or $16 \times 16$ matrices over $\mathbb{F}_{2^8}$ while preserving optimal branch number and low XOR overhead.
2. **Automated Differential Hull Enumeration**:
   - Formulate the full MLMC-G16 differential propagation equations into an SMT/SAT model (using Z3 or CryptoMiniSat) or a Mixed-Integer Linear Programming (MILP) model (using Gurobi) to search for exact multi-round differential and linear hulls.
3. **Key Schedule Co-Design**:
   - Analyze potential slide attacks and related-key differential trails arising from interactions between the proposed diffusion matrix and the MLMC-G16 key schedule round constants.
4. **Fault Injection Susceptibility**:
   - Because $R_{\text{diff}} = 1$, a single bit flip fault injected prior to the diffusion layer spreads to all 16 output bits in a single round. The implications for Differential Fault Analysis (DFA) warrant systematic formal verification.

