# MLMC Diffusion Architecture: Research Objectives & Evaluation Methodology

**Project**: MediLink-DL Experimental Cryptography Research Sprint  
**Target Cipher**: MediLink Mini Cipher (MLMC-G16 Educational Prototype)  
**Document**: Research Objectives, Metric Definitions, and Evaluation Framework  
**Status**: Academic Research Document — Non-Production Candidate  

---

## 1. Context and Problem Statement

The educational specification of **MLMC-G16** is a 16-bit toy block cipher with a 16-bit key operating on four 4-bit branches ($X_0, X_1, X_2, X_3$) over 14 rounds.
Its current diffusion layer relies on:
- Two 4-bit rotate-XOR linear transformations:
  $$L_0(y) = y \oplus \text{rotl}_4(y, 1) \oplus \text{rotl}_4(y, 2)$$
  $$L_1(y) = y \oplus \text{rotl}_4(y, 1) \oplus \text{rotl}_4(y, 3)$$
- A Type-II Generalized Feistel Network (GFN) structure where two branches are updated per round and shuffled.

### Critical Limitations of the Existing Architecture:
1. **Intra-Nibble Isolation**: $L_0$ and $L_1$ diffuse bits strictly *within* an individual 4-bit nibble. Inter-nibble diffusion depends entirely on the Feistel branch shuffle.
2. **Low Branch Number**: The 1-round linear transformation has a nibble branch number of only $\mathcal{B}_{\text{nibble}} = 2$ and a bit branch number of $\mathcal{B}_{\text{bit}} = 2$.
3. **Slow Multi-Round Diffusion**: A difference injected into one branch takes **4 rounds** to affect all 16 bits of the state ($R_{\text{diff}} = 4$).
4. **Weak Active S-Box Bounds**: Across 2 rounds, a differential characteristic can activate as few as **1 active S-box**, leaving the cipher vulnerable to structural shortcut attacks unless many rounds are executed.

---

## 2. Primary Research Objectives

This research investigates a new family of lightweight invertible diffusion transformations for the 16-bit / 4-nibble block cipher state that aims to achieve a balanced, Pareto-optimal combination of:

1. **High Branch Numbers**:
   - Maximize nibble-level branch number: target $\mathcal{B}_{\text{nibble}} = 5$ (the optimal Maximum Distance Separable / MDS bound).
   - Maximize bit-level branch number: target $\mathcal{B}_{\text{bit}} \in [6, 8]$ (where 8 is the theoretical upper bound for any $[32, 16]$ binary code).
2. **Minimal XOR Complexity**:
   - Minimize direct XOR operations and sequential XORs under Common Subexpression Elimination (CSE).
3. **Hardware Efficiency**:
   - Minimize standard CMOS Gate Equivalents (GE).
   - Investigate **involutory (self-inverse)** transformations ($M = M^{-1}$) where the decryption circuit requires **0 additional XOR gates**.
4. **Software Efficiency**:
   - Enable fast bit-parallel, nibble-sliced, or SIMD execution with low instruction counts.
5. **Rapid Diffusion Speed**:
   - Achieve full bit-level dependency ($R_{\text{diff}} = 1$ or $2$) compared to $R_{\text{diff}} = 4$ in Feistel networks.
6. **Provable Differential & Linear Trail Bounds**:
   - Maximize the minimum number of active S-boxes across multiple rounds ($r \in [1, 14]$), driving the maximum differential characteristic probability $p_{\text{diff}}^{14R} \ll 2^{-16}$ and linear correlation bias $\epsilon_{\text{linear}}^{14R} \ll 2^{-8}$.

---

## 3. Strict Research Constraints & Non-Objectives

> [!WARNING]
> - **NOT A PROOF OF PRODUCTION SECURITY**: Passing branch number tests, avalanche tests, or active S-box counts does **not** prove cryptographic security against advanced attacks (e.g., algebraic attacks, integral attacks, or side-channel leakage).
> - **NO FABRICATED NOVELTY**: Every proposed candidate must be rigorously compared against published literature (AES, PRESENT, GIFT, SKINNY, RECTANGLE, Ascon, PHOTON, LED).
> - **EDUCATIONAL SCOPE**: MLMC-G16 is an educational research cipher. Production telemedicine image transmission in MediLink-DL strictly retains **NIST SP 800-38D (AES-GCM)** and **NIST SP 800-232 (Ascon-AEAD128)**.

