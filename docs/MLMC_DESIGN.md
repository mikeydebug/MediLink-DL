# MediLink Mini Cipher (MLMC): Design & Specification Document

**Candidate Name**: MediLink Mini Cipher (MLMC-128)  
**Authors**: MediLink-DL Cryptographic Research Team (IIIT Raichur)  
**Status**: Academic Research Prototype — **NOT FOR PRODUCTION CLINICAL USE**  
**Version**: 1.0 (Experimental)  

---

## 1. Executive Summary & Research Disclaimer

> [!CAUTION]
> **ACADEMIC RESEARCH PROTOTYPE DISCLAIMER**  
> The MediLink Mini Cipher (MLMC) is an educational and exploratory cryptographic candidate designed specifically for investigating lightweight block cipher and authenticated encryption characteristics in resource-constrained rural telemedicine nodes.  
> **IT HAS NOT UNDERGONE YEARS OF PEER-REVIEWED OPEN CRYPTANALYSIS.**  
> It must **NEVER** be deployed to protect actual patient health records, diagnostic imagery, or production systems. MediLink-DL strictly retains NIST SP 800-38D (AES-256-GCM / AES-128-GCM) and NIST SP 800-232 (Ascon-AEAD128) as its trusted production cryptographic standards.

The goal of MLMC is to explore whether a streamlined Substitution-Permutation Network (SPN) block cipher with an experimental Counter-with-GHASH authenticated mode can provide transparent, auditable performance characteristics for small medical-image chunks (256 B to 4096 B).

---

## 2. Mathematical Foundation: Finite Field $GF(2^8)$

MLMC operates entirely over the Galois Field $GF(2^8)$ defined by the standard Rijndael irreducible polynomial:

$$P(x) = x^8 + x^4 + x^3 + x + 1 \quad (\text{hex: } \texttt{0x11B}, \text{ dec: } 283)$$

### 2.1 Addition in $GF(2^8)$
Addition of two field elements $a, b \in GF(2^8)$ corresponds to polynomial addition over $\mathbb{Z}_2$, computed via bitwise XOR:

$$a \oplus b$$

### 2.2 Multiplication in $GF(2^8)$
Multiplication is polynomial multiplication modulo $P(x)$, accelerated using precomputed base-3 generator exponentiation ($\text{exp}$) and logarithm ($\text{log}$) lookup tables:

$$a \odot b = \begin{cases} 0 & \text{if } a = 0 \text{ or } b = 0 \\ \text{exp}[\text{log}[a] + \text{log}[b]] & \text{otherwise} \end{cases}$$

---

## 3. State Matrix Architecture

MLMC is a **128-bit block cipher** operating on a $4 \times 4$ matrix of bytes in column-major order:

$$S = \begin{bmatrix} s_{0,0} & s_{0,1} & s_{0,2} & s_{0,3} \\ s_{1,0} & s_{1,1} & s_{1,2} & s_{1,3} \\ s_{2,0} & s_{2,1} & s_{2,2} & s_{2,3} \\ s_{3,0} & s_{3,1} & s_{3,2} & s_{3,3} \end{bmatrix}$$

A 16-byte block $B = (b_0, b_1, \dots, b_{15})$ maps to $S$ via:

$$s_{r, c} = b_{r + 4c}, \quad r \in \{0, 1, 2, 3\}, \; c \in \{0, 1, 2, 3\}$$

---

## 4. Parameter Overview

| Parameter | MLMC-128 (Standard Profile) | MLMC-32 (Miniature Educational) |
| :--- | :--- | :--- |
| **Block Size** | 128 bits (16 bytes) | 32 bits (4 bytes) |
| **Key Size** | 128 bits (16 bytes) | 32 bits (4 bytes) |
| **State Dimensions** | $4 \times 4$ bytes | $2 \times 2$ bytes |
| **Rounds ($N_r$)** | 10 rounds | 4 rounds |
| **Permutation Structure** | SPN (Substitution-Permutation) | SPN |
| **S-Box Size** | $8 \times 8$ bits | $8 \times 8$ bits (or $4 \times 4$) |
| **Linear Diffusion** | Circulant MDS Matrix ($\mathcal{B}=5$) | Circulant MDS Matrix ($\mathcal{B}=3$) |

---

## 5. Round Transformations

For $N_r = 10$, encryption proceeds as follows:

```
[Plaintext Block (16B)]
         |
         v
+------------------+
| AddRoundKey(K_0) |  <-- Key Whitening
+------------------+
         |
         +---------------------------------------+
         |                                       |
         v (Rounds 1 to 9)                       v (Round 10)
+------------------+                    +------------------+
|    SubBytes      |                    |    SubBytes      |
+------------------+                    +------------------+
         |                                       |
         v                                       v
+------------------+                    +------------------+
|    ShiftRows     |                    |    ShiftRows     |
+------------------+                    +------------------+
         |                                       |
         v                                       v
+------------------+                    +------------------+
|    MixColumns    |                    | AddRoundKey(K_10)|
+------------------+                    +------------------+
         |                                       |
         v                                       v
+------------------+                    [Ciphertext (16B)]
| AddRoundKey(K_r) |
+------------------+
         |
         +--> Loop r = 1..9
```

### 5.1 SubBytes ($S$-Box Transformation)
Every byte $s_{r, c}$ in the state is mapped through an $8 \times 8$ non-linear S-box $S: GF(2^8) \to GF(2^8)$ constructed via multiplicative inversion in $GF(2^8)$ followed by an affine mapping over $\mathbb{Z}_2$:

$$b'_i = b_i \oplus b_{(i+4)\%8} \oplus b_{(i+5)\%8} \oplus b_{(i+6)\%8} \oplus b_{(i+7)\%8} \oplus c_i$$

with $c = \texttt{0x63}$.

**Properties**:
- Bijective permutation of $\{0, \dots, 255\}$
- Maximum differential probability: $\max DP = 4/256 = 2^{-6}$
- Maximum linear bias: $\max \epsilon = 16/256 = 2^{-4}$
- Zero fixed points: $S(x) \neq x$ for all $x \in [0, 255]$

### 5.2 ShiftRows
Row $r$ is cyclically shifted to the left by $r$ byte positions:

$$\begin{aligned}
\text{Row 0}: & \quad (s_{0,0}, s_{0,1}, s_{0,2}, s_{0,3}) \to (s_{0,0}, s_{0,1}, s_{0,2}, s_{0,3}) \\
\text{Row 1}: & \quad (s_{1,0}, s_{1,1}, s_{1,2}, s_{1,3}) \to (s_{1,1}, s_{1,2}, s_{1,3}, s_{1,0}) \\
\text{Row 2}: & \quad (s_{2,0}, s_{2,1}, s_{2,2}, s_{2,3}) \to (s_{2,2}, s_{2,3}, s_{2,0}, s_{2,1}) \\
\text{Row 3}: & \quad (s_{3,0}, s_{3,1}, s_{3,2}, s_{3,3}) \to (s_{3,3}, s_{3,0}, s_{3,1}, s_{3,2})
\end{aligned}$$

### 5.3 MixColumns
Each column $c$ is multiplied by a circulant Maximum Distance Separable (MDS) matrix $M \in GF(2^8)^{4 \times 4}$:

$$\begin{bmatrix} s'_{0,c} \\ s'_{1,c} \\ s'_{2,c} \\ s'_{3,c} \end{bmatrix} = \begin{bmatrix} 02 & 03 & 01 & 01 \\ 01 & 02 & 03 & 01 \\ 01 & 01 & 02 & 03 \\ 03 & 01 & 01 & 02 \end{bmatrix} \begin{bmatrix} s_{0,c} \\ s_{1,c} \\ s_{2,c} \\ s_{3,c} \end{bmatrix}$$

**Branch Number**: $\mathcal{B} = \min_{v \neq 0} (\text{wt}(v) + \text{wt}(Mv)) = 5$. This guarantees that across any two consecutive rounds, any single active byte activates at least 5 active S-boxes, achieving the wide-trail strategy bound.

---

## 6. Key Schedule Expansion

The 128-bit master key $K$ is expanded into $(N_r + 1) = 11$ round keys $K_0, K_1, \dots, K_{10}$ represented as 32-bit words $w_0, \dots, w_{43}$:

1. For $i \in \{0, 1, 2, 3\}$, $w_i = (K_{4i} \ll 24) \mid (K_{4i+1} \ll 16) \mid (K_{4i+2} \ll 8) \mid K_{4i+3}$.
2. For $i = 4, \dots, 43$:
   $$\text{temp} = w_{i-1}$$
   - If $i \pmod 4 == 0$:
     $$\text{temp} = \text{SubWord}(\text{RotWord}(\text{temp})) \oplus (Rcon[i/4] \ll 24)$$
   - Otherwise:
     $$w_i = w_{i-4} \oplus \text{temp}$$

Round constants $Rcon[j] = x^{j-1} \in GF(2^8)$ prevent symmetry and slide attacks.

---

## 7. Experimental AEAD Mode (`MLMC-GCM-Exp`)

To support variable-length medical chunks without leaking contour patterns (as raw ECB would), MLMC is wrapped in an experimental AEAD container:

1. **Encryption**: Counter (CTR) mode with a 96-bit CSPRNG nonce and 32-bit counter starting at 2.
2. **Authentication**: GHASH polynomial evaluation over $GF(2^{128})$ modulo $x^{128} + x^7 + x^2 + x + 1$ using hash key $H = E_K(0^{128})$.
3. **Tag Masking**: $T = \text{GHASH}(H, \text{AAD}, C) \oplus E_K(\text{Nonce} \parallel \texttt{0x00000001})$.
4. **Envelope**: `[12-byte Nonce] + [Ciphertext] + [16-byte Tag]`.

---

## 8. Honest Attribution: What is Novel vs Borrowed

| Element | Classification | Source / Literature Reference |
| :--- | :--- | :--- |
| **Field $GF(2^8)$ mod $x^8+x^4+x^3+x+1$** | Established Standard | Rijndael / FIPS 197 |
| **S-Box Construction** | Established Design | Nyberg (1993), Rijndael S-box |
| **MDS Diffusion Matrix** | Established Technique | Daemen & Rijmen, Wide-Trail Design |
| **SPN Architecture ($4 \times 4$)** | Established Framework | Shannon (1949), Daemen (2002) |
| **Micro-Benchmark Adaptation for Telemedicine** | **Proposed Candidate** | MediLink-DL Sprint Research |
| **Miniature 32-bit Educational Profile (MLMC-32)** | **Proposed Variant** | For exhaustive algebraic trail visualization |
