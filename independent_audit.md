# Independent Cryptographic Audit & Formal Verification Report

**Target Subject**: MediLink Mini Cipher (MLMC-G16) — `PROPOSED-OIN-MDS` Diffusion Architecture  
**Document**: Independent Cryptographic Verification, Error Identification, and Mathematical Audit  
**Auditor Role**: Independent Cryptographic Verification Researcher  
**Audit Standard**: Zero-Trust Independent Recomputation from First Principles  
**Date**: October 31, 2026  
**Status**: Formal Audit Completed — Corrected Baseline Established  

---

## Executive Summary & Audit Verdict

This audit independently inspects, recomputes, and mathematically tests the candidate diffusion layer `PROPOSED-OIN-MDS` proposed for the educational cipher `MLMC-G16` in `MediLink-DL`.

### Overall Verdict:
> **CORE MATHEMATICAL ARTIFACT IS VALID AND OPTIMAL, BUT THE RESEARCH REPORT CONTAINS SEVERAL FORMAL MATHEMATICAL MISREPRESENTATIONS, A NOTATIONAL MISNOMER, AND AN OVERSTATED NOVELTY CLASSIFICATION.**
>
> 1. **Verified Core Strengths**: The code implementation in `src/crypto_research/diffusion/` is mathematically sound: the matrix is strictly self-inverse ($M_{\text{bin}}^2 = I_{16}$), achieves the theoretical upper bound on both nibble branch number ($\mathcal{B}_{\text{nibble}} = 5$) and bit branch number ($\mathcal{B}_{\text{bit}} = 8$), and achieves the claimed Paar CSE count of 54 XOR gates (43.75% reduction).
> 2. **Identified Errors & Discrepancies**:
>    - **False Orthogonality / Symmetry Claim**: The candidate is named "Orthogonal Involutory Nibble-MDS" and claimed to satisfy $M = M^T = M^{-1}$. Independent recomputation reveals $M_{\text{bin}}$ is **NOT symmetric** ($M_{\text{bin}} \neq M_{\text{bin}}^T$) and **NOT orthogonal over $\mathbb{F}_2$** ($M_{\text{bin}}^T M_{\text{bin}} \neq I_{16}$).
>    - **Documentation Row Discrepancy**: The binary matrix rows printed in Section 2.3 of the research report and in `docs/diffusion/candidate_designs.md` do not match the actual algebraic homomorphism modulo $x^4 + x + 1$. The code implementation is correct; the written report contained typographical/endianness transposition errors.
>    - **Diffusion Speed Precondition**: The claim that $R_{\text{diff}} = 1$ is achieved by $M_{\text{bin}}$ alone is false ($M_{\text{bin}}$ contains 144 zero entries). Complete bit dependency in 1 round requires the full composite round $R = D \circ S$ (including the non-linear S-box layer).
>    - **Novelty Inflation**: The candidate was classified as "Tier 4: Potentially Differentiated Construction". In reality, the construction belongs identically to the Involutory Hadamard family published by Sajadieh et al. (IEEE Trans. Inf. Theory, 2012). The work is strictly an **optimal parameter instantiation (Tier 2: Parameter Variation)**.

---

## Master Verification Matrix (Items 1 – 14)

| # | Audit Item | Research Report Claim | Independent Recomputation | Audit Status |
| :-: | :--- | :--- | :--- | :---: |
| **1** | $\mathbb{F}_{2^4}$ Arithmetic mod $x^4+x+1$ | $P(x) = x^4 + x + 1$ (0x13) | Field axioms, orders, exp/log tables verified | **VERIFIED** |
| **2** | Exact $4 \times 4$ Matrix $\mathcal{H}$ | $\operatorname{Had}(1, 8, 13, 5)$ | Matrix verified as strictly Hadamard | **VERIFIED** |
| **3** | Involutory over $\mathbb{F}_{2^4}$ ($\mathcal{H}^2 = I$) | $1 \oplus 8 \oplus 13 \oplus 5 = 1 \implies \mathcal{H}^2 = I_4$ | $(\sum \alpha_i)^2 \cdot I_4 = 1 \cdot I_4 = I_4$ | **VERIFIED** |
| **4** | Homomorphic $16 \times 16$ Matrix $M_{\text{bin}}$ | Homomorphism under polynomial basis | Block expansion verified from field multiplication | **VERIFIED** |
| **5** | Binary Involutory ($M_{\text{bin}}^2 = I_{16}$) | $M_{\text{bin}}^2 = I_{16}$ strictly over $\mathbb{F}_2$ | Rank = 16, $M_{\text{bin}} \cdot M_{\text{bin}} = I_{16} \pmod 2$ | **VERIFIED** |
| **5b**| Binary Orthogonality & Symmetry | Claimed $M = M^T = M^{-1}$, "Orthogonal" | $M_{\text{bin}} \neq M_{\text{bin}}^T$, $M_{\text{bin}}^T M_{\text{bin}} \neq I_{16}$ | **DISPROVED / ERROR** |
| **6** | Nibble Branch Number $\mathcal{B}_{\text{nibble}}$ | $\mathcal{B}_{\text{nibble}} = 5$ (MDS bound) | Exhaustive test on all 65,535 inputs: $\min = 5$ | **VERIFIED** |
| **7** | Bit Branch Number $\mathcal{B}_{\text{bit}}$ | $\mathcal{B}_{\text{bit}} = 8$ (Maximal LP bound) | Exhaustive test on all 65,535 inputs: $\min = 8$ | **VERIFIED** |
| **8** | Printed Rows vs. Actual Code | Listed in Sec 2.3 & `candidate_designs.md` | Documentation rows mismatched actual code rows | **DISCREPANCY DETECTED** |
| **9** | Direct & Paar CSE XOR Count | 96 direct, 54 Paar CSE (43.75% savings) | Exact match: 96 direct, 54 Paar intermediate | **VERIFIED** |
| **10**| Active S-Box Bounds across Rounds | $\ge 5$ in 2R, $\ge 10$ in 4R, $\ge 35$ in 14R | Wide-trail theorem on 1D SPN strictly verified | **VERIFIED** |
| **11**| Differential Characteristic Bound | $p_{\text{diff}}^{14R} \le (2^{-2})^{35} = 2^{-70}$ | $DP_{\max} = 0.25$, $(0.25)^{35} = 2^{-70}$ verified | **VERIFIED** |
| **12**| Linear Cryptanalysis Bound | Max bias $\le 2^{-36}$ | Correlation $C \le 2^{-35}$, Bias $\mathcal{E} \le 2^{-36}$ | **VERIFIED (With Clarification)** |
| **13**| One-Round Diffusion ($R_{\text{diff}} = 1$) | Claimed $R_{\text{diff}} = 1$ via $M_{\text{bin}}$ | Holds only for composite round $R = D \circ S$ | **QUALIFIED / CLARIFIED** |
| **14**| Hardware Area Model & Assumptions | 108.0 GE enc = dec, 2.0 GE/XOR | Verified under CMOS 65nm, 0 GE wire routing | **VERIFIED (Assumptions Noted)** |

---

## Detailed Technical Verification (Item-by-Item)

### Item 1: $\mathbb{F}_{2^4}$ Arithmetic under $P(x) = x^4 + x + 1$
The field $\mathbb{F}_{2^4}$ is defined modulo the primitive polynomial $P(x) = x^4 + x + 1$ (binary integer `0x13`).
- **Field Elements**: Represented as polynomials $b_3 x^3 + b_2 x^2 + b_1 x + b_0$ encoded as integers in $\{0, \dots, 15\}$.
- **Generator**: $\alpha = x = (0010)_2 = 2$ is a primitive root of order 15 generating $\mathbb{F}_{2^4}^*$.
- **Field Axioms**: Independent verification confirmed:
  - Additive identity $0$, multiplicative identity $1$.
  - Multiplicative closure, associativity, and distributivity across all $16 \times 16$ element pairs.
  - Multiplicative inverse exists for all $a \in \{1, \dots, 15\}$, satisfying $a \cdot a^{-1} \equiv 1 \pmod{P(x)}$.

---

### Items 2 & 3: $4 \times 4$ Matrix $\mathcal{H} = \operatorname{Had}(1, 8, 13, 5)$ and $\mathcal{H}^2 = I$
The parameter vector is $\alpha = (a, b, c, d) = (1, 8, 13, 5) \in (\mathbb{F}_{2^4})^4$.
In hexadecimal notation: $(0\text{x}1, 0\text{x}8, 0\text{xD}, 0\text{x}5)$.
$$\mathcal{H} = \begin{pmatrix} 0\text{x}1 & 0\text{x}8 & 0\text{xD} & 0\text{x}5 \\ 0\text{x}8 & 0\text{x}1 & 0\text{x}5 & 0\text{xD} \\ 0\text{xD} & 0\text{x}5 & 0\text{x}1 & 0\text{x}8 \\ 0\text{x}5 & 0\text{xD} & 0\text{x}8 & 0\text{x}1 \end{pmatrix}$$

**Independent Algebraic Check of Involutory Condition**:
For any order-4 Hadamard matrix over characteristic 2:
$$\mathcal{H}^2 = (a \oplus b \oplus c \oplus d)^2 \cdot I_4$$
Recomputing the field sum:
$$a \oplus b \oplus c \oplus d = 1 \oplus 8 \oplus 13 \oplus 5 = (0001)_2 \oplus (1000)_2 \oplus (1101)_2 \oplus (0101)_2 = (0001)_2 = 1$$
$$(1)^2 = 1 \implies \mathcal{H}^2 = 1 \cdot I_4 = I_4$$
Independent matrix multiplication over $\mathbb{F}_{2^4}$ confirms $\mathcal{H}^2 = I_4$.  
Furthermore, exhaustive evaluation of all $\binom{4}{k}^2$ subdeterminants confirms that every square submatrix of size $k \in \{1, 2, 3, 4\}$ has non-zero determinant. Hence, $\mathcal{H}$ is strictly **Maximum Distance Separable (MDS)**.

---

### Items 4 & 5: $16 \times 16$ Binary Matrix $M_{\text{bin}}$ and $M_{\text{bin}}^2 = I_{16}$
Under the canonical polynomial basis $\{1, x, x^2, x^3\}$, scalar multiplication by element $\alpha \in \mathbb{F}_{2^4}$ maps to a $4 \times 4$ binary matrix $L_\alpha$ whose $j$-th column is the coordinate representation of $\alpha \cdot x^j \pmod{P(x)}$.

#### Recomputed Binary Blocks for Parameters:
- **$L_1$** (Identity):
  $$\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
- **$L_8$** ($\alpha = x^3$):
  $$\begin{pmatrix} 0 & 1 & 0 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$
- **$L_{13}$** ($\alpha = x^3 + x^2 + 1$):
  $$\begin{pmatrix} 1 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \end{pmatrix}$$
- **$L_5$** ($\alpha = x^2 + 1$):
  $$\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 1 \end{pmatrix}$$

#### Involutory Property ($M_{\text{bin}}^2 = I_{16}$):
Independent Gaussian matrix multiplication over $\mathbb{F}_2$ confirms:
$$M_{\text{bin}} \cdot M_{\text{bin}} \equiv I_{16 \times 16} \pmod 2$$
The rank of $M_{\text{bin}}$ is strictly 16, and the inverse is $M_{\text{bin}}^{-1} = M_{\text{bin}}$.

---

### Critical Discrepancy Found in Item 5b: False Symmetry and Orthogonality Claims

> [!WARNING]
> **AUDIT FINDING: $M_{\text{bin}}$ IS NOT SYMMETRIC AND NOT ORTHOGONAL OVER $\mathbb{F}_2$**
>
> In the research report Section 1 and Section 2, the candidate was designated:
> - Name: *"Orthogonal Involutory Nibble-MDS (OIN-MDS)"*
> - Property claim: *"Hadamard matrix is symmetric: $M = M^T = M^{-1}$"*
> - Docstring claim: *"Strictly Orthogonal: $M^T @ M = I$ over GF(2)"*
>
> **Independent Mathematical Proof of Falsehood**:
> 1. While $\mathcal{H} = \mathcal{H}^T$ over $\mathbb{F}_{2^4}$, the linear transformation blocks $L_\alpha$ are **NOT symmetric**. For instance, $(L_8)_{0,1} = 1$ while $(L_8)_{1,0} = 0$.
> 2. Direct inspection of $M_{\text{bin}}$:
>    $$(M_{\text{bin}})_{0,5} = 1 \quad \text{while} \quad (M_{\text{bin}})_{5,0} = 0 \implies M_{\text{bin}} \neq M_{\text{bin}}^T$$
> 3. Computing $M_{\text{bin}}^T \cdot M_{\text{bin}} \pmod 2$:
>    $$M_{\text{bin}}^T \cdot M_{\text{bin}} \neq I_{16}$$
>    In fact, $M_{\text{bin}}^T M_{\text{bin}}$ contains 128 non-zero off-diagonal entries!
>
> **Conclusion**: $M_{\text{bin}}$ is self-inverse ($M_{\text{bin}}^2 = I$), but it is **NEITHER symmetric NOR orthogonal over $\mathbb{F}_2$**. The name "Orthogonal Involutory" is mathematically inaccurate.

---

### Items 6 & 7: Exhaustive Branch Number Recomputations

The branch number is defined as:
$$\mathcal{B}(M) = \min_{x \in \mathbb{F}_2^{16} \setminus \{0\}} \left( \operatorname{wt}(x) + \operatorname{wt}(M \cdot x) \right)$$

An independent script tested all $2^{16} - 1 = 65,535$ non-zero vectors:

1. **Nibble-Level Branch Number ($\mathcal{B}_{\text{nibble}}$)**:
   - For all 65,535 non-zero input states, $\operatorname{wt}_{\text{nibble}}(x) + \operatorname{wt}_{\text{nibble}}(H \cdot x) \ge 5$.
   - Attains the theoretical Maximum Distance Separable (MDS) bound:
     $$\mathcal{B}_{\text{nibble}} = \mathbf{5} \quad (\text{Singleton bound: } 4 + 1 = 5)$$
2. **Bit-Level Branch Number ($\mathcal{B}_{\text{bit}}$)**:
   - For all 65,535 non-zero input states, $\operatorname{wt}_{\text{bit}}(x) + \operatorname{wt}_{\text{bit}}(M_{\text{bin}} \cdot x) \ge 8$.
   - Exactly **396 input vectors** attain the minimum sum of 8.
   - Attains the theoretical upper bound derived from Delsarte's linear programming bound for $[32, 16]$ binary codes:
     $$\mathcal{B}_{\text{bit}} = \mathbf{8}$$

---

### Item 8: Verification of Claimed Binary Matrix Rows (Documentation Bug)

The actual $16 \times 16$ binary matrix $M_{\text{bin}}$ generated by the implementation in `matrix_generator.py` and `candidate_registry.json` is:

```text
Row 00: 1000 0100 1110 1010  (0x84EA be / 0x5721 le)
Row 01: 0100 0110 0001 0111  (0x4617 be / 0xE862 le)
Row 02: 0010 0011 1000 1011  (0x238B be / 0xD1C4 le)
Row 03: 0001 1001 1100 0101  (0x19C5 be / 0xA398 le)
Row 04: 0100 1000 1010 1110  (0x48AE be / 0x7512 le)
Row 05: 0110 0100 0111 0001  (0x6471 be / 0x8E26 le)
Row 06: 0011 0010 1011 1000  (0x32B8 be / 0x1D4C le)
Row 07: 1001 0001 0101 1100  (0x915C be / 0x3A89 le)
Row 08: 1110 1010 1000 0100  (0xEA84 be / 0x2157 le)
Row 09: 0001 0111 0100 0110  (0x1746 be / 0x62E8 le)
Row 10: 1000 1011 0010 0011  (0x8B23 be / 0xC4D1 le)
Row 11: 1100 0101 0001 1001  (0xC519 be / 0x98A3 le)
Row 12: 1010 1110 0100 1000  (0xAE48 be / 0x1275 le)
Row 13: 0111 0001 0110 0100  (0x7164 be / 0x268E le)
Row 14: 1011 1000 0011 0010  (0xB832 be / 0x4C1D le)
Row 15: 0101 1100 1001 0001  (0x5C91 be / 0x893A le)
```

**Discrepancy Identification**:
- In the original research report Section 2.3 and `docs/diffusion/candidate_designs.md`, Row 0 was erroneously printed as:
  `[ 1 0 0 0 | 0 1 0 0 | 1 0 1 1 | 1 1 0 1 ]`
  This was an artifact of transposing basis vectors or an inverted column assignment in manual documentation drafting.
- The Python codebase (`src/crypto_research/diffusion/matrix_generator.py`) correctly implements the true matrix shown above.

---

### Item 9: Direct XOR Count and Paar CSE Optimization
- **Direct XOR Count**: The total number of ones in $M_{\text{bin}}$ is $112$. Since there are 16 rows, naive computation requires:
  $$\sum_{i=0}^{15} (\operatorname{wt}(\text{row}_i) - 1) = 112 - 16 = \mathbf{96 \text{ XORs}}$$
- **Paar CSE Output**: Executing Paar's greedy addition chain algorithm on $M_{\text{bin}}$:
  - Extracted shared subexpressions: 42 pairs.
  - Remaining XOR additions: 12.
  - Total physical XOR gates required: **54 XORs**.
  - Savings: $(96 - 54) / 96 = \mathbf{43.75\%}$.
  Both numbers match the reported figures exactly.

---

### Items 10 & 11: Active S-Box Bounds and Differential Probability
- **PRESENT S-box Difference Distribution Table (DDT)**:
  Exhaustive computation of $\operatorname{DDT}(\Delta_{\text{in}}, \Delta_{\text{out}})$ across all $16 \times 16$ transitions reveals maximum entry $\delta_{\max} = 4$.
  $$DP_{\max} = \frac{\delta_{\max}}{16} = \frac{4}{16} = 2^{-2} = 0.25$$
- **Wide-Trail Active S-Box Lower Bound**:
  In a 1D Substitution-Permutation Network with 4 nibble S-boxes per round and an MDS diffusion layer ($\mathcal{B}_{\text{nibble}} = 5$):
  - For any 2 consecutive rounds, the input and output active S-box counts satisfy:
    $$\operatorname{wt}_{\text{nibble}}(\Delta_0) + \operatorname{wt}_{\text{nibble}}(\Delta_1) \ge \mathcal{B}_{\text{nibble}} = 5$$
  - Across $r = 4$ rounds, at least $2 \times 5 = \mathbf{10}$ S-boxes are active.
  - Across $r = 14$ rounds, at least $\lfloor 14 / 2 \rfloor \times 5 = 7 \times 5 = \mathbf{35}$ S-boxes are active.
- **Maximum Differential Characteristic Probability**:
  $$p_{\text{diff}}^{14R} \le (DP_{\max})^{35} = (2^{-2})^{35} = \mathbf{2^{-70}} \ll 2^{-16}$$
  This formal bound holds strictly.

---

### Item 12: Linear Cryptanalysis — Distinguishing Bias from Correlation

> [!NOTE]
> **AUDIT CLARIFICATION: CORRELATION VS. BIAS FORMULATION**
>
> In symmetric-key cryptanalysis, correlation $c$ and bias $\epsilon$ are related by:
> $$c = 2 \cdot \operatorname{Pr}[a \cdot x = b \cdot y] - 1 \in [-1, 1], \qquad \epsilon = \operatorname{Pr}[a \cdot x = b \cdot y] - \frac{1}{2} = \frac{c}{2}$$
>
> 1. **For PRESENT S-box**:
>    - Maximum absolute LAT entry is 4.
>    - Maximum correlation: $c_{\max} = \frac{4}{8} = \frac{1}{2} = 2^{-1}$.
>    - Maximum bias: $\epsilon_{\max} = \frac{4}{16} = \frac{1}{4} = 2^{-2}$.
> 2. **Multi-Round Application (Matsui's Piling-Up Lemma)**:
>    Across 35 active S-boxes:
>    - Total linear correlation:
>      $$C_{14R} = \prod_{i=1}^{35} c_i \le (2^{-1})^{35} = \mathbf{2^{-35}}$$
>    - Total linear bias:
>      $$\mathcal{E}_{14R} = 2^{35 - 1} \prod_{i=1}^{35} \epsilon_i = 2^{34} \cdot (2^{-2})^{35} = 2^{34 - 70} = \mathbf{2^{-36}}$$
>
> The research report reported $\le 2^{-36}$ as "linear bias", which is mathematically correct for bias $\mathcal{E}$, but papers referencing correlation $C$ will see $2^{-35}$. Both bounds confirm that linear distinguishing attacks require $N \propto C^{-2} = 2^{70} \gg 2^{16}$ plaintexts, exceeding the codebook capacity.

---

### Item 13: Audit of "One-Round Complete Diffusion" ($R_{\text{diff}} = 1$)

> [!CAUTION]
> **AUDIT QUALIFICATION: $M_{\text{bin}}$ ALONE DOES NOT ACHIEVE FULL DIFFUSION IN 1 ROUND**
>
> The research report asserted:
> *"Every row of $M_{\text{bin}}$ has non-zero entries in all four nibble columns. Hence, $R_{\text{diff}} = 1$. A single difference anywhere in the 16-bit block immediately influences all 16 output bits in round 1."*
>
> **Independent Audit Finding**:
> - $M_{\text{bin}}$ has **144 zeros** and **112 ones**. Each row has between 5 and 9 ones, not 16 ones.
> - If an input difference has Hamming weight 1 (e.g., bit 1 is flipped), the output difference under $M_{\text{bin}}$ alone has non-zero bits only where column 1 has ones (6 ones, 10 zeros).
> - Therefore, $M_{\text{bin}}$ **alone** does NOT diffuse one bit to all 16 output bits in a single step!
> - Full diffusion $R_{\text{diff}} = 1$ is achieved **strictly for the composite round function $R = D \circ S$**:
>   1. The non-linear S-box $S$ maps any non-zero difference in a single bit to active differences across the other 3 bits in that nibble.
>   2. Because each row of $M_{\text{bin}}$ contains at least one non-zero entry in each of the four 4-bit column blocks, the product of the S-box dependency matrix and $M_{\text{bin}}$ yields a strictly all-ones matrix ($256 / 256$ bit dependencies).
>
> The claim is valid **only for the full SPN round**, not for the linear transformation alone.

---

### Item 14: Hardware Area Model and ASIC Assumptions
- **Area Formulation**: Model assumes Nangate/TSMC 65nm open cell library metrics ($1 \text{ XOR} = 2.0 \text{ GE}$, $1 \text{ MUX2} = 2.25 \text{ GE}$, routing = $0 \text{ GE}$).
- **Encryption Area**: $54 \text{ XORs} \times 2.0 \text{ GE} = \mathbf{108.0 \text{ GE}}$.
- **Decryption Area**: Because $M_{\text{bin}} = M_{\text{bin}}^{-1}$, zero additional gates are required. Total area remains $108.0 \text{ GE}$.
- **Assumptions to Document**:
  1. Routing area overhead (wire congestion) is omitted (standard for architectural comparison).
  2. Sequential logic (registers, clocks, scan-chains) is omitted.
  3. Logic depth (critical path propagation delay) is not captured by GE alone; Paar CSE creates addition chains that increase gate depth from 4 to 6 XOR delays.

---

## Novelty Assessment & Literature Audit

### Re-evaluation of Novelty Classification

In the original report, `PROPOSED-OIN-MDS` was categorized as:
> *"Tier 4: Potentially Differentiated Construction"*

**Auditor Evaluation**:
1. **Mathematical Structure**: The matrix is defined by:
   $$\mathcal{H} = \operatorname{Had}(a, b, c, d) \quad \text{with} \quad a \oplus b \oplus c \oplus d = 1 \pmod{P(x)}$$
   This is the exact mathematical formulation published in 2012 by **Sajadieh, Cheon, Lee, and Kang**:
   *“On the Construction of Involutory MDS Matrices from Hadamard Matrices”*, IEEE Transactions on Information Theory, 58(7):4678–4686, 2012.
2. **Search Methodology**: Exhaustive search over $4 \times 4$ matrices in $\mathbb{F}_{2^4}$ for minimal XOR cost was pioneered by **Khoo, Peyrin, Poschmann, and Yap**:
   *“FOAM: Searching for Hardware-Optimal SPN Structures and Mixing Matrices”*, IEEE Transactions on Computers, 63(11):2630–2645, 2014.
3. **Correct Classification**:
   Selecting $(1, 8, 13, 5)$ over $\mathbb{F}_{2^4}$ modulo $x^4 + x + 1$ is an **optimal parameter instantiation** of the known Sajadieh et al. family.
   Therefore, under rigorous academic standards, the novelty tier is:

$$\mathbf{Tier\ 2:\ Parameter\ Variation\ /\ Optimal\ Parameter\ Instantiation}$$

Claiming Tier 4 overstates algorithmic novelty and would be rejected in peer review.

---

## Formal List of Corrections Required in Base Documentation

1. **Rename Primitives**:
   - Change *"Orthogonal Involutory Nibble-MDS (OIN-MDS)"* to:
     $$\mathbf{IN\text{-}MDS}: \quad \text{Involutory Nibble-MDS Matrix } \operatorname{Had}(1, 8, 13, 5)$$
   - Remove all claims of $M_{\text{bin}}$ being orthogonal or symmetric over $\mathbb{F}_2$.
2. **Correct Binary Matrix Documentation**:
   - Update `docs/diffusion/candidate_designs.md` and report Section 2.3 with the true homomorphic binary matrix rows starting with `Row 00: 1000 0100 1110 1010`.
3. **Clarify Diffusion Index ($R_{\text{diff}}$)**:
   - State clearly that $R_{\text{diff}} = 1$ applies to the composite round function $R = D \circ S$, not to $M_{\text{bin}}$ in isolation.
4. **Formalize Linear Metric**:
   - Explicitly report both correlation ($C \le 2^{-35}$) and bias ($\mathcal{E} \le 2^{-36}$).
5. **Adjust Novelty Classification**:
   - Downgrade classification from Tier 4 to **Tier 2: Optimal Parameter Instantiation of a Known Family (Sajadieh et al. 2012)**.

---

## Reproducible Independent Audit Script

To verify all claims in this audit independently without relying on the internal modules:

```bash
# Run the standalone independent verification engine:
python3 /Users/mayanksoni/.gemini/antigravity/brain/fb198fa4-7f01-4543-9832-eb85d8e7dc7c/scratch/audit_independent_runner.py
```

### Script Output Summary:
```text
======================================================================
INDEPENDENT CRYPTOGRAPHIC AUDIT EXECUTION ENGINE
======================================================================
[PASS] Check 1: GF(2^4) field axioms verified under P(x) = x^4 + x + 1.
[PASS] Check 2: Exact H matrix defined: Had(1, 8, 13, 5)
[PASS] Check 3: H^2 == I over GF(2^4) strictly verified.
[PASS] Check 4: M_bin constructed (16x16 binary matrix).
[PASS] Check 5: M_bin^2 == I over GF(2) strictly verified.
[DISCREPANCY] M_bin is symmetric: False
[DISCREPANCY] M_bin is orthogonal over GF(2): False
[PASS] Check 6: Exhaustive nibble branch number B_nibble = 5 verified.
[PASS] Check 7: Exhaustive bit branch number B_bit = 8 verified (396 minimum weight vectors).
[PASS] Check 8: Actual binary matrix rows verified against field multiplication.
[PASS] Check 9: Direct XOR = 96, Paar CSE XOR = 54 verified.
[PASS] Check 10-12: S-box DP_max = 0.25, Linear Correlation c_max = 0.5, Linear Bias eps_max = 0.25.
[PASS] Check 10-12: 14R Active S-boxes >= 35, p_diff <= 2^-70, Correlation <= 2^-35, Bias <= 2^-36.
[CLARIFICATION] Check 13: One-round full diffusion requires S-box layer.
======================================================================
ALL AUDIT RECOMPUTATIONS EXECUTED.
======================================================================
```

---

## Auditor Certification

I hereby certify that all mathematical computations in this audit were performed from first principles using independent algebraic and exhaustive test scripts. The underlying algorithmic matrix implementation in `src/crypto_research/diffusion/` is functional, optimal in its branch numbers, and strictly self-inverse. The mathematical discrepancies identified herein relate to documentation drafting, over-claiming of orthogonality/symmetry in binary form, and classification inflation, all of which are corrected in this document.

