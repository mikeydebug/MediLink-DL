# MediLink Mini Cipher (MLMC): Cryptanalytic & Statistical Security Analysis

**Candidate**: MediLink Mini Cipher (MLMC-128)  
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
**Date**: November 2026  
**Evaluator**: MediLink-DL Cryptographic Analysis Framework  
**Status**: Experimental Academic Analysis — **NOT A PROOF OF PRODUCTION SECURITY**  

---

## 1. Important Cryptographic Caveat

> [!WARNING]
> Passing avalanche tests, statistical randomness batteries, or unit tests does **NOT** constitute formal proof of cryptographic security.
> While MLMC exhibits the mathematical properties of a classical SPN cipher, **it has not undergone multi-year public cryptanalysis by the international cryptographic community.**
> This analysis establishes baseline resistance against textbook attacks; it does not warrant clinical deployment.

---

## 2. Strict Avalanche Criterion (SAC) & Diffusion Progression

The Strict Avalanche Criterion (Webster & Tavares, 1985) states that flipping any single input bit must cause each output bit to flip with probability $p = 0.5$.

### 2.1 Full 10-Round Avalanche Distribution

Empirical evaluation across 4,800 random single-bit perturbation trials (`num_samples=300`, 16 bit flips per block):

| Metric | Measured Value | Ideal Theoretical | Interpretation |
| :--- | :--- | :--- | :--- |
| **Mean Bit-Flip Ratio** | **49.92%** | **50.00%** | Deviation: 0.08% (Optimal SAC compliance) |
| **Standard Deviation ($\sigma$)** | **4.45%** | $\approx 4.41\%$ | Tightly bounded normal distribution |
| **Minimum Bit-Flip Ratio** | **33.59%** | — | No catastrophic zero-diffusion events |
| **Maximum Bit-Flip Ratio** | **67.19%** | — | Symmetric spread around mean |

### 2.2 Round-by-Round Diffusion Progression

To determine the minimum number of rounds required to achieve full avalanche saturation, bit-flip propagation was tracked through each round $r \in [1, 10]$:

```
Round 1: [==                  ]  3.14% (Localized to single active column)
Round 2: [======              ] 12.16% (Cross-column diffusion via MixColumns)
Round 3: [====================] 50.05% (FULL DIFFUSION SATURATION ACHIEVED)
Round 4: [====================] 49.88% (Avalanche Plateau)
Round 5: [====================] 50.49% (Avalanche Plateau)
Round 6: [====================] 49.48% (Avalanche Plateau)
Round 7: [====================] 49.36% (Avalanche Plateau)
Round 8: [====================] 50.24% (Avalanche Plateau)
Round 9: [====================] 50.04% (Avalanche Plateau)
Round 10:[====================] 49.79% (Full Security Margin)
```

**Observation**: MLMC achieves full diffusion saturation by **Round 3**. With 10 standard rounds, MLMC maintains a **7-round security margin** against linear and differential structural shortcut attacks.

---

## 3. Differential Cryptanalysis & Difference Distribution Table (DDT)

Differential cryptanalysis (Biham & Shamir, 1991) exploits high-probability input-output difference pairs $\Delta X \to \Delta Y$ across cipher rounds.

### 3.1 S-Box Differential Uniformity

The Difference Distribution Table (DDT) of size $256 \times 256$ was exhaustively calculated for all $\Delta x, \Delta y \in [0, 255]$:

$$\text{DDT}[\Delta x, \Delta y] = \#\{ x \in [0, 255] \mid S(x) \oplus S(x \oplus \Delta x) = \Delta y \}$$

- **Trivial Difference**: $\text{DDT}[0, 0] = 256$.
- **Maximum Non-Trivial Difference**: $\max_{\Delta x \neq 0, \Delta y} \text{DDT}[\Delta x, \Delta y] = \mathbf{4}$.
- **Differential Uniformity**: $\delta = 4$.
- **Maximum Differential Probability ($DP_{\max}$)**:
  $$DP_{\max} = \frac{4}{256} = 2^{-6} \approx 0.015625$$

### 3.2 Wide-Trail Strategy Bound Across Rounds

Because the MDS matrix has branch number $\mathcal{B} = 5$, any two consecutive rounds contain at least 5 active S-boxes. Across 4 rounds, the minimum number of active S-boxes is bounded by:

$$N_{\text{active}} \ge \mathcal{B}^2 = 25$$

The upper bound on the maximum differential characteristic probability over 4 rounds is:

$$p_{\text{diff}}^{4R} \le (DP_{\max})^{25} = (2^{-6})^{25} = 2^{-150} \ll 2^{-128}$$

This confirms that 10-round MLMC has strong structural theoretical resistance against standard differential cryptanalysis.

---

## 4. Linear Cryptanalysis & Linear Approximation Table (LAT)

Linear cryptanalysis (Matsui, 1993) seeks affine approximations $\alpha \cdot X = \beta \cdot Y$ with high correlation bias $\epsilon$.

### 4.1 S-Box Linear Bias

The Linear Approximation Table (LAT) was exhaustively calculated:

$$\text{LAT}[\alpha, \beta] = \#\{ x \in [0, 255] \mid \alpha \cdot x = \beta \cdot S(x) \} - 128$$

- **Maximum Linear Count Bias**: $\max_{\alpha, \beta \neq 0} |\text{LAT}[\alpha, \beta]| = \mathbf{16}$.
- **Maximum Correlation Bias ($\epsilon_{\max}$)**:
  $$\epsilon_{\max} = \frac{16}{256} = 2^{-4} = 0.0625$$

### 4.2 Linear Trail Bound Across Rounds

Applying Piling-Up Lemma across 25 active S-boxes over 4 rounds:

$$\epsilon_{\text{total}}^{4R} \le 2^{24} \cdot (\epsilon_{\max})^{25} = 2^{24} \cdot (2^{-4})^{25} = 2^{24} \cdot 2^{-100} = 2^{-76} \ll 2^{-64}$$

The data complexity required to exploit linear trails across 10 rounds exceeds $2^{128}$ known plaintexts, rendering standard linear cryptanalysis infeasible.

---

## 5. Statistical Randomness Evaluation (NIST SP 800-22 Tests)

A ciphertext stream of 300 encrypted blocks (4,800 bytes = 38,400 bits) generated by MLMC-128 was evaluated:

| Test | Measured Statistic | $p$-value | Decision Criterion | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Monobit Frequency** | $S_{\text{obs}} = 0.0816$, 1s Ratio = 0.5002 | **0.9349** | $p \ge 0.01$ | **PASSED** |
| **Runs Test** | Total runs = 19,252 | **0.2931** | $p \ge 0.01$ | **PASSED** |
| **Chi-Square Uniformity** | $\chi^2 = 258.99$ (dof = 255) | $\approx 0.42$ | $200.0 \le \chi^2 \le 310.0$ | **PASSED** |

---

## 6. Structural & Key Schedule Resistance

1. **Slide Attacks**: Round constants $Rcon[r] = x^{r-1} \in GF(2^8)$ differ in every round, breaking round symmetry.
2. **Symmetry Attacks**: Unique word transformations in the key schedule eliminate weak-key classes.
3. **Algebraic Degree**: The S-box algebraic degree is 7 (near maximal for an 8-bit permutation), preventing low-order differential and interpolation attacks.
