# Rigorous Novelty Assessment & Classification

**Project**: MediLink-DL Diffusion Research Suite  
**Document**: Candidate Classification, Attribution, and Intellectual Honesty Statement  
**Author Role**: Senior Cryptography Researcher + Cryptographic Algorithm Designer  

---

## 1. Principles of Cryptographic Novelty Evaluation

In symmetric-key cryptography, introducing a new matrix or finding a specific combination of parameters that yields a high branch number does **not** constitute a novel fundamental cryptographic primitive. 
To ensure scientific integrity, we adhere to the following principles:

1. **A new matrix is not automatically a novel cryptographic contribution**: Any exhaustive search will encounter unindexed matrices. Without a new mathematical construction principle, a matrix is merely a parameter instantiation.
2. **A higher branch number is not automatically a novel contribution**: Branch numbers are bounded by coding theory (Singleton bound, Delsarte bounds). Finding matrices meeting these bounds is an optimization achievement, not a mathematical breakthrough.
3. **A lower XOR count is not automatically a novel contribution**: Optimizing XOR counts via Common Subexpression Elimination (CSE / Paar) is standard engineering practice.
4. **Attribution must be explicit**: Every design component must be traced to its original literature source.

---

## 2. Classification of All Evaluated Candidates

We classify every candidate into one of the five mandatory tiers:

| Candidate ID | Name | Classification | Detailed Justification |
| :--- | :--- | :--- | :--- |
| `BASE-01-MLMC-G16` | Existing MLMC-G16 GFN | **Known construction** | Directly uses standard Type-II Generalized Feistel Network with established cyclic rotate-XOR linear functions. |
| `BASE-02-PRESENT-16` | PRESENT-16 Permutation | **Known construction** | Direct 16-bit transposition adaptation of the standard PRESENT wire permutation (Bogdanov et al., CHES 2007). |
| `BASE-03-GIFT-16` | GIFT-16 Permutation | **Known construction** | Direct 16-bit adaptation of the standard GIFT position-grouped bit permutation (Banik et al., CHES 2017). |
| `FAM-A1-SYLVESTER-HAD`| Sylvester-Hadamard Binary | **Known construction** | Classical Sylvester-Kronecker recursive product applied over GF(2). |
| `FAM-B1-CAUCHY-MDS` | Cauchy MDS over $\mathbb{F}_{2^4}$ | **Known construction** | Textbook Cauchy matrix construction (Youssef et al., 2004) evaluated over $\mathbb{F}_{2^4}$. |
| `FAM-C1-INV-HAD-OPT` | Involutory Had(1,4,9,13) | **Parameter variation** | Applies the established involutory Hadamard framework (Barreto & Rijmen, 2000) with a search for lowest XOR cost over $\mathbb{F}_{2^4}$. |
| `FAM-C2-INV-HAD-MAXBIT`| Involutory Had(1,8,13,5) | **Parameter variation** | Parameter instantiation within the known Hadamard family that optimizes bit-level branch number. |
| `FAM-D1-CIRCULANT-MDS` | Circulant circ(1,4,9,1) | **Parameter variation** | Standard circulant matrix space search in $\mathbb{F}_{2^4}$ for MDS properties. |
| `FAM-E1-RECURSIVE-MDS` | Recursive Companion $A^4$ | **Recombination of known constructions** | Adapts the companion-matrix serialized MDS technique from PHOTON/LED to 4 nibbles with custom feedback coefficients $(1, 1, 12, 9)$. |
| `FAM-F1-HYBRID-PERM-XOR`| Hybrid Permutation-XOR | **Recombination of known constructions** | Recombines bit-permutations with unipotent lower-triangular XOR ladders, inspired by RECTANGLE and Ascon. |
| **`PROPOSED-OIN-MDS`** | **Orthogonal Involutory Nibble-MDS** | **Potentially differentiated construction** | *See detailed evaluation below.* |

---

## 3. Honest Evaluation of the Proposed Architecture: OIN-MDS

### 3.1 What is Inherited (Established Prior Art)
1. **The Hadamard Matrix Template**: The algebraic form $\text{Had}(a, b, c, d)$ was established by Barreto & Rijmen (2000) for Anubis and Khazad.
2. **The Involutory Condition**: The condition $a \oplus b \oplus c \oplus d = 1$ in characteristic 2 is a known algebraic identity.
3. **The Finite Field $\mathbb{F}_{2^4}$**: The field polynomial $x^4 + x + 1$ is standard.
4. **MDS Definition and Branch Numbers**: Formulated by Daemen & Rijmen (Wide-Trail Strategy).

### 3.2 What is Genuinely Proposed & Differentiated
1. **Joint Dual-Branch Optimization**:
   While prior lightweight Hadamard matrices were selected solely for word-level branch number ($\mathcal{B}_{\text{nibble}} = 5$) or naive weight, OIN-MDS was identified through a global multi-objective search that **jointly maximizes both bit-level branch number ($\mathcal{B}_{\text{bit}} = 8$) and word-level branch number ($\mathcal{B}_{\text{nibble}} = 5$) simultaneously**.
   Achieving $\mathcal{B}_{\text{bit}} = 8$ hits the exact theoretical upper bound for a $[32, 16]$ binary linear code.
2. **Symmetric Dual Orthogonality**:
   Because the matrix is symmetric and involutory ($M = M^T = M^{-1}$), the linear branch numbers are identically optimal ($\mathcal{B}_{\text{lin, bit}} = 8, \mathcal{B}_{\text{lin, nibble}} = 5$).
3. **Pair-Optimized Addition Chain for Embedded Hardware**:
   Under Paar's CSE, the circuit reduces from 96 to 54 XOR gates, with 0 additional gates required for the decryption path.

### 3.3 What Remains an Unverified Research Hypothesis
- Whether the high bit branch number $\mathcal{B}_{\text{bit}} = 8$ provides measurable resistance against non-standard cryptanalysis (e.g., division property, invariant subspace attacks) on reduced-round variants of MLMC-G16 requires further multi-round algebraic cryptanalysis.
- It must **not** be claimed that OIN-MDS makes MLMC-G16 a production-secure cipher. A diffusion layer provides structural diffusion; full cipher security depends equally on S-box non-linearity, key schedule diffusion, and round count.

