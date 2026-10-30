# Mathematical Foundations of Linear Diffusion Layers

**Project**: MediLink-DL Diffusion Research Suite  
**Document**: Finite Field Arithmetic, Matrix Homomorphisms, Coding Theory Bounds, and Wide-Trail Proofs  
**State Dimension**: 16-bit block ($n = 16$), partitioned into 4 nibbles of 4 bits each ($m = 4, k = 4$)  

---

## 1. Algebraic Structures

### 1.1 Finite Field $\mathbb{F}_{2^4}$
The finite field of 16 elements $\mathbb{F}_{2^4}$ is constructed as the quotient ring:

$$\mathbb{F}_{2^4} \cong \mathbb{F}_2[x] / (P(x))$$

where $P(x) = x^4 + x + 1$ (irreducible over $\mathbb{F}_2$, hex representation `0x13`).
Any element $\alpha \in \mathbb{F}_{2^4}$ is represented as a polynomial:

$$\alpha = a_3 x^3 + a_2 x^2 + a_1 x + a_0, \quad a_i \in \{0, 1\}$$

- **Addition**: Polynomial addition over $\mathbb{Z}_2$, executed as bitwise XOR ($\oplus$).
- **Multiplication**: Polynomial multiplication modulo $P(x)$, accelerated via generator $g = x = \texttt{0x02}$ using logarithm and exponentiation tables of length 32:
  $$\alpha \odot \beta = \text{exp}[(\text{log}[\alpha] + \text{log}[\beta]) \bmod 15]$$

### 1.2 Binary Matrix Homomorphism
Multiplication of an element $v \in \mathbb{F}_{2^4}$ by a fixed scalar $\alpha \in \mathbb{F}_{2^4}$ is a linear mapping over the vector space $\mathbb{F}_2^4$.
Therefore, every field element $\alpha \in \mathbb{F}_{2^4}$ has an exact $4 \times 4$ binary matrix representation $L(\alpha) \in \mathbb{F}_2^{4 \times 4}$, where column $j$ is the coordinate vector of $\alpha \odot x^j$:

$$L(\alpha) = \begin{bmatrix} (\alpha \odot 1)_0 & (\alpha \odot x)_0 & (\alpha \odot x^2)_0 & (\alpha \odot x^3)_0 \\ (\alpha \odot 1)_1 & (\alpha \odot x)_1 & (\alpha \odot x^2)_1 & (\alpha \odot x^3)_1 \\ (\alpha \odot 1)_2 & (\alpha \odot x)_2 & (\alpha \odot x^2)_2 & (\alpha \odot x^3)_2 \\ (\alpha \odot 1)_3 & (\alpha \odot x)_3 & (\alpha \odot x^2)_3 & (\alpha \odot x^3)_3 \end{bmatrix}$$

For any $4 \times 4$ matrix $\mathbf{M} \in \mathbb{F}_{2^4}^{4 \times 4}$, the expanded $16 \times 16$ binary matrix $M \in \mathbb{F}_2^{16 \times 16}$ is the block matrix:

$$M = \begin{bmatrix} L(\mathbf{M}_{0,0}) & L(\mathbf{M}_{0,1}) & L(\mathbf{M}_{0,2}) & L(\mathbf{M}_{0,3}) \\ L(\mathbf{M}_{1,0}) & L(\mathbf{M}_{1,1}) & L(\mathbf{M}_{1,2}) & L(\mathbf{M}_{1,3}) \\ L(\mathbf{M}_{2,0}) & L(\mathbf{M}_{2,1}) & L(\mathbf{M}_{2,2}) & L(\mathbf{M}_{2,3}) \\ L(\mathbf{M}_{3,0}) & L(\mathbf{M}_{3,1}) & L(\mathbf{M}_{3,2}) & L(\mathbf{M}_{3,3}) \end{bmatrix}$$

This homomorphism guarantees that linear algebra properties over $\mathbb{F}_{2^4}$ translate directly to hardware-level binary operations.

---

## 2. Branch Numbers & Coding Theory Bounds

A linear diffusion mapping $y = M x$ can be viewed as the generator of a $[2k, k, d]$ linear block code with generator matrix $G = [I \mid M]$.
The minimum distance $d$ of this code equals the branch number of $M$.

### 2.1 Nibble-Level Branch Number $\mathcal{B}_{\text{nibble}}$
Let $X \in (\mathbb{F}_2^4)^4$. The nibble Hamming weight $\text{wt}_{\text{nibble}}(X)$ is the number of non-zero 4-bit nibbles:

$$\mathcal{B}_{\text{nibble}}(M) = \min_{X \in \mathbb{F}_2^{16} \setminus \{0\}} \left( \text{wt}_{\text{nibble}}(X) + \text{wt}_{\text{nibble}}(M X) \right)$$

**Singleton Bound**: For an $[8, 4, d]$ code over alphabet $\mathbb{F}_{2^4}$:
$$d \le n - k + 1 = 8 - 4 + 1 = 5$$
Therefore:
$$\mathcal{B}_{\text{nibble}}(M) \le 5$$
When $\mathcal{B}_{\text{nibble}}(M) = 5$, the matrix is **Maximum Distance Separable (MDS)**.

### 2.2 Bit-Level Branch Number $\mathcal{B}_{\text{bit}}$
The bit-level branch number measures diffusion across individual bits:

$$\mathcal{B}_{\text{bit}}(M) = \min_{X \in \mathbb{F}_2^{16} \setminus \{0\}} \left( \text{wt}_{\text{bit}}(X) + \text{wt}_{\text{bit}}(M X) \right)$$

where $\text{wt}_{\text{bit}}(X)$ is the standard Hamming weight.
This corresponds to the minimum distance of the $[32, 16, d]$ binary linear code $[I_{16} \mid M]$.
By the linear programming bound (Delsarte, 1973) and the classification of binary self-dual codes (MacWilliams & Sloane):
$$\max d([32, 16]) = 8$$
Therefore, for **any** $16 \times 16$ binary matrix $M$:
$$\mathcal{B}_{\text{bit}}(M) \le 8$$
A matrix achieving $\mathcal{B}_{\text{bit}}(M) = 8$ attains the **absolute mathematical upper bound** on bit diffusion for a 16-bit block cipher.

### 2.3 Linear Branch Number (Duality)
For a linear approximation $\alpha \cdot X = \beta \cdot Y$, substituting $Y = M X$ yields:
$$\alpha \cdot X = \beta \cdot (M X) = (M^T \beta) \cdot X \implies \alpha = M^T \beta$$
Therefore, the linear branch number is the branch number of the transpose matrix $M^T$:
$$\mathcal{B}_{\text{lin, nibble}}(M) = \mathcal{B}_{\text{nibble}}(M^T)$$
$$\mathcal{B}_{\text{lin, bit}}(M) = \mathcal{B}_{\text{bit}}(M^T)$$
If $M$ is **symmetric** ($M = M^T$), the differential and linear branch numbers are mathematically identical.

---

## 3. Involutory Symmetry: $M = M^{-1}$

A matrix is **involutory (self-inverse)** if:

$$M^2 = I \iff M = M^{-1}$$

### Involutory Hadamard Matrices
A Hadamard matrix over $\mathbb{F}_{2^4}$ has the circulant-like symmetric form:

$$H = \begin{bmatrix} a & b & c & d \\ b & a & d & c \\ c & d & a & b \\ d & c & b & a \end{bmatrix}$$

Computing $H^2$ over $\mathbb{F}_{2^4}$ (where $2 \equiv 0$ and $xy = yx$):
- Diagonal entries: $a^2 \oplus b^2 \oplus c^2 \oplus d^2 = (a \oplus b \oplus c \oplus d)^2$
- Off-diagonal entry $(0, 1)$: $ab \oplus ba \oplus cd \oplus dc = 0$
- Off-diagonal entry $(0, 2)$: $ac \oplus bd \oplus ca \oplus db = 0$
- Off-diagonal entry $(0, 3)$: $ad \oplus bc \oplus cb \oplus da = 0$

All off-diagonal entries cancel out identically! Therefore:

$$H^2 = (a \oplus b \oplus c \oplus d)^2 \cdot I_4$$

**Theorem**: A Hadamard matrix $H(a, b, c, d)$ over $\mathbb{F}_{2^4}$ is involutory ($H = H^{-1}$) if and only if:

$$a \oplus b \oplus c \oplus d = 1$$

---

## 4. Provable Active S-Box Lower Bounds (Wide-Trail Strategy)

Let an SPN cipher execute rounds consisting of S-box substitution $S$ followed by linear diffusion $M$.
Let $w_r$ denote the number of active S-boxes in round $r$.

**Theorem (Wide-Trail Active S-Box Lower Bound)**:
If $M$ has nibble branch number $\mathcal{B}_{\text{nibble}}$, then for any two consecutive active rounds $r$ and $r+1$:

$$w_r + w_{r+1} \ge \mathcal{B}_{\text{nibble}}$$

*Proof*:
In round $r$, the active S-boxes produce an output difference $\Delta Y^{(r)}$ with non-zero nibble count $\text{wt}_{\text{nibble}}(\Delta Y^{(r)}) = w_r$.
The linear layer maps $\Delta Y^{(r)}$ to input difference $\Delta X^{(r+1)} = M \Delta Y^{(r)}$ for round $r+1$.
The number of active S-boxes in round $r+1$ is $w_{r+1} = \text{wt}_{\text{nibble}}(\Delta X^{(r+1)})$.
By definition of branch number:
$$w_r + w_{r+1} = \text{wt}_{\text{nibble}}(\Delta Y^{(r)}) + \text{wt}_{\text{nibble}}(M \Delta Y^{(r)}) \ge \mathcal{B}_{\text{nibble}}$$
$\blacksquare$

### Multi-Round Corollaries for MDS Diffusion ($\mathcal{B}_{\text{nibble}} = 5$):
- Across **4 rounds**:
  $$\sum_{r=1}^4 w_r = (w_1 + w_2) + (w_3 + w_4) \ge 5 + 5 = \mathbf{10} \text{ active S-boxes}$$
- Across **14 rounds** (7 pairs of rounds):
  $$\sum_{r=1}^{14} w_r \ge 7 \times 5 = \mathbf{35} \text{ active S-boxes}$$

### Comparison with Bit-Permutations and Feistel Networks:
- **Bit-Permutations (PRESENT / GIFT)**: $\mathcal{B}_{\text{nibble}} = 2$.
  Across 4 rounds: $w_1 + \dots + w_4 \ge \mathbf{4}$ active S-boxes.
  Across 14 rounds: $w_1 + \dots + w_{14} \ge \mathbf{14}$ active S-boxes.
- **Feistel Networks (Existing MLMC-G16)**:
  Across 4 rounds: active S-boxes $\ge \mathbf{2}$.
  Across 14 rounds: active S-boxes $\ge \mathbf{7}$.

### Probability Bounds with PRESENT S-Box ($DP_{\max} = 2^{-2}$):
- MDS diffusion: $p_{\text{diff}}^{14R} \le (2^{-2})^{35} = \mathbf{2^{-70}} \ll 2^{-16}$ (42 orders of magnitude below exhaustive search).
- PRESENT diffusion: $p_{\text{diff}}^{14R} \le (2^{-2})^{14} = 2^{-28}$.
- GFN diffusion: $p_{\text{diff}}^{14R} \le (2^{-2})^7 = 2^{-14} > 2^{-16}$ (vulnerable to differential attacks within 14 rounds!).

