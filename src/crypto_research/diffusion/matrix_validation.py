"""
src/crypto_research/diffusion/matrix_validation.py

Mathematical foundations and matrix validation utilities over GF(2) and GF(2^4).
Academic Research Framework for MLMC Diffusion Architecture.

Implements:
  1. Finite field arithmetic over GF(2^4) with irreducible polynomial P(x) = x^4 + x + 1 (0x13).
  2. Gaussian elimination, rank, determinant, and matrix inversion over GF(2).
  3. Gaussian elimination, determinant, and matrix inversion over GF(2^4).
  4. Homomorphism mapping 4x4 matrices over GF(2^4) to 16x16 binary matrices over GF(2).
  5. Deterministic validation of matrix properties (singularity, involutory, orthogonality).
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple, Union
import numpy as np


# =====================================================================
# 1. Finite Field GF(2^4) Arithmetic
# =====================================================================

# Irreducible polynomial P(x) = x^4 + x + 1 (0x13 in binary: 10011)
POLY_GF16 = 0x13


class GF16:
    """
    Finite Field GF(2^4) arithmetic modulo P(x) = x^4 + x + 1.
    Elements are represented as 4-bit integers in [0, 15].
    """

    def __init__(self, poly: int = POLY_GF16):
        self.poly = poly
        self.exp = [0] * 32
        self.log = [0] * 16
        self._init_tables()

    def _init_tables(self) -> None:
        """Initialize exp and log tables using generator g = x (0x02)."""
        # In GF(2^4) mod x^4 + x + 1, alpha = 0x02 is a primitive root of order 15
        val = 1
        for i in range(15):
            self.exp[i] = val
            self.exp[i + 15] = val
            self.log[val] = i
            # Multiply by x (0x02)
            val <<= 1
            if val & 0x10:
                val ^= self.poly
        self.log[0] = 0  # Convention: log(0) undefined

    def add(self, a: int, b: int) -> int:
        """Addition in GF(2^4) is bitwise XOR."""
        return (a ^ b) & 0x0F

    def multiply(self, a: int, b: int) -> int:
        """Multiplication in GF(2^4) via log/exp tables."""
        a &= 0x0F
        b &= 0x0F
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def inverse(self, a: int) -> int:
        """Multiplicative inverse in GF(2^4), inv(0) = 0."""
        a &= 0x0F
        if a == 0:
            return 0
        return self.exp[15 - self.log[a]]

    def divide(self, a: int, b: int) -> int:
        """Division in GF(2^4): a / b = a * inv(b)."""
        b &= 0x0F
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(2^4)")
        return self.multiply(a, self.inverse(b))

    def to_binary_matrix(self, alpha: int) -> np.ndarray:
        """
        Converts scalar multiplication by alpha in GF(2^4) to a 4x4 binary matrix over GF(2).
        For x in GF(2^4), alpha * x is a linear mapping over GF(2)^4.
        Column j of the matrix is alpha * (1 << j).
        """
        alpha &= 0x0F
        mat = np.zeros((4, 4), dtype=np.uint8)
        for j in range(4):
            basis_vector = 1 << j
            prod = self.multiply(alpha, basis_vector)
            for i in range(4):
                mat[i, j] = (prod >> i) & 1
        return mat


# Global GF(2^4) instance
gf16 = GF16()


# =====================================================================
# 2. Linear Algebra over GF(2) (Binary Matrices)
# =====================================================================

def matmul_gf2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication over GF(2): (A @ B) % 2."""
    return (np.dot(A.astype(np.int32), B.astype(np.int32)) % 2).astype(np.uint8)


def matvec_gf2(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector multiplication over GF(2): (A @ x) % 2."""
    return (np.dot(A.astype(np.int32), x.astype(np.int32)) % 2).astype(np.uint8)


def rank_gf2(M: np.ndarray) -> int:
    """Computes the rank of a binary matrix over GF(2) using Gaussian elimination."""
    A = M.copy().astype(np.uint8) % 2
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # Find pivot
        pivot = -1
        for i in range(r, rows):
            if A[i, c] == 1:
                pivot = i
                break
        if pivot == -1:
            continue
        # Swap rows
        A[[r, pivot]] = A[[pivot, r]]
        # Eliminate below
        for i in range(r + 1, rows):
            if A[i, c] == 1:
                A[i] ^= A[r]
        r += 1
    return r


def is_invertible_gf2(M: np.ndarray) -> bool:
    """Checks if square binary matrix M is non-singular over GF(2)."""
    rows, cols = M.shape
    if rows != cols:
        return False
    return rank_gf2(M) == rows


def inverse_gf2(M: np.ndarray) -> np.ndarray:
    """
    Computes the exact inverse of a square binary matrix over GF(2)
    using Gauss-Jordan elimination on [M | I].
    Raises ValueError if matrix is singular.
    """
    rows, cols = M.shape
    if rows != cols:
        raise ValueError(f"Matrix must be square, got {rows}x{cols}")
    n = rows
    aug = np.zeros((n, 2 * n), dtype=np.uint8)
    aug[:, :n] = M.copy() % 2
    aug[:, n:] = np.eye(n, dtype=np.uint8)

    for c in range(n):
        pivot = -1
        for r in range(c, n):
            if aug[r, c] == 1:
                pivot = r
                break
        if pivot == -1:
            raise ValueError("Matrix is singular over GF(2) (det = 0)")
        if pivot != c:
            aug[[c, pivot]] = aug[[pivot, c]]

        # Eliminate all other rows
        for r in range(n):
            if r != c and aug[r, c] == 1:
                aug[r] ^= aug[c]

    inv = aug[:, n:].copy()
    # Sanity verification
    prod = matmul_gf2(M, inv)
    assert np.array_equal(prod, np.eye(n, dtype=np.uint8)), "Inversion verification failed!"
    return inv


# =====================================================================
# 3. Linear Algebra over GF(2^4) (4x4 Word Matrices)
# =====================================================================

def matmul_gf16(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication over GF(2^4)."""
    r_a, c_a = A.shape
    r_b, c_b = B.shape
    assert c_a == r_b, f"Shape mismatch: {A.shape} and {B.shape}"
    out = np.zeros((r_a, c_b), dtype=np.uint8)
    for i in range(r_a):
        for j in range(c_b):
            acc = 0
            for k in range(c_a):
                acc ^= gf16.multiply(int(A[i, k]), int(B[k, j]))
            out[i, j] = acc
    return out


def det_gf16(M: np.ndarray) -> int:
    """Computes determinant of a square matrix over GF(2^4) via Gaussian elimination."""
    n = M.shape[0]
    A = M.copy().astype(np.uint8)
    det = 1

    for c in range(n):
        pivot = -1
        for r in range(c, n):
            if A[r, c] != 0:
                pivot = r
                break
        if pivot == -1:
            return 0
        if pivot != c:
            A[[c, pivot]] = A[[pivot, c]]
            # In characteristic 2, row swap sign (-1) == 1, so det sign does not flip

        pivot_val = int(A[c, c])
        det = gf16.multiply(det, pivot_val)
        pivot_inv = gf16.inverse(pivot_val)

        # Normalize pivot row
        for j in range(c, n):
            A[c, j] = gf16.multiply(int(A[c, j]), pivot_inv)

        # Eliminate below
        for r in range(c + 1, n):
            factor = int(A[r, c])
            if factor != 0:
                for j in range(c, n):
                    A[r, j] ^= gf16.multiply(factor, int(A[c, j]))

    return det


def inverse_gf16(M: np.ndarray) -> np.ndarray:
    """Computes inverse of square matrix over GF(2^4) via Gauss-Jordan elimination."""
    n = M.shape[0]
    aug = np.zeros((n, 2 * n), dtype=np.uint8)
    aug[:, :n] = M.copy()
    aug[:, n:] = np.eye(n, dtype=np.uint8)

    for c in range(n):
        pivot = -1
        for r in range(c, n):
            if aug[r, c] != 0:
                pivot = r
                break
        if pivot == -1:
            raise ValueError("Matrix is singular over GF(2^4)")
        if pivot != c:
            aug[[c, pivot]] = aug[[pivot, c]]

        pivot_val = int(aug[c, c])
        pivot_inv = gf16.inverse(pivot_val)

        # Normalize row
        for j in range(2 * n):
            aug[c, j] = gf16.multiply(int(aug[c, j]), pivot_inv)

        # Eliminate other rows
        for r in range(n):
            if r != c and aug[r, c] != 0:
                factor = int(aug[r, c])
                for j in range(2 * n):
                    aug[r, j] ^= gf16.multiply(factor, int(aug[c, j]))

    inv = aug[:, n:].copy()
    assert np.array_equal(matmul_gf16(M, inv), np.eye(n, dtype=np.uint8)), "GF16 inversion error"
    return inv


# =====================================================================
# 4. Homomorphism: GF(2^4)^(4x4) -> GF(2)^(16x16)
# =====================================================================

def gf16_matrix_to_binary_matrix(mat_gf16: np.ndarray) -> np.ndarray:
    """
    Converts a 4x4 matrix over GF(2^4) into an equivalent 16x16 binary matrix over GF(2).
    State vector X in GF(2)^16 is partitioned into 4 nibbles X = (X0, X1, X2, X3).
    Nibble k is bits [4*k : 4*k + 4].
    Each entry M[r, c] in GF(2^4) expands to a 4x4 binary block L(M[r, c]).
    """
    assert mat_gf16.shape == (4, 4), "Expected 4x4 matrix over GF(2^4)"
    bin_mat = np.zeros((16, 16), dtype=np.uint8)

    for r in range(4):
        for c in range(4):
            coeff = int(mat_gf16[r, c])
            block_4x4 = gf16.to_binary_matrix(coeff)
            bin_mat[4 * r : 4 * r + 4, 4 * c : 4 * c + 4] = block_4x4

    return bin_mat


def is_involutory_gf2(M: np.ndarray) -> bool:
    """Checks if binary matrix M is self-inverse: M @ M == I over GF(2)."""
    n = M.shape[0]
    prod = matmul_gf2(M, M)
    return np.array_equal(prod, np.eye(n, dtype=np.uint8))


def is_orthogonal_gf2(M: np.ndarray) -> bool:
    """Checks if binary matrix M is orthogonal: M^T @ M == I over GF(2)."""
    n = M.shape[0]
    prod = matmul_gf2(M.T, M)
    return np.array_equal(prod, np.eye(n, dtype=np.uint8))

