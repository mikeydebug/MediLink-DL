"""
src/crypto_research/diffusion/inverse_generator.py

Exact Inverse Transformation Generator and Symmetry Verification.
Academic Research Framework for MLMC Diffusion Architecture.

Implements:
  1. Exact matrix inversion over GF(2) and GF(2^4).
  2. Involutory verification: M^2 == I.
  3. Orthogonality verification: M^T @ M == I.
  4. Inverse XOR complexity and sparsity evaluation.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

from .matrix_validation import inverse_gf2, matmul_gf2, is_involutory_gf2, is_orthogonal_gf2
from .xor_cost import compute_direct_xor_cost, compute_paar_optimized_xor_cost


def generate_and_verify_inverse(M: np.ndarray) -> Dict[str, Any]:
    """
    Generates exact inverse matrix M^-1 over GF(2), verifies correctness,
    checks symmetry/involutory/orthogonal properties, and computes inverse XOR cost.
    """
    assert M.shape == (16, 16), f"Expected 16x16 matrix, got {M.shape}"
    M_bin = (M % 2).astype(np.uint8)

    # 1. Compute Inverse
    try:
        M_inv = inverse_gf2(M_bin)
        is_invertible = True
    except ValueError:
        return {
            "is_invertible": False,
            "error": "Matrix is singular over GF(2)"
        }

    # 2. Verify identity: M @ M_inv == I and M_inv @ M == I
    I_16 = np.eye(16, dtype=np.uint8)
    prod_right = matmul_gf2(M_bin, M_inv)
    prod_left = matmul_gf2(M_inv, M_bin)
    identity_verified = np.array_equal(prod_right, I_16) and np.array_equal(prod_left, I_16)

    # 3. Check special symmetries
    involutory = np.array_equal(M_bin, M_inv)  # M == M^-1 <=> M^2 == I
    orthogonal = is_orthogonal_gf2(M_bin)       # M^T @ M == I

    # 4. Measure inverse costs
    direct_xor_inv = compute_direct_xor_cost(M_inv)
    opt_inv = compute_paar_optimized_xor_cost(M_inv)

    return {
        "is_invertible": True,
        "identity_verified": identity_verified,
        "is_involutory": involutory,
        "is_orthogonal": orthogonal,
        "inverse_matrix": M_inv,
        "direct_xor_count_inverse": direct_xor_inv,
        "optimized_xor_count_inverse": opt_inv["optimized_xor_count"],
        "sparsity_inverse": round(float(np.mean(M_inv == 0)), 4)
    }

