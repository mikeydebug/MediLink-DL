"""
src/crypto_research/diffusion/branch_number.py

Exact Branch Number and Differential/Linear Metric Calculation Framework.
Academic Research Framework for MLMC Diffusion Architecture.

Distinguishes:
  1. Bit-level branch number: B_bit(M) = min_{x != 0} (wt_bit(x) + wt_bit(Mx))
  2. Nibble-level branch number: B_nibble(M) = min_{x != 0} (wt_nibble(x) + wt_nibble(Mx))
  3. Linear branch numbers: B_lin_bit(M) = B_bit(M^T), B_lin_nibble(M) = B_nibble(M^T)
  4. MDS condition testing (B_nibble == 5 over 4 nibbles).

Employs fast, vectorized exact enumeration across all 2^16 - 1 = 65,535 non-zero states.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Any

from .matrix_validation import matmul_gf2


# Precomputed bit-table for 16-bit states
def _generate_all_nonzero_vectors_16() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a 16 x 65535 binary matrix of all non-zero 16-bit vectors,
    together with their precomputed bit weights and nibble weights.
    """
    # 65535 non-zero integers from 1 to 65535
    ints = np.arange(1, 65536, dtype=np.uint32)
    # Shape (16, 65535): row i is bit i
    vectors = np.zeros((16, 65535), dtype=np.uint8)
    for i in range(16):
        vectors[i] = ((ints >> i) & 1).astype(np.uint8)

    bit_weights = vectors.sum(axis=0).astype(np.uint8)

    # Nibble weights: nibble k is bits [4*k : 4*k+4]
    n0 = ((vectors[0] | vectors[1] | vectors[2] | vectors[3]) > 0).astype(np.uint8)
    n1 = ((vectors[4] | vectors[5] | vectors[6] | vectors[7]) > 0).astype(np.uint8)
    n2 = ((vectors[8] | vectors[9] | vectors[10] | vectors[11]) > 0).astype(np.uint8)
    n3 = ((vectors[12] | vectors[13] | vectors[14] | vectors[15]) > 0).astype(np.uint8)
    nibble_weights = n0 + n1 + n2 + n3

    return vectors, bit_weights, nibble_weights


# Precompute globally once for ultra-fast evaluations
_ALL_VECTORS_16, _BIT_WEIGHTS_16, _NIBBLE_WEIGHTS_16 = _generate_all_nonzero_vectors_16()


def compute_exact_branch_numbers(M: np.ndarray) -> Dict[str, Any]:
    """
    Computes exact bit-level and nibble-level differential and linear branch numbers
    for a 16x16 binary matrix M over GF(2).

    Returns:
      dict with:
        - branch_number_bit: int (1 to 8)
        - branch_number_nibble: int (1 to 5)
        - linear_branch_number_bit: int (1 to 8)
        - linear_branch_number_nibble: int (1 to 5)
        - is_mds_nibble: bool (True if branch_number_nibble == 5)
        - min_diff_weight_inputs: list of worst-case vectors
    """
    assert M.shape == (16, 16), f"Expected 16x16 matrix, got {M.shape}"
    M_bin = (M % 2).astype(np.uint8)

    # 1. Forward Differential Branch Numbers: Y = (M @ X) % 2
    Y = (np.dot(M_bin.astype(np.int32), _ALL_VECTORS_16.astype(np.int32)) % 2).astype(np.uint8)

    # Bit weights of Y
    y_bit_weights = Y.sum(axis=0).astype(np.uint8)
    total_bit_weights = _BIT_WEIGHTS_16 + y_bit_weights
    b_bit = int(np.min(total_bit_weights))

    # Nibble weights of Y
    y_n0 = ((Y[0] | Y[1] | Y[2] | Y[3]) > 0).astype(np.uint8)
    y_n1 = ((Y[4] | Y[5] | Y[6] | Y[7]) > 0).astype(np.uint8)
    y_n2 = ((Y[8] | Y[9] | Y[10] | Y[11]) > 0).astype(np.uint8)
    y_n3 = ((Y[12] | Y[13] | Y[14] | Y[15]) > 0).astype(np.uint8)
    y_nibble_weights = y_n0 + y_n1 + y_n2 + y_n3

    total_nibble_weights = _NIBBLE_WEIGHTS_16 + y_nibble_weights
    b_nibble = int(np.min(total_nibble_weights))

    # 2. Linear Branch Numbers via Dual Matrix M^T: Y_lin = (M^T @ X) % 2
    M_T = M_bin.T.copy()
    Y_lin = (np.dot(M_T.astype(np.int32), _ALL_VECTORS_16.astype(np.int32)) % 2).astype(np.uint8)

    y_lin_bit_weights = Y_lin.sum(axis=0).astype(np.uint8)
    total_lin_bit_weights = _BIT_WEIGHTS_16 + y_lin_bit_weights
    b_lin_bit = int(np.min(total_lin_bit_weights))

    y_lin_n0 = ((Y_lin[0] | Y_lin[1] | Y_lin[2] | Y_lin[3]) > 0).astype(np.uint8)
    y_lin_n1 = ((Y_lin[4] | Y_lin[5] | Y_lin[6] | Y_lin[7]) > 0).astype(np.uint8)
    y_lin_n2 = ((Y_lin[8] | Y_lin[9] | Y_lin[10] | Y_lin[11]) > 0).astype(np.uint8)
    y_lin_n3 = ((Y_lin[12] | Y_lin[13] | Y_lin[14] | Y_lin[15]) > 0).astype(np.uint8)
    y_lin_nibble_weights = y_lin_n0 + y_lin_n1 + y_lin_n2 + y_lin_n3

    total_lin_nibble_weights = _NIBBLE_WEIGHTS_16 + y_lin_nibble_weights
    b_lin_nibble = int(np.min(total_lin_nibble_weights))

    return {
        "branch_number_bit": b_bit,
        "branch_number_nibble": b_nibble,
        "linear_branch_number_bit": b_lin_bit,
        "linear_branch_number_nibble": b_lin_nibble,
        "is_mds_nibble": (b_nibble == 5),
        "is_optimal_bit": (b_bit >= 8)
    }

