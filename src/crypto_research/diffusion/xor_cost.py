"""
src/crypto_research/diffusion/xor_cost.py

Implementation Complexity, XOR Optimization, and Hardware Area Estimation.
Academic Research Framework for MLMC Diffusion Architecture.

Implements:
  1. Direct (naive) XOR count: sum_i max(0, wt(row_i) - 1).
  2. Sequential XOR (s-XOR) optimization via Paar's Common Subexpression Elimination (CSE).
  3. ASIC Hardware Gate Equivalent (GE) estimation based on standard CMOS 65nm cell metrics.
  4. Software operation count (CPU instructions, SIMD register usage).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Set, Tuple, Any
import numpy as np


def compute_direct_xor_cost(M: np.ndarray) -> int:
    """
    Computes the naive (direct) XOR operation count required to compute y = M x over GF(2).
    For each row, if weight is k > 0, it requires k - 1 XOR gates.
    """
    M_bin = (M % 2).astype(np.uint8)
    row_weights = M_bin.sum(axis=1)
    return int(sum(max(0, int(w) - 1) for w in row_weights))


def compute_paar_optimized_xor_cost(M: np.ndarray) -> Dict[str, Any]:
    """
    Computes optimized XOR count using Paar's Common Subexpression Elimination (CSE) algorithm.
    Reference: C. Paar, 'Optimized Addition-Chain Methods for Generating Low-Complexity Linear Transformations',
    IEEE Transactions on Computers, 1997.

    Algorithm:
      1. Represent each row as a set of active input variable indices {x_j}.
      2. Count frequency of all pairs {u, v} across all expressions.
      3. Identify the most frequent pair with frequency >= 2.
      4. Introduce a new intermediate variable t_k = u ^ v, consuming 1 XOR.
      5. Replace {u, v} with t_k in all expressions containing both.
      6. Repeat until no pair appears >= 2 times.
      7. Count remaining XORs.
    """
    M_bin = (M % 2).astype(np.uint8)
    rows, cols = M_bin.shape

    # Each target output is a set of active input variables
    expressions: List[Set[int]] = []
    for r in range(rows):
        active_vars = {c for c in range(cols) if M_bin[r, c] == 1}
        expressions.append(active_vars)

    intermediate_count = 0
    intermediate_nodes: List[Tuple[int, int, int]] = []
    next_var_id = cols

    while True:
        # Count frequency of all pairs {u, v} across expressions
        pair_counts: Dict[Tuple[int, int], int] = {}
        for expr in expressions:
            if len(expr) < 2:
                continue
            expr_list = sorted(list(expr))
            for i in range(len(expr_list)):
                for j in range(i + 1, len(expr_list)):
                    pair = (expr_list[i], expr_list[j])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # Find best pair
        best_pair = max(pair_counts.keys(), key=lambda p: pair_counts[p])
        max_freq = pair_counts[best_pair]

        if max_freq < 2:
            # No pair occurs more than once; no further sharing possible
            break

        # Create intermediate variable
        u, v = best_pair
        new_var = next_var_id
        next_var_id += 1
        intermediate_count += 1
        intermediate_nodes.append((new_var, u, v))

        # Substitute {u, v} -> {new_var} in expressions
        for expr in expressions:
            if u in expr and v in expr:
                expr.remove(u)
                expr.remove(v)
                expr.add(new_var)

    # Remaining XORs needed to combine surviving terms in each expression
    remaining_xors = sum(max(0, len(expr) - 1) for expr in expressions)
    total_optimized_xors = intermediate_count + remaining_xors

    return {
        "direct_xor_count": compute_direct_xor_cost(M),
        "optimized_xor_count": total_optimized_xors,
        "intermediate_subexpressions": intermediate_count,
        "xor_savings_percentage": round((1.0 - total_optimized_xors / max(1, compute_direct_xor_cost(M))) * 100.0, 2)
    }


def estimate_hardware_ge(
    xor_count: int,
    is_involutory: bool,
    inv_xor_count: Optional[int] = None
) -> Dict[str, float]:
    """
    Estimates ASIC Gate Equivalents (GE) for encryption and combined encryption+decryption circuits.
    Based on standard TSMC / NanGate CMOS 65nm / 45nm standard cell benchmarks:
      - 2-input XOR (XOR2_X1): 2.0 GE
      - 2-to-1 Multiplexer (MUX2_X1): 2.25 GE (needed for non-involutory dual-path)
      - Wire routing (permutations): 0.0 GE

    Involutory Advantage:
      If M is self-inverse (M == M^-1), the decryption path reuses the exact same XOR network!
      Additional decryption logic area is 0 GE.
    """
    ge_per_xor = 2.0
    ge_enc = xor_count * ge_per_xor

    if is_involutory:
        ge_dec = 0.0  # Fully reused
        ge_total_enc_dec = ge_enc  # Zero extra area for inverse
    else:
        inv_xors = inv_xor_count if inv_xor_count is not None else xor_count
        ge_dec = inv_xors * ge_per_xor
        # Need multiplexer to switch between forward and inverse in combined core: 16 MUXes
        ge_mux = 16 * 2.25
        ge_total_enc_dec = ge_enc + ge_dec + ge_mux

    return {
        "ge_encryption_only": round(ge_enc, 2),
        "ge_decryption_overhead": round(ge_dec, 2),
        "ge_combined_enc_dec": round(ge_total_enc_dec, 2)
    }

