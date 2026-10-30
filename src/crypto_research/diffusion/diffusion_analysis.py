"""
src/crypto_research/diffusion/diffusion_analysis.py

Diffusion Speed, Bit Dependency Matrix Propagation, and Multi-Round Avalanche Evaluation.
Academic Research Framework for MLMC Diffusion Architecture.

Implements:
  1. Exact boolean dependency matrix propagation across rounds.
  2. Complete Diffusion Round Index (R_diff): minimal rounds until all output bits depend on all input bits.
  3. Empirical avalanche effect progression across 1 to 10 rounds using PRESENT S-box.
  4. Comparison of diffusion speed across all candidate architectures.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, Any
import numpy as np

from .matrix_validation import matmul_gf2


# PRESENT 4-bit S-box
PRESENT_SBOX = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2], dtype=np.uint8)


def compute_diffusion_rounds_index(M: np.ndarray, max_rounds: int = 10) -> Dict[str, Any]:
    """
    Computes the exact bit-level dependency propagation across successive rounds.
    In a round R = D o S:
      - S is the 4-bit PRESENT S-box layer: each output bit of nibble k depends on all 4 input bits of nibble k.
        A_S is a block-diagonal matrix of 4x4 all-ones blocks.
      - D is the linear diffusion layer with matrix M: A_D = (M != 0).
      - Round dependency matrix: A_R = (A_D @ A_S) in Boolean semiring.
      - r-round dependency: A^(r) = (A_R)^r in Boolean semiring.
    R_diff is the minimal round r where A^(r) is strictly all-ones.
    """
    assert M.shape == (16, 16)
    # A_S: block diagonal 4x4 all-ones
    A_S = np.zeros((16, 16), dtype=bool)
    for k in range(4):
        A_S[4 * k : 4 * k + 4, 4 * k : 4 * k + 4] = True

    # A_D: binary adjacency of M
    A_D = (M % 2) != 0

    # 1-round dependency matrix
    A_round = np.dot(A_D.astype(int), A_S.astype(int)) > 0

    curr = np.eye(16, dtype=bool)
    round_dependencies: List[int] = []
    r_diff = None

    for r in range(1, max_rounds + 1):
        curr = (np.dot(curr.astype(int), A_round.astype(int)) > 0)
        num_dependencies = int(np.sum(curr))
        round_dependencies.append(num_dependencies)
        if num_dependencies == 256 and r_diff is None:
            r_diff = r

    return {
        "rounds_to_full_diffusion": r_diff if r_diff is not None else -1,
        "dependency_count_per_round": round_dependencies,
        "full_diffusion_achieved": (r_diff is not None)
    }


def evaluate_candidate_avalanche(
    M: np.ndarray,
    num_samples: int = 400,
    max_rounds: int = 8,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Empirically evaluates the Strict Avalanche Criterion (SAC) progression across rounds.
    Uses PRESENT 4-bit S-box and linear layer M.
    For each round r in 1..max_rounds, flips 1 bit in input and measures % of output bits that flip.
    """
    np.random.seed(seed)
    M_bin = (M % 2).astype(np.uint8)

    def apply_sbox_layer(state_16bit: int) -> int:
        out = 0
        for k in range(4):
            nibble = (state_16bit >> (4 * k)) & 0x0F
            sub = PRESENT_SBOX[nibble]
            out |= (int(sub) << (4 * k))
        return out

    def apply_linear_layer(state_16bit: int) -> int:
        # Convert 16-bit int to bit vector
        v = np.array([(state_16bit >> i) & 1 for i in range(16)], dtype=np.uint8)
        y = matmul_gf2(M_bin, v)
        out = 0
        for i in range(16):
            out |= (int(y[i]) << i)
        return out

    def encrypt_rounds(state_16bit: int, rounds: int) -> int:
        s = state_16bit
        for _ in range(rounds):
            s = apply_sbox_layer(s)
            s = apply_linear_layer(s)
        return s

    progression = []
    for r in range(1, max_rounds + 1):
        flip_percentages = []
        for _ in range(num_samples):
            pt = int(np.random.randint(0, 65536, dtype=np.uint32))
            ct = encrypt_rounds(pt, r)

            # Flip random bit
            bit_to_flip = int(np.random.randint(0, 16))
            pt_flipped = pt ^ (1 << bit_to_flip)
            ct_flipped = encrypt_rounds(pt_flipped, r)

            diff = ct ^ ct_flipped
            flipped_bits = bin(diff).count("1")
            flip_percentages.append((flipped_bits / 16.0) * 100.0)

        arr = np.array(flip_percentages)
        progression.append({
            "round": r,
            "mean_flip_pct": round(float(np.mean(arr)), 2),
            "std_flip_pct": round(float(np.std(arr)), 2),
            "min_flip_pct": round(float(np.min(arr)), 2),
            "max_flip_pct": round(float(np.max(arr)), 2)
        })

    return progression

