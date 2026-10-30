"""
src/crypto_research/diffusion/linear_analysis.py

Linear Cryptanalysis and Linear Trail Lower Bounds.
Academic Research Framework for MLMC Diffusion Architecture.

Implements:
  1. Linear Approximation Table (LAT) for 4-bit S-boxes (PRESENT).
  2. Maximum linear correlation bias (epsilon_max).
  3. Matsui's Piling-Up Lemma application over active S-boxes across 1 to 14 rounds.
  4. Provable wide-trail security margins against linear approximations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np

from .branch_number import compute_exact_branch_numbers


# PRESENT 4-bit S-Box
PRESENT_SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


def compute_sbox_lat(sbox: List[int] = PRESENT_SBOX) -> Tuple[np.ndarray, int, float]:
    """
    Computes 16x16 Linear Approximation Table (LAT) for a 4-bit S-box.
    LAT[alpha, beta] = #{ x in [0..15] | alpha . x == beta . S(x) } - 8.
    Returns:
      - lat: 16x16 numpy array
      - max_bias_count: maximum absolute bias count
      - epsilon_max: max_bias_count / 16.0
    """
    lat = np.zeros((16, 16), dtype=np.int32)
    for alpha in range(16):
        for beta in range(16):
            matches = 0
            for x in range(16):
                # Parity inner product
                in_parity = bin(alpha & x).count("1") % 2
                out_parity = bin(beta & sbox[x]).count("1") % 2
                if in_parity == out_parity:
                    matches += 1
            lat[alpha, beta] = matches - 8

    # Exclude trivial (0, 0)
    non_trivial_mask = np.ones((16, 16), dtype=bool)
    non_trivial_mask[0, 0] = False
    max_bias_count = int(np.max(np.abs(lat[non_trivial_mask])))
    epsilon_max = max_bias_count / 16.0
    return lat, max_bias_count, epsilon_max


def evaluate_linear_trails_bounds(
    M: np.ndarray,
    max_rounds: int = 14,
    sbox: List[int] = PRESENT_SBOX
) -> Dict[str, Any]:
    """
    Calculates provable upper bounds on linear trail correlation bias across 1 to max_rounds.
    For an SPN with linear layer M having linear nibble branch number B_lin = B_lin_nibble(M):
      - In any 2 consecutive rounds, active S-boxes in linear trail >= B_lin.
      - Over 2k rounds: active S-boxes >= k * B_lin.
      - Over 14 rounds: active S-boxes >= 7 * B_lin.
    Matsui's Piling-Up Lemma for A active S-boxes with maximum bias epsilon_max:
      |epsilon_trail| <= 2^(A - 1) * (epsilon_max)^A
      Data complexity required for linear attack: N_L ~ 1 / (epsilon_trail)^2
    """
    lat, max_bias_count, epsilon_max = compute_sbox_lat(sbox)
    b_metrics = compute_exact_branch_numbers(M)
    b_lin = b_metrics["linear_branch_number_nibble"]

    active_sboxes_by_round = []
    log2_bias_bound_by_round = []

    for r in range(1, max_rounds + 1):
        if r == 1:
            active_lb = 1
        elif r == 2:
            active_lb = b_lin
        elif r == 3:
            active_lb = b_lin + 1
        elif r % 2 == 0:
            active_lb = (r // 2) * b_lin
        else:
            active_lb = (r // 2) * b_lin + 1

        active_sboxes_by_round.append(active_lb)
        # |epsilon_trail| <= 2^(active_lb - 1) * (epsilon_max)^active_lb
        # Since epsilon_max = 2^-2: 2^(active_lb - 1) * 2^(-2 * active_lb) = 2^(-active_lb - 1)
        log2_bias = (active_lb - 1) + active_lb * np.log2(epsilon_max)
        log2_bias_bound_by_round.append(round(float(log2_bias), 2))

    return {
        "linear_branch_number_nibble": b_lin,
        "epsilon_max": epsilon_max,
        "max_lat_bias_count": max_bias_count,
        "active_sboxes_lower_bound_per_round": active_sboxes_by_round,
        "log2_max_correlation_bias_per_round": log2_bias_bound_by_round,
        "log2_bias_round_14": log2_bias_bound_by_round[13] if len(log2_bias_bound_by_round) >= 14 else -1,
        "is_immune_14_rounds": (log2_bias_bound_by_round[13] <= -8.0) if len(log2_bias_bound_by_round) >= 14 else False
    }

