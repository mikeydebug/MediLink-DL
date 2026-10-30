"""
src/crypto_research/diffusion/differential_analysis.py

Differential Cryptanalysis and Provable Active S-Box Lower Bounds.
Academic Research Framework for MLMC Diffusion Architecture.

Implements:
  1. Difference Distribution Table (DDT) generation for 4-bit S-boxes (PRESENT).
  2. Maximum Differential Probability (DP_max) and differential uniformity computation.
  3. Exact and bounded active S-box lower bound tracking across 1 to 14 rounds.
  4. Provable wide-trail security margins against differential characteristics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np

from .branch_number import compute_exact_branch_numbers


# PRESENT 4-bit S-Box
PRESENT_SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


def compute_sbox_ddt(sbox: List[int] = PRESENT_SBOX) -> Tuple[np.ndarray, int, float]:
    """
    Computes 16x16 Difference Distribution Table (DDT) for a 4-bit S-box.
    Returns:
      - ddt: 16x16 numpy array
      - max_diff_count: maximum non-trivial entry
      - dp_max: max_diff_count / 16.0
    """
    ddt = np.zeros((16, 16), dtype=np.int32)
    for dx in range(16):
        for x in range(16):
            dy = sbox[x] ^ sbox[x ^ dx]
            ddt[dx, dy] += 1

    max_diff_count = int(np.max(ddt[1:, :]))
    dp_max = max_diff_count / 16.0
    return ddt, max_diff_count, dp_max


def evaluate_active_sboxes_bounds(
    M: np.ndarray,
    max_rounds: int = 14,
    sbox: List[int] = PRESENT_SBOX
) -> Dict[str, Any]:
    """
    Calculates provable lower bounds on the number of active S-boxes across 1 to max_rounds.
    For an SPN with linear layer M having nibble branch number B = B_nibble(M):
      - In any 2 consecutive rounds (r, r+1), w_r + w_{r+1} >= B.
      - Over 4 rounds: total active S-boxes >= 2 * B.
      - Over 2k rounds: total active S-boxes >= k * B.
      - Over 14 rounds (7 pairs): total active S-boxes >= 7 * B.
    Computes upper bound on differential characteristic probability:
      P_diff(r) <= (DP_max)^(active_sboxes(r))
    """
    ddt, max_diff, dp_max = compute_sbox_ddt(sbox)
    b_metrics = compute_exact_branch_numbers(M)
    b_nibble = b_metrics["branch_number_nibble"]

    active_sboxes_by_round = []
    prob_bound_by_round = []

    for r in range(1, max_rounds + 1):
        if r == 1:
            active_lb = 1
        elif r == 2:
            active_lb = b_nibble
        elif r == 3:
            active_lb = b_nibble + 1
        elif r % 2 == 0:
            active_lb = (r // 2) * b_nibble
        else:
            active_lb = (r // 2) * b_nibble + 1

        active_sboxes_by_round.append(active_lb)
        # log2 probability bound: active_lb * log2(dp_max)
        log2_prob = active_lb * np.log2(dp_max)
        prob_bound_by_round.append(round(float(log2_prob), 2))

    return {
        "branch_number_nibble": b_nibble,
        "dp_max": dp_max,
        "max_differential_uniformity": max_diff,
        "active_sboxes_lower_bound_per_round": active_sboxes_by_round,
        "log2_max_characteristic_prob_per_round": prob_bound_by_round,
        "active_sboxes_round_4": active_sboxes_by_round[3] if len(active_sboxes_by_round) >= 4 else -1,
        "active_sboxes_round_14": active_sboxes_by_round[13] if len(active_sboxes_by_round) >= 14 else -1,
        "log2_prob_round_14": prob_bound_by_round[13] if len(prob_bound_by_round) >= 14 else -1,
        "is_immune_14_rounds": (prob_bound_by_round[13] <= -16.0) if len(prob_bound_by_round) >= 14 else False
    }

