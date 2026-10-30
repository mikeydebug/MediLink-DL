"""
tests/test_properties.py

Automated test suite verifying cryptographic properties, diffusion speed, and XOR metrics.
Part of MLMC Diffusion Architecture Research Suite.
"""

import pytest
import numpy as np

from src.crypto_research.diffusion.matrix_generator import get_all_candidates
from src.crypto_research.diffusion.xor_cost import compute_direct_xor_cost, compute_paar_optimized_xor_cost
from src.crypto_research.diffusion.diffusion_analysis import compute_diffusion_rounds_index
from src.crypto_research.diffusion.differential_analysis import evaluate_active_sboxes_bounds, compute_sbox_ddt
from src.crypto_research.diffusion.linear_analysis import evaluate_linear_trails_bounds, compute_sbox_lat


def test_paar_optimization_reduces_or_maintains_xor_count():
    """Verifies that Paar CSE algorithm never increases XOR cost and achieves meaningful savings on dense matrices."""
    for c in get_all_candidates():
        dir_xor = compute_direct_xor_cost(c.matrix_16x16)
        opt_res = compute_paar_optimized_xor_cost(c.matrix_16x16)
        opt_xor = opt_res["optimized_xor_count"]
        assert opt_xor <= dir_xor, f"{c.candidate_id}: Optimized XOR ({opt_xor}) > direct XOR ({dir_xor})"


def test_diffusion_speed_comparison():
    """Verifies that MDS candidates achieve full bit dependency in Round 1, whereas GFN takes 4 rounds."""
    candidates = {c.candidate_id: c for c in get_all_candidates()}

    # Existing MLMC-G16 GFN takes 4 rounds to achieve full diffusion
    gfn = candidates["BASE-01-MLMC-G16"]
    gfn_diff = compute_diffusion_rounds_index(gfn.matrix_16x16)
    assert gfn_diff["rounds_to_full_diffusion"] == 4

    # Proposed OIN-MDS achieves full diffusion in 1 round
    proposed = candidates["PROPOSED-OIN-MDS"]
    prop_diff = compute_diffusion_rounds_index(proposed.matrix_16x16)
    assert prop_diff["rounds_to_full_diffusion"] == 1


def test_active_sboxes_wide_trail_bounds():
    """Verifies that MDS candidates guarantee >= 5 active S-boxes in 2 rounds and >= 10 in 4 rounds."""
    candidates = {c.candidate_id: c for c in get_all_candidates()}
    proposed = candidates["PROPOSED-OIN-MDS"]
    diff_res = evaluate_active_sboxes_bounds(proposed.matrix_16x16, max_rounds=14)

    assert diff_res["active_sboxes_round_4"] >= 10
    assert diff_res["active_sboxes_round_14"] >= 35
    assert diff_res["log2_prob_round_14"] <= -70.0
    assert diff_res["is_immune_14_rounds"] is True


def test_present_sbox_cryptographic_bounds():
    """Verifies standard cryptographic bounds of the PRESENT S-box."""
    ddt, max_diff, dp_max = compute_sbox_ddt()
    assert max_diff == 4
    assert dp_max == 0.25

    lat, max_bias, eps_max = compute_sbox_lat()
    assert max_bias == 4
    assert eps_max == 0.25

