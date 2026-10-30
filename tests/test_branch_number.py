"""
tests/test_branch_number.py

Automated test suite verifying exact bit-level and nibble-level branch numbers.
Part of MLMC Diffusion Architecture Research Suite.
"""

import pytest
import numpy as np

from src.crypto_research.diffusion.matrix_generator import get_all_candidates
from src.crypto_research.diffusion.branch_number import compute_exact_branch_numbers


@pytest.mark.parametrize("candidate", get_all_candidates(), ids=lambda c: c.candidate_id)
def test_branch_number_consistency(candidate):
    """Verifies that branch numbers match precomputed values and satisfy mathematical bounds."""
    metrics = compute_exact_branch_numbers(candidate.matrix_16x16)

    assert metrics["branch_number_nibble"] == candidate.branch_number_nibble
    assert metrics["branch_number_bit"] == candidate.branch_number_bit

    # Mathematical bounds:
    # Nibble branch number for 4 nibbles must be in [2, 5]
    assert 2 <= metrics["branch_number_nibble"] <= 5
    # Bit branch number for 16-bit block must be in [2, 8]
    assert 2 <= metrics["branch_number_bit"] <= 8


def test_mds_candidates_achieve_optimal_nibble_branch():
    """Verifies that all MDS candidate families strictly achieve B_nibble = 5."""
    candidates = {c.candidate_id: c for c in get_all_candidates()}
    mds_ids = [
        "FAM-B1-CAUCHY-MDS",
        "FAM-C1-INV-HAD-OPT",
        "FAM-C2-INV-HAD-MAXBIT",
        "FAM-D1-CIRCULANT-MDS",
        "FAM-E1-RECURSIVE-MDS",
        "PROPOSED-OIN-MDS"
    ]

    for cid in mds_ids:
        c = candidates[cid]
        assert c.branch_number_nibble == 5, f"{cid} failed to achieve B_nibble = 5"


def test_proposed_oin_mds_achieves_maximal_bit_branch():
    """Verifies that PROPOSED-OIN-MDS achieves the theoretical maximum bit branch number B_bit = 8."""
    candidates = {c.candidate_id: c for c in get_all_candidates()}
    proposed = candidates["PROPOSED-OIN-MDS"]
    assert proposed.branch_number_bit == 8
    assert proposed.linear_branch_number_bit == 8
    assert proposed.branch_number_nibble == 5

