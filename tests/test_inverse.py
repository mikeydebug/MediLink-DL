"""
tests/test_inverse.py

Automated test suite verifying exact inverse correctness and symmetry properties.
Part of MLMC Diffusion Architecture Research Suite.
"""

import pytest
import numpy as np

from src.crypto_research.diffusion.matrix_generator import get_all_candidates
from src.crypto_research.diffusion.inverse_generator import generate_and_verify_inverse
from src.crypto_research.diffusion.matrix_validation import matmul_gf2, is_involutory_gf2


@pytest.mark.parametrize("candidate", get_all_candidates(), ids=lambda c: c.candidate_id)
def test_all_candidates_exact_inverse_correctness(candidate):
    """Verifies that M @ M^-1 == I and M^-1 @ M == I over GF(2) for all candidates."""
    res = generate_and_verify_inverse(candidate.matrix_16x16)
    assert res["is_invertible"] is True
    assert res["identity_verified"] is True
    M_inv = res["inverse_matrix"]
    I16 = np.eye(16, dtype=np.uint8)
    assert np.array_equal(matmul_gf2(candidate.matrix_16x16, M_inv), I16)
    assert np.array_equal(matmul_gf2(M_inv, candidate.matrix_16x16), I16)


def test_involutory_candidates_self_inverse():
    """Verifies that candidates declared involutory satisfy M == M^-1."""
    candidates = {c.candidate_id: c for c in get_all_candidates()}

    involutory_ids = [
        "BASE-02-PRESENT-16",
        "BASE-03-GIFT-16",
        "FAM-C1-INV-HAD-OPT",
        "FAM-C2-INV-HAD-MAXBIT",
        "PROPOSED-OIN-MDS"
    ]

    for cid in involutory_ids:
        c = candidates[cid]
        assert c.is_involutory is True, f"{cid} should be flagged as involutory"
        assert is_involutory_gf2(c.matrix_16x16) is True
        inv_res = generate_and_verify_inverse(c.matrix_16x16)
        assert np.array_equal(c.matrix_16x16, inv_res["inverse_matrix"])

