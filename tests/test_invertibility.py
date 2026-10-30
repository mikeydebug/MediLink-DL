"""
tests/test_invertibility.py

Automated test suite verifying non-singularity and invertibility of all diffusion candidates.
Part of MLMC Diffusion Architecture Research Suite.
"""

import pytest
import numpy as np

from src.crypto_research.diffusion.matrix_generator import get_all_candidates
from src.crypto_research.diffusion.matrix_validation import is_invertible_gf2, rank_gf2


@pytest.mark.parametrize("candidate", get_all_candidates(), ids=lambda c: c.candidate_id)
def test_all_candidates_invertibility_over_gf2(candidate):
    """Verifies that every candidate diffusion matrix is strictly non-singular over GF(2)."""
    M = candidate.matrix_16x16
    assert M.shape == (16, 16), f"{candidate.candidate_id}: Expected 16x16 matrix"
    assert rank_gf2(M) == 16, f"{candidate.candidate_id}: Matrix rank is not full (rank={rank_gf2(M)})"
    assert is_invertible_gf2(M) is True, f"{candidate.candidate_id}: is_invertible_gf2 returned False"


def test_singular_matrix_rejection():
    """Verifies that non-invertible test matrices are correctly identified."""
    singular_mat = np.zeros((16, 16), dtype=np.uint8)
    assert is_invertible_gf2(singular_mat) is False

    rank15_mat = np.eye(16, dtype=np.uint8)
    rank15_mat[15, 15] = 0
    assert is_invertible_gf2(rank15_mat) is False

