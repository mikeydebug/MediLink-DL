"""
tests/test_non_hadamard.py

Unit and property verification suite for Non-Hadamard Involutory Diffusion Architectures.
Phase 2 Research: Validates ABC-MDS family, TBOI-16, and theoretical impossibility proofs.
"""

import pytest
import numpy as np

from src.crypto_research.diffusion.non_hadamard import (
    get_candidate_abc_mds_opt,
    get_candidate_abc_mds_maxbit,
    get_candidate_tboi_16,
    verify_anti_circulant_mds_impossibility,
    verify_galois_binary_symmetry_obstruction
)
from src.crypto_research.diffusion.matrix_validation import (
    matmul_gf16,
    matmul_gf2,
    is_involutory_gf2,
    is_orthogonal_gf2
)


def test_abc_mds_opt_mathematical_properties():
    """Verify ABC-MDS-OPT candidate achieves B_nibble=5, involutory, low XOR, and is strictly non-Hadamard."""
    cand = get_candidate_abc_mds_opt()
    mat_gf16 = cand["matrix_gf16"]
    mat_bin = cand["matrix_16x16"]

    # Involutory checks
    assert cand["is_involutory"] is True
    assert np.array_equal(matmul_gf16(mat_gf16, mat_gf16), np.eye(4, dtype=np.uint8))
    assert is_involutory_gf2(mat_bin) is True

    # Non-Hadamard and Asymmetric verification
    assert not np.array_equal(mat_gf16, mat_gf16.T), "ABC-MDS must be asymmetric"
    row0_set = set(mat_gf16[0, :])
    row2_set = set(mat_gf16[2, :])
    assert row0_set != row2_set, "Rows must not be permutations of identical elements (non-Hadamard)"

    # Branch numbers
    assert cand["branch_number_nibble"] == 5, "Must achieve optimal MDS bound B_nibble=5"
    assert cand["branch_number_bit"] == 5
    assert cand["linear_branch_nibble"] == 5

    # XOR complexities
    assert cand["direct_xor_count"] == 64, "Direct XOR must be exactly 64 (lower than Had(1,4,9,13)=72)"
    assert cand["optimized_xor_count"] == 49, "Paar CSE XOR must be 49"
    assert cand["hardware_ge"]["ge_combined_enc_dec"] == 98.0


def test_abc_mds_maxbit_properties():
    """Verify ABC-MDS-MAXBIT candidate achieves B_bit=7, B_nibble=5, and involutory symmetry."""
    cand = get_candidate_abc_mds_maxbit()
    mat_gf16 = cand["matrix_gf16"]
    mat_bin = cand["matrix_16x16"]

    # Involutory checks
    assert cand["is_involutory"] is True
    assert np.array_equal(matmul_gf16(mat_gf16, mat_gf16), np.eye(4, dtype=np.uint8))
    assert is_involutory_gf2(mat_bin) is True

    # Non-Hadamard verification
    assert not np.array_equal(mat_gf16, mat_gf16.T)
    assert set(mat_gf16[0, :]) != set(mat_gf16[2, :])

    # Branch numbers
    assert cand["branch_number_nibble"] == 5
    assert cand["branch_number_bit"] == 7
    assert cand["linear_branch_nibble"] == 5
    assert cand["linear_branch_bit"] == 7

    # Complexity
    assert cand["direct_xor_count"] == 94
    assert cand["optimized_xor_count"] == 58


def test_tboi_16_true_binary_symmetry():
    """Verify TBOI-16 achieves genuine binary symmetry, orthogonality, and involution over GF(2)."""
    cand = get_candidate_tboi_16()
    mat_bin = cand["matrix_16x16"]

    # Genuine binary symmetry and orthogonality
    assert cand["is_symmetric_gf2"] is True
    assert np.array_equal(mat_bin, mat_bin.T), "M must be strictly symmetric over GF(2)"
    assert is_involutory_gf2(mat_bin) is True, "M must be strictly involutory over GF(2)"
    assert is_orthogonal_gf2(mat_bin) is True, "M must be strictly orthogonal over GF(2)"

    # Branch numbers
    assert cand["branch_number_nibble"] == 4
    assert cand["branch_number_bit"] == 6

    # High Paar CSE compression
    assert cand["direct_xor_count"] == 128
    assert cand["optimized_xor_count"] == 60, "Paar CSE achieves 60 XORs (53.1% reduction from 128)"
    assert cand["hardware_ge"]["ge_combined_enc_dec"] == 120.0


def test_impossibility_theorems():
    """Verify formal impossibility theorems independently."""
    assert verify_anti_circulant_mds_impossibility() is True, "Involutory Anti-Circulant MDS must not exist"
    assert verify_galois_binary_symmetry_obstruction() is True, "Only {0, 1} can have symmetric binary blocks"
