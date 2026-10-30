"""
tests/test_phase2_1_audit.py

Independent Mathematical and Cryptographic Audit Suite for Phase 2.1.
Verifies all claims, counterexamples, minors, branch numbers, and theorems independently.
"""

import itertools
import pytest
import numpy as np

# ---------------------------------------------------------------------
# Independent First-Principles GF(16) Arithmetic mod x^4 + x + 1 (0x13)
# ---------------------------------------------------------------------
def gf16_mul_audit(a: int, b: int) -> int:
    res = 0
    for _ in range(4):
        if b & 1:
            res ^= a
        b >>= 1
        carry = a & 8
        a = (a << 1) & 0xF
        if carry:
            a ^= 3
    return res

def det2_audit(m00, m01, m10, m11):
    return gf16_mul_audit(m00, m11) ^ gf16_mul_audit(m01, m10)

def det3_audit(M):
    d0 = det2_audit(M[1, 1], M[1, 2], M[2, 1], M[2, 2])
    d1 = det2_audit(M[1, 0], M[1, 2], M[2, 0], M[2, 2])
    d2 = det2_audit(M[1, 0], M[1, 1], M[2, 0], M[2, 1])
    return gf16_mul_audit(int(M[0, 0]), d0) ^ gf16_mul_audit(int(M[0, 1]), d1) ^ gf16_mul_audit(int(M[0, 2]), d2)

def det4_audit(M):
    res = 0
    for c in range(4):
        sub = M[1:, [j for j in range(4) if j != c]]
        d3 = det3_audit(sub)
        res ^= gf16_mul_audit(int(M[0, c]), d3)
    return res

def to_bin_matrix_audit(M_gf):
    M_bin = np.zeros((16, 16), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            val = int(M_gf[r, c])
            block = np.zeros((4, 4), dtype=np.uint8)
            for j in range(4):
                col_val = gf16_mul_audit(val, 1 << j)
                for i in range(4):
                    block[i, j] = (col_val >> i) & 1
            M_bin[4*r:4*r+4, 4*c:4*c+4] = block
    return M_bin


# =====================================================================
# 1. ABC-MDS All 69 Minors & Involution
# =====================================================================

def test_abc_mds_opt_all_69_minors_and_involution():
    """Verify ABC-MDS-OPT is strictly involutory and all 69 minors are non-zero."""
    # Parameters: a=(1, 2), b=(1, 4), gamma=2 => c=(2, 8)
    M = np.array([
        [1, 2, 1, 4],
        [2, 1, 4, 1],
        [2, 8, 1, 2],
        [8, 2, 2, 1]
    ], dtype=np.uint8)

    # 1. Involutory check
    M_sq = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            s = 0
            for k in range(4):
                s ^= gf16_mul_audit(int(M[i, k]), int(M[k, j]))
            M_sq[i, j] = s
    assert np.array_equal(M_sq, np.eye(4, dtype=np.uint8)), "M^2 must be I_4"

    # 2. Check all 16 1x1 minors
    for r in range(4):
        for c in range(4):
            assert M[r, c] != 0, f"1x1 minor ({r},{c}) is zero"

    # 3. Check all 36 2x2 minors
    count_2x2 = 0
    for r in itertools.combinations(range(4), 2):
        for c in itertools.combinations(range(4), 2):
            sub = M[np.ix_(r, c)]
            d2 = det2_audit(sub[0, 0], sub[0, 1], sub[1, 0], sub[1, 1])
            assert d2 != 0, f"2x2 minor rows={r} cols={c} is zero"
            count_2x2 += 1
    assert count_2x2 == 36

    # 4. Check all 16 3x3 minors
    count_3x3 = 0
    for r in itertools.combinations(range(4), 3):
        for c in itertools.combinations(range(4), 3):
            sub = M[np.ix_(r, c)]
            d3 = det3_audit(sub)
            assert d3 != 0, f"3x3 minor rows={r} cols={c} is zero"
            count_3x3 += 1
    assert count_3x3 == 16

    # 5. Check 4x4 minor
    assert det4_audit(M) != 0, "4x4 determinant is zero"


def test_abc_mds_maxbit_all_69_minors_and_involution():
    """Verify ABC-MDS-MAXBIT is strictly involutory and all 69 minors are non-zero."""
    # Parameters: a=(1, 7), b=(1, 9), gamma=9 => c=(9, 13)
    M = np.array([
        [1, 7, 1, 9],
        [7, 1, 9, 1],
        [9, 13, 1, 7],
        [13, 9, 7, 1]
    ], dtype=np.uint8)

    # Involutory
    M_sq = np.zeros((4, 4), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            s = 0
            for k in range(4):
                s ^= gf16_mul_audit(int(M[i, k]), int(M[k, j]))
            M_sq[i, j] = s
    assert np.array_equal(M_sq, np.eye(4, dtype=np.uint8))

    # All 69 minors
    for r in range(4):
        for c in range(4):
            assert M[r, c] != 0
    for r in itertools.combinations(range(4), 2):
        for c in itertools.combinations(range(4), 2):
            sub = M[np.ix_(r, c)]
            assert det2_audit(sub[0, 0], sub[0, 1], sub[1, 0], sub[1, 1]) != 0
    for r in itertools.combinations(range(4), 3):
        for c in itertools.combinations(range(4), 3):
            sub = M[np.ix_(r, c)]
            assert det3_audit(sub) != 0
    assert det4_audit(M) != 0


# =====================================================================
# 2. Exhaustive Branch Number Tests
# =====================================================================

def test_exhaustive_branch_numbers_audit():
    """Exhaustively verify bit and nibble branch numbers across all 65,535 non-zero inputs."""
    all_x = np.arange(1, 65536, dtype=np.uint16)
    x_bits = np.zeros((16, 65535), dtype=np.uint8)
    for bit in range(16):
        x_bits[bit, :] = (all_x >> bit) & 1
    wt_in_bit = np.sum(x_bits, axis=0)

    wt_in_nib = ((x_bits[0:4, :].sum(axis=0) > 0).astype(int) +
                 (x_bits[4:8, :].sum(axis=0) > 0).astype(int) +
                 (x_bits[8:12, :].sum(axis=0) > 0).astype(int) +
                 (x_bits[12:16, :].sum(axis=0) > 0).astype(int))

    # Test ABC-MDS-OPT
    M_opt = np.array([[1, 2, 1, 4], [2, 1, 4, 1], [2, 8, 1, 2], [8, 2, 2, 1]], dtype=np.uint8)
    M_bin_opt = to_bin_matrix_audit(M_opt)
    Z_opt = (np.dot(M_bin_opt.astype(int), x_bits.astype(int)) % 2).astype(np.uint8)
    wt_out_bit_opt = np.sum(Z_opt, axis=0)
    wt_out_nib_opt = ((Z_opt[0:4, :].sum(axis=0) > 0).astype(int) +
                      (Z_opt[4:8, :].sum(axis=0) > 0).astype(int) +
                      (Z_opt[8:12, :].sum(axis=0) > 0).astype(int) +
                      (Z_opt[12:16, :].sum(axis=0) > 0).astype(int))
    assert np.min(wt_in_nib + wt_out_nib_opt) == 5, "ABC-MDS-OPT must have B_nibble = 5"
    assert np.min(wt_in_bit + wt_out_bit_opt) == 5, "ABC-MDS-OPT must have B_bit = 5"

    # Test ABC-MDS-MAXBIT
    M_maxbit = np.array([[1, 7, 1, 9], [7, 1, 9, 1], [9, 13, 1, 7], [13, 9, 7, 1]], dtype=np.uint8)
    M_bin_maxbit = to_bin_matrix_audit(M_maxbit)
    Z_maxbit = (np.dot(M_bin_maxbit.astype(int), x_bits.astype(int)) % 2).astype(np.uint8)
    wt_out_bit_max = np.sum(Z_maxbit, axis=0)
    wt_out_nib_max = ((Z_maxbit[0:4, :].sum(axis=0) > 0).astype(int) +
                      (Z_maxbit[4:8, :].sum(axis=0) > 0).astype(int) +
                      (Z_maxbit[8:12, :].sum(axis=0) > 0).astype(int) +
                      (Z_maxbit[12:16, :].sum(axis=0) > 0).astype(int))
    assert np.min(wt_in_nib + wt_out_nib_max) == 5, "ABC-MDS-MAXBIT must have B_nibble = 5"
    assert np.min(wt_in_bit + wt_out_bit_max) == 7, "ABC-MDS-MAXBIT must have B_bit = 7"


# =====================================================================
# 3. Theorem 3 Counterexample & Corrected Lemma Verification
# =====================================================================

def test_theorem3_flawed_step_counterexample():
    """
    Demonstrates that the claim in beyond_hadamard.md:
      'the constant term of x^(2k) mod P(x) is 0 for any primitive polynomial of degree m >= 2'
    is FALSE for k=2.
    """
    # For k=2, basis element is x^2 (represented by integer 4)
    # (x^2)^2 = x^4. Modulo P(x) = x^4 + x + 1, x^4 == x + 1.
    # The polynomial x + 1 has constant term 1!
    x2_sq = gf16_mul_audit(4, 4) # (x^2)^2
    constant_term = x2_sq & 1
    # The original report claimed this constant term is 0. We prove it is 1!
    assert constant_term == 1, "Counterexample: (x^2)^2 mod (x^4+x+1) has constant term 1, NOT 0!"


def test_theorem3_corrected_lemma():
    """
    Verifies the corrected lemma:
      For all (a, b) in GF(16)^2, L_a = L_b^T if and only if a = b in {0, 1}.
    """
    transpose_pairs = []
    for a in range(16):
        # Build L_a
        La = np.zeros((4, 4), dtype=np.uint8)
        for j in range(4):
            val = gf16_mul_audit(a, 1 << j)
            for i in range(4):
                La[i, j] = (val >> i) & 1
        for b in range(16):
            Lb = np.zeros((4, 4), dtype=np.uint8)
            for j in range(4):
                val = gf16_mul_audit(b, 1 << j)
                for i in range(4):
                    Lb[i, j] = (val >> i) & 1
            if np.array_equal(La, Lb.T):
                transpose_pairs.append((a, b))

    # The ONLY solutions are (0, 0) and (1, 1)
    assert transpose_pairs == [(0, 0), (1, 1)]


# =====================================================================
# 4. TBOI-16 Near-MDS Qualification
# =====================================================================

def test_tboi_16_near_mds_qualification():
    """
    Verifies that TBOI-16 achieves B_nibble = 4 (Near-MDS, NOT MDS),
    and is strictly symmetric, involutory, and orthogonal over GF(2).
    """
    K4 = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ], dtype=np.uint8)
    M = np.kron(K4, K4).astype(np.uint8)

    # Symmetry, involution, orthogonality
    assert np.array_equal(M, M.T)
    M_sq = (np.dot(M.astype(int), M.astype(int)) % 2).astype(np.uint8)
    assert np.array_equal(M_sq, np.eye(16, dtype=np.uint8))
    M_T_M = (np.dot(M.T.astype(int), M.astype(int)) % 2).astype(np.uint8)
    assert np.array_equal(M_T_M, np.eye(16, dtype=np.uint8))

    # Branch numbers
    all_x = np.arange(1, 65536, dtype=np.uint16)
    x_bits = np.zeros((16, 65535), dtype=np.uint8)
    for bit in range(16):
        x_bits[bit, :] = (all_x >> bit) & 1
    wt_in_bit = np.sum(x_bits, axis=0)
    wt_in_nib = ((x_bits[0:4, :].sum(axis=0) > 0).astype(int) +
                 (x_bits[4:8, :].sum(axis=0) > 0).astype(int) +
                 (x_bits[8:12, :].sum(axis=0) > 0).astype(int) +
                 (x_bits[12:16, :].sum(axis=0) > 0).astype(int))

    Z = (np.dot(M.astype(int), x_bits.astype(int)) % 2).astype(np.uint8)
    wt_out_bit = np.sum(Z, axis=0)
    wt_out_nib = ((Z[0:4, :].sum(axis=0) > 0).astype(int) +
                  (Z[4:8, :].sum(axis=0) > 0).astype(int) +
                  (Z[8:12, :].sum(axis=0) > 0).astype(int) +
                  (Z[12:16, :].sum(axis=0) > 0).astype(int))

    b_bit = int(np.min(wt_in_bit + wt_out_bit))
    b_nib = int(np.min(wt_in_nib + wt_out_nib))

    assert b_bit == 6, "Bit branch number must be 6"
    assert b_nib == 4, "Nibble branch number must be 4 (Near-MDS, not MDS 5)"

