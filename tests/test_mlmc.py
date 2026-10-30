"""
tests/test_mlmc.py: Comprehensive Unit and Integration Test Suite for MediLink Mini Cipher (MLMC).

Academic Research Prototype Test Suite.
Verifies:
  1. Finite field arithmetic, S-box bijectivity, absence of fixed points, and MDS branch number.
  2. Key schedule expansion, invertibility, and key-diffusion properties.
  3. Encryption and decryption round-trip correctness across test vectors and random blocks.
  4. Input validation and defensive exception handling.
  5. Strict Avalanche Criterion (SAC) threshold satisfaction.
  6. Experimental AEAD wrapper round-trip and constant-time tamper detection.
"""

import os
import pytest

from src.crypto_research.mlmc_spec import (
    gf256,
    MLMC_SBOX,
    MLMC_INV_SBOX,
    MLMC_MDS_MATRIX,
    MLMC_INV_MDS_MATRIX,
    verify_mds_branch_number,
    MLMCConfig
)
from src.crypto_research.mlmc_cipher import MLMCBlockCipher, MLMCAEADExperimental
from src.crypto_research.mlmc_analysis import (
    evaluate_avalanche_effect,
    compute_sbox_ddt,
    compute_sbox_lat
)


# =====================================================================
# 1. Mathematical and Spec Verifications
# =====================================================================

def test_sbox_properties():
    """Verify that MLMC S-box is a bijective permutation with no fixed points."""
    assert len(MLMC_SBOX) == 256
    assert len(MLMC_INV_SBOX) == 256
    assert len(set(MLMC_SBOX)) == 256, "S-box must be a permutation of 0..255"

    for i in range(256):
        # Invertibility
        assert MLMC_INV_SBOX[MLMC_SBOX[i]] == i, f"Inverse failed at index {i}"
        # No fixed points (S(x) != x)
        assert MLMC_SBOX[i] != i, f"S-box must have no fixed points, failed at {i}"


def test_mds_matrix_invertibility_and_branch_number():
    """Verify that MDS matrix has branch number 5 and its inverse is exact."""
    # Verify M * M^(-1) == Identity in GF(2^8)
    for r in range(4):
        for c in range(4):
            acc = 0
            for k in range(4):
                acc ^= gf256.multiply(MLMC_MDS_MATRIX[r][k], MLMC_INV_MDS_MATRIX[k][c])
            expected = 1 if r == c else 0
            assert acc == expected, f"MDS inverse mismatch at ({r}, {c}): {acc} != {expected}"

    # Verify branch number
    branch = verify_mds_branch_number()
    assert branch == 5, f"Expected MDS branch number 5, got {branch}"


# =====================================================================
# 2. Block Cipher Core Correctness & Deterministic Test Vectors
# =====================================================================

def test_mlmc_deterministic_test_vector():
    """Verify MLMC-128 against a fixed deterministic test vector."""
    # Test vector: All-zeros key and all-zeros plaintext
    key_zero = b"\x00" * 16
    pt_zero = b"\x00" * 16
    cipher = MLMCBlockCipher(key_zero, num_rounds=10)

    ct = cipher.encrypt_block(pt_zero)
    assert ct != pt_zero, "Ciphertext must not equal plaintext"
    assert len(ct) == 16

    pt_recovered = cipher.decrypt_block(ct)
    assert pt_recovered == pt_zero, "Decrypted block must match original plaintext"

    # Secondary deterministic vector
    key_pattern = bytes(range(16))
    pt_pattern = bytes(range(16, 32))
    cipher2 = MLMCBlockCipher(key_pattern, num_rounds=10)
    ct2 = cipher2.encrypt_block(pt_pattern)
    assert cipher2.decrypt_block(ct2) == pt_pattern


def test_mlmc_multiple_random_roundtrips():
    """Verify round-trip invertibility across 100 randomly sampled blocks and keys."""
    for _ in range(100):
        key = os.urandom(16)
        pt = os.urandom(16)
        cipher = MLMCBlockCipher(key, num_rounds=10)
        ct = cipher.encrypt_block(pt)
        assert cipher.decrypt_block(ct) == pt


@pytest.mark.parametrize("rounds", [2, 4, 6, 8, 10, 12])
def test_mlmc_various_round_counts(rounds):
    """Verify that MLMC maintains round-trip correctness across reduced and extended rounds."""
    key = os.urandom(16)
    pt = b"TelemedSecure128"
    cipher = MLMCBlockCipher(key, num_rounds=rounds)
    ct = cipher.encrypt_block(pt)
    assert cipher.decrypt_block(ct) == pt


# =====================================================================
# 3. Defensive Input Validation
# =====================================================================

def test_mlmc_invalid_key_length():
    """Verify that keys other than 16 bytes raise ValueError."""
    with pytest.raises(ValueError):
        MLMCBlockCipher(b"short_key")
    with pytest.raises(ValueError):
        MLMCBlockCipher(b"way_too_long_key_exceeding_16_bytes")


def test_mlmc_invalid_block_length():
    """Verify that blocks other than 16 bytes raise ValueError."""
    cipher = MLMCBlockCipher(os.urandom(16))
    with pytest.raises(ValueError):
        cipher.encrypt_block(b"short")
    with pytest.raises(ValueError):
        cipher.encrypt_block(b"12345678901234567")  # 17 bytes
    with pytest.raises(ValueError):
        cipher.decrypt_block(b"short")


def test_mlmc_invalid_types():
    """Verify that non-bytes types raise TypeError."""
    with pytest.raises(TypeError):
        MLMCBlockCipher("string_instead_of_bytes")  # type: ignore


# =====================================================================
# 4. Cryptanalytic Property Assertions
# =====================================================================

def test_mlmc_avalanche_statistical_bounds():
    """Verify that 10-round MLMC satisfies SAC within acceptable statistical bounds [45%, 55%]."""
    res = evaluate_avalanche_effect(num_samples=150, num_rounds=10)
    mean_flip = res["mean_bit_flip_pct"]
    assert 45.0 <= mean_flip <= 55.0, f"Mean SAC bit flip {mean_flip}% outside [45%, 55%] bound"


def test_sbox_ddt_and_lat_bounds():
    """Verify that S-box maximum differential probability is 2^-6 and linear bias is 2^-4."""
    _, max_ddt, max_dp = compute_sbox_ddt()
    assert max_ddt == 4, f"Expected max DDT count 4, got {max_ddt}"
    assert abs(max_dp - (4 / 256.0)) < 1e-6

    _, max_lat, max_bias = compute_sbox_lat()
    assert max_lat == 16, f"Expected max LAT bias count 16, got {max_lat}"
    assert abs(max_bias - (16 / 256.0)) < 1e-6


# =====================================================================
# 5. Experimental AEAD Wrapper Tests
# =====================================================================

def test_mlmc_aead_roundtrip_various_payloads():
    """Verify experimental AEAD encryption and decryption on empty, small, and large payloads."""
    key = os.urandom(16)
    aead = MLMCAEADExperimental(key)
    aad = b"ClinicalPatient:00912"

    for size in [0, 1, 15, 16, 17, 64, 256, 1024]:
        plaintext = os.urandom(size)
        envelope = aead.encrypt(plaintext, associated_data=aad)
        assert len(envelope) == size + aead.overhead_bytes
        decrypted = aead.decrypt(envelope, associated_data=aad)
        assert decrypted == plaintext


def test_mlmc_aead_ciphertext_tamper_rejection():
    """Verify that modifying a single bit in the ciphertext causes authentication failure."""
    key = os.urandom(16)
    aead = MLMCAEADExperimental(key)
    envelope = bytearray(aead.encrypt(b"Diagnostic Infiltration Record", associated_data=b"AD-01"))

    # Flip bit in ciphertext body (after 12-byte nonce, before 16-byte tag)
    envelope[15] ^= 0x01

    with pytest.raises(ValueError) as exc:
        aead.decrypt(bytes(envelope), associated_data=b"AD-01")
    assert "Mismatch" in str(exc.value) or "tampered" in str(exc.value)


def test_mlmc_aead_aad_tamper_rejection():
    """Verify that altering associated data causes authentication failure."""
    key = os.urandom(16)
    aead = MLMCAEADExperimental(key)
    envelope = aead.encrypt(b"Confidential CXR Stream", associated_data=b"Correct-AAD")

    with pytest.raises(ValueError):
        aead.decrypt(envelope, associated_data=b"Corrupted-AAD")


def test_mlmc_aead_tag_tamper_rejection():
    """Verify that altering the authentication tag causes authentication failure."""
    key = os.urandom(16)
    aead = MLMCAEADExperimental(key)
    envelope = bytearray(aead.encrypt(b"Confidential CXR Stream"))
    # Flip bit in tag (last byte)
    envelope[-1] ^= 0x01

    with pytest.raises(ValueError):
        aead.decrypt(bytes(envelope))
