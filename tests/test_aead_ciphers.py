"""test_aead_ciphers.py: Comprehensive security and integrity tests for AES and Ascon AEAD ciphers."""

import pytest
import os
from src.security import (
    get_aead_cipher,
    generate_secure_key,
    encrypt_payload,
    decrypt_payload,
    DecryptionTamperedError,
    SecurityError
)

CIPHER_NAMES = ["AES-256-GCM", "AES-128-GCM", "ASCON-128"]


@pytest.mark.parametrize("cipher_name", CIPHER_NAMES)
def test_cipher_roundtrip(cipher_name):
    """Test encryption and decryption round-trip for each cipher."""
    cipher = get_aead_cipher(cipher_name)
    key = cipher.generate_key()
    plaintext = b"Clinical report: No active focal infiltrates detected."
    aad = b"Patient:90184|Session:882"
    
    envelope = cipher.encrypt(plaintext, key, associated_data=aad)
    assert len(envelope) == len(plaintext) + cipher.overhead_bytes
    
    decrypted = cipher.decrypt(envelope, key, associated_data=aad)
    assert decrypted == plaintext


@pytest.mark.parametrize("cipher_name", CIPHER_NAMES)
def test_cipher_tampered_ciphertext(cipher_name):
    """Test that modifying even one bit of ciphertext fails authentication."""
    cipher = get_aead_cipher(cipher_name)
    key = cipher.generate_key()
    plaintext = b"Critical CXR diagnostic ROI bitstream payload"
    aad = b"Metadata:Header"
    
    envelope = bytearray(cipher.encrypt(plaintext, key, associated_data=aad))
    
    # Flip bit in ciphertext body (after nonce, before tag)
    flip_pos = cipher.nonce_length_bytes + 5
    envelope[flip_pos] ^= 0x01
    
    with pytest.raises(DecryptionTamperedError):
        cipher.decrypt(bytes(envelope), key, associated_data=aad)


@pytest.mark.parametrize("cipher_name", CIPHER_NAMES)
def test_cipher_tampered_tag(cipher_name):
    """Test that tampering with authentication tag fails authentication."""
    cipher = get_aead_cipher(cipher_name)
    key = cipher.generate_key()
    plaintext = b"Confidential diagnosis payload"
    
    envelope = bytearray(cipher.encrypt(plaintext, key))
    # Flip bit in last byte (tag region)
    envelope[-1] ^= 0x01
    
    with pytest.raises(DecryptionTamperedError):
        cipher.decrypt(bytes(envelope), key)


@pytest.mark.parametrize("cipher_name", CIPHER_NAMES)
def test_cipher_wrong_key(cipher_name):
    """Test that decryption with wrong key is rejected."""
    cipher = get_aead_cipher(cipher_name)
    key1 = cipher.generate_key()
    key2 = cipher.generate_key()
    plaintext = b"Patient payload"
    
    envelope = cipher.encrypt(plaintext, key1)
    with pytest.raises(DecryptionTamperedError):
        cipher.decrypt(envelope, key2)


@pytest.mark.parametrize("cipher_name", CIPHER_NAMES)
def test_cipher_wrong_associated_data(cipher_name):
    """Test that associated data mismatch triggers authentication rejection."""
    cipher = get_aead_cipher(cipher_name)
    key = cipher.generate_key()
    plaintext = b"Patient payload"
    
    envelope = cipher.encrypt(plaintext, key, associated_data=b"CorrectAAD")
    with pytest.raises(DecryptionTamperedError):
        cipher.decrypt(envelope, key, associated_data=b"WrongAAD")


@pytest.mark.parametrize("cipher_name", CIPHER_NAMES)
def test_cipher_truncated_payload(cipher_name):
    """Test that truncated envelopes shorter than minimum headers are rejected."""
    cipher = get_aead_cipher(cipher_name)
    key = cipher.generate_key()
    
    short_envelope = b"short"
    with pytest.raises(SecurityError):
        cipher.decrypt(short_envelope, key)


@pytest.mark.parametrize("cipher_name", CIPHER_NAMES)
def test_multiple_chunks_nonce_uniqueness(cipher_name):
    """Verify that encrypting multiple chunks generates fresh, unique nonces."""
    cipher = get_aead_cipher(cipher_name)
    key = cipher.generate_key()
    plaintext = b"Identical chunk data"
    
    envelopes = [cipher.encrypt(plaintext, key) for _ in range(20)]
    nonces = [env[:cipher.nonce_length_bytes] for env in envelopes]
    
    # All nonces must be strictly distinct
    assert len(set(nonces)) == len(nonces)
