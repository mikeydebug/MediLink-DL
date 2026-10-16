"""test_security.py: Unit tests for cryptographic security module."""

import pytest

from src.security import (
    generate_secure_key,
    encrypt_payload,
    decrypt_payload,
    demonstrate_tamper_detection,
    DecryptionTamperedError,
    SecurityError
)


def test_key_generation():
    key = generate_secure_key(256)
    assert len(key) == 32
    # Ensure consecutive keys are unique
    key2 = generate_secure_key(256)
    assert key != key2


def test_encryption_decryption_roundtrip():
    key = generate_secure_key()
    plaintext = b"MediLink-DL: Radiologist diagnostic report payload"
    envelope = encrypt_payload(plaintext, key)
    
    assert len(envelope) > len(plaintext)
    decrypted = decrypt_payload(envelope, key)
    assert decrypted == plaintext


def test_nonce_freshness():
    key = generate_secure_key()
    plaintext = b"Identical clinical payload for nonce testing"
    env1 = encrypt_payload(plaintext, key)
    env2 = encrypt_payload(plaintext, key)
    
    # Nonces (first 12 bytes) must be distinct
    assert env1[:12] != env2[:12]
    # Complete envelopes must be distinct
    assert env1 != env2


def test_tamper_detection_single_bit_flip():
    key = generate_secure_key()
    plaintext = b"Confidential Chest X-Ray Diagnosis: Clear Lungs"
    envelope = bytearray(encrypt_payload(plaintext, key))
    
    # Invert single bit in ciphertext region
    envelope[15] ^= 0x01
    
    with pytest.raises(DecryptionTamperedError):
        decrypt_payload(bytes(envelope), key)


def test_associated_data_binding():
    key = generate_secure_key()
    plaintext = b"Patient 10293 DICOM metadata"
    aad_correct = b"PatientID:10293"
    aad_wrong = b"PatientID:99999"
    
    envelope = encrypt_payload(plaintext, key, associated_data=aad_correct)
    
    # Correct AAD succeeds
    assert decrypt_payload(envelope, key, associated_data=aad_correct) == plaintext
    
    # Tampered / mismatched AAD must fail
    with pytest.raises(DecryptionTamperedError):
        decrypt_payload(envelope, key, associated_data=aad_wrong)


def test_demonstrate_tamper_detection_helper():
    payload = b"Research prototype payload verification"
    audit = demonstrate_tamper_detection(payload)
    
    assert audit["unmodified_decryption_successful"] is True
    assert audit["tamper_detected"] is True
    assert audit["is_safe"] is True
