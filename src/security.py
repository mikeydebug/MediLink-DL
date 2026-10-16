"""security.py: Authenticated Encryption and Integrity Protection for Medical Imagery.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Implements AES-256-GCM authenticated encryption, per-operation fresh nonce generation,
and cryptographic tamper detection.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag


# Protocol constants
NONCE_LENGTH_BYTES = 12   # 96-bit nonce standard for AES-GCM (NIST SP 800-38D)
KEY_LENGTH_BITS = 256     # AES-256 provides 256-bit post-quantum resistant symmetric security


class SecurityError(Exception):
    """Base exception for cryptographic operations."""
    pass


class DecryptionTamperedError(SecurityError):
    """Raised when ciphertext authentication fails (tampering or wrong key)."""
    pass


def generate_secure_key(bit_length: int = KEY_LENGTH_BITS) -> bytes:
    """Generate a cryptographically secure random symmetric key.
    
    ACADEMIC & SECURITY ARCHITECTURE NOTE:
    In this prototype, keys are generated via an OS-level CSPRNG (os.urandom).
    In a real rural telemedicine production deployment, symmetric keys should NOT
    be transmitted in plaintext or shared statically. Instead, deploy:
    1. Hybrid Public-Key Encryption (HPKE, RFC 9180) or Ephemeral Elliptic Curve
       Diffie-Hellman (ECDH over Curve25519) for key agreement between the rural
       health center and the regional hospital PACS.
    2. Mutual TLS 1.3 (mTLS) with hardware-backed client certificates (PKCS#11 / TPM).
    
    Args:
        bit_length: Key length in bits (must be 128, 192, or 256).
        
    Returns:
        bytes: 32 random bytes for AES-256.
    """
    if bit_length not in (128, 192, 256):
        raise ValueError(f"Unsupported key length {bit_length}. Must be 128, 192, or 256.")
    return AESGCM.generate_key(bit_length=bit_length)


def encrypt_payload(
    plaintext: bytes,
    key: bytes,
    associated_data: Optional[bytes] = None
) -> bytes:
    """Encrypt and authenticate a medical image payload using AES-256-GCM.
    
    Format of returned envelope:
    [12-byte Nonce] + [Ciphertext + 16-byte Poly1305/GHASH Auth Tag]
    
    Args:
        plaintext: Raw data bytes to encrypt.
        key: 32-byte AES-256 key.
        associated_data: Optional unencrypted metadata bound to the auth tag (e.g. Patient ID hash).
        
    Returns:
        bytes: Complete encrypted package (nonce prepended).
    """
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError(f"Plaintext must be bytes, got {type(plaintext)}")
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be exactly 32 bytes, got {len(key)}")
        
    # Generate fresh cryptographically random 96-bit nonce for EVERY invocation
    nonce = os.urandom(NONCE_LENGTH_BYTES)
    aesgcm = AESGCM(key)
    
    # AESGCM.encrypt returns ciphertext with appended 16-byte authentication tag
    ciphertext_and_tag = aesgcm.encrypt(nonce, bytes(plaintext), associated_data)
    
    return nonce + ciphertext_and_tag


def decrypt_payload(
    envelope: bytes,
    key: bytes,
    associated_data: Optional[bytes] = None
) -> bytes:
    """Decrypt and verify ciphertext envelope authenticity.
    
    Args:
        envelope: [12-byte Nonce] + [Ciphertext + 16-byte Auth Tag].
        key: 32-byte AES-256 key.
        associated_data: Must match associated data supplied during encryption.
        
    Returns:
        bytes: Original decrypted plaintext.
        
    Raises:
        DecryptionTamperedError: If authentication tag fails verification (tampering detected).
        SecurityError: If envelope is too short.
    """
    if len(envelope) < (NONCE_LENGTH_BYTES + 16):
        raise SecurityError(
            f"Envelope length ({len(envelope)}) is shorter than minimum nonce + tag header."
        )
    if len(key) != 32:
        raise ValueError(f"AES-256 key must be exactly 32 bytes, got {len(key)}")
        
    nonce = envelope[:NONCE_LENGTH_BYTES]
    ciphertext_and_tag = envelope[NONCE_LENGTH_BYTES:]
    
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext_and_tag, associated_data)
        return plaintext
    except InvalidTag as exc:
        raise DecryptionTamperedError(
            "Cryptographic integrity verification failed: Authentication tag mismatch or corrupted ciphertext."
        ) from exc


def demonstrate_tamper_detection(
    sample_payload: bytes,
    key: Optional[bytes] = None
) -> Dict[str, Any]:
    """Demonstrate encryption, successful decryption, and tamper detection.
    
    Returns an audit dictionary proving that modifying even one bit of the ciphertext
    causes authentication verification to strictly fail.
    """
    if key is None:
        key = generate_secure_key()
        
    # Step 1: Encrypt
    aad = b"MediLink-DL:ClinicalHeader-V1"
    envelope = encrypt_payload(sample_payload, key, associated_data=aad)
    
    # Step 2: Decrypt unmodified
    decrypted = decrypt_payload(envelope, key, associated_data=aad)
    success_unmodified = (decrypted == sample_payload)
    
    # Step 3: Tamper with ciphertext by flipping a single bit at the midpoint
    tampered_envelope = bytearray(envelope)
    flip_index = NONCE_LENGTH_BYTES + (len(envelope) - NONCE_LENGTH_BYTES) // 2
    tampered_envelope[flip_index] ^= 0x01  # Invert lowest bit
    
    tamper_detected = False
    error_message = ""
    try:
        decrypt_payload(bytes(tampered_envelope), key, associated_data=aad)
    except DecryptionTamperedError as exc:
        tamper_detected = True
        error_message = str(exc)
        
    return {
        "payload_length_bytes": len(sample_payload),
        "envelope_length_bytes": len(envelope),
        "overhead_bytes": len(envelope) - len(sample_payload),
        "encryption_algorithm": "AES-256-GCM",
        "nonce_length_bytes": NONCE_LENGTH_BYTES,
        "auth_tag_length_bytes": 16,
        "unmodified_decryption_successful": success_unmodified,
        "tamper_injection_index": flip_index,
        "tamper_detected": tamper_detected,
        "tamper_rejection_reason": error_message,
        "is_safe": success_unmodified and tamper_detected
    }
