"""security.py: Authenticated Encryption and Integrity Protection for Medical Imagery.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Implements standardized authenticated encryption (AEAD) ciphers:
  1. AES-256-GCM (NIST SP 800-38D baseline)
  2. AES-128-GCM (NIST SP 800-38D lightweight baseline)
  3. Ascon-AEAD128 (NIST SP 800-232 standardized lightweight AEAD)

CRITICAL CRYPTOGRAPHIC AND INTEGRITY SPECIFICATIONS:
  - Standard AES-GCM authenticates via GHASH over GF(2^128) producing a 128-bit tag.
  - Ascon-AEAD128 operates on a 320-bit permutation state with 128-bit tag and rate 64 bits.
  - Per-operation fresh nonces are strictly enforced via OS-level CSPRNG.
  - Zero plaintext is released upon authentication or tag verification failure.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Dict, Optional, Tuple, Union
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from src.ascon import ascon_aead128_encrypt, ascon_aead128_decrypt


class SecurityError(Exception):
    """Base exception for cryptographic operations."""
    pass


class DecryptionTamperedError(SecurityError):
    """Raised when ciphertext authentication fails (tampering or corrupted tag)."""
    pass


class AEADCipher(abc.ABC):
    """Abstract Base Class defining a unified Authenticated Encryption with Associated Data (AEAD) interface."""
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Official standardized cipher name."""
        pass

    @property
    @abc.abstractmethod
    def key_length_bytes(self) -> int:
        """Key length in bytes."""
        pass

    @property
    @abc.abstractmethod
    def nonce_length_bytes(self) -> int:
        """Nonce length in bytes."""
        pass

    @property
    @abc.abstractmethod
    def tag_length_bytes(self) -> int:
        """Authentication tag length in bytes."""
        pass

    @property
    def overhead_bytes(self) -> int:
        """Total envelope overhead added to plaintext payload (nonce + tag)."""
        return self.nonce_length_bytes + self.tag_length_bytes

    def generate_key(self) -> bytes:
        """Generate a cryptographically secure random key."""
        return os.urandom(self.key_length_bytes)

    @abc.abstractmethod
    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None,
        nonce: Optional[bytes] = None
    ) -> bytes:
        """Encrypt and authenticate payload, returning complete envelope (nonce + ciphertext + tag)."""
        pass

    @abc.abstractmethod
    def decrypt(
        self,
        envelope: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """Authenticate and decrypt envelope. Raises DecryptionTamperedError if tag fails."""
        pass


class AESGCMCipherBase(AEADCipher):
    """AES-GCM Authenticated Encryption implementation (NIST SP 800-38D)."""
    
    def __init__(self, key_bytes: int):
        self._key_bytes = key_bytes
        self._nonce_bytes = 12  # Standard 96-bit nonce for AES-GCM
        self._tag_bytes = 16    # 128-bit GHASH authentication tag

    @property
    def key_length_bytes(self) -> int:
        return self._key_bytes

    @property
    def nonce_length_bytes(self) -> int:
        return self._nonce_bytes

    @property
    def tag_length_bytes(self) -> int:
        return self._tag_bytes

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None,
        nonce: Optional[bytes] = None
    ) -> bytes:
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError(f"Plaintext must be bytes, got {type(plaintext)}")
        if len(key) != self.key_length_bytes:
            raise ValueError(f"{self.name} requires {self.key_length_bytes}-byte key, got {len(key)}")
            
        if nonce is None:
            nonce = os.urandom(self.nonce_length_bytes)
        elif len(nonce) != self.nonce_length_bytes:
            raise ValueError(f"{self.name} requires {self.nonce_length_bytes}-byte nonce, got {len(nonce)}")
            
        aesgcm = AESGCM(key)
        # AESGCM.encrypt returns [Ciphertext] + [16-byte GHASH tag]
        ciphertext_and_tag = aesgcm.encrypt(nonce, bytes(plaintext), associated_data)
        return nonce + ciphertext_and_tag

    def decrypt(
        self,
        envelope: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        min_len = self.nonce_length_bytes + self.tag_length_bytes
        if len(envelope) < min_len:
            raise SecurityError(f"Envelope length ({len(envelope)}) shorter than header + tag minimum ({min_len}).")
        if len(key) != self.key_length_bytes:
            raise ValueError(f"{self.name} requires {self.key_length_bytes}-byte key, got {len(key)}")
            
        nonce = envelope[:self.nonce_length_bytes]
        ciphertext_and_tag = envelope[self.nonce_length_bytes:]
        aesgcm = AESGCM(key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext_and_tag, associated_data)
            return plaintext
        except InvalidTag as exc:
            raise DecryptionTamperedError(
                f"{self.name} integrity verification failed: GHASH tag mismatch or corrupted ciphertext."
            ) from exc


class AES256GCMCipher(AESGCMCipherBase):
    """AES-256-GCM: 256-bit key, 96-bit nonce, 128-bit GHASH tag."""
    def __init__(self):
        super().__init__(key_bytes=32)

    @property
    def name(self) -> str:
        return "AES-256-GCM"


class AES128GCMCipher(AESGCMCipherBase):
    """AES-128-GCM: 128-bit key, 96-bit nonce, 128-bit GHASH tag."""
    def __init__(self):
        super().__init__(key_bytes=16)

    @property
    def name(self) -> str:
        return "AES-128-GCM"


class Ascon128Cipher(AEADCipher):
    """Ascon-AEAD128: 128-bit key, 128-bit nonce, 128-bit tag (NIST SP 800-232)."""
    
    @property
    def name(self) -> str:
        return "Ascon-AEAD128"

    @property
    def key_length_bytes(self) -> int:
        return 16

    @property
    def nonce_length_bytes(self) -> int:
        return 16

    @property
    def tag_length_bytes(self) -> int:
        return 16

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None,
        nonce: Optional[bytes] = None
    ) -> bytes:
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError(f"Plaintext must be bytes, got {type(plaintext)}")
        if len(key) != 16:
            raise ValueError(f"Ascon-AEAD128 requires 16-byte key, got {len(key)}")
            
        if nonce is None:
            nonce = os.urandom(16)
        elif len(nonce) != 16:
            raise ValueError(f"Ascon-AEAD128 requires 16-byte nonce, got {len(nonce)}")
            
        ad = associated_data if associated_data is not None else b""
        ciphertext, tag = ascon_aead128_encrypt(key, nonce, ad, bytes(plaintext))
        # Format: [16-byte Nonce] + [Ciphertext] + [16-byte Tag]
        return nonce + ciphertext + tag

    def decrypt(
        self,
        envelope: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        min_len = 32  # 16-byte nonce + 16-byte tag
        if len(envelope) < min_len:
            raise SecurityError(f"Envelope length ({len(envelope)}) shorter than nonce+tag minimum ({min_len}).")
        if len(key) != 16:
            raise ValueError(f"Ascon-AEAD128 requires 16-byte key, got {len(key)}")
            
        nonce = envelope[:16]
        tag = envelope[-16:]
        ciphertext = envelope[16:-16]
        ad = associated_data if associated_data is not None else b""
        
        try:
            plaintext = ascon_aead128_decrypt(key, nonce, ad, ciphertext, tag)
            return plaintext
        except ValueError as exc:
            raise DecryptionTamperedError(
                f"Ascon-AEAD128 integrity verification failed: Tag mismatch or corrupted ciphertext."
            ) from exc


class MLMC128Cipher(AEADCipher):
    """Academic Research Prototype: MLMC-128 (MediLink Mini Cipher).
    
    128-bit block cipher candidate with 10 rounds, PRESENT 4-bit S-box,
    non-Hadamard involutory MDS diffusion, and CTR-GHASH authenticated encryption mode.
    Note: For academic research evaluation; clinical data defaults to AES/Ascon.
    """

    @property
    def name(self) -> str:
        return "MLMC-128"

    @property
    def key_length_bytes(self) -> int:
        return 16  # 128-bit key

    @property
    def nonce_length_bytes(self) -> int:
        return 12  # 96-bit nonce

    @property
    def tag_length_bytes(self) -> int:
        return 16  # 128-bit GHASH tag

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None,
        nonce: Optional[bytes] = None
    ) -> bytes:
        from src.crypto_research.mlmc_cipher import MLMCAEADExperimental
        cipher = MLMCAEADExperimental(key)
        return cipher.encrypt(plaintext, key, associated_data, nonce)

    def decrypt(
        self,
        envelope: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        from src.crypto_research.mlmc_cipher import MLMCAEADExperimental
        cipher = MLMCAEADExperimental(key)
        try:
            return cipher.decrypt(envelope, key, associated_data)
        except ValueError as exc:
            raise DecryptionTamperedError(
                f"MLMC-128 integrity verification failed: Tag mismatch or corrupted ciphertext."
            ) from exc


# Cipher Registry
CIPHER_REGISTRY: Dict[str, AEADCipher] = {
    "AES-256-GCM": AES256GCMCipher(),
    "AES-128-GCM": AES128GCMCipher(),
    "ASCON-128": Ascon128Cipher(),
    "ASCON-AEAD128": Ascon128Cipher(),
    "MLMC-128": MLMC128Cipher(),
    "MLMC-AEAD128": MLMC128Cipher()
}


def get_aead_cipher(algorithm: str = "AES-256-GCM") -> AEADCipher:
    """Retrieve AEAD cipher instance by standardized name.
    
    Args:
        algorithm: One of 'AES-256-GCM', 'AES-128-GCM', 'ASCON-128' / 'Ascon-AEAD128'.
    """
    key = algorithm.strip().upper()
    if key not in CIPHER_REGISTRY:
        supported = list(CIPHER_REGISTRY.keys())
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported: {supported}")
    return CIPHER_REGISTRY[key]


# Top-level functional API for backwards compatibility
NONCE_LENGTH_BYTES = 12
KEY_LENGTH_BITS = 256


def generate_secure_key(bit_length: int = 256, algorithm: Optional[str] = None) -> bytes:
    """Generate cryptographically secure symmetric key.
    
    Args:
        bit_length: Desired key length in bits (128, 192, or 256).
        algorithm: If supplied, overrides bit_length with cipher's native key requirement.
    """
    if algorithm is not None:
        cipher = get_aead_cipher(algorithm)
        return cipher.generate_key()

    # Gracefully accept byte-lengths (16, 24, 32) as well as bit-lengths (128, 192, 256)
    if bit_length in (16, 24, 32):
        bit_length *= 8
        
    if bit_length not in (128, 192, 256):
        raise ValueError(f"Unsupported key length {bit_length}. Must be 128, 192, or 256 bits (or 16, 24, 32 bytes).")
    return os.urandom(bit_length // 8)


def encrypt_payload(
    plaintext: bytes,
    key: bytes,
    associated_data: Optional[bytes] = None,
    algorithm: str = "AES-256-GCM"
) -> bytes:
    """Encrypt and authenticate a medical image payload using specified AEAD cipher."""
    cipher = get_aead_cipher(algorithm)
    return cipher.encrypt(plaintext, key, associated_data=associated_data)


def decrypt_payload(
    envelope: bytes,
    key: bytes,
    associated_data: Optional[bytes] = None,
    algorithm: str = "AES-256-GCM"
) -> bytes:
    """Decrypt and verify ciphertext envelope authenticity."""
    cipher = get_aead_cipher(algorithm)
    return cipher.decrypt(envelope, key, associated_data=associated_data)


def demonstrate_tamper_detection(
    sample_payload: bytes,
    key: Optional[bytes] = None,
    algorithm: str = "AES-256-GCM"
) -> Dict[str, Any]:
    """Demonstrate encryption, successful decryption, and tamper detection."""
    cipher = get_aead_cipher(algorithm)
    if key is None:
        key = cipher.generate_key()
        
    aad = b"MediLink-DL:ClinicalHeader-V1"
    envelope = cipher.encrypt(sample_payload, key, associated_data=aad)
    
    # Decrypt unmodified
    decrypted = cipher.decrypt(envelope, key, associated_data=aad)
    success_unmodified = (decrypted == sample_payload)
    
    # Flip single bit at midpoint of ciphertext region
    tampered_envelope = bytearray(envelope)
    nonce_len = cipher.nonce_length_bytes
    tag_len = cipher.tag_length_bytes
    ct_len = len(envelope) - nonce_len - tag_len
    
    flip_index = nonce_len + ct_len // 2 if ct_len > 0 else nonce_len
    tampered_envelope[flip_index] ^= 0x01
    
    tamper_detected = False
    error_message = ""
    try:
        cipher.decrypt(bytes(tampered_envelope), key, associated_data=aad)
    except DecryptionTamperedError as exc:
        tamper_detected = True
        error_message = str(exc)
        
    return {
        "payload_length_bytes": len(sample_payload),
        "envelope_length_bytes": len(envelope),
        "overhead_bytes": len(envelope) - len(sample_payload),
        "encryption_algorithm": cipher.name,
        "nonce_length_bytes": cipher.nonce_length_bytes,
        "auth_tag_length_bytes": cipher.tag_length_bytes,
        "unmodified_decryption_successful": success_unmodified,
        "tamper_injection_index": flip_index,
        "tamper_detected": tamper_detected,
        "tamper_rejection_reason": error_message,
        "is_safe": success_unmodified and tamper_detected
    }
