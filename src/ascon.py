"""ascon.py: NIST SP 800-232 Standardized Ascon-AEAD128 Implementation.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Implements the Ascon-AEAD128 lightweight authenticated cipher specified in
NIST Special Publication 800-232 (Final, 2025):
"Ascon-Based Lightweight Cryptography Standard".
Reference: https://csrc.nist.gov/pubs/sp/800/232/final

SPECIFICATION PARAMETERS:
  - Algorithm: Ascon-AEAD128
  - Key length (k): 128 bits (16 bytes)
  - Nonce length (n): 128 bits (16 bytes)
  - Tag length (t): 128 bits (16 bytes)
  - Rate (r): 64 bits (8 bytes)
  - Internal state: 320 bits (5 x 64-bit words x0..x4)
  - Permutation rounds: a = 12 (init / finalization), b = 6 (intermediate)
"""

from __future__ import annotations

import hmac
import os
import struct
from typing import Optional, Tuple


# NIST SP 800-232 Round Constants for Ascon Permutation
ROUND_CONSTANTS = (
    0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b
)


def _ror(val: int, r: int) -> int:
    """64-bit bitwise right rotation."""
    return ((val >> r) | (val << (64 - r))) & 0xffffffffffffffff


def ascon_permutation(s: list[int], rounds: int = 12) -> list[int]:
    """Execute Ascon 320-bit state permutation p^rounds.
    
    Args:
        s: State represented as 5 64-bit unsigned integers [x0, x1, x2, x3, x4].
        rounds: Number of rounds (12 for p^a, 6 for p^b).
        
    Returns:
        Updated state list of 5 64-bit unsigned integers.
    """
    x0, x1, x2, x3, x4 = s
    start_round = 12 - rounds
    
    for r in range(start_round, 12):
        # 1. Constant addition layer
        x2 ^= ROUND_CONSTANTS[r]
        
        # 2. 5-bit Non-linear Substitution layer (S-box)
        x0 ^= x4
        x4 ^= x3
        x2 ^= x1
        t0 = (~x0) & x1
        t1 = (~x1) & x2
        t2 = (~x2) & x3
        t3 = (~x3) & x4
        t4 = (~x4) & x0
        x0 ^= t1
        x1 ^= t2
        x2 ^= t3
        x3 ^= t4
        x4 ^= t0
        x1 ^= x0
        x0 ^= x4
        x3 ^= x2
        x2 = (~x2) & 0xffffffffffffffff
        
        # 3. Linear Diffusion layer
        x0 ^= _ror(x0, 19) ^ _ror(x0, 28)
        x1 ^= _ror(x1, 61) ^ _ror(x1, 39)
        x2 ^= _ror(x2, 1)  ^ _ror(x2, 6)
        x3 ^= _ror(x3, 10) ^ _ror(x3, 17)
        x4 ^= _ror(x4, 7)  ^ _ror(x4, 41)
        
    return [x0, x1, x2, x3, x4]


def ascon_aead128_encrypt(
    key: bytes,
    nonce: bytes,
    associated_data: bytes,
    plaintext: bytes
) -> Tuple[bytes, bytes]:
    """Encrypt and compute 128-bit authentication tag using Ascon-AEAD128.
    
    Args:
        key: 16-byte (128-bit) symmetric key.
        nonce: 16-byte (128-bit) unique nonce.
        associated_data: Arbitrary unencrypted bytes bound to authentication tag.
        plaintext: Raw data bytes to encrypt.
        
    Returns:
        Tuple containing:
            - ciphertext (bytes): Encrypted payload of identical length to plaintext.
            - tag (bytes): 16-byte authentication tag.
    """
    if len(key) != 16:
        raise ValueError(f"Ascon-AEAD128 requires 16-byte key, got {len(key)}")
    if len(nonce) != 16:
        raise ValueError(f"Ascon-AEAD128 requires 16-byte nonce, got {len(nonce)}")
        
    rate = 8
    # Initialization: IV = 0x80400c0600000000 (k=128, r=64, a=12, b=6)
    iv = 0x80400c0600000000
    k0, k1 = struct.unpack(">QQ", key)
    n0, n1 = struct.unpack(">QQ", nonce)
    s = [iv, k0, k1, n0, n1]
    s = ascon_permutation(s, 12)
    s[3] ^= k0
    s[4] ^= k1
    
    # Associated Data absorption
    if associated_data:
        pad_len = rate - (len(associated_data) % rate)
        ad_padded = associated_data + b"\x80" + b"\x00" * (pad_len - 1)
        for i in range(0, len(ad_padded), rate):
            block = struct.unpack(">QQ", ad_padded[i:i+8] + b"\x00"*8)[0]
            s[0] ^= block
            s = ascon_permutation(s, 6)
    s[4] ^= 1  # 1-bit domain separation between AD and plaintext
    
    # Plaintext absorption and Ciphertext generation
    ciphertext = bytearray()
    plen = len(plaintext)
    full_blocks = plen // rate
    
    for i in range(full_blocks):
        block = struct.unpack(">Q", plaintext[i*8:(i+1)*8])[0]
        s[0] ^= block
        ciphertext.extend(struct.pack(">Q", s[0]))
        if i < full_blocks - 1 or plen % rate > 0:
            s = ascon_permutation(s, 6)
            
    rem = plen % rate
    if rem > 0:
        if full_blocks > 0:
            s = ascon_permutation(s, 6)
        last_block = plaintext[full_blocks*8:] + b"\x80" + b"\x00" * (rate - rem - 1)
        last_word = struct.unpack(">Q", last_block)[0]
        s[0] ^= last_word
        c_word = struct.pack(">Q", s[0])[:rem]
        ciphertext.extend(c_word)
    elif plen == 0 or full_blocks == 0:
        last_block = b"\x80" + b"\x00" * 7
        last_word = struct.unpack(">Q", last_block)[0]
        s[0] ^= last_word
    else:
        s = ascon_permutation(s, 6)
        last_block = b"\x80" + b"\x00" * 7
        last_word = struct.unpack(">Q", last_block)[0]
        s[0] ^= last_word
        
    # Finalization
    s[1] ^= k0
    s[2] ^= k1
    s = ascon_permutation(s, 12)
    s[3] ^= k0
    s[4] ^= k1
    tag = struct.pack(">QQ", s[3], s[4])
    
    return bytes(ciphertext), tag


def ascon_aead128_decrypt(
    key: bytes,
    nonce: bytes,
    associated_data: bytes,
    ciphertext: bytes,
    tag: bytes
) -> bytes:
    """Verify 128-bit authentication tag and decrypt using Ascon-AEAD128.
    
    Args:
        key: 16-byte (128-bit) symmetric key.
        nonce: 16-byte (128-bit) unique nonce.
        associated_data: Arbitrary unencrypted bytes bound to authentication tag.
        ciphertext: Raw ciphertext payload.
        tag: 16-byte authentication tag.
        
    Returns:
        bytes: Decrypted original plaintext.
        
    Raises:
        ValueError: If tag verification fails (integrity or authentication failure).
    """
    if len(key) != 16:
        raise ValueError(f"Ascon-AEAD128 requires 16-byte key, got {len(key)}")
    if len(nonce) != 16:
        raise ValueError(f"Ascon-AEAD128 requires 16-byte nonce, got {len(nonce)}")
    if len(tag) != 16:
        raise ValueError(f"Ascon-AEAD128 requires 16-byte tag, got {len(tag)}")
        
    rate = 8
    iv = 0x80400c0600000000
    k0, k1 = struct.unpack(">QQ", key)
    n0, n1 = struct.unpack(">QQ", nonce)
    s = [iv, k0, k1, n0, n1]
    s = ascon_permutation(s, 12)
    s[3] ^= k0
    s[4] ^= k1
    
    if associated_data:
        pad_len = rate - (len(associated_data) % rate)
        ad_padded = associated_data + b"\x80" + b"\x00" * (pad_len - 1)
        for i in range(0, len(ad_padded), rate):
            block = struct.unpack(">QQ", ad_padded[i:i+8] + b"\x00"*8)[0]
            s[0] ^= block
            s = ascon_permutation(s, 6)
    s[4] ^= 1
    
    plaintext = bytearray()
    clen = len(ciphertext)
    full_blocks = clen // rate
    
    for i in range(full_blocks):
        c_block = struct.unpack(">Q", ciphertext[i*8:(i+1)*8])[0]
        p_block = s[0] ^ c_block
        plaintext.extend(struct.pack(">Q", p_block))
        s[0] = c_block
        if i < full_blocks - 1 or clen % rate > 0:
            s = ascon_permutation(s, 6)
            
    rem = clen % rate
    if rem > 0:
        if full_blocks > 0:
            s = ascon_permutation(s, 6)
        c_rem = ciphertext[full_blocks*8:]
        s0_bytes = bytearray(struct.pack(">Q", s[0]))
        p_rem = bytearray()
        for j in range(rem):
            p_rem.append(s0_bytes[j] ^ c_rem[j])
            s0_bytes[j] = c_rem[j]
        s0_bytes[rem] ^= 0x80
        s[0] = struct.unpack(">Q", s0_bytes)[0]
        plaintext.extend(p_rem)
    elif clen == 0 or full_blocks == 0:
        last_block = b"\x80" + b"\x00" * 7
        last_word = struct.unpack(">Q", last_block)[0]
        s[0] ^= last_word
    else:
        s = ascon_permutation(s, 6)
        last_block = b"\x80" + b"\x00" * 7
        last_word = struct.unpack(">Q", last_block)[0]
        s[0] ^= last_word
        
    s[1] ^= k0
    s[2] ^= k1
    s = ascon_permutation(s, 12)
    s[3] ^= k0
    s[4] ^= k1
    expected_tag = struct.pack(">QQ", s[3], s[4])
    
    # Constant-time tag comparison to prevent timing side-channel attacks
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("Cryptographic authentication failed: Tag mismatch or corrupted ciphertext.")
        
    return bytes(plaintext)
