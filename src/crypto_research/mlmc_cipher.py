"""
src/crypto_research/mlmc_cipher.py

Core Implementation of MediLink Mini Cipher (MLMC-128).
Academic Research Prototype - NOT FOR PRODUCTION MEDICAL USE.

Classes:
  - MLMCBlockCipher: 128-bit block, 128-bit key Substitution-Permutation Network.
  - MLMCAEADExperimental: Experimental authenticated encryption wrapper using
    CTR mode with GHASH authentication over GF(2^128).
"""

from __future__ import annotations

import os
import struct
from typing import List, Optional, Tuple

from .mlmc_spec import (
    MLMCConfig,
    MLMC_SBOX,
    MLMC_INV_SBOX,
    MLMC_MDS_MATRIX,
    MLMC_INV_MDS_MATRIX,
    MLMC_RCON,
    gf256,
)


class MLMCBlockCipher:
    """
    MediLink Mini Cipher (MLMC) 128-bit Block Cipher.
    
    Structure: Substitution-Permutation Network (SPN).
    Parameters:
      - Block Size: 128 bits (16 bytes, 4x4 state matrix in GF(2^8))
      - Key Size: 128 bits (16 bytes)
      - Standard Rounds: 10
      - Reduced Rounds: 2, 4, 6, 8 (supported for cryptanalytic profiling)
    """

    def __init__(self, key: bytes, num_rounds: int = 10):
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError(f"Key must be bytes or bytearray, got {type(key)}")
        if len(key) != 16:
            raise ValueError(f"MLMC-128 requires exactly 16 bytes (128 bits) key, got {len(key)}")
        if not (1 <= num_rounds <= 16):
            raise ValueError(f"num_rounds must be between 1 and 16, got {num_rounds}")

        self.key = bytes(key)
        self.num_rounds = num_rounds
        self.round_keys = self._expand_key(self.key, self.num_rounds)

    @staticmethod
    def _expand_key(master_key: bytes, num_rounds: int) -> List[List[List[int]]]:
        """
        Expands 128-bit master key into (num_rounds + 1) 4x4 round key matrices.
        
        Key schedule algorithm:
          - Operates on 32-bit words w[0..4*num_rounds + 3].
          - For i % 4 == 0:
              w[i] = w[i-4] ^ SubWord(RotWord(w[i-1])) ^ (Rcon[i/4] << 24)
          - For i % 4 != 0:
              w[i] = w[i-4] ^ w[i-1]
        """
        words: List[int] = []
        for i in range(4):
            word = (
                (master_key[4 * i] << 24)
                | (master_key[4 * i + 1] << 16)
                | (master_key[4 * i + 2] << 8)
                | master_key[4 * i + 3]
            )
            words.append(word)

        total_words = 4 * (num_rounds + 1)
        for i in range(4, total_words):
            temp = words[i - 1]
            if i % 4 == 0:
                # RotWord: circular 8-bit left shift
                rot = ((temp << 8) & 0xFFFFFFFF) | (temp >> 24)
                # SubWord: byte-wise S-Box substitution
                sub = (
                    (MLMC_SBOX[(rot >> 24) & 0xFF] << 24)
                    | (MLMC_SBOX[(rot >> 16) & 0xFF] << 16)
                    | (MLMC_SBOX[(rot >> 8) & 0xFF] << 8)
                    | MLMC_SBOX[rot & 0xFF]
                )
                rcon_byte = MLMC_RCON[i // 4] if (i // 4) < len(MLMC_RCON) else 0x01
                temp = sub ^ (rcon_byte << 24)
            words.append(words[i - 4] ^ temp)

        # Convert words into 4x4 round key matrices (column-major)
        round_keys = []
        for r in range(num_rounds + 1):
            matrix = [[0] * 4 for _ in range(4)]
            for c in range(4):
                word = words[4 * r + c]
                matrix[0][c] = (word >> 24) & 0xFF
                matrix[1][c] = (word >> 16) & 0xFF
                matrix[2][c] = (word >> 8) & 0xFF
                matrix[3][c] = word & 0xFF
            round_keys.append(matrix)

        return round_keys

    @staticmethod
    def _bytes_to_state(b: bytes) -> List[List[int]]:
        """Map 16 bytes to 4x4 state matrix in column-major order."""
        state = [[0] * 4 for _ in range(4)]
        for c in range(4):
            for r in range(4):
                state[r][c] = b[r + 4 * c]
        return state

    @staticmethod
    def _state_to_bytes(state: List[List[int]]) -> bytes:
        """Map 4x4 state matrix in column-major order back to 16 bytes."""
        out = bytearray(16)
        for c in range(4):
            for r in range(4):
                out[r + 4 * c] = state[r][c] & 0xFF
        return bytes(out)

    @staticmethod
    def _sub_bytes(state: List[List[int]]) -> None:
        """Apply non-linear S-Box substitution to every byte in state."""
        for r in range(4):
            for c in range(4):
                state[r][c] = MLMC_SBOX[state[r][c]]

    @staticmethod
    def _inv_sub_bytes(state: List[List[int]]) -> None:
        """Apply inverse non-linear S-Box substitution to every byte in state."""
        for r in range(4):
            for c in range(4):
                state[r][c] = MLMC_INV_SBOX[state[r][c]]

    @staticmethod
    def _shift_rows(state: List[List[int]]) -> None:
        """Cyclically left-shift row r by r byte positions."""
        state[1][0], state[1][1], state[1][2], state[1][3] = (
            state[1][1], state[1][2], state[1][3], state[1][0]
        )
        state[2][0], state[2][1], state[2][2], state[2][3] = (
            state[2][2], state[2][3], state[2][0], state[2][1]
        )
        state[3][0], state[3][1], state[3][2], state[3][3] = (
            state[3][3], state[3][0], state[3][1], state[3][2]
        )

    @staticmethod
    def _inv_shift_rows(state: List[List[int]]) -> None:
        """Cyclically right-shift row r by r byte positions."""
        state[1][0], state[1][1], state[1][2], state[1][3] = (
            state[1][3], state[1][0], state[1][1], state[1][2]
        )
        state[2][0], state[2][1], state[2][2], state[2][3] = (
            state[2][2], state[2][3], state[2][0], state[2][1]
        )
        state[3][0], state[3][1], state[3][2], state[3][3] = (
            state[3][1], state[3][2], state[3][3], state[3][0]
        )

    @staticmethod
    def _mix_columns(state: List[List[int]]) -> None:
        """Multiply each column by MDS diffusion matrix over GF(2^8)."""
        for c in range(4):
            col = [state[r][c] for r in range(4)]
            for r in range(4):
                acc = 0
                for k in range(4):
                    acc ^= gf256.multiply(MLMC_MDS_MATRIX[r][k], col[k])
                state[r][c] = acc

    @staticmethod
    def _inv_mix_columns(state: List[List[int]]) -> None:
        """Multiply each column by inverse MDS diffusion matrix over GF(2^8)."""
        for c in range(4):
            col = [state[r][c] for r in range(4)]
            for r in range(4):
                acc = 0
                for k in range(4):
                    acc ^= gf256.multiply(MLMC_INV_MDS_MATRIX[r][k], col[k])
                state[r][c] = acc

    @staticmethod
    def _add_round_key(state: List[List[int]], round_key: List[List[int]]) -> None:
        """XOR state with 128-bit round subkey."""
        for r in range(4):
            for c in range(4):
                state[r][c] ^= round_key[r][c]

    def encrypt_block(self, block: bytes) -> bytes:
        """
        Encrypt a single 16-byte (128-bit) plaintext block.
        
        Args:
            block: Exactly 16 bytes of plaintext.
            
        Returns:
            Exactly 16 bytes of ciphertext.
        """
        if len(block) != 16:
            raise ValueError(f"Block size must be 16 bytes, got {len(block)}")

        state = self._bytes_to_state(block)

        # Round 0: Key whitening
        self._add_round_key(state, self.round_keys[0])

        # Main rounds 1 to num_rounds - 1
        for r in range(1, self.num_rounds):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state)
            self._add_round_key(state, self.round_keys[r])

        # Final round (omits MixColumns to match standard SPN symmetry)
        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, self.round_keys[self.num_rounds])

        return self._state_to_bytes(state)

    def decrypt_block(self, block: bytes) -> bytes:
        """
        Decrypt a single 16-byte (128-bit) ciphertext block.
        
        Args:
            block: Exactly 16 bytes of ciphertext.
            
        Returns:
            Exactly 16 bytes of recovered plaintext.
        """
        if len(block) != 16:
            raise ValueError(f"Block size must be 16 bytes, got {len(block)}")

        state = self._bytes_to_state(block)

        # Initial inverse round
        self._add_round_key(state, self.round_keys[self.num_rounds])
        self._inv_shift_rows(state)
        self._inv_sub_bytes(state)

        # Main rounds in reverse order
        for r in range(self.num_rounds - 1, 0, -1):
            self._add_round_key(state, self.round_keys[r])
            self._inv_mix_columns(state)
            self._inv_shift_rows(state)
            self._inv_sub_bytes(state)

        # Final round: unwhiten
        self._add_round_key(state, self.round_keys[0])

        return self._state_to_bytes(state)


# =====================================================================
# GF(2^128) Universal Hashing (GHASH) Helper for Experimental AEAD
# =====================================================================

def _ghash_multiply(x: int, y: int) -> int:
    """
    Multiplication in GF(2^128) modulo R = x^128 + x^7 + x^2 + x + 1 (NIST SP 800-38D).
    x and y are 128-bit unsigned integers.
    """
    r_poly = 0xE1000000000000000000000000000000  # Shifted representation
    z = 0
    v = x
    for i in range(128):
        if (y >> (127 - i)) & 1:
            z ^= v
        if v & 1:
            v = (v >> 1) ^ r_poly
        else:
            v >>= 1
    return z


def _compute_ghash(h_key: bytes, aad: bytes, ciphertext: bytes) -> bytes:
    """
    Computes standard GHASH tag over (AAD || pad || Ciphertext || pad || len(AAD) || len(C)).
    """
    h_int = int.from_bytes(h_key, byteorder="big")
    y = 0

    def process_data(data: bytes) -> None:
        nonlocal y
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            if len(chunk) < 16:
                chunk = chunk + b"\x00" * (16 - len(chunk))
            chunk_int = int.from_bytes(chunk, byteorder="big")
            y = _ghash_multiply(y ^ chunk_int, h_int)

    process_data(aad)
    process_data(ciphertext)

    # Length block: [64-bit bitlen(AAD)] + [64-bit bitlen(C)]
    len_block = struct.pack(">QQ", len(aad) * 8, len(ciphertext) * 8)
    len_int = int.from_bytes(len_block, byteorder="big")
    y = _ghash_multiply(y ^ len_int, h_int)

    return y.to_bytes(16, byteorder="big")


# =====================================================================
# Experimental AEAD Wrapper
# =====================================================================

class MLMCAEADExperimental:
    """
    Experimental Authenticated Encryption with Associated Data (AEAD) for MLMC-128.
    
    WARNING:
      THIS IS AN UNREVIEWED EDUCATIONAL / RESEARCH CONSTRUCTION.
      DO NOT USE FOR CLINICAL OR REAL PATIENT DATA PROTECTION.
      
    Construction:
      - Encryption: Counter (CTR) mode with 96-bit nonce and 32-bit big-endian counter.
      - Authentication: GHASH over GF(2^128) using hash key H = MLMC_Encrypt(0^128).
      - Tag generation: GHASH(H, AAD, C) ^ MLMC_Encrypt(Nonce || 0x00000001).
      - Envelope Format: [12-byte Nonce] + [Ciphertext] + [16-byte Auth Tag].
    """

    def __init__(self, key: bytes, num_rounds: int = 10):
        if len(key) != 16:
            raise ValueError(f"MLMCAEADExperimental requires 16-byte key, got {len(key)}")
        self.key = bytes(key)
        self.cipher = MLMCBlockCipher(self.key, num_rounds=num_rounds)
        # Precompute GHASH subkey H = E_K(0^128)
        self.h_key = self.cipher.encrypt_block(b"\x00" * 16)

    @property
    def name(self) -> str:
        return "MLMC-GCM-Experimental"

    @property
    def key_length_bytes(self) -> int:
        return 16

    @property
    def nonce_length_bytes(self) -> int:
        return 12

    @property
    def tag_length_bytes(self) -> int:
        return 16

    @property
    def overhead_bytes(self) -> int:
        return self.nonce_length_bytes + self.tag_length_bytes

    def generate_key(self) -> bytes:
        return os.urandom(16)

    def encrypt(
        self,
        plaintext: bytes,
        key: Optional[bytes] = None,
        associated_data: Optional[bytes] = None,
        nonce: Optional[bytes] = None
    ) -> bytes:
        """
        Encrypts plaintext and returns authenticated envelope: [Nonce (12B)] + [Ciphertext] + [Tag (16B)].
        """
        if key is not None and key != self.key:
            cipher = MLMCBlockCipher(key, num_rounds=self.cipher.num_rounds)
            h_key = cipher.encrypt_block(b"\x00" * 16)
        else:
            cipher = self.cipher
            h_key = self.h_key

        if nonce is None:
            nonce = os.urandom(12)
        elif len(nonce) != 12:
            raise ValueError(f"Nonce must be 12 bytes, got {len(nonce)}")

        ad = associated_data if associated_data is not None else b""

        # CTR Mode Encryption
        ciphertext = bytearray(len(plaintext))
        counter = 2  # Counter 1 is reserved for tag masking
        for offset in range(0, len(plaintext), 16):
            ctr_block = nonce + struct.pack(">I", counter)
            keystream = cipher.encrypt_block(ctr_block)
            block_len = min(16, len(plaintext) - offset)
            for j in range(block_len):
                ciphertext[offset + j] = plaintext[offset + j] ^ keystream[j]
            counter += 1

        ct_bytes = bytes(ciphertext)

        # GHASH Tag Generation
        ghash_val = _compute_ghash(h_key, ad, ct_bytes)
        mask = cipher.encrypt_block(nonce + struct.pack(">I", 1))
        tag = bytes(a ^ b for a, b in zip(ghash_val, mask))

        return nonce + ct_bytes + tag

    def decrypt(
        self,
        envelope: bytes,
        key: Optional[bytes] = None,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Authenticates and decrypts envelope.
        Raises ValueError if authentication tag does not match in constant time.
        """
        if len(envelope) < 28:
            raise ValueError(f"Envelope length ({len(envelope)}) shorter than Nonce+Tag (28 bytes)")

        if key is not None and key != self.key:
            cipher = MLMCBlockCipher(key, num_rounds=self.cipher.num_rounds)
            h_key = cipher.encrypt_block(b"\x00" * 16)
        else:
            cipher = self.cipher
            h_key = self.h_key

        nonce = envelope[:12]
        tag = envelope[-16:]
        ciphertext = envelope[12:-16]
        ad = associated_data if associated_data is not None else b""

        # Verify authentication tag in constant time
        ghash_val = _compute_ghash(h_key, ad, ciphertext)
        mask = cipher.encrypt_block(nonce + struct.pack(">I", 1))
        expected_tag = bytes(a ^ b for a, b in zip(ghash_val, mask))

        # Constant-time comparison
        diff = 0
        for a, b in zip(tag, expected_tag):
            diff |= (a ^ b)

        if diff != 0:
            raise ValueError("MLMC Authentication Tag Mismatch: Ciphertext or AAD has been tampered with!")

        # CTR Mode Decryption
        plaintext = bytearray(len(ciphertext))
        counter = 2
        for offset in range(0, len(ciphertext), 16):
            ctr_block = nonce + struct.pack(">I", counter)
            keystream = cipher.encrypt_block(ctr_block)
            block_len = min(16, len(ciphertext) - offset)
            for j in range(block_len):
                plaintext[offset + j] = ciphertext[offset + j] ^ keystream[j]
            counter += 1

        return bytes(plaintext)
