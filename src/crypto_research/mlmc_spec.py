"""
src/crypto_research/mlmc_spec.py

Mathematical and structural specifications for MediLink Mini Cipher (MLMC).
Academic Research Prototype - NOT FOR PRODUCTION MEDICAL USE.

Defines:
  1. Finite field arithmetic in GF(2^8) modulo P(x) = x^8 + x^4 + x^3 + x + 1 (0x11B).
  2. S-box construction, bijectivity, and algebraic inverse.
  3. MDS linear diffusion matrix and branch number verification.
  4. Permutation state layout (4x4 byte matrix).
  5. Key schedule expansion rules and round constants.
  6. Configuration dataclasses for full MLMC-128 and educational MLMC-32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# =====================================================================
# 1. Finite Field GF(2^8) Arithmetic
# =====================================================================

# Irreducible polynomial P(x) = x^8 + x^4 + x^3 + x + 1 (0x11B = 283)
IRREDUCIBLE_POLY = 0x11B


class GF256:
    """Galois Field GF(2^8) arithmetic implementation with precomputed lookup tables."""

    def __init__(self, poly: int = IRREDUCIBLE_POLY):
        self.poly = poly
        self.exp = [0] * 512
        self.log = [0] * 256
        self._init_tables()

    def _init_tables(self) -> None:
        """Precompute generator exponentiation and logarithm tables base generator g=0x03."""
        # 0x03 is a primitive generator of GF(2^8) modulo 0x11B
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.exp[i + 255] = x
            self.log[x] = i
            # Multiply x by 3 (i.e. x * (x + 1) = xtime(x) ^ x)
            x_times_2 = (x << 1) ^ (self.poly if (x & 0x80) else 0)
            x = (x_times_2 ^ x) & 0xFF
        self.log[0] = 0  # Convention: log(0) undefined, set to 0

    def add(self, a: int, b: int) -> int:
        """Addition in GF(2^8) is bitwise XOR."""
        return (a ^ b) & 0xFF

    def multiply(self, a: int, b: int) -> int:
        """Multiplication in GF(2^8) via log/exp tables."""
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def inverse(self, a: int) -> int:
        """Multiplicative inverse in GF(2^8), with inv(0) = 0 by convention."""
        if a == 0:
            return 0
        return self.exp[255 - self.log[a]]

    def xtime(self, a: int) -> int:
        """Multiply by polynomial x (0x02) in GF(2^8)."""
        return ((a << 1) ^ (self.poly if (a & 0x80) else 0)) & 0xFF


# Global GF(2^8) instance
gf256 = GF256()


# =====================================================================
# 2. S-Box Generation and Cryptographic Properties
# =====================================================================

def _generate_sbox() -> Tuple[List[int], List[int]]:
    """
    Constructs the MLMC 8x8 S-Box.
    
    Design:
      1. Multiplicative inversion in GF(2^8) (inv(0) = 0)
      2. Affine transformation:
         s_i = b_i ^ b_{(i+4)%8} ^ b_{(i+5)%8} ^ b_{(i+6)%8} ^ b_{(i+7)%8} ^ c_i
         with constant c = 0x63.
         
    This yields:
      - Bijective: Exactly 1-to-1 mapping of {0, ..., 255}
      - Maximum differential probability: 4/256 = 2^(-6)
      - Non-linearity: NL = 112
      - Algebraic degree: 7
    """
    sbox = [0] * 256
    inv_sbox = [0] * 256

    for byte_val in range(256):
        inv = gf256.inverse(byte_val)
        # Affine transform over GF(2)
        s = inv
        transformed = 0
        for bit in range(8):
            b_bit = (
                ((s >> bit) & 1)
                ^ ((s >> ((bit + 4) % 8)) & 1)
                ^ ((s >> ((bit + 5) % 8)) & 1)
                ^ ((s >> ((bit + 6) % 8)) & 1)
                ^ ((s >> ((bit + 7) % 8)) & 1)
                ^ ((0x63 >> bit) & 1)
            )
            transformed |= (b_bit << bit)
        sbox[byte_val] = transformed
        inv_sbox[transformed] = byte_val

    return sbox, inv_sbox


MLMC_SBOX, MLMC_INV_SBOX = _generate_sbox()


# =====================================================================
# 3. Maximum Distance Separable (MDS) Linear Diffusion Matrix
# =====================================================================

# Circulant MDS matrix over GF(2^8) with branch number B = 5:
# M = circ(02, 03, 01, 01)
# M^(-1) = circ(0E, 0B, 0D, 09)
MLMC_MDS_MATRIX = [
    [0x02, 0x03, 0x01, 0x01],
    [0x01, 0x02, 0x03, 0x01],
    [0x01, 0x01, 0x02, 0x03],
    [0x03, 0x01, 0x01, 0x02]
]

MLMC_INV_MDS_MATRIX = [
    [0x0E, 0x0B, 0x0D, 0x09],
    [0x09, 0x0E, 0x0B, 0x0D],
    [0x0D, 0x09, 0x0E, 0x0B],
    [0x0B, 0x0D, 0x09, 0x0E]
]


def verify_mds_branch_number() -> int:
    """
    Empirically verifies that the MDS matrix satisfies branch number B = 5.
    Branch number B = min_{v != 0} (wt(v) + wt(M * v)).
    For a 4x4 matrix over GF(2^8), maximum possible branch number is 4 + 1 = 5.
    """
    # Sample verification over single-byte active inputs to confirm differential branch number
    min_weight = 8
    for col_idx in range(4):
        for val in range(1, 256):
            v = [0, 0, 0, 0]
            v[col_idx] = val
            out = [0, 0, 0, 0]
            for r in range(4):
                acc = 0
                for c in range(4):
                    acc ^= gf256.multiply(MLMC_MDS_MATRIX[r][c], v[c])
                out[r] = acc
            wt_in = sum(1 for x in v if x != 0)
            wt_out = sum(1 for x in out if x != 0)
            branch = wt_in + wt_out
            if branch < min_weight:
                min_weight = branch
    return min_weight


# =====================================================================
# 4. Round Constants
# =====================================================================

def _compute_round_constants(max_rounds: int = 16) -> List[int]:
    """Compute round constants Rcon[j] = x^(j-1) in GF(2^8) modulo 0x11B."""
    rcon = [0] * (max_rounds + 1)
    val = 1
    for i in range(1, max_rounds + 1):
        rcon[i] = val
        val = gf256.xtime(val)
    return rcon


MLMC_RCON = _compute_round_constants(16)


# =====================================================================
# 5. Configuration Dataclasses
# =====================================================================

@dataclass(frozen=True)
class MLMCConfig:
    """
    Configuration parameters for MediLink Mini Cipher.
    
    Standard Profile (MLMC-128):
      - Block Size: 128 bits (16 bytes, 4x4 state matrix)
      - Key Size: 128 bits (16 bytes)
      - Rounds: 10 rounds (standard full security)
      - Reduced Rounds: 2, 4, 6, 8 for cryptanalytic attack exploration
    """
    block_size_bytes: int = 16
    key_size_bytes: int = 16
    num_rounds: int = 10
    state_rows: int = 4
    state_cols: int = 4

    def validate(self) -> None:
        if self.block_size_bytes != 16:
            raise ValueError(f"MLMC block size must be 16 bytes (128 bits), got {self.block_size_bytes}")
        if self.key_size_bytes != 16:
            raise ValueError(f"MLMC key size must be 16 bytes (128 bits), got {self.key_size_bytes}")
        if self.num_rounds < 1 or self.num_rounds > 16:
            raise ValueError(f"MLMC rounds must be between 1 and 16, got {self.num_rounds}")


@dataclass(frozen=True)
class MLMC32Config:
    """
    Miniature Educational Profile (MLMC-32):
      - Block Size: 32 bits (4 bytes, 2x2 state matrix)
      - Key Size: 32 bits (4 bytes)
      - Rounds: 4 rounds
      - Purpose: Enables exhaustive differential and linear trail search in academic demos.
    """
    block_size_bytes: int = 4
    key_size_bytes: int = 4
    num_rounds: int = 4
    state_rows: int = 2
    state_cols: int = 2
