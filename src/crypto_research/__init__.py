"""
src/crypto_research/__init__.py

Experimental Cryptographic Research Module for MediLink-DL.
Candidate: MediLink Mini Cipher (MLMC)

NOTE: ACADEMIC RESEARCH PROTOTYPE ONLY.
NOT SUITABLE FOR PRODUCTION USE OR REAL PATIENT DATA.
"""

from .mlmc_spec import (
    MLMCConfig,
    GF256,
    MLMC_SBOX,
    MLMC_INV_SBOX,
    MLMC_MDS_MATRIX,
    MLMC_INV_MDS_MATRIX,
)
from .mlmc_cipher import MLMCBlockCipher, MLMCAEADExperimental

__all__ = [
    "MLMCConfig",
    "GF256",
    "MLMC_SBOX",
    "MLMC_INV_SBOX",
    "MLMC_MDS_MATRIX",
    "MLMC_INV_MDS_MATRIX",
    "MLMCBlockCipher",
    "MLMCAEADExperimental",
]
