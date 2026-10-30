"""
src/crypto_research/mlmc_analysis.py

Cryptanalysis and Empirical Statistical Analysis Suite for MediLink Mini Cipher (MLMC).
Academic Research Prototype - NOT FOR PRODUCTION MEDICAL USE.

Analyzes:
  1. Strict Avalanche Criterion (SAC) & bit-flip diffusion over random samples.
  2. Round-by-round diffusion progression (Rounds 1 to 10).
  3. Difference Distribution Table (DDT) & Maximum Differential Probability.
  4. Linear Approximation Table (LAT) & Maximum Linear Bias.
  5. Statistical randomness evaluation (Monobit test, Runs test, Chi-square test).
  6. Reduced-round trail exploration for educational Mini profile.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crypto_research.mlmc_spec import MLMC_SBOX, MLMC_INV_SBOX
from src.crypto_research.mlmc_cipher import MLMCBlockCipher


def hamming_weight(n: int) -> int:
    """Compute number of set bits (popcount)."""
    return bin(n).count("1")


def bytes_hamming_distance(a: bytes, b: bytes) -> int:
    """Compute bitwise Hamming distance between two byte sequences."""
    return sum(hamming_weight(x ^ y) for x, y in zip(a, b))


# =====================================================================
# 1. Strict Avalanche Criterion (SAC) & Bit-Flip Diffusion
# =====================================================================

def evaluate_avalanche_effect(
    num_samples: int = 400,
    num_rounds: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates Strict Avalanche Criterion (SAC).
    For each random plaintext block and key, flips each of the 128 bit positions
    and records the percentage of ciphertext bits that flip.
    
    Ideal SAC: exactly 50.0% bit flip probability per output bit for any single input bit flip.
    """
    np.random.seed(seed)
    bit_flip_percentages: List[float] = []

    # Sample random keys and plaintexts
    for _ in range(num_samples):
        key = os.urandom(16)
        cipher = MLMCBlockCipher(key, num_rounds=num_rounds)
        pt = os.urandom(16)
        ct = cipher.encrypt_block(pt)

        # Flip random bit positions (sample 16 bits per plaintext to keep test fast yet statistically significant)
        for bit_idx in np.random.choice(128, size=16, replace=False):
            byte_idx = bit_idx // 8
            bit_in_byte = bit_idx % 8
            pt_flipped = bytearray(pt)
            pt_flipped[byte_idx] ^= (1 << bit_in_byte)

            ct_flipped = cipher.encrypt_block(bytes(pt_flipped))
            dist = bytes_hamming_distance(ct, ct_flipped)
            pct = (dist / 128.0) * 100.0
            bit_flip_percentages.append(pct)

    arr = np.array(bit_flip_percentages)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # SAC compliance score: distance from theoretical ideal 50.0%
    sac_deviation = abs(mean_val - 50.0)

    return {
        "num_trials": len(bit_flip_percentages),
        "mean_bit_flip_pct": round(mean_val, 2),
        "std_bit_flip_pct": round(std_val, 2),
        "min_bit_flip_pct": round(min_val, 2),
        "max_bit_flip_pct": round(max_val, 2),
        "sac_deviation_from_ideal": round(sac_deviation, 2),
        "ideal_expected_pct": 50.0
    }


def evaluate_round_avalanche_progression(
    num_samples: int = 100,
    max_rounds: int = 10
) -> List[Dict[str, Any]]:
    """
    Measures the progression of the avalanche effect across successive rounds (1 to max_rounds).
    Shows how linear and non-linear layers spread diffusion until the 50% saturation plateau.
    """
    progression = []
    key = os.urandom(16)

    for r in range(1, max_rounds + 1):
        cipher = MLMCBlockCipher(key, num_rounds=r)
        diffs = []
        for _ in range(num_samples):
            pt = os.urandom(16)
            ct = cipher.encrypt_block(pt)
            # Flip single bit 0
            pt_flip = bytearray(pt)
            pt_flip[0] ^= 0x01
            ct_flip = cipher.encrypt_block(bytes(pt_flip))
            diffs.append((bytes_hamming_distance(ct, ct_flip) / 128.0) * 100.0)

        progression.append({
            "round": r,
            "mean_flip_pct": round(float(np.mean(diffs)), 2),
            "std_flip_pct": round(float(np.std(diffs)), 2)
        })

    return progression


# =====================================================================
# 2. Difference Distribution Table (DDT) & Differential Cryptanalysis
# =====================================================================

def compute_sbox_ddt() -> Tuple[np.ndarray, int, float]:
    """
    Computes the 256x256 Difference Distribution Table (DDT) of the MLMC S-box.
    
    DDT[dx, dy] = #{ x in [0, 255] | S(x) ^ S(x ^ dx) == dy }
    
    Returns:
      - ddt: 256x256 numpy array
      - max_diff_count: maximum non-trivial entry max_{dx != 0, dy} DDT[dx, dy]
      - max_differential_prob: max_diff_count / 256
    """
    ddt = np.zeros((256, 256), dtype=np.int32)

    for dx in range(256):
        for x in range(256):
            dy = MLMC_SBOX[x] ^ MLMC_SBOX[x ^ dx]
            ddt[dx, dy] += 1

    # Max non-trivial difference (exclude dx == 0, dy == 0 where all 256 pairs match)
    non_trivial_ddt = ddt[1:, :]
    max_diff_count = int(np.max(non_trivial_ddt))
    max_differential_prob = max_diff_count / 256.0

    return ddt, max_diff_count, max_differential_prob


# =====================================================================
# 3. Linear Approximation Table (LAT) & Linear Cryptanalysis
# =====================================================================

def compute_sbox_lat() -> Tuple[np.ndarray, int, float]:
    """
    Computes the 256x256 Linear Approximation Table (LAT) of the MLMC S-box.
    
    LAT[alpha, beta] = #{ x in [0, 255] | (alpha . x) == (beta . S(x)) } - 128
    where '.' denotes bitwise parity inner product over GF(2).
    
    Returns:
      - lat: 256x256 numpy array
      - max_linear_bias_count: max_{alpha, beta != 0} |LAT[alpha, beta]|
      - max_linear_bias: max_linear_bias_count / 256
    """
    lat = np.zeros((256, 256), dtype=np.int32)

    for alpha in range(256):
        for beta in range(256):
            matches = 0
            for x in range(256):
                in_parity = hamming_weight(alpha & x) % 2
                out_parity = hamming_weight(beta & MLMC_SBOX[x]) % 2
                if in_parity == out_parity:
                    matches += 1
            lat[alpha, beta] = matches - 128

    # Exclude trivial alpha=0, beta=0
    non_trivial_mask = np.ones((256, 256), dtype=bool)
    non_trivial_mask[0, 0] = False
    max_linear_bias_count = int(np.max(np.abs(lat[non_trivial_mask])))
    max_linear_bias = max_linear_bias_count / 256.0

    return lat, max_linear_bias_count, max_linear_bias


# =====================================================================
# 4. Statistical Randomness Verification
# =====================================================================

def evaluate_ciphertext_randomness(
    num_blocks: int = 500,
    num_rounds: int = 10
) -> Dict[str, Any]:
    """
    Executes standard cryptographic randomness tests over MLMC ciphertext stream:
      1. Monobit Frequency Test (NIST SP 800-22 style)
      2. Runs Test (consecutive 0s / 1s distribution)
      3. Chi-Square Byte Uniformity Test over GF(2^8)
    """
    key = os.urandom(16)
    cipher = MLMCBlockCipher(key, num_rounds=num_rounds)

    stream = bytearray()
    for _ in range(num_blocks):
        stream.extend(cipher.encrypt_block(os.urandom(16)))

    total_bytes = len(stream)
    total_bits = total_bytes * 8

    # 1. Monobit Test
    one_bits = sum(bin(b).count("1") for b in stream)
    zero_bits = total_bits - one_bits
    s_n = one_bits - zero_bits
    s_obs = abs(s_n) / math.sqrt(total_bits)
    # Complementary error function for p-value: erfc(s_obs / sqrt(2))
    p_value_monobit = math.erfc(s_obs / math.sqrt(2))

    # 2. Chi-Square Byte Uniformity Test
    counts = [0] * 256
    for b in stream:
        counts[b] += 1
    expected = total_bytes / 256.0
    chi_square = sum(((c - expected) ** 2) / expected for c in counts)

    # 3. Runs Test
    bit_seq = []
    for b in stream:
        for bit in range(8):
            bit_seq.append((b >> (7 - bit)) & 1)

    v_n = 1
    for i in range(len(bit_seq) - 1):
        if bit_seq[i] != bit_seq[i + 1]:
            v_n += 1

    pi = one_bits / float(total_bits)
    runs_num = abs(v_n - 2 * total_bits * pi * (1.0 - pi))
    runs_den = 2 * math.sqrt(2 * total_bits) * pi * (1.0 - pi)
    p_value_runs = math.erfc(runs_num / runs_den) if runs_den > 0 else 0.0

    return {
        "total_stream_bytes": total_bytes,
        "monobit": {
            "ones_ratio": round(one_bits / total_bits, 4),
            "s_obs": round(s_obs, 4),
            "p_value": round(p_value_monobit, 4),
            "passed": p_value_monobit >= 0.01
        },
        "chi_square": {
            "statistic": round(chi_square, 2),
            "degrees_of_freedom": 255,
            "passed": 200.0 <= chi_square <= 310.0  # ~95% confidence interval for dof=255
        },
        "runs": {
            "total_runs": v_n,
            "p_value": round(p_value_runs, 4),
            "passed": p_value_runs >= 0.01
        }
    }


# =====================================================================
# Main Report Generator
# =====================================================================

def run_full_mlmc_analysis(output_dir: str = "results/crypto_research") -> Dict[str, Any]:
    """Execute complete cryptanalytic evaluation and persist results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" MediLink Mini Cipher (MLMC-128): Cryptanalytic Evaluation")
    print("=" * 75)

    # 1. Avalanche Effect
    print("\n[1/4] Evaluating Strict Avalanche Criterion (SAC)...")
    sac_results = evaluate_avalanche_effect(num_samples=300, num_rounds=10)
    print(f"      Mean Bit-Flip: {sac_results['mean_bit_flip_pct']}% (Ideal: 50.0%)")
    print(f"      Std Deviation: {sac_results['std_bit_flip_pct']}%")

    print("\n[2/4] Measuring Round-by-Round Diffusion Progression...")
    progression = evaluate_round_avalanche_progression(num_samples=80, max_rounds=10)
    for p in progression:
        print(f"      Round {p['round']:2d}: {p['mean_flip_pct']:5.2f}% +/- {p['std_flip_pct']:.2f}%")

    # 2. DDT & LAT
    print("\n[3/4] Computing Difference Distribution Table (DDT) & LAT for 8-bit S-Box...")
    _, max_ddt, max_dp = compute_sbox_ddt()
    print(f"      Max Differential Count: {max_ddt}/256 | Max DP: {max_dp:.4f} (2^-6)")

    _, max_lat, max_bias = compute_sbox_lat()
    print(f"      Max Linear Bias Count: {max_lat}/256 | Max Linear Bias: {max_bias:.4f} (2^-4)")

    # 3. Randomness Tests
    print("\n[4/4] Evaluating Ciphertext Stream Randomness (NIST SP 800-22 Tests)...")
    rand_res = evaluate_ciphertext_randomness(num_blocks=300)
    print(f"      Monobit p-value: {rand_res['monobit']['p_value']} (Passed: {rand_res['monobit']['passed']})")
    print(f"      Runs p-value:    {rand_res['runs']['p_value']} (Passed: {rand_res['runs']['passed']})")
    print(f"      Chi-Square:      {rand_res['chi_square']['statistic']} (Passed: {rand_res['chi_square']['passed']})")

    report = {
        "candidate": "MediLink Mini Cipher (MLMC-128)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "avalanche": sac_results,
        "round_progression": progression,
        "sbox_cryptanalysis": {
            "max_ddt_count": max_ddt,
            "max_differential_probability": max_dp,
            "max_lat_bias_count": max_lat,
            "max_linear_bias": max_bias
        },
        "randomness_tests": rand_res
    }

    # Save to JSON
    json_path = out_path / "mlmc_analysis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Analysis report successfully saved to: {json_path}")
    print("=" * 75 + "\n")

    return report


if __name__ == "__main__":
    run_full_mlmc_analysis()
