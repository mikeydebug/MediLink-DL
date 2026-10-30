"""
src/crypto_research/benchmark_mlmc.py

Rigorous Empirical Benchmarking Harness for MediLink Mini Cipher (MLMC).
Compares MLMC against standardized AEAD ciphers (AES-256-GCM, AES-128-GCM, Ascon-AEAD128)
across representative medical chunk payload sizes (256, 512, 1024, 2048, 4096 bytes).

Academic Research Prototype - NOT FOR PRODUCTION MEDICAL USE.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cryptography
from src.security import get_aead_cipher
from src.crypto_research.mlmc_cipher import MLMCBlockCipher, MLMCAEADExperimental


def run_comprehensive_benchmarks(
    payload_sizes: List[int] = (256, 512, 1024, 2048, 4096),
    iterations: int = 100,
    output_dir: str = "results/crypto_research"
) -> List[Dict[str, Any]]:
    """
    Executes controlled benchmarking comparing:
      1. AES-256-GCM (NIST SP 800-38D, C/OpenSSL HW-accelerated)
      2. AES-128-GCM (NIST SP 800-38D, C/OpenSSL HW-accelerated)
      3. Ascon-AEAD128 (NIST SP 800-232, Pure Python reference)
      4. MLMC-128 (Raw Block Cipher, Pure Python reference)
      5. MLMC-GCM-Experimental (Experimental AEAD, Pure Python reference)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    aad = b"MediLink-DL:ResearchChunk:v1"

    print("=" * 80)
    print(" MediLink-DL: Empirical Cryptographic Benchmarking Suite")
    print(f" Platform: {platform.system()} {platform.machine()} | Python {platform.python_version()}")
    print(f" Cryptography library: {cryptography.__version__}")
    print("=" * 80)

    for size in payload_sizes:
        print(f"\n[BENCHMARK] Payload Size: {size} bytes ({iterations} iterations per candidate)...")
        plaintext = os.urandom(size)

        # -------------------------------------------------------------
        # 1. Standard Ciphers (AES-256-GCM, AES-128-GCM, Ascon-AEAD128)
        # -------------------------------------------------------------
        for name in ["AES-256-GCM", "AES-128-GCM", "ASCON-128"]:
            cipher = get_aead_cipher(name)
            key = cipher.generate_key()

            # Warm-up
            for _ in range(5):
                env = cipher.encrypt(plaintext, key, associated_data=aad)
                _ = cipher.decrypt(env, key, associated_data=aad)

            enc_times = []
            dec_times = []

            for _ in range(iterations):
                t0 = time.perf_counter_ns()
                env = cipher.encrypt(plaintext, key, associated_data=aad)
                t1 = time.perf_counter_ns()

                t2 = time.perf_counter_ns()
                dec = cipher.decrypt(env, key, associated_data=aad)
                t3 = time.perf_counter_ns()

                assert dec == plaintext
                enc_times.append((t1 - t0) / 1000.0)  # Microseconds
                dec_times.append((t3 - t2) / 1000.0)

            enc_med = float(np.median(enc_times))
            dec_med = float(np.median(dec_times))
            tot_sec = (enc_med + dec_med) * 1e-6
            throughput = (size / (1024.0 * 1024.0)) / tot_sec if tot_sec > 0 else 0.0

            results.append({
                "algorithm": name,
                "type": "Standardized AEAD",
                "implementation": "OpenSSL C / Hardware Accelerated" if "AES" in name else "Pure Python Reference",
                "payload_bytes": size,
                "overhead_bytes": cipher.overhead_bytes,
                "enc_time_us": round(enc_med, 2),
                "dec_time_us": round(dec_med, 2),
                "throughput_mb_s": round(throughput, 2)
            })
            print(f"  {name:15s} | Enc: {enc_med:7.1f} us | Dec: {dec_med:7.1f} us | Throughput: {throughput:6.2f} MB/s")

        # -------------------------------------------------------------
        # 2. MLMC-128 Raw Block Cipher (Unauthenticated, Multiple Blocks)
        # -------------------------------------------------------------
        mlmc_raw_key = os.urandom(16)
        mlmc_raw = MLMCBlockCipher(mlmc_raw_key, num_rounds=10)
        num_blocks = (size + 15) // 16
        padded_pt = plaintext + (b"\x00" * (num_blocks * 16 - size))

        enc_times_raw = []
        dec_times_raw = []

        for _ in range(iterations):
            # Encrypt all blocks
            t0 = time.perf_counter_ns()
            ct_blocks = bytearray()
            for b_idx in range(num_blocks):
                ct_blocks.extend(mlmc_raw.encrypt_block(padded_pt[b_idx * 16 : (b_idx + 1) * 16]))
            t1 = time.perf_counter_ns()

            # Decrypt all blocks
            t2 = time.perf_counter_ns()
            rec_blocks = bytearray()
            for b_idx in range(num_blocks):
                rec_blocks.extend(mlmc_raw.decrypt_block(bytes(ct_blocks[b_idx * 16 : (b_idx + 1) * 16])))
            t3 = time.perf_counter_ns()

            assert rec_blocks == padded_pt
            enc_times_raw.append((t1 - t0) / 1000.0)
            dec_times_raw.append((t3 - t2) / 1000.0)

        enc_med_raw = float(np.median(enc_times_raw))
        dec_med_raw = float(np.median(dec_times_raw))
        tot_sec_raw = (enc_med_raw + dec_med_raw) * 1e-6
        throughput_raw = (size / (1024.0 * 1024.0)) / tot_sec_raw if tot_sec_raw > 0 else 0.0

        results.append({
            "algorithm": "MLMC-128 (Raw Block)",
            "type": "Experimental Block Cipher",
            "implementation": "Pure Python SPN Prototype",
            "payload_bytes": size,
            "overhead_bytes": 0,
            "enc_time_us": round(enc_med_raw, 2),
            "dec_time_us": round(dec_med_raw, 2),
            "throughput_mb_s": round(throughput_raw, 2)
        })
        print(f"  {'MLMC-128 Raw':15s} | Enc: {enc_med_raw:7.1f} us | Dec: {dec_med_raw:7.1f} us | Throughput: {throughput_raw:6.2f} MB/s")

        # -------------------------------------------------------------
        # 3. MLMC-GCM-Experimental (Full AEAD Wrapper)
        # -------------------------------------------------------------
        mlmc_aead_key = os.urandom(16)
        mlmc_aead = MLMCAEADExperimental(mlmc_aead_key, num_rounds=10)

        enc_times_aead = []
        dec_times_aead = []

        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            env = mlmc_aead.encrypt(plaintext, associated_data=aad)
            t1 = time.perf_counter_ns()

            t2 = time.perf_counter_ns()
            rec = mlmc_aead.decrypt(env, associated_data=aad)
            t3 = time.perf_counter_ns()

            assert rec == plaintext
            enc_times_aead.append((t1 - t0) / 1000.0)
            dec_times_aead.append((t3 - t2) / 1000.0)

        enc_med_aead = float(np.median(enc_times_aead))
        dec_med_aead = float(np.median(dec_times_aead))
        tot_sec_aead = (enc_med_aead + dec_med_aead) * 1e-6
        throughput_aead = (size / (1024.0 * 1024.0)) / tot_sec_aead if tot_sec_aead > 0 else 0.0

        results.append({
            "algorithm": "MLMC-GCM-Exp",
            "type": "Experimental AEAD Candidate",
            "implementation": "Pure Python SPN + GHASH",
            "payload_bytes": size,
            "overhead_bytes": mlmc_aead.overhead_bytes,
            "enc_time_us": round(enc_med_aead, 2),
            "dec_time_us": round(dec_med_aead, 2),
            "throughput_mb_s": round(throughput_aead, 2)
        })
        print(f"  {'MLMC-GCM-Exp':15s} | Enc: {enc_med_aead:7.1f} us | Dec: {dec_med_aead:7.1f} us | Throughput: {throughput_aead:6.2f} MB/s")

    # Save to CSV
    csv_file = out_path / "mlmc_benchmark_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "type", "implementation", "payload_bytes",
            "overhead_bytes", "enc_time_us", "dec_time_us", "throughput_mb_s"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Save to JSON
    json_file = out_path / "mlmc_benchmark_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmarks saved to: {csv_file}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_comprehensive_benchmarks()
