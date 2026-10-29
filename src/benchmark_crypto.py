"""benchmark_crypto.py: Microbenchmarking AEAD Ciphers for Medical Chunk Transmission.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Empirically benchmarks AES-256-GCM, AES-128-GCM, and Ascon-AEAD128 (NIST SP 800-232)
across representative medical chunk payload sizes (256, 512, 1024, 2048, 4096 bytes).
Measures encryption latency, decryption latency, throughput, and overhead.
"""

from __future__ import annotations

import csv
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cryptography
from src.security import get_aead_cipher, CIPHER_REGISTRY


def run_crypto_benchmarks(
    payload_sizes: List[int] = (256, 512, 1024, 2048, 4096),
    iterations: int = 150,
    output_dir: str = "results"
) -> List[Dict[str, Any]]:
    """Execute rigorous cryptographic benchmarking across all candidate AEAD ciphers."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    ciphers = ["AES-256-GCM", "AES-128-GCM", "ASCON-128"]
    results = []
    
    system_info = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cryptography_version": cryptography.__version__
    }
    
    print("=" * 80)
    print(" MediLink-DL: AEAD Cryptographic Benchmark for Telemedicine Chunking")
    print(f" Environment: {system_info['os']} ({system_info['machine']}), Python {system_info['python_version']}")
    print(f" Cryptography library: {system_info['cryptography_version']}")
    print("=" * 80)
    
    aad = b"MediLink-DL:PacketHeader:v1"
    
    for size in payload_sizes:
        print(f"\n[BENCHMARK] Payload Size: {size} bytes ({iterations} trials per cipher)...")
        # Generate random representative clinical data chunk
        sample_plaintext = os.urandom(size)
        
        for c_name in ciphers:
            cipher = get_aead_cipher(c_name)
            key = cipher.generate_key()
            
            enc_times = []
            dec_times = []
            
            # Warm-up (10 iterations)
            for _ in range(10):
                env = cipher.encrypt(sample_plaintext, key, associated_data=aad)
                _ = cipher.decrypt(env, key, associated_data=aad)
                
            # Benchmark loop
            for _ in range(iterations):
                # Measure encryption
                t0 = time.perf_counter_ns()
                envelope = cipher.encrypt(sample_plaintext, key, associated_data=aad)
                t1 = time.perf_counter_ns()
                
                # Measure decryption
                t2 = time.perf_counter_ns()
                decrypted = cipher.decrypt(envelope, key, associated_data=aad)
                t3 = time.perf_counter_ns()
                
                if decrypted != sample_plaintext:
                    raise RuntimeError(f"Integrity failure in {c_name}")
                    
                enc_times.append((t1 - t0) / 1000.0)  # Microseconds
                dec_times.append((t3 - t2) / 1000.0)  # Microseconds
                
            enc_median = float(np.median(enc_times))
            enc_std = float(np.std(enc_times))
            dec_median = float(np.median(dec_times))
            dec_std = float(np.std(dec_times))
            combined_median = enc_median + dec_median
            
            # Throughput in MB/s (based on combined enc+dec time)
            total_time_sec = (enc_median + dec_median) * 1e-6
            throughput_mb_s = (size / (1024.0 * 1024.0)) / total_time_sec if total_time_sec > 0 else 0.0
            
            overhead = len(envelope) - size
            
            res_row = {
                "Algorithm": cipher.name,
                "Payload_Size_Bytes": size,
                "Key_Size_Bits": cipher.key_length_bytes * 8,
                "Nonce_Size_Bytes": cipher.nonce_length_bytes,
                "Tag_Size_Bytes": cipher.tag_length_bytes,
                "Total_Overhead_Bytes": overhead,
                "Total_Envelope_Bytes": len(envelope),
                "Enc_Latency_Median_us": round(enc_median, 2),
                "Enc_Latency_Std_us": round(enc_std, 2),
                "Dec_Latency_Median_us": round(dec_median, 2),
                "Dec_Latency_Std_us": round(dec_std, 2),
                "Combined_Latency_Median_us": round(combined_median, 2),
                "Throughput_MB_s": round(throughput_mb_s, 2),
                "Trials": iterations
            }
            results.append(res_row)
            print(f"  {cipher.name:<14} | Enc: {enc_median:6.2f} us (±{enc_std:4.2f}) | "
                  f"Dec: {dec_median:6.2f} us (±{dec_std:4.2f}) | "
                  f"Overhead: {overhead} B | {throughput_mb_s:6.1f} MB/s")
                  
    # Export to CSV
    csv_path = out_path / "crypto_benchmark.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    print(f"\n[BENCHMARK] Exported benchmark results to {csv_path.resolve()}")
    
    return results


if __name__ == "__main__":
    run_crypto_benchmarks()
