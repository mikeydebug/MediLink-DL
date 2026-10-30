"""
src/crypto_research/diffusion/benchmark.py

Comprehensive Benchmarking, Multi-Objective Evaluation, and Registry Generator.
Academic Research Framework for MLMC Diffusion Architecture.

Runs:
  1. Full candidate suite evaluation across all 6 families and baselines.
  2. Software runtime benchmarking (latency per 20,000 blocks, throughput).
  3. Multi-objective Pareto scoring (Branch number vs XOR cost vs Involutory symmetry vs Diffusion speed).
  4. Exports structured JSON registry and comparative CSV records.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crypto_research.diffusion.matrix_generator import get_all_candidates, DiffusionCandidate
from src.crypto_research.diffusion.matrix_validation import matmul_gf2
from src.crypto_research.diffusion.diffusion_analysis import compute_diffusion_rounds_index
from src.crypto_research.diffusion.differential_analysis import evaluate_active_sboxes_bounds
from src.crypto_research.diffusion.linear_analysis import evaluate_linear_trails_bounds


def benchmark_software_throughput(
    M: np.ndarray,
    num_blocks: int = 20000,
    iterations: int = 5
) -> Dict[str, float]:
    """
    Measures pure software execution latency and throughput for computing y = M @ x over GF(2)
    across num_blocks random 16-bit blocks.
    """
    M_bin = (M % 2).astype(np.uint8)
    # Generate random test blocks: shape (16, num_blocks)
    blocks = np.random.randint(0, 2, size=(16, num_blocks), dtype=np.uint8)

    # Warm-up
    for _ in range(2):
        _ = matmul_gf2(M_bin, blocks)

    times_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        _ = matmul_gf2(M_bin, blocks)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)

    med_time_ms = float(np.median(times_ms))
    # Throughput in MB/s (16 bits = 2 bytes per block)
    total_bytes = num_blocks * 2
    throughput_mb_s = (total_bytes / (1024.0 * 1024.0)) / (med_time_ms / 1000.0) if med_time_ms > 0 else 0.0
    latency_per_block_ns = (med_time_ms * 1e6) / num_blocks

    return {
        "median_time_ms": round(med_time_ms, 3),
        "latency_per_block_ns": round(latency_per_block_ns, 2),
        "throughput_mb_s": round(throughput_mb_s, 2)
    }


def run_comprehensive_diffusion_benchmark(
    output_dir: str = "experiments/diffusion/results",
    registry_file: str = "experiments/diffusion/candidate_registry.json"
) -> List[Dict[str, Any]]:
    """
    Executes full evaluation on all baseline and candidate diffusion transformations.
    Saves outputs to CSV and candidate_registry.json.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    reg_path = Path(registry_file)
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = get_all_candidates()
    records: List[Dict[str, Any]] = []

    print("=" * 95)
    print(" MLMC-G16 Diffusion Architecture Research Benchmark Suite")
    print(f" Platform: {platform.system()} {platform.machine()} | Python {platform.python_version()}")
    print("=" * 95)

    for c in candidates:
        print(f"\nEvaluating: [{c.candidate_id}] {c.name}...")

        # 1. Diffusion speed
        diff_info = compute_diffusion_rounds_index(c.matrix_16x16, max_rounds=10)
        r_diff = diff_info["rounds_to_full_diffusion"]

        # 2. Differential bounds
        diff_bounds = evaluate_active_sboxes_bounds(c.matrix_16x16, max_rounds=14)

        # 3. Linear bounds
        lin_bounds = evaluate_linear_trails_bounds(c.matrix_16x16, max_rounds=14)

        # 4. Software benchmark
        sw_perf = benchmark_software_throughput(c.matrix_16x16, num_blocks=20000, iterations=5)

        # 5. Multi-objective research score (Pareto fitness)
        # Components:
        #   + B_nibble / 5.0 (weight 0.35)
        #   + B_bit / 8.0 (weight 0.25)
        #   + Involutory bonus (weight 0.20)
        #   - Optimized XOR / 100.0 (weight 0.10)
        #   - R_diff / 4.0 (weight 0.10)
        score = (
            0.35 * (c.branch_number_nibble / 5.0)
            + 0.25 * (c.branch_number_bit / 8.0)
            + 0.20 * (1.0 if c.is_involutory else 0.0)
            - 0.10 * (c.optimized_xor_count / 100.0)
            - 0.10 * (min(4, r_diff) / 4.0)
        )

        record = {
            "candidate_id": c.candidate_id,
            "name": c.name,
            "family": c.family,
            "branch_number_nibble": c.branch_number_nibble,
            "branch_number_bit": c.branch_number_bit,
            "linear_branch_nibble": c.linear_branch_number_nibble,
            "linear_branch_bit": c.linear_branch_number_bit,
            "is_involutory": c.is_involutory,
            "is_orthogonal": c.is_orthogonal,
            "direct_xor_count": c.direct_xor_count,
            "optimized_xor_count": c.optimized_xor_count,
            "ge_encryption_only": c.hardware_ge["ge_encryption_only"],
            "ge_combined_enc_dec": c.hardware_ge["ge_combined_enc_dec"],
            "rounds_to_full_diffusion": r_diff,
            "active_sboxes_r4": diff_bounds["active_sboxes_round_4"],
            "active_sboxes_r14": diff_bounds["active_sboxes_round_14"],
            "log2_diff_prob_r14": diff_bounds["log2_prob_round_14"],
            "log2_linear_bias_r14": lin_bounds["log2_bias_round_14"],
            "sw_latency_ns": sw_perf["latency_per_block_ns"],
            "sw_throughput_mb_s": sw_perf["throughput_mb_s"],
            "research_fitness_score": round(score, 4),
            "novelty_classification": c.novelty_classification,
            "novelty_rationale": c.novelty_rationale
        }
        records.append(record)

        print(f"  -> B_nibble={c.branch_number_nibble}, B_bit={c.branch_number_bit}, Involutory={c.is_involutory}, Opt_XOR={c.optimized_xor_count}, R_diff={r_diff}, Score={score:.4f}")

    # Sort by research fitness score descending
    records.sort(key=lambda x: x["research_fitness_score"], reverse=True)

    # Save to CSV
    csv_file = out_path / "diffusion_benchmark_results.csv"
    fieldnames = list(records[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Save complete candidate registry to JSON
    registry_data = [c.to_dict() for c in candidates]
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

    print(f"\n[OK] Benchmark results written to: {csv_file}")
    print(f"[OK] Candidate registry written to: {reg_path}")
    print("=" * 95 + "\n")

    return records


if __name__ == "__main__":
    run_comprehensive_diffusion_benchmark()

