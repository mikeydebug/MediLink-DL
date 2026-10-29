"""optimization.py: Multi-Objective Cross-Layer Optimization across DL, DIP, and CN.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Executes an empirical multi-objective grid search across:
  - DL Localization: Ground-truth reference mask vs Trained Grad-CAM saliency mask
  - DIP Compression: Quality parameters (q_roi, q_bg) vs rate-matched uniform baseline
  - CN Transport: Priority Min-Heap vs Standard FIFO scheduling, and chunk sizes (512, 1024, 2048 B)
  - Cryptography: AES-256-GCM, AES-128-GCM, and Ascon-AEAD128 (NIST SP 800-232)

Evaluates actual measured distortion (PSNR/SSIM), wire overhead, triage latency,
crypto processing latency, and extracts the non-dominated Pareto frontier.
"""

from __future__ import annotations

import csv
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_utils import generate_synthetic_cxr_phantom
from src.roi_compression import compress_roi_aware, find_comparable_uniform_quality
from src.image_metrics import evaluate_quality_metrics
from src.security import get_aead_cipher
from src.transmission import simulate_transmission
from src.roi_localization import (
    build_demonstration_cxr_model,
    generate_roi_mask_from_heatmap
)
from src.localization_loss import DifferentiableGradCAM


def is_pareto_dominated(candidate: Dict[str, Any], others: List[Dict[str, Any]]) -> bool:
    """Check if candidate configuration is dominated by any other configuration.
    
    Objectives to MAXIMIZE:
      - roi_ssim
    Objectives to MINIMIZE:
      - total_time_sec
      - first_roi_sec
      - total_wire_bytes
      - crypto_ms
    """
    for o in others:
        if o is candidate:
            continue
        # Check if 'o' is at least as good in all objectives
        better_or_equal = (
            o["ssim_roi"] >= candidate["ssim_roi"] and
            o["total_time_sec"] <= candidate["total_time_sec"] and
            o["first_roi_time_sec"] <= candidate["first_roi_time_sec"] and
            o["total_wire_bytes"] <= candidate["total_wire_bytes"] and
            o["crypto_processing_time_ms"] <= candidate["crypto_processing_time_ms"]
        )
        # Check if 'o' is strictly better in at least one objective
        strictly_better = (
            o["ssim_roi"] > candidate["ssim_roi"] or
            o["total_time_sec"] < candidate["total_time_sec"] or
            o["first_roi_time_sec"] < candidate["first_roi_time_sec"] or
            o["total_wire_bytes"] < candidate["total_wire_bytes"] or
            o["crypto_processing_time_ms"] < candidate["crypto_processing_time_ms"]
        )
        if better_or_equal and strictly_better:
            return True
    return False


def run_cross_layer_optimization(
    output_dir: str = "results",
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute empirical cross-layer grid search and extract Pareto frontier."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(" MediLink-DL: Cross-Layer Multi-Objective Optimization (DL x DIP x CN x Crypto)")
    print("=" * 80)
    
    # Generate canonical test radiograph
    image, gt_mask, _ = generate_synthetic_cxr_phantom(width=256, height=256, seed=seed)
    
    # Load trained localization model for Grad-CAM ROI extraction
    model, target_layer, _ = build_demonstration_cxr_model()
    weights_path = out_path / "best_localization_model.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"[OPT] Loaded trained localization weights from {weights_path.name}")
    model.eval()
    
    gradcam = DifferentiableGradCAM(model, target_layer)
    img_t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.enable_grad():
        _, heatmaps = gradcam.compute_heatmap(img_t, target_classes=torch.tensor([1]), create_graph=False)
    gradcam.remove_hook()
    
    gc_heatmap = heatmaps.squeeze().detach().cpu().numpy()
    gradcam_mask = generate_roi_mask_from_heatmap(gc_heatmap, threshold=0.45)
    
    # Decision Variable Spaces
    roi_methods = [
        ("synthetic", gt_mask),
        ("gradcam", gradcam_mask)
    ]
    compression_configs = [
        ("ROI-Aware-High", 90, 25),
        ("ROI-Aware-Medium", 85, 20),
        ("ROI-Aware-Aggressive", 80, 15),
    ]
    chunk_sizes = [512, 1024, 2048]
    priority_modes = [True, False]
    ciphers = ["AES-256-GCM", "AES-128-GCM", "ASCON-128"]
    
    # Network Link Configuration: Rural 2.5G/3G link (64 kbps, 120ms latency, 5% drop)
    bw_bps = 64 * 1024
    lat_ms = 120.0
    loss = 0.05
    
    all_results = []
    
    total_combinations = len(roi_methods) * len(compression_configs) * len(chunk_sizes) * len(priority_modes) * len(ciphers)
    print(f"[OPT] Evaluating {total_combinations} full pipeline configurations under measured metrics...")
    
    count = 0
    for (r_name, mask_arr), (c_label, q_roi, q_bg), chunk_sz, prio, cipher_name in itertools.product(
        roi_methods, compression_configs, chunk_sizes, priority_modes, ciphers
    ):
        count += 1
        # Step 1: Compress
        recon, payload, c_info = compress_roi_aware(image, mask_arr, q_roi=q_roi, q_bg=q_bg)
        
        # Step 2: Quality metrics
        q_metrics = evaluate_quality_metrics(image, recon, mask=mask_arr)
        
        # Step 3: Transmit with Authenticated Encryption
        tx = simulate_transmission(
            payload_bytes=payload,
            bandwidth_bps=bw_bps,
            latency_ms=lat_ms,
            loss_rate=loss,
            chunk_size=chunk_sz,
            priority_mode=prio,
            crypto_algorithm=cipher_name,
            seed=seed + count
        )
        
        # Hard constraint check: must verify cryptographic and transmission integrity
        if not tx["transmission_successful"] or not tx["integrity_verified"]:
            print(f"      [WARN] Config {count} failed integrity checks - rejected from feasible set.")
            continue
            
        first_roi = float(tx["first_roi_time_sec"]) if tx["first_roi_time_sec"] != "N/A" else float(tx["total_time_sec"])
        
        row = {
            "config_id": count,
            "roi_source": r_name,
            "compression_profile": c_label,
            "q_roi": q_roi,
            "q_bg": q_bg,
            "chunk_size_bytes": chunk_sz,
            "priority_scheduling": prio,
            "crypto_algorithm": cipher_name,
            "compressed_payload_bytes": len(payload),
            "total_wire_bytes": tx["total_bytes_transmitted"],
            "wire_overhead_ratio": tx["wire_overhead_ratio"],
            "psnr_global_db": q_metrics["psnr_global_db"],
            "ssim_global": q_metrics["ssim_global"],
            "psnr_roi_db": q_metrics["psnr_roi_db"],
            "ssim_roi": q_metrics["ssim_roi"],
            "psnr_bg_db": q_metrics["psnr_bg_db"],
            "ssim_bg": q_metrics["ssim_bg"],
            "total_time_sec": tx["total_time_sec"],
            "first_roi_time_sec": round(first_roi, 3),
            "crypto_processing_time_ms": tx["crypto_processing_time_ms"],
            "retransmissions": tx["retransmissions"],
            "throughput_kbps": tx["effective_throughput_kbps"]
        }
        all_results.append(row)
        
    print(f"[OPT] Successfully completed {len(all_results)} feasible measurements.")
    
    # Export full optimization results
    opt_csv = out_path / "optimization_results.csv"
    if all_results:
        with open(opt_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
    print(f"[OPT] Exported full configuration dataset to {opt_csv}")
    
    # Compute Pareto Frontier
    pareto_frontier = [r for r in all_results if not is_pareto_dominated(r, all_results)]
    print(f"[OPT] Identified {len(pareto_frontier)} non-dominated Pareto-optimal configurations.")
    
    pareto_csv = out_path / "pareto_frontier.csv"
    if pareto_frontier:
        with open(pareto_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pareto_frontier[0].keys()))
            writer.writeheader()
            writer.writerows(pareto_frontier)
    print(f"[OPT] Exported Pareto frontier to {pareto_csv}")
    
    # Generate Pareto Visualization
    _generate_pareto_plot(all_results, pareto_frontier, out_path / "pareto_frontier.png")
    
    return all_results, pareto_frontier


def _generate_pareto_plot(
    all_res: List[Dict[str, Any]],
    pareto_res: List[Dict[str, Any]],
    output_fig: Path
) -> None:
    """Generate publication-ready 2-panel trade-off scatter plot showing the Pareto frontier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Panel 1: ROI SSIM vs First ROI Triage Latency (Priority Benefit)
    x_all = [r["first_roi_time_sec"] for r in all_res]
    y_all = [r["ssim_roi"] for r in all_res]
    ax1.scatter(x_all, y_all, c="#94a3b8", alpha=0.45, s=40, label="Feasible Configurations")
    
    x_p = [r["first_roi_time_sec"] for r in pareto_res]
    y_p = [r["ssim_roi"] for r in pareto_res]
    ax1.scatter(x_p, y_p, c="#dc2626", s=90, edgecolors="#7f1d1d", linewidths=1.5, zorder=5, label="Pareto Frontier")
    ax1.set_xlabel("First ROI Arrival Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Diagnostic ROI SSIM", fontsize=11, fontweight="bold")
    ax1.set_title("Multi-Objective Trade-Off: ROI Fidelity vs Triage Speed", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")
    
    # Panel 2: Total Wire Bytes vs End-to-End Latency
    b_all = [r["total_wire_bytes"] / 1024.0 for r in all_res]
    t_all = [r["total_time_sec"] for r in all_res]
    ax2.scatter(b_all, t_all, c="#94a3b8", alpha=0.45, s=40, label="Feasible Configurations")
    
    b_p = [r["total_wire_bytes"] / 1024.0 for r in pareto_res]
    t_p = [r["total_time_sec"] for r in pareto_res]
    ax2.scatter(b_p, t_p, c="#2563eb", s=90, edgecolors="#1e3a8a", linewidths=1.5, zorder=5, label="Pareto Frontier")
    ax2.set_xlabel("Total Transmitted Payload (KB)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Complete Delivery Time (seconds)", fontsize=11, fontweight="bold")
    ax2.set_title("Payload Size vs Complete Delivery Time", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_fig, bbox_inches="tight")
    plt.close()
    print(f"[OPT] Saved Pareto trade-off figure to {output_fig}")


if __name__ == "__main__":
    run_cross_layer_optimization()
