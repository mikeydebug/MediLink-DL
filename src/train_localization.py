"""train_localization.py: Controlled Localization-Aware vs Classification-Only Training.

Part of MediLink-DL: Bandwidth-Adaptive Medical Image Transmission.
Empirically investigates whether adding a differentiable localization penalty (alignment + soft-Dice)
to CNN training guides Grad-CAM saliency toward the true diagnostic lesion mask.

EXPERIMENTAL REGIMES EVALUATED:
  Regime A: Classification-only (lambda_loc = 0.0) - Standard cross-entropy baseline
  Regime B: Combined localization (lambda_loc = 1.0, beta_dice = 0.5) - Alignment + Dice
  Regime C: Alignment-only loss (lambda_loc = 1.0) - Energy penalty
  Regime D: Soft-Dice-only loss (lambda_loc = 1.0) - Shape overlap penalty
"""

from __future__ import annotations

import os
import sys
import time
import csv
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.image_utils import generate_synthetic_cxr_phantom
from src.roi_localization import build_demonstration_cxr_model
from src.localization_loss import (
    DifferentiableGradCAM,
    LocalizationAwareLoss,
    evaluate_saliency_metrics
)


def create_synthetic_dataset(
    num_samples: int = 40,
    width: int = 128,
    height: int = 128,
    base_seed: int = 100
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate balanced synthetic CXR dataset with varying lesion coordinates.
    
    Returns:
        List of tuples: (image_tensor [1, 1, H, W], mask_tensor [1, 1, H, W], label_tensor [1])
    """
    dataset = []
    rng = np.random.default_rng(base_seed)
    
    for i in range(num_samples):
        has_lesion = (i % 2 == 1)
        seed = base_seed + i * 13
        
        # Position lesion within plausible lung field bounds
        row_norm = rng.uniform(0.35, 0.70)
        col_norm = rng.choice([rng.uniform(0.20, 0.40), rng.uniform(0.60, 0.80)])
        center = (int(height * row_norm), int(width * col_norm))
        radius = int(width * rng.uniform(0.06, 0.12))
        
        img, mask, _ = generate_synthetic_cxr_phantom(
            width=width,
            height=height,
            seed=seed,
            add_lesion=has_lesion,
            lesion_center=center,
            lesion_radius=radius
        )
        
        # Convert to tensors
        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, H, W) in [0, 1]
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        target_t = torch.tensor([1 if has_lesion else 0], dtype=torch.long)
        
        dataset.append((img_t, mask_t, target_t))
        
    return dataset


def train_single_regime(
    regime_name: str,
    loss_mode: str,
    lambda_loc: float,
    beta_dice: float,
    train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 15,
    lr: float = 0.002,
    seed: int = 42
) -> Dict[str, Any]:
    """Train a demonstration model under a specified loss configuration.
    
    Employs validation checkpoint tracking and gradient norm clipping to ensure
    stable second-order backpropagation through the DifferentiableGradCAM graph.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model, target_layer, meta = build_demonstration_cxr_model()
    model.train()
    
    gradcam = DifferentiableGradCAM(model, target_layer)
    criterion = LocalizationAwareLoss(
        mode=loss_mode,
        lambda_loc=lambda_loc,
        beta_dice=beta_dice
    )
    trainable_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    
    start_time = time.time()
    train_losses = []
    
    best_score = -1.0
    best_state = None
    best_metrics = None
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        # Permute training data
        perm = np.random.permutation(len(train_data))
        
        # Batch size 4
        batch_size = 4
        for b_start in range(0, len(train_data), batch_size):
            indices = perm[b_start:b_start + batch_size]
            b_imgs = torch.cat([train_data[idx][0] for idx in indices], dim=0)
            b_masks = torch.cat([train_data[idx][1] for idx in indices], dim=0)
            b_targets = torch.cat([train_data[idx][2] for idx in indices], dim=0)
            
            optimizer.zero_grad()
            logits, heatmaps = gradcam.compute_heatmap(
                b_imgs,
                target_classes=b_targets,
                create_graph=(lambda_loc > 0.0)
            )
            
            loss_dict = criterion(logits, b_targets, heatmaps, b_masks)
            loss_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss_dict["loss_total"].item() * len(indices)
            
        train_losses.append(epoch_loss / len(train_data))
        
        # Validation Evaluation per epoch for checkpoint selection
        model.eval()
        val_correct = 0
        val_total = len(val_data)
        val_loss = 0.0
        
        dices, ious, pointing_hits, energies = [], [], [], []
        sample_heatmaps = []
        
        with torch.enable_grad():
            for img_t, mask_t, target_t in val_data:
                logits, heatmaps = gradcam.compute_heatmap(
                    img_t,
                    target_classes=target_t,
                    create_graph=False
                )
                pred = torch.argmax(logits, dim=1).item()
                if pred == target_t.item():
                    val_correct += 1
                    
                loss_dict = criterion(logits, target_t, heatmaps, mask_t)
                val_loss += loss_dict["loss_total"].item()
                
                h_np = heatmaps.squeeze().detach().cpu().numpy()
                m_np = mask_t.squeeze().cpu().numpy()
                sample_heatmaps.append((h_np, m_np, img_t.squeeze().cpu().numpy()))
                
                # Evaluate localization metrics for pathology cases (label == 1)
                if target_t.item() == 1:
                    metrics = evaluate_saliency_metrics(h_np, m_np, threshold=0.35)
                    dices.append(metrics["dice"])
                    ious.append(metrics["iou"])
                    pointing_hits.append(metrics["pointing_game_hit"])
                    energies.append(metrics["energy_inside_roi"])
                    
        curr_dice = float(np.mean(dices)) if dices else 0.0
        score = curr_dice if lambda_loc > 0.0 else (val_correct / val_total)
        if score > best_score or best_state is None:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_loss": round(val_loss / val_total, 4),
                "val_accuracy": round(val_correct / val_total, 4),
                "mean_roi_dice": round(curr_dice, 4),
                "mean_roi_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
                "pointing_game_acc": round(float(np.mean(pointing_hits)), 4) if pointing_hits else 0.0,
                "energy_in_roi": round(float(np.mean(energies)), 4) if energies else 0.0,
                "sample_heatmaps": sample_heatmaps
            }
            
    elapsed = time.time() - start_time
    gradcam.remove_hook()
    
    # Restore best validated checkpoint weights
    model.load_state_dict(best_state)
    
    return {
        "regime": regime_name,
        "loss_mode": loss_mode,
        "lambda_loc": lambda_loc,
        "beta_dice": beta_dice,
        "epochs": epochs,
        "runtime_sec": round(elapsed, 3),
        "final_train_loss": round(train_losses[-1], 4),
        "val_loss": best_metrics["val_loss"],
        "val_accuracy": best_metrics["val_accuracy"],
        "mean_roi_dice": best_metrics["mean_roi_dice"],
        "mean_roi_iou": best_metrics["mean_roi_iou"],
        "pointing_game_acc": best_metrics["pointing_game_acc"],
        "energy_in_roi": best_metrics["energy_in_roi"],
        "model_state": best_state,
        "sample_heatmaps": best_metrics["sample_heatmaps"]
    }


def run_localization_experiments(
    output_dir: str = "results",
    seed: int = 42
) -> pd.DataFrame:
    """Run all 4 training regimes, save CSV, figures, and best model weights."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("[DL-EXP] Generating synthetic CXR dataset (60 train, 20 val)...")
    train_data = create_synthetic_dataset(num_samples=60, width=128, height=128, base_seed=seed)
    val_data = create_synthetic_dataset(num_samples=20, width=128, height=128, base_seed=seed + 500)
    
    regimes = [
        ("A: Classification-Only", "cls_only", 0.0, 0.0),
        ("B: Combined Loc Loss", "combined", 0.5, 2.0),
        ("C: Alignment-Only", "align_only", 0.3, 0.0),
        ("D: Dice-Only", "dice_only", 0.5, 0.0)
    ]
    
    results = []
    trained_artifacts = {}
    
    print(f"[DL-EXP] Benchmarking {len(regimes)} training regimes on CPU...")
    for name, mode, lam, beta in regimes:
        print(f"         Running Regime '{name}' (lambda={lam}, beta={beta})...")
        res = train_single_regime(
            regime_name=name,
            loss_mode=mode,
            lambda_loc=lam,
            beta_dice=beta,
            train_data=train_data,
            val_data=val_data,
            epochs=15,
            lr=0.002,
            seed=seed
        )
        trained_artifacts[name] = res
        results.append({
            "Regime": res["regime"],
            "Loss_Mode": res["loss_mode"],
            "Lambda_Loc": res["lambda_loc"],
            "Beta_Dice": res["beta_dice"],
            "Val_Accuracy": res["val_accuracy"],
            "ROI_Dice": res["mean_roi_dice"],
            "ROI_IoU": res["mean_roi_iou"],
            "Pointing_Game_Acc": res["pointing_game_acc"],
            "Energy_in_ROI": res["energy_in_roi"],
            "Final_Train_Loss": res["final_train_loss"],
            "Val_Loss": res["val_loss"],
            "Runtime_Sec": res["runtime_sec"]
        })
        print(f"         -> Accuracy: {res['val_accuracy']*100:.1f}%, Dice: {res['mean_roi_dice']:.4f}, "
              f"IoU: {res['mean_roi_iou']:.4f}, Pointing: {res['pointing_game_acc']*100:.1f}%, Time: {res['runtime_sec']}s")
    csv_file = out_path / "gradcam_localization_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    print(f"[DL-EXP] Saved empirical metrics to {csv_file}")
    
    # Save the best localization-aware model state dict (highest ROI Dice)
    best_name = max(trained_artifacts.keys(), key=lambda k: trained_artifacts[k]["mean_roi_dice"])
    best_regime = trained_artifacts[best_name]
    best_weights_file = out_path / "best_localization_model.pt"
    torch.save(best_regime["model_state"], best_weights_file)
    print(f"[DL-EXP] Saved best localization model weights ({best_name}) to {best_weights_file}")
    
    # Generate 5-panel comparative visualization
    fig_file = out_path / "gradcam_training_comparison.png"
    _generate_comparison_figure(
        trained_artifacts["A: Classification-Only"],
        best_regime,
        fig_file
    )
    print(f"[DL-EXP] Generated visual comparison figure at {fig_file}")
    
    return results


def _generate_comparison_figure(
    baseline_res: Dict[str, Any],
    loc_res: Dict[str, Any],
    save_path: Path
) -> None:
    """Generate high-resolution 5-panel figure comparing baseline vs localization-trained Grad-CAM."""
    # Find a validation sample with label == 1 (pathology)
    sample_idx = 1  # Index with ground truth lesion
    h_base, m_gt, img = baseline_res["sample_heatmaps"][sample_idx]
    h_loc, _, _ = loc_res["sample_heatmaps"][sample_idx]
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), dpi=300)
    
    # Panel 1: Grayscale Input
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("(a) Synthetic CXR Input", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 2: Ground Truth ROI Mask
    axes[1].imshow(m_gt, cmap="Blues_r", vmin=0, vmax=1)
    axes[1].set_title("(b) Reference ROI Mask M", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    # Panel 3: Baseline Grad-CAM (Classification Only)
    im3 = axes[2].imshow(h_base, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title(f"(c) Baseline Grad-CAM\n(Dice={evaluate_saliency_metrics(h_base, m_gt)['dice']:.2f})", fontsize=10)
    axes[2].axis("off")
    
    # Panel 4: Localization-Aware Grad-CAM (Combined Loss)
    axes[3].imshow(h_loc, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title(f"(d) Localization-Aware\n(Dice={evaluate_saliency_metrics(h_loc, m_gt)['dice']:.2f})", fontsize=10, fontweight="bold")
    axes[3].axis("off")
    
    # Panel 5: Difference Map (Localization - Baseline)
    diff = h_loc - h_base
    im5 = axes[4].imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    axes[4].set_title("(e) Heatmap Shift (d - c)\n(Red=Gain, Blue=Suppressed)", fontsize=10)
    axes[4].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_localization_experiments()
