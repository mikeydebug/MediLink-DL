"""
Training and Evaluation Pipeline on Genuine Chest X-rays.

Evaluates 4 Experimental Regimes:
- Regime A: Classification-Only Baseline (L = L_cls)
- Regime B: Combined Classification + Alignment + Dice Loss (L = L_cls + lambda * (L_align + L_dice))
- Regime C: Alignment-Only Ablation (L = L_cls + lambda * L_align)
- Regime D: Dice-Only Ablation (L = L_cls + lambda * L_dice)

Hardware: Supports Apple Silicon MPS, CUDA, and CPU.
Guarantees: Strict patient-level stratification (zero patient leakage).
"""

import sys
import os
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import csv
import copy
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset_real import get_real_cxr_dataloaders, RealCXRDataset
from src.localization_loss import (
    DifferentiableGradCAM,
    LocalizationAwareLoss,
    evaluate_saliency_metrics
)


class MedicalResNet18(nn.Module):
    """
    ResNet-18 adapted for 1-channel grayscale medical radiographs.
    Target layer for Grad-CAM: backbone.layer4[-1]
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        # Use torchvision ResNet-18 architecture
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Modify conv1 from 3 channels to 1 channel (grayscale CXR)
        orig_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(orig_conv1.weight.mean(dim=1, keepdim=True))
                
        # Adapt final classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # Target layer for Grad-CAM
        self.target_layer = self.backbone.layer4[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def compute_binary_metrics(y_true: List[int], y_preds: List[int], y_scores: List[float]) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, specificity, F1, and ROC-AUC.
    """
    y_true_arr = np.array(y_true)
    y_preds_arr = np.array(y_preds)
    
    tp = int(np.sum((y_true_arr == 1) & (y_preds_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_preds_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_preds_arr == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_preds_arr == 0)))
    
    total = len(y_true_arr)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # Compute ROC-AUC via trapezoidal integration
    desc_idx = np.argsort(y_scores)[::-1]
    sorted_true = y_true_arr[desc_idx]
    n_pos = np.sum(sorted_true == 1)
    n_neg = np.sum(sorted_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        tps = np.cumsum(sorted_true == 1)
        fps = np.cumsum(sorted_true == 0)
        tpr = np.r_[0, tps / n_pos]
        fpr = np.r_[0, fps / n_neg]
        auc = float(np.trapz(tpr, fpr))
        
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "roc_auc": float(auc)
    }


def compute_pointing_game_accuracy(heatmaps: np.ndarray, masks: np.ndarray) -> float:
    """
    Computes Pointing Game Accuracy (Hit rate).
    A Hit occurs if the max heatmap coordinate falls inside the ground-truth mask.
    """
    hits = 0
    total = len(heatmaps)
    if total == 0:
        return 0.0
        
    for i in range(total):
        h = heatmaps[i, 0]  # (H, W)
        m = masks[i, 0]     # (H, W)
        
        if m.max() == 0:
            # Empty mask
            continue
            
        peak_y, peak_x = np.unravel_index(np.argmax(h), h.shape)
        if m[peak_y, peak_x] > 0.5:
            hits += 1
            
    return float(hits / total)


def train_single_regime(
    regime_name: str,
    loss_mode: str,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-4,
    lambda_loc: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains and evaluates a single experimental regime.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = MedicalResNet18(num_classes=2, pretrained=False).to(device)
    gradcam = DifferentiableGradCAM(model, model.target_layer)
    criterion = LocalizationAwareLoss(lambda_loc=lambda_loc, mode=loss_mode).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_score = -1.0
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_acc": []}
    
    print(f"\n--- Training {regime_name} (Mode: {loss_mode}) on {device} ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_batches = 0
        
        for imgs, labels, masks, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            if loss_mode == "cls_only":
                logits = model(imgs)
                # dummy cam for loss signature
                cam = torch.zeros_like(masks)
                loss_dict = criterion(logits, labels, cam, masks)
            else:
                logits, cam = gradcam.compute_heatmap(imgs, target_classes=labels, create_graph=True)
                loss_dict = criterion(logits, labels, cam, masks)
                
            loss = loss_dict["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_accum += loss.item()
            train_batches += 1
            
        avg_train_loss = train_loss_accum / max(1, train_batches)
        
        # Validation evaluation
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        val_dices = []
        val_correct = 0
        val_total = 0
        
        for imgs, labels, masks, _ in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                
            # Compute evaluation Grad-CAM
            with torch.enable_grad():
                _, eval_cam = gradcam.compute_heatmap(imgs, target_classes=labels, create_graph=False)
                loss_dict = criterion(logits, labels, eval_cam, masks)
                
            val_loss_accum += loss_dict["loss_total"].item()
            val_batches += 1
            
            # Measure continuous Dice
            cam_np = eval_cam.detach().cpu().numpy()
            mask_np = masks.detach().cpu().numpy()
            for b in range(len(cam_np)):
                metrics = evaluate_saliency_metrics(cam_np[b, 0], mask_np[b, 0], threshold=0.35)
                val_dices.append(metrics["dice"])
                
        avg_val_loss = val_loss_accum / max(1, val_batches)
        avg_val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        val_acc = val_correct / max(1, val_total)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_acc"].append(val_acc)
        
        # Selection criterion: validation accuracy + localization dice
        val_score = val_acc + avg_val_dice if loss_mode != "cls_only" else val_acc
        if val_score > best_val_score:
            best_val_score = val_score
            best_weights = copy.deepcopy(model.state_dict())
            
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Dice: {avg_val_dice:.4f}")
        
    duration = time.time() - start_time
    print(f"Completed {regime_name} in {duration:.1f}s. Best Val Score: {best_val_score:.4f}")
    
    # Load best model weights for held-out test evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    # Final evaluation on held-out test split
    model.eval()
    test_true = []
    test_preds = []
    test_scores = []
    test_heatmaps = []
    test_masks = []
    test_imgs_list = []
    test_dices = []
    test_aligns = []
    test_ious = []
    
    for imgs, labels, masks, metas in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        with torch.no_grad():
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
        # Evaluation Grad-CAM
        with torch.enable_grad():
            _, eval_cam = gradcam.compute_heatmap(imgs, target_classes=labels, create_graph=False)
            
        cams_np = eval_cam.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        imgs_np = imgs.detach().cpu().numpy()
        
        test_true.extend(labels.cpu().tolist())
        test_preds.extend(preds.cpu().tolist())
        test_scores.extend(probs[:, 1].cpu().tolist())
        test_heatmaps.append(cams_np)
        test_masks.append(masks_np)
        test_imgs_list.append(imgs_np)
        
        for b in range(len(cams_np)):
            m_res = evaluate_saliency_metrics(cams_np[b, 0], masks_np[b, 0], threshold=0.35)
            test_dices.append(m_res["dice"])
            test_aligns.append(m_res["energy_inside_roi"])
            test_ious.append(m_res["iou"])
            
    all_heatmaps = np.concatenate(test_heatmaps, axis=0)
    all_masks = np.concatenate(test_masks, axis=0)
    all_imgs = np.concatenate(test_imgs_list, axis=0)
    
    cls_metrics = compute_binary_metrics(test_true, test_preds, test_scores)
    pointing_game_acc = compute_pointing_game_accuracy(all_heatmaps, all_masks)
    
    results = {
        "regime": regime_name,
        "loss_mode": loss_mode,
        "test_accuracy": cls_metrics["accuracy"],
        "test_precision": cls_metrics["precision"],
        "test_recall": cls_metrics["recall"],
        "test_specificity": cls_metrics["specificity"],
        "test_f1": cls_metrics["f1"],
        "test_roc_auc": cls_metrics["roc_auc"],
        "test_roi_dice": float(np.mean(test_dices)),
        "test_roi_iou": float(np.mean(test_ious)),
        "test_roi_alignment": float(np.mean(test_aligns)),
        "pointing_game_acc": pointing_game_acc,
        "training_time_s": duration,
        "model_state": best_weights,
        "history": history,
        "sample_heatmaps": all_heatmaps,
        "sample_masks": all_masks,
        "sample_imgs": all_imgs
    }
    
    return results


def run_all_real_data_experiments(
    data_dir: str = "data/real_cxr",
    results_dir: str = "results",
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1e-4,
    lambda_loc: float = 1.0,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Executes all 4 regimes on the real COVID-19 Chest X-ray dataset.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Executing Deep Learning experiments on compute device: {device}")
    
    train_loader, val_loader, test_loader = get_real_cxr_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        target_size=(224, 224),
        in_channels=1
    )
    
    regimes = [
        ("Regime A (Classification-Only)", "cls_only"),
        ("Regime B (Combined Loss)", "combined"),
        ("Regime C (Alignment-Only)", "align_only"),
        ("Regime D (Dice-Only)", "dice_only"),
    ]
    
    all_results = []
    for name, mode in regimes:
        res = train_single_regime(
            regime_name=name,
            loss_mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            lambda_loc=lambda_loc,
            seed=seed
        )
        all_results.append(res)
        
    # Save checkpoints:
    # 1. Baseline checkpoint (Regime A)
    baseline_res = all_results[0]
    torch.save(baseline_res["model_state"], os.path.join(results_dir, "real_cxr_classifier.pt"))
    print(f"Saved baseline classifier checkpoint to {results_dir}/real_cxr_classifier.pt")
    
    # 2. Best localization model (highest test_roi_dice)
    best_loc_res = max(all_results, key=lambda r: r["test_roi_dice"])
    torch.save(best_loc_res["model_state"], os.path.join(results_dir, "real_cxr_localization_model.pt"))
    print(f"Saved best localization model ({best_loc_res['regime']}) checkpoint to {results_dir}/real_cxr_localization_model.pt")
    
    # Export results to CSV
    csv_path = os.path.join(results_dir, "real_dataset_training_results.csv")
    fieldnames = [
        "regime", "loss_mode", "test_accuracy", "test_precision", "test_recall",
        "test_specificity", "test_f1", "test_roc_auc", "test_roi_dice",
        "test_roi_iou", "test_roi_alignment", "pointing_game_acc", "training_time_s"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: round(r[k], 4) if isinstance(r[k], float) else r[k] for k in fieldnames})
    print(f"Saved experiment results to {csv_path}")
    
    # Generate Training Curves Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]
    
    for idx, r in enumerate(all_results):
        ep_range = range(1, len(r["history"]["train_loss"]) + 1)
        axes[0].plot(ep_range, r["history"]["train_loss"], label=r["regime"], color=colors[idx], lw=2)
        axes[1].plot(ep_range, r["history"]["val_loss"], label=r["regime"], color=colors[idx], lw=2)
        axes[2].plot(ep_range, r["history"]["val_dice"], label=r["regime"], color=colors[idx], lw=2)
        
    axes[0].set_title("Training Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    
    axes[1].set_title("Validation Loss vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title("Validation ROI Dice vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("ROI Dice Score")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_plot_path = os.path.join(results_dir, "real_cxr_training_curves.png")
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {curve_plot_path}")
    
    # Generate Visual Comparison Plot across Test Samples
    # Display 4 samples from test split
    base_heatmaps = baseline_res["sample_heatmaps"]
    best_heatmaps = best_loc_res["sample_heatmaps"]
    test_masks = baseline_res["sample_masks"]
    test_imgs = baseline_res["sample_imgs"]
    
    n_vis = min(4, len(test_imgs))
    fig, axes = plt.subplots(n_vis, 4, figsize=(14, 3.5 * n_vis))
    if n_vis == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(n_vis):
        # Denormalize image: (x * 0.229) + 0.485
        img_raw = np.clip((test_imgs[i, 0] * 0.229) + 0.485, 0.0, 1.0)
        gt_mask = test_masks[i, 0]
        base_cam = base_heatmaps[i, 0]
        best_cam = best_heatmaps[i, 0]
        
        # Col 0: Original Radiograph
        axes[i, 0].imshow(img_raw, cmap="gray")
        axes[i, 0].set_title(f"Test CXR #{i+1}: Original")
        axes[i, 0].axis("off")
        
        # Col 1: Ground-truth Target Bounding Box Mask
        axes[i, 1].imshow(img_raw, cmap="gray")
        axes[i, 1].imshow(gt_mask, cmap="Reds", alpha=0.45)
        axes[i, 1].set_title("Ground-Truth Lung ROI")
        axes[i, 1].axis("off")
        
        # Col 2: Baseline Grad-CAM Heatmap Overlay
        axes[i, 2].imshow(img_raw, cmap="gray")
        axes[i, 2].imshow(base_cam, cmap="jet", alpha=0.5)
        m_base = evaluate_saliency_metrics(base_cam, gt_mask, threshold=0.35)
        axes[i, 2].set_title(f"Baseline (Dice: {m_base['dice']:.2f})")
        axes[i, 2].axis("off")
        
        # Col 3: Localization-Aware Grad-CAM Heatmap Overlay
        axes[i, 3].imshow(img_raw, cmap="gray")
        axes[i, 3].imshow(best_cam, cmap="jet", alpha=0.5)
        m_best = evaluate_saliency_metrics(best_cam, gt_mask, threshold=0.35)
        axes[i, 3].set_title(f"Proposed Loc-Aware (Dice: {m_best['dice']:.2f})")
        axes[i, 3].axis("off")
        
    plt.tight_layout()
    vis_plot_path = os.path.join(results_dir, "real_cxr_gradcam_comparison.png")
    plt.savefig(vis_plot_path, dpi=300)
    plt.close()
    print(f"Saved qualitative visualization to {vis_plot_path}")
    
    return all_results


if __name__ == "__main__":
    run_all_real_data_experiments(epochs=8, batch_size=8, lr=1e-4, lambda_loc=1.0)
