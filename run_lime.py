"""
Compute LIME attributions for a trained SENN model on FashionMNIST.

Saves:
    - lime_attributions.pt         : full test-set attributions (N, 1, 28, 28)
    - lime_predictions.pt          : model predictions per sample  (N,)
    - lime_labels.pt               : ground-truth labels            (N,)
    - lime_ablation_drops.npy      : per-sample confidence drop after masking top-20% pixels
    - lime_sp_drop_top.npy         : per-sample drop masking top-20% superpixels (if enabled)
    - lime_sp_drop_rand.npy        : per-sample drop masking random-20% superpixels (if enabled)
    - lime_sp_drop_relative.npy    : top minus random superpixel drop (if enabled)
    - lime_meta.json               : timing + hyperparams

Usage:
    python run_lime.py --config configs/fashion_mnist_lambda1e-2_c5_seed29.json
    python run_lime.py --config configs/fashion_mnist_lambda1e-2_c5_seed29.json --max_images 200
    python run_lime.py --config configs/fashion_mnist_lambda1e-2_c5_seed29.json --use_superpixels
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from skimage.segmentation import slic

from senn.trainer import SENN_Trainer
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLinearRegression


# FashionMNIST normalisation constants
FMNIST_MEAN = 0.2860
FMNIST_STD  = 0.3530


# ── helpers ──────────────────────────────────────────────────────────────────

def load_senn(config_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)
    config["device"] = device
    config["train"] = False
    config = SimpleNamespace(**config)

    ckpt_path = Path("results") / config.exp_name / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    trainer = SENN_Trainer(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.model.eval()
    print(f"[LIME] Model loaded — best valid acc: {ckpt['best_accuracy']*100:.2f}%")
    return trainer


class SENNWrapper(nn.Module):
    """Expose only the log-softmax output for Captum."""
    def __init__(self, senn_model):
        super().__init__()
        self.senn = senn_model
    #from senn output (y_pred, (concepts, relevances), x_reconstructed), we want only the predictions
    def forward(self, x):
        predictions, _, _ = self.senn(x)
        return predictions


def pixel_ablation_confidence_drop(wrapper, images, attributions, pred_labels,
                                   top_fraction=0.20):
    """Mask top-k% pixels (by |attribution|) with the background value; return per-sample confidence drop."""
    
    # Valore del pixel nero (0.0 originale) dopo la normalizzazione: (0.0 - 0.2860) / 0.3530
    fill_value = -0.8102 
    # (fill_value = 0.0 per usare la media del dataset)

    wrapper.eval()
    with torch.no_grad():
        probs_orig = torch.softmax(wrapper(images), dim=1)
        conf_orig = probs_orig[torch.arange(len(pred_labels)), pred_labels]

        images_abl = images.clone()
        for i in range(len(images)):
            attr_flat = attributions[i].sum(dim=0).abs().flatten()
            k = int(top_fraction * len(attr_flat))
            topk_idx = attr_flat.topk(k).indices
            img_flat = images_abl[i].view(images.shape[1], -1)
            img_flat[:, topk_idx] = fill_value # Applica il nero/sfondo

        probs_abl = torch.softmax(wrapper(images_abl), dim=1)
        conf_abl = probs_abl[torch.arange(len(pred_labels)), pred_labels]

    return (conf_orig - conf_abl).cpu().numpy()


def build_superpixel_mask(image_tensor, num_segments=50, compactness=0.1):
    """Create a per-image superpixel mask compatible with Captum LIME."""
    img_denorm = image_tensor * FMNIST_STD + FMNIST_MEAN
    img_np = img_denorm.squeeze(0).detach().cpu().numpy()
    segments = slic(
        img_np,
        n_segments=num_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=None,
    )
    return torch.from_numpy(segments).long().unsqueeze(0).to(image_tensor.device)


def superpixel_ablation_confidence_drop(wrapper, image, attribution, pred_label,
                                        superpixel_mask, top_fraction=0.20,
                                        fill_value=-0.8102):
    """Mask top-k superpixels (by |attribution|) and random-k superpixels."""
    wrapper.eval()
    with torch.no_grad():
        probs_orig = torch.softmax(wrapper(image.unsqueeze(0)), dim=1)
        conf_orig = probs_orig[0, pred_label]

        sp_labels = superpixel_mask.squeeze(0)
        n_sp = int(sp_labels.max().item()) + 1
        k = max(1, int(top_fraction * n_sp))

        attr_map = attribution.sum(dim=0).abs()
        sp_scores = torch.zeros(n_sp, device=attr_map.device)
        for sp_id in range(n_sp):
            sp_scores[sp_id] = attr_map[sp_labels == sp_id].sum()

        top_sp = sp_scores.topk(k).indices
        rand_sp = torch.randperm(n_sp, device=attr_map.device)[:k]

        image_top = image.clone()
        image_rand = image.clone()
        for sp_id in top_sp:
            image_top[:, sp_labels == sp_id] = fill_value
        for sp_id in rand_sp:
            image_rand[:, sp_labels == sp_id] = fill_value

        conf_top = torch.softmax(wrapper(image_top.unsqueeze(0)), dim=1)[0, pred_label]
        conf_rand = torch.softmax(wrapper(image_rand.unsqueeze(0)), dim=1)[0, pred_label]

    return (conf_orig - conf_top).item(), (conf_orig - conf_rand).item()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LIME on a SENN model")
    parser.add_argument("--config", required=True, help="Path to SENN config JSON")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of LIME perturbation samples per image")
    parser.add_argument("--max_images", type=int, default=500,
                        help="Max test images to process (0 = all)")
    parser.add_argument("--use_superpixels", action="store_true",
                        help="Use SLIC superpixels as interpretable features")
    parser.add_argument("--sp_num_segments", type=int, default=50,
                        help="Number of SLIC superpixels per image")
    parser.add_argument("--sp_compactness", type=float, default=0.1,
                        help="SLIC compactness (higher = more regular superpixels)")
    parser.add_argument("--device", default="", help="Device (auto-detect if empty)")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[LIME] Device: {device}")

    # Load model
    trainer = load_senn(args.config, device=device)
    model = trainer.model
    test_loader = trainer.test_loader
    wrapper = SENNWrapper(model).to(device)
    wrapper.eval()

    lime_method = Lime(
        wrapper,
        interpretable_model=SkLearnLinearRegression(),
    )

    # Output dir
    with open(args.config, "r") as f:
        exp_name = json.load(f)["exp_name"]
    out_dir = Path("results") / exp_name / "posthoc_superpixels"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect attributions batch by batch
    all_attrs, all_preds, all_labels, all_drops = [], [], [], []
    all_sp_drop_top, all_sp_drop_rand = [], []
    n_processed = 0
    t_total = 0.0

    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.float().to(device)
        y = y.long().to(device)

        with torch.no_grad():
            preds = model(x)[0].argmax(1)

        # LIME is per-image
        batch_attrs = []
        t0 = time.perf_counter()
        for i in range(len(x)):
            img = x[i].unsqueeze(0)

            feature_mask = None
            if args.use_superpixels:
                feature_mask = build_superpixel_mask(
                    img.squeeze(0),
                    num_segments=args.sp_num_segments,
                    compactness=args.sp_compactness,
                ).unsqueeze(0)

            attr = lime_method.attribute(
                img,
                target=preds[i].item(),
                n_samples=args.n_samples,
                show_progress=False,
                feature_mask=feature_mask,
            )
            attr_map = attr.squeeze(0)
            batch_attrs.append(attr_map)

            if args.use_superpixels:
                drop_top, drop_rand = superpixel_ablation_confidence_drop(
                    wrapper,
                    x[i],
                    attr_map,
                    preds[i].item(),
                    feature_mask.squeeze(0),
                    fill_value=-0.8102,
                )
                all_sp_drop_top.append(drop_top)
                all_sp_drop_rand.append(drop_rand)
        batch_attrs = torch.stack(batch_attrs)
        t_total += time.perf_counter() - t0
        drops = pixel_ablation_confidence_drop(wrapper, x, batch_attrs, preds)

        all_attrs.append(batch_attrs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        all_drops.append(drops)

        n_processed += len(x)
        print(f"  Batch {batch_idx+1}: {n_processed} samples done "
              f"({t_total:.1f}s elapsed, "
              f"~{t_total/n_processed:.2f} s/sample)")

        if args.max_images > 0 and n_processed >= args.max_images:
            break

    all_attrs  = torch.cat(all_attrs)
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_drops  = np.concatenate(all_drops)

    # Save
    torch.save(all_attrs,  out_dir / "lime_attributions.pt")
    torch.save(all_preds,  out_dir / "lime_predictions.pt")
    torch.save(all_labels, out_dir / "lime_labels.pt")
    np.save(out_dir / "lime_ablation_drops.npy", all_drops)

    if args.use_superpixels:
        sp_drop_top = np.array(all_sp_drop_top)
        sp_drop_rand = np.array(all_sp_drop_rand)
        sp_drop_rel = sp_drop_top - sp_drop_rand
        np.save(out_dir / "lime_sp_drop_top.npy", sp_drop_top)
        np.save(out_dir / "lime_sp_drop_rand.npy", sp_drop_rand)
        np.save(out_dir / "lime_sp_drop_relative.npy", sp_drop_rel)

    meta = {
        "method": "LIME",
        "config": args.config,
        "exp_name": exp_name,
        "n_lime_samples": args.n_samples,
        "n_samples": int(len(all_labels)),
        "total_time_s": round(t_total, 3),
        "time_per_sample_s": round(t_total / len(all_labels), 5),
        "mean_confidence_drop": round(float(all_drops.mean()), 6),
        "std_confidence_drop": round(float(all_drops.std()), 6),
        "device": device,
        "use_superpixels": bool(args.use_superpixels),
        "sp_num_segments": args.sp_num_segments if args.use_superpixels else None,
        "sp_compactness": args.sp_compactness if args.use_superpixels else None,
    }
    if args.use_superpixels:
        meta["sp_drop_top_mean"] = round(float(sp_drop_top.mean()), 6)
        meta["sp_drop_rand_mean"] = round(float(sp_drop_rand.mean()), 6)
        meta["sp_drop_relative_mean"] = round(float(sp_drop_rel.mean()), 6)
    with open(out_dir / "lime_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[LIME] Done — {len(all_labels)} samples in {t_total:.1f}s")
    print(f"       Mean confidence drop (top-20% ablation): {all_drops.mean():.4f}")
    if args.use_superpixels:
        print(f"       Superpixel drop (top-20%): {sp_drop_top.mean():.4f}")
        print(f"       Superpixel drop (rand-20%): {sp_drop_rand.mean():.4f}")
        print(f"       Superpixel advantage (top-rand): {sp_drop_rel.mean():.4f}")
    print(f"       Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
