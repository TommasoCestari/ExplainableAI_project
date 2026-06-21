import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Keep these aligned with your dataloader normalization
MEAN = 0.2860
STD = 0.3530

def denormalize(x_norm, mean=MEAN, std=STD):
    # normalized -> pixel space [0,1]
    return x_norm * std + mean


def normalize(x_pix, mean=MEAN, std=STD):
    # pixel space [0,1] -> normalized space used by the model
    return (x_pix - mean) / std


def ensure_batch(x, device):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise ValueError("Input must be [1,28,28] or [1,1,28,28]")
    return x.to(device)


def add_white_noise(x_norm, sigma=0.05, mean=MEAN, std=STD):
    # normalized -> pixel space
    x_pix = denormalize(x_norm, mean, std)

    # add Gaussian white noise in pixel space
    noise = torch.randn_like(x_pix) * sigma # same shape as x_pix with noise[i, j] ~ N(0, 1)
    x_noisy_pix = x_pix + noise

    # clip to valid pixel range
    x_noisy_pix = torch.clamp(x_noisy_pix, 0.0, 1.0) # for visualization
    # pixel -> normalized (for model input)
    x_noisy_norm = normalize(x_noisy_pix, mean, std) # for the model

    return x_noisy_norm, x_noisy_pix


def plot_stability_comparison(
    model,
    x_i_norm,   # [1,28,28] or [1,1,28,28]
    x_j_norm,   # [1,28,28] or [1,1,28,28]
    mean,
    std,
):
    model.eval()
    device = next(model.parameters()).device

    x_i_b = ensure_batch(x_i_norm, device)
    x_j_b = ensure_batch(x_j_norm, device)

    # If a batch is passed accidentally, keep first neighbor for the plot
    if x_j_b.size(0) > 1:
        x_j_b = x_j_b[:1]

    inputs = torch.cat([x_i_b, x_j_b], dim=0)  # [2,1,28,28]
    with torch.no_grad():
        y_logp, (_, relevances), _ = model(inputs)
        preds = y_logp.argmax(dim=1)

    c_i = preds[0].item()
    if preds[1].item() != c_i:
        print(f"[Warning] Prediction changed: {c_i} → {preds[1].item()}")

    n_concepts = relevances.shape[1]
    if n_concepts == 28 * 28:
        rel_i = relevances[0, :, c_i].view(28, 28).cpu() # Take the relevances for all concepts (that correspond to pixels) for one class
        rel_j = relevances[1, :, c_i].view(28, 28).cpu()
    else:
        raise ValueError(
            f"Expected 784 concepts for pixel alignment, got {n_concepts}"
        )

    img_i = (x_i_b[0, 0].cpu() * std + mean).clamp(0, 1) # [28, 28]
    img_j = (x_j_b[0, 0].cpu() * std + mean).clamp(0, 1)

    vals = torch.cat([rel_i.flatten(), rel_j.flatten()])
    max_abs = torch.quantile(torch.abs(vals), 0.99).item() # without this if rel_i was in the range [-1, 1] and rel_j in [-10, 10] they would look equally strong
    # 99th quantile so we avoid extreme pixels from dominating making the others invisible

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img_i, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].imshow(rel_i, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, alpha=0.6, interpolation="nearest")
    axes[0].set_title(f"x_i | pred={preds[0].item()} | map class={c_i}")
    axes[0].axis("off")

    axes[1].imshow(img_j, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].imshow(rel_j, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, alpha=0.6, interpolation="nearest")
    axes[1].set_title(f"x_j | pred={preds[1].item()} | map class={c_i}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()