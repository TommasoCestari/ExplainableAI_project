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


def add_white_noise(x_norm, sigma=0.05, mean=MEAN, std=STD):
    # 1) normalized -> pixel space
    x_pix = denormalize(x_norm, mean, std)

    # 2) add Gaussian white noise in pixel space
    noise = torch.randn_like(x_pix) * sigma
    x_noisy_pix = x_pix + noise

    # 3) clip to valid pixel range
    x_noisy_pix = torch.clamp(x_noisy_pix, 0.0, 1.0) # for visualization

    # 4) pixel -> normalized (for model input)
    x_noisy_norm = normalize(x_noisy_pix, mean, std) # for the model

    return x_noisy_norm, x_noisy_pix


def plot_stability_comparison(
    model,
    x_i_norm,            # expected shape [1, 28, 28] or [1, 1, 28, 28]
    neighbors_norm,      # expected shape [N, 1, 28, 28]
    j_star,              # index from your argmax(ratio)
    mean,
    std,
):
    model.eval()
    device = next(model.parameters()).device

    # Prepare x_i as [1, 1, 28, 28]
    if x_i_norm.dim() == 3:
        x_i_b = x_i_norm.unsqueeze(0).to(device)
    elif x_i_norm.dim() == 4:
        x_i_b = x_i_norm.to(device)
    else:
        raise ValueError("x_i_norm must have shape [1,28,28] or [1,1,28,28]")

    # Pick worst neighbor and keep shape [1, 1, 28, 28]
    x_j_b = neighbors_norm[j_star].unsqueeze(0).to(device)

    # Forward pass on both samples together
    inputs = torch.cat([x_i_b, x_j_b], dim=0)  # [2,1,28,28]
    with torch.no_grad():
        y_logp, (concepts, relevances), _ = model(inputs)
        preds = y_logp.argmax(dim=1)

    c_i = preds[0].item()
    c_j = c_i

    # Relevances -> heatmaps (Identity conceptizer: 784 concepts)
    rel_i = relevances[0, :, c_i].view(28, 28).cpu()
    rel_j = relevances[1, :, c_j].view(28, 28).cpu()

    # Denormalize images for display
    img_i = (x_i_b[0].cpu() * std + mean).squeeze()
    img_j = (x_j_b[0].cpu() * std + mean).squeeze()

    # Shared color scale for fair visual comparison
    max_abs = torch.max(torch.abs(torch.cat([rel_i.reshape(-1), rel_j.reshape(-1)]))).item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        rel_i, cmap="RdBu_r", center=0, vmin=-max_abs, vmax=max_abs,
        ax=axes[0], cbar_kws={"label": "Relevance (theta)"}, alpha=0.6
    )
    axes[0].imshow(img_i, cmap="gray", alpha=0.4)
    axes[0].set_title(f"x_i | pred={preds[0].item()} | map class={c_i}")
    axes[0].axis("off")

    sns.heatmap(
        rel_j, cmap="RdBu_r", center=0, vmin=-max_abs, vmax=max_abs,
        ax=axes[1], cbar_kws={"label": "Relevance (theta)"}, alpha=0.6
    )
    axes[1].imshow(img_j, cmap="gray", alpha=0.4)
    axes[1].set_title(f"x_j* | pred={preds[1].item()} | map class={c_j}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()