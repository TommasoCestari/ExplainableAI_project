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


def compute_stability_adversarial(model, x_i, device, epsilon=0.01, steps=20, lr=0.001,
                                  use_fixed_class=True, data_min=None, data_max=None):
    """
    Otimizes x_j through Gradient Ascent 
    """
    model.eval()
    x_i = x_i.detach().to(device)
    
    # 1. Inizializziamo x_j come x_i + un piccolo rumore casuale (per rompere la simmetria)
    x_j = x_i.clone().detach() + torch.randn_like(x_i) * 0.001
    x_j.requires_grad = True
    
    # Calcoliamo i riferimenti per x_i (fissi)
    with torch.no_grad():
        y_logp_i, (concepts_i, relevances_i), _ = model(x_i)
        pred_i = y_logp_i.argmax(dim=1)
        h_i = concepts_i.squeeze(-1) # [1, num_concepts]
        
        # Estraiamo theta_i (f_i) per la classe di riferimento
        idx_i = pred_i.view(-1, 1, 1).expand(-1, relevances_i.size(1), 1)
        f_i = torch.gather(relevances_i, dim=2, index=idx_i).squeeze(-1) # [1, num_concepts]

    # 2. Loop di ottimizzazione (Gradient Ascent)
    for _ in range(steps):
        y_logp_j, (concepts_j, relevances_j), _ = model(x_j)
        h_j = concepts_j.squeeze(-1)
        
        # Strategia per la classe di x_j
        if use_fixed_class:
            idx_j = pred_i.view(1, 1, 1).expand(relevances_j.size(0), relevances_j.size(1), 1)
        else:
            pred_j = y_logp_j.argmax(dim=1)
            idx_j = pred_j.view(-1, 1, 1).expand(-1, relevances_j.size(1), 1)
            
        f_j = torch.gather(relevances_j, dim=2, index=idx_j).squeeze(-1)

        # Calcolo del rapporto (Formula 5 del paper)
        num = torch.norm(f_i - f_j, p=2)
        den = torch.norm(h_i - h_j, p=2)
        ratio = num / (den + 1e-12)

        # Vogliamo MASSIMIZZARE il rapporto -> usiamo il gradiente della loss negativa
        loss = -ratio 
        model.zero_grad()
        loss.backward()

        # Update x_j
        with torch.no_grad():
            # Spostiamo x_j nella direzione del gradiente
            x_j -= lr * x_j.grad.sign() 
            
            # L_inf projection around x_i
            diff = x_j - x_i
            diff = torch.clamp(diff, -epsilon, epsilon)
            x_j.data = x_i + diff

            # Clamp to valid data range (normalized range if inputs are normalized)
            if data_min is not None and data_max is not None:
                x_j.data = torch.clamp(x_j.data, data_min, data_max)
            
            x_j.grad.zero_()

    # Risultato finale dopo l'ottimizzazione
    with torch.no_grad():
        final_ratio = ratio.item()
        final_pred_j = y_logp_j.argmax(dim=1).item()
        
    return x_j.detach(), final_ratio, final_pred_j


def plot_input_comparison(x_i, x_j, title_i="Originale (x_i)", title_j="Perturbata (x_j)"):
    """
    Visualizza l'immagine originale e quella perturbata (o adversarial).
    Assume che i tensor siano in formato [1, 1, 28, 28] o [1, 28, 28].
    """
    # Rimuoviamo le dimensioni extra e portiamo in CPU
    img_i = x_i.detach().cpu().squeeze()
    img_j = x_j.detach().cpu().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Plot x_i
    axes[0].imshow(img_i, cmap='gray')
    axes[0].set_title(title_i)
    axes[0].axis('off')
    
    # Plot x_j
    axes[1].imshow(img_j, cmap='gray')
    axes[1].set_title(title_j)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()