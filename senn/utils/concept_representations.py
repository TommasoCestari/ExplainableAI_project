import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def highest_activations(model, test_loader, num_concepts=5, num_prototypes=9, save_path=None):
    """Creates concept representation via highest activation.

    The concepts are represented by the most prototypical data samples.
    (The samples that yield the highest activation for each concept)

    Parameters
    ----------
    model: torch.nn.Module
      The trained model with all its parameters.
    test_loader: DataLoader object
       Data loader that iterates over the test set.
    num_concepts: int
       Number of concepts of the model.
    num_prototypes: int
        Number of prototypical examples that should be displayed for each concept.
    save_path: str
        Path to the location where the bar plot should be saved.
    """
    model.eval()
    activations = []
    for x, _ in test_loader:
        x = x.float().to("cuda:0" if next(model.parameters()).is_cuda else "cpu")
        with torch.no_grad():
            _, (concepts, _), _ = model(x)
            activations.append(concepts.squeeze())
    activations = torch.cat(activations)

    _, top_test_idx = torch.topk(activations, num_prototypes, 0)

    top_examples = [test_loader.dataset.data[top_test_idx[:, concept]] for concept in range(num_concepts)]
    # flatten list and ensure correct image shape
    top_examples = [img.unsqueeze(0) if len(img.size()) == 2 else img for sublist in top_examples for img in sublist]

    plt.rcdefaults()
    fig, ax = plt.subplots()
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]

    start = 0.0
    end = num_concepts * x.size(-1)
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} most prototypical data examples per concept'.format(num_prototypes))
    ax.set_title('Concept Prototypes: ')
    save_or_show(make_grid(top_examples, nrow=num_prototypes, pad_value=1), save_path)
    plt.rcdefaults()


def highest_contrast(model, test_loader, num_concepts=5, num_prototypes=9, save_path=None):
    """Creates concept representation via highest contrast.

    The concepts are represented by the most data samples that are most specific to a concept.
    (The sample that yield the highest activation for each concept while at the same time
    not activating the other concepts)

    Parameters
    ----------
    model: torch.nn.Module
        The trained model with all its parameters.
    test_loader: DataLoader object
        Data loader that iterates over the test set.
    num_concepts: int
        Number of concepts of the model.
    num_prototypes: int
        Number of prototypical examples that should be displayed for each concept.
    save_path: str
        Path to the location where the bar plot should be saved.
    """
    model.eval()
    activations = []
    for x, _ in test_loader:
        x = x.float().to("cuda:0" if next(model.parameters()).is_cuda else "cpu")
        with torch.no_grad():
            _, (concepts, _), _ = model(x)
            activations.append(concepts.squeeze())
    activations = torch.cat(activations)

    contrast_scores = torch.empty_like(activations)
    for c in range(num_concepts - 1):
        contrast_scores[:, c] = activations[:, c] - (activations[:, :c].sum(dim=1) + activations[:, c + 1:].sum(dim=1))
    contrast_scores[:, num_concepts - 1] = activations[:, num_concepts - 1] - activations[:, :num_concepts - 1].sum(dim=1)

    _, top_test_idx = torch.topk(contrast_scores, num_prototypes, 0)

    top_examples = [test_loader.dataset.data[top_test_idx[:, concept]] for concept in range(num_concepts)]
    # flatten list and ensure correct image shape
    top_examples = [img.unsqueeze(0) if len(img.size()) == 2 else img for sublist in top_examples for img in sublist]

    plt.rcdefaults()
    fig, ax = plt.subplots()
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]

    start = 0.0
    end = num_concepts * x.size(-1)
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} data examples with highest contrast per concept'.format(num_prototypes))
    ax.set_title('Concept Prototypes: ')
    save_or_show(make_grid(top_examples, nrow=num_prototypes, pad_value=1), save_path)
    plt.rcdefaults()


def filter_concepts(model, num_concepts=5, num_prototypes=10, save_path=None):
    """Creates concept representation via filter visualization.

    The concepts are represented by the filters of the last layer of the concept encoder.
    (This option for visualization requires the concept_visualization field in
    configs to have the value 'filter'.See documentation of ConvConceptizer for more details)

    Parameters
    ----------
    model: torch.nn.Module
        The trained model with all its parameters.
    num_concepts: int
        Number of concepts of the model.
    num_prototypes: int
        Number of channels that each of the filters representing a concept has.
    save_path: str
        Path to the location where the bar plot should be saved.
    """
    model.eval()
    plt.rcdefaults()
    fig, ax = plt.subplots()
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]

    filters = [f for f in model.conceptizer.encoder[-2][0].weight.data.clone()]
    imgs = [dim.unsqueeze(0) for f in filters for dim in f]

    start = 0.0
    end = num_concepts * filters[0].size(-1) + 2
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} dimensions of concept filters'.format(num_prototypes))
    ax.set_title('Concept Prototypes: ')
    save_or_show(make_grid(imgs, nrow=num_prototypes, normalize=True, padding=1, pad_value=1), save_path)
    plt.rcdefaults()


def highest_activations_with_values(model, test_loader, num_concepts=5, num_prototypes=9, save_path=None, plot=True, print_values=False):
    """
    Creates concept representation via highest activation, displaying/printing the exact h(x) values.
    
    Parameters
    ----------
    model: torch.nn.Module
        The trained model with all its parameters.
    test_loader: DataLoader object
        Data loader that iterates over the test set.
    num_concepts: int
        Number of concepts of the model.
    num_prototypes: int
        Number of prototypical examples that should be displayed for each concept.
    save_path: str
        Path to the location where the plot should be saved.
    plot: bool
        If True, displays the grid of images with h(x) values as titles.
    print_values: bool
        If True, prints the arrays in the console.
        
    Returns
    -------
    dict
        A dictionary containing the h(x) arrays for the top prototypes of each concept.
    """
    model.eval()
    activations = []
    device = "cuda:0" if next(model.parameters()).is_cuda else "cpu"

    for x, _ in test_loader:
        x = x.float().to(device)
        with torch.no_grad():
            _, (concepts, _), _ = model(x)
            activations.append(concepts.squeeze().cpu()) 
    activations = torch.cat(activations)

    _, top_test_idx = torch.topk(activations, num_prototypes, dim=0)
    extracted_h_values = {}

    for concept in range(num_concepts):
        extracted_h_values[f"Concept_{concept + 1}"] = []
        if print_values:
            print(f"\n--- Top {num_prototypes} Prototypes (Highest Activation) for Concept {concept + 1} ---")
            
        for p in range(num_prototypes):
            idx = top_test_idx[p, concept]
            h_values = activations[idx].numpy()
            extracted_h_values[f"Concept_{concept + 1}"].append(h_values)
            
            if print_values:
                h_str = "[" + ", ".join([f"{v:.4f}" for v in h_values]) + "]"
                print(f"  Prototype {p + 1}: {h_str}")

    if plot:
        fig, axes = plt.subplots(num_concepts, num_prototypes, figsize=(num_prototypes * 2.5, num_concepts * 2.5))
        if num_concepts == 1: axes = np.expand_dims(axes, axis=0)
        if num_prototypes == 1: axes = np.expand_dims(axes, axis=1)

        for concept in range(num_concepts):
            for p in range(num_prototypes):
                idx = top_test_idx[p, concept]
                img = test_loader.dataset.data[idx]
                h_values = extracted_h_values[f"Concept_{concept + 1}"][p]
                h_str = "[" + ", ".join([f"{v:.2f}" for v in h_values]) + "]"

                ax = axes[concept, p]
                if len(img.size()) == 3 and img.size(0) in [1, 3]: 
                    ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
                else:
                    ax.imshow(img.squeeze(), cmap='gray')

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{h_str}", fontsize=9)
                if p == 0:
                    ax.set_ylabel(f'Concept {concept + 1}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        else: plt.show()
        plt.close()

    return extracted_h_values


def highest_contrast_with_values(model, test_loader, num_concepts=5, num_prototypes=9, save_path=None, plot=True, print_values=False):
    """
    Creates concept representation via highest contrast, displaying/printing the exact h(x) values.
    
    Parameters
    ----------
    model: torch.nn.Module
        The trained model with all its parameters.
    test_loader: DataLoader object
        Data loader that iterates over the test set.
    num_concepts: int
        Number of concepts of the model.
    num_prototypes: int
        Number of prototypical examples that should be displayed for each concept.
    save_path: str
        Path to the location where the plot should be saved.
    plot: bool
        If True, displays the grid of images with h(x) values as titles.
    print_values: bool
        If True, prints the arrays and contrast scores in the console.
        
    Returns
    -------
    dict
        A dictionary containing the h(x) arrays for the top prototypes of each concept.
    """
    model.eval()
    activations = []
    device = "cuda:0" if next(model.parameters()).is_cuda else "cpu"
    
    for x, _ in test_loader:
        x = x.float().to(device)
        with torch.no_grad():
            _, (concepts, _), _ = model(x)
            activations.append(concepts.squeeze().cpu())
    activations = torch.cat(activations)

    contrast_scores = torch.empty_like(activations)
    for c in range(num_concepts - 1):
        contrast_scores[:, c] = activations[:, c] - (activations[:, :c].sum(dim=1) + activations[:, c + 1:].sum(dim=1))
    contrast_scores[:, num_concepts - 1] = activations[:, num_concepts - 1] - activations[:, :num_concepts - 1].sum(dim=1)

    _, top_test_idx = torch.topk(contrast_scores, num_prototypes, dim=0)
    extracted_h_values = {}

    for concept in range(num_concepts):
        extracted_h_values[f"Concept_{concept + 1}"] = []
        if print_values:
            print(f"\n--- Top {num_prototypes} Prototypes (Highest Contrast) for Concept {concept + 1} ---")
            
        for p in range(num_prototypes):
            idx = top_test_idx[p, concept]
            h_values = activations[idx].numpy()
            extracted_h_values[f"Concept_{concept + 1}"].append(h_values)
            
            if print_values:
                c_score = contrast_scores[idx, concept].item()
                h_str = "[" + ", ".join([f"{v:.4f}" for v in h_values]) + "]"
                print(f"  Prototype {p + 1}: {h_str} | Contrast Score: {c_score:.4f}")

    if plot:
        fig, axes = plt.subplots(num_concepts, num_prototypes, figsize=(num_prototypes * 2.5, num_concepts * 2.5))
        if num_concepts == 1: axes = np.expand_dims(axes, axis=0)
        if num_prototypes == 1: axes = np.expand_dims(axes, axis=1)

        for concept in range(num_concepts):
            for p in range(num_prototypes):
                idx = top_test_idx[p, concept]
                img = test_loader.dataset.data[idx]
                h_values = extracted_h_values[f"Concept_{concept + 1}"][p]
                h_str = "[" + ", ".join([f"{v:.2f}" for v in h_values]) + "]"

                ax = axes[concept, p]
                if len(img.size()) == 3 and img.size(0) in [1, 3]: 
                    ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
                else:
                    ax.imshow(img.squeeze(), cmap='gray')

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{h_str}", fontsize=9)
                if p == 0:
                    ax.set_ylabel(f'Concept {concept + 1}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        else: plt.show()
        plt.close()
        
    return extracted_h_values


def save_or_show(img, save_path):
    """Saves an image or displays it.
    
    Parameters
    ----------
    img: torch.Tensor
        Tensor containing the image data that should be saved.
    save_path: str
        Path to the location where the bar plot should be saved. If None is passed image is showed instead.
    """
    # TODO: redesign me
    img = img.clone().squeeze()
    npimg = img.cpu().numpy()
    if len(npimg.shape) == 2:
        if save_path is None:
            plt.imshow(npimg, cmap='Greys')
            plt.show()
        else:
            plt.imsave(save_path, npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
    plt.clf()
