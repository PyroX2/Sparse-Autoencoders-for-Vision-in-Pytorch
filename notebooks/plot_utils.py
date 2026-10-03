import torch
from torch.utils import data
from captum import attr
import matplotlib.pyplot as plt
import numpy as np
import typing


def show_from_dataset(dataset, idx: int | typing.List = 0):
    if isinstance(idx, int):
        plt.imshow(dataset[idx][0].permute(1, 2, 0).numpy())
        plt.title(f"Label: {dataset[idx][1].item()}")
        plt.axis("off")
    else:
        current_idx = 0
        if len(idx) >= 5:
            fig, axs = plt.subplots(len(idx) // 5, 5)
            for i, subaxs in enumerate(axs):
                for j, ax in enumerate(subaxs):
                    img_idx = idx[current_idx]
                    ax.imshow(dataset[img_idx][0].permute(1, 2, 0).numpy())
                    ax.set_title(f"Label: {dataset[img_idx][1].item()}")
                    ax.axis("off")
                    current_idx += 1
        else:
            fig, axs = plt.subplots(1, len(idx))
            for i, ax in enumerate(axs):
                img_idx = idx[current_idx]
                ax.imshow(dataset[img_idx][0].permute(1, 2, 0).numpy())
                ax.set_title(f"Label: {dataset[img_idx][1].item()}")
                ax.axis("off")
                current_idx += 1
    plt.show()


def show_ig_from_dataset(
    dataset: data.Dataset,
    ig,
    ig_target_idx: int,
    idx: int | list = 0,
):
    def ig_inference(dataset_idx: int):
        inputs, targets = dataset[dataset_idx]
        inputs = inputs.unsqueeze(0)
        baseline = torch.zeros_like(inputs)

        # Integrated gradients calculation
        attributions = ig.attribute(inputs, baselines=(baseline), target=ig_target_idx)
        normalized_attributions = normalize_min_max(
            attributions[0].permute(1, 2, 0).cpu().numpy()
        )
        return normalized_attributions

    if isinstance(idx, int):
        fig, axs = plt.subplots(1, 2)  # Plot input image with ig attributions for topk
        normalized_attributions = ig_inference(idx)
        axs[0].imshow(dataset[idx][0].permute(1, 2, 0).numpy())
        axs[0].axis("off")
        axs[1].imshow(normalized_attributions)
        axs[1].axis("off")
    elif isinstance(idx, list):
        fig, axs = plt.subplots(len(idx), 2, figsize=(4, len(idx) * 2))
        for i, subaxs in enumerate(axs):
            img_idx = idx[i]
            normalized_attributions = ig_inference(img_idx)
            subaxs[0].imshow(dataset[img_idx][0].permute(1, 2, 0).numpy())
            subaxs[0].axis("off")
            subaxs[1].imshow(normalized_attributions)
            subaxs[1].axis("off")
    else:
        raise TypeError(f"idx must be int or list not {type(idx)}")
    plt.show()


def plot_loss(loss: typing.List, title: str | None = None) -> None:
    x_data = np.arange(1, len(loss) + 1, 1)
    plt.plot(x_data, loss)

    if title is not None:
        plt.title(title)

    plt.show()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def normalize_min_max(data: torch.Tensor) -> torch.Tensor:
    return (data - data.min()) / (data.max() - data.min())
