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


def plot_loss(loss: typing.List, title: str | None = None) -> None:
    x_data = np.arange(1, len(loss) + 1, 1)
    plt.plot(x_data, loss)

    if title is not None:
        plt.title(title)

    plt.show()
