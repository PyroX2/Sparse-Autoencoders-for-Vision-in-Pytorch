import torch
from torch import nn
from torch.utils import data
import tqdm


@torch.no_grad()
def generate_sae_dataset(
    model: nn.Module, dataset: data.Dataset, device="cpu", layer_idx: int = -1
) -> data.TensorDataset:
    """
    Function for generating SAE dataset based on a model and a dataset.
    Uses model's last layer input as a reconstruction target for SAE.

    Args:
            model: Model which embeddings are to be extracted.
            dataset: Dataset with input images to be encoded by the model.

    Returns:
            data.TensorDataset: SAE dataset with encoded images.
    """
    sae_input_shape = model.fc[layer_idx].out_features
    sae_dataset = torch.empty(len(dataset), sae_input_shape)

    idx = 0
    for inputs, targets in tqdm.tqdm(dataset):
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(0)
        outputs = model.fc[: layer_idx + 1](inputs).squeeze()
        sae_dataset[idx] = outputs
        idx += 1

    return data.TensorDataset(sae_dataset)
