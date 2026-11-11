from torch.utils import data

from src.configs import ClsDataConfig
from src.dataset.datasets import CelebADataset, ColoredMNISTDataset, WaterBirdsDataset

dataset_registry = {
    "celeba": CelebADataset,
    "coloredmnist": ColoredMNISTDataset,
    "waterbirds": WaterBirdsDataset,
}


def get_dataset(dataset_config: ClsDataConfig) -> data.Dataset:
    dataset_id = dataset_config.dataset
    if dataset_id.lower() not in dataset_registry:
        raise ValueError(
            f"{dataset_id} not recognized as valid dataset name. Valid names are {dataset_registry.keys()}"
        )
    return dataset_registry[dataset_id.lower()](dataset_config)
