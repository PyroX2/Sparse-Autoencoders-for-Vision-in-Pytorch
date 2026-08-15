import logging
import torch
import sklearn
from src.configs import configs
from src.data import data
from src.models import ClsModel
import utils


def main():
    config = configs.ClsTrainConfig.from_yaml("config/train_cls_config.yaml")
    logger = logging.Logger("Training logger", config.general.logging_level)

    device = utils.get_device()

    train_dataset = data.get_dataset(config.dataset.name, split="train")
    test_dataset = data.get_dataset(config.dataset.name, split="test")

    logger.info(f"""Loading dataset: 
                {config.dataset.name}\n
                Training dataset:\n{train_dataset}\n
                Test dataset:\n{test_dataset}""")

    # Split train dataset into train and val while maintaining labels ratio
    labels = torch.tensor([y for x, y in train_dataset])
    train_split, val_split = sklearn.model_selection.train_test_split(
        train_dataset,
        test_size=config.hyperparams.val_size,
        stratify=labels,
        random_state=config.general.seed,
    )


if __name__ == "__main__":
    main()
