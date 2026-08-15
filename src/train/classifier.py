import torch
import logging
from src.configs import configs
from src.data import data


def main():
    config = configs.ClsTrainConfig.from_yaml('config/train_cls_config.yaml')
    logger = logging.Logger('Training logger', config.general.logging_level)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_dataset = data.get_dataset(config.dataset.name, split="train")
    test_dataset = data.get_dataset(config.dataset.name, split="test")

    logger.info(f"""Loading dataset: 
                {config.dataset.name}\n
                Training dataset:\n{train_dataset}\n
                Test dataset:\n{test_dataset}""")



if __name__ == "__main__":
    main()