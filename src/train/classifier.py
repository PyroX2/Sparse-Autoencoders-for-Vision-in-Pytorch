import logging
import torch
import sklearn
from src import configs
from src.data import data
from src.models import ClsModel
from src.data import datasets
from src.train import ClsTrainer
from src import utils


def main():
    train_config = configs.ClsTrainConfig.from_yaml("config/train_cls_config.yaml")
    display_config = configs.DisplayConfig.from_yaml("config/display_config.yaml")
    logger = logging.Logger("Training logger", train_config.general.logging_level)

    device = utils.get_device()

    train_dataset = data.get_dataset(train_config.dataset.name, split="train")
    test_dataset = data.get_dataset(train_config.dataset.name, split="test")

    logger.info(f"""Loading dataset: 
                {train_config.dataset.name}\n
                Training dataset:\n{train_dataset}\n
                Test dataset:\n{test_dataset}""")

    # Split train dataset into train and val while maintaining labels ratio
    labels = torch.tensor([y for x, y in train_dataset])
    train_split, val_split = sklearn.model_selection.train_test_split(
        train_dataset,
        test_size=train_config.dataset.val_split_size,
        stratify=labels,
        random_state=train_config.general.seed,
    )

    colored_train_dataset = datasets.ColoredDataset(
        train_split,
        colored_proportions=train_config.dataset.colored_percentage,
        colors=display_config.colors,
    )
    colored_val_dataset = datasets.ColoredDataset(
        val_split,
        colored_proportions=train_config.dataset.colored_percentage,
        colors=display_config.colors,
    )
    clean_val_dataset = datasets.ColoredDataset(
        val_split, colored_proportions=0.0, colors=display_config.colors
    )
    fully_colored_val_dataset = datasets.ColoredDataset(
        val_split, colored_proportions=1.0, colors=display_config.colors
    )

    colored_train_dataloader = torch.utils.data.DataLoader(
        colored_train_dataset,
        batch_size=train_config.hyperparams.train_batch_size,
        shuffle=True,
    )
    colored_val_dataloader = torch.utils.data.DataLoader(
        colored_val_dataset,
        batch_size=train_config.hyperparams.val_batch_size,
        shuffle=False,
    )
    clean_val_dataloader = torch.utils.data.DataLoader(
        clean_val_dataset,
        batch_size=train_config.hyperparams.val_batch_size,
        shuffle=False,
    )
    fully_colored_val_dataloader = torch.utils.data.DataLoader(
        fully_colored_val_dataset,
        batch_size=train_config.hyperparams.val_batch_size,
        shuffle=False,
    )

    model = ClsModel()
    model.to(device)

    n_classes = len(labels.unique())

    trainer = ClsTrainer(
        train_config,
        model=model,
        train_dataloader=colored_train_dataloader,
        val_dataloader=colored_val_dataloader,
        n_classes=n_classes,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
