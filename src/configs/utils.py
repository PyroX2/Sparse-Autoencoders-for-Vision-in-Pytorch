import yaml
from src.configs.general_config import GeneralConfig
from src.configs.model_config import ModelConfig


def read_general_config(path: str):
    with open(path, "r") as f:
        general_dict = yaml.safe_load(f)

    return GeneralConfig.from_dict(general_dict)


def read_model_config(path: str):
    with open(path, "r") as f:
        model_dict = yaml.safe_load(f)

    return ModelConfig.from_dict(model_dict)
