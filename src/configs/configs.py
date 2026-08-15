import yaml
import pydantic


class GeneralTrainConfig(pydantic.BaseModel):
    seed: int
    logging_level: int

class HyperparamsConfig(pydantic.BaseModel):
    lr: float
    train_batch_size: int
    val_batch_size: int
    num_epochs: int

class DatasetConfig(pydantic.BaseModel):
    name: str                   # Name of the dataset
    colored_percentage: float   # Between 0 and 1

class ClsTrainConfig(pydantic.BaseModel):
    dataset: DatasetConfig
    hyperparams: HyperparamsConfig
    general: GeneralTrainConfig

        
    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class SaeTrainConfig:
    """Configuration class for SAE training."""

    # TODO: Implement this class similarly to ClsTrainConfig, based on the SAE training requirements.
