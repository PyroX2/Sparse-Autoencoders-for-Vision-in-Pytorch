import pydantic
from typing import List
import yaml


class DisplayConfig(pydantic.BaseModel):
    colors: List

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
