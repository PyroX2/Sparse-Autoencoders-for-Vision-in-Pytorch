from pydantic import BaseModel, Field


class GeneralConfig(BaseModel):
    project_name: str = Field()
    dataset: str = Field(default="coloredmnist")

    @classmethod
    def from_dict(cls, kwargs_dict: dict):
        return GeneralConfig(**kwargs_dict)
