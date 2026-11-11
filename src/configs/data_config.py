from pydantic import BaseModel, Field


class ClsDataConfig(BaseModel):
    dataset: str = Field(
        default="coloredmnist", description="Dataset used for training."
    )
