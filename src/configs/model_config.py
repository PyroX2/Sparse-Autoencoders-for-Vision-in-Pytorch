from pydantic import BaseModel, Field


class ClsModelConfig(BaseModel):
    type: str = Field(default="mlp")
    lr: float = Field(default=1e-03)
    batch_size: int = Field(default=16)


class SaeModelConfig(BaseModel):
    type: str = Field(default="l1")
    lr: float = Field(default=1e-03)
    batch_size: int = Field(default=16)
    sae_input_layer: str = Field()


class TrimmedClsModelConfig(BaseModel):
    method: str = Field()


class ModelConfig(BaseModel):
    cls_model: ClsModelConfig = Field()
    sae_model: SaeModelConfig = Field()
    trimmed_cls_model: TrimmedClsModelConfig = Field()

    @classmethod
    def from_dict(cls, model_dict: dict):
        return cls(**model_dict)
