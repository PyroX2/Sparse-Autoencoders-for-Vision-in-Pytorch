from pydantic import BaseModel, Field


class ClsModelConfig(BaseModel):
    type: str = Field(default="mlp", description="Type of classifier model.")
    n_outputs: int = Field(default=10, description="Number of model output neurons.")
    lr: float = Field(
        default=1e-03, description="Classifier model training learning rate."
    )
    batch_size: int = Field(
        default=16, description="Classifier model training batch size."
    )
    epochs: int = Field(default=10, description="Number of classifier training epochs.")


class SaeModelConfig(BaseModel):
    type: str = Field(default="l1", descriptio="Type of SAE model.")
    lr: float = Field(default=1e-03, description="SAE model training learning rate.")
    batch_size: int = Field(default=16, description="SAE model training batch size.")
    sae_input_layer: str = Field(
        description="Name of classifier layer which activations will be used as SAE inputs."
    )


class TrimmedClsModelConfig(BaseModel):
    method: str = Field()


class ModelConfig(BaseModel):
    cls_model: ClsModelConfig = Field()
    sae_model: SaeModelConfig = Field()
    trimmed_cls_model: TrimmedClsModelConfig = Field()

    @classmethod
    def from_dict(cls, model_dict: dict):
        return cls(**model_dict)
