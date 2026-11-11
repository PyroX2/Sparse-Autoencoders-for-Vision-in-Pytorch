import torch
from torch import nn

from src.configs.model_config import ModelConfig
from src.models.model_registry import classifier_models_registry


class MLP(nn.Module):
    def __init__(self, n_outputs: int = 10) -> None:
        super().__init__()
        self.fc = nn.Sequential()
        self.fc.add_module("flatten", nn.Flatten())
        self.fc.add_module("linear_01", nn.Linear(3 * 784, 512))
        self.fc.add_module("tanh_01", nn.Tanh())
        self.fc.add_module("linear_02", nn.Linear(512, 64))
        self.fc.add_module("tanh_02", nn.Tanh())
        self.fc.add_module("linear_03", nn.Linear(64, 8))
        self.fc.add_module("tanh_03", nn.Tanh())
        self.fc.add_module("output", nn.Linear(8, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("CNN model class is not implemented.")

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ViT model class is not implemented.")

class ClsModel:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = self._resolve_model()

    def _resolve_model(self):
        model_id = self.model_config.cls_model.type

        if model_id.lower() in classifier_models_registry.keys():
            model_cls = classifier_models_registry[model_id.lower()]
        else:
            raise ValueError(f"{model_id} is not recognized as valid classifier model type. Valid ids are: {classifier_models_registry.keys()}")

        return model_cls()
