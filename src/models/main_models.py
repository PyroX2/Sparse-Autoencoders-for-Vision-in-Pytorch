from src.configs.model_config import ModelConfig
from src.models.model_registry import classifier_models_registry


class ClsModel:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = self._resolve_model()

    def _resolve_model(self):
        model_id = self.model_config.type

        if model_id.lower() in classifier_models_registry:
            model_cls = classifier_models_registry[model_id.lower()]
        else:
            raise ValueError(
                f"{model_id} is not recognized as valid classifier model type. Valid ids are: {classifier_models_registry.keys()}"
            )

        return model_cls(n_outputs=self.model_config.n_outputs)


class SaeModel:
    def __init__(self):
        pass
