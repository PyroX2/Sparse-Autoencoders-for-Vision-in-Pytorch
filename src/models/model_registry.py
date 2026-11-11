from src.models.classifier import MLP, CNN, ViT
from src.models.sae import SAE, TopkSAE, BatchTopkSAE, MatryoshkaBatchTopkSAE


classifier_models_registry = {
    "mlp": MLP,
    "cnn": CNN,
    "vit": ViT,
}

sae_models_registry = {
    "l1": SAE,
    "topk": TopkSAE,
    "batchtopk": BatchTopkSAE,
    "matryoshka": MatryoshkaBatchTopkSAE,
}