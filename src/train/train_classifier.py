from src.parsers.config_parser import ClsTrainConfig


def main():
    config = ClsTrainConfig.from_yaml('config/train_cls_config.yaml')
    