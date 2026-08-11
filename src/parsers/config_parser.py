import yaml


class ClsTrainConfig:
    def __init__(self, config_dict):
        self.hyperparams = config_dict.get('hyperparams', {})
        self.mnist = config_dict.get('mnist', {})
        self.learning_rate = self.hyperparams.get('learning_rate', 0.001)
        self.batch_size = self.hyperparams.get('batch_size', 1)
        self.num_epochs = self.hyperparams.get('num_epochs', 50)
        self.colored_percentage = self.mnist.get('colored_percentage', 0.95)

        self.optimizer = self.hyperparams.get('optimizer', 'adam')
        self.loss_function = self.hyperparams.get('loss_function', 'cross_entropy')
    
    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)


class SaeTrainConfig:
    """Configuration class for SAE training."""
    # TODO: Implement this class similarly to ClsTrainConfig, based on the SAE training requirements.