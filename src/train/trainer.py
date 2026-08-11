import torch
import tqdm
from src.logging.metrics_logger import MetricsLogger


class BaseTrainer:
    def __init__(self):
        self.epoch = 0
        self.metrics_logger = MetricsLogger()

    def resolve_optimizer(optimizer_name, model_parameters, learning_rate):
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(model_parameters, lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(model_parameters, lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def resolve_loss_function(loss_function_name):
        if loss_function_name.lower() == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_function_name.lower() == 'mse':
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def train_one_epoch(self):
        self.train_metrics_calculator.reset()
        self.model.train()
        train_epoch_loss = 0
        n_instances = 0
        
        for inputs, targets in tqdm.tqdm(self.train_dataloader, desc=f"Training epoch: {self.epoch}"):
            self.optimizer.zero_grad()
    
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
    
            outputs = self.model(inputs)
    
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
            train_epoch_loss += loss.item()
            self.train_metrics_calculator.update(outputs.detach(), targets.detach())
            n_instances += inputs.shape[0]
        
        mean_epoch_loss = train_epoch_loss / n_instances
        return mean_epoch_loss

    def log_results(self):
        self.metrics_logger.show()


class ClsTrainer(BaseTrainer):
    def __init__(self, config: ClsTrainConfig, model: torch.nn.Module, train_dataloader, val_dataloader):
        super().__init__()
        self.config = config
        self.model = model

        self.optimizer = self.resolve_optimizer(self.config.optimizer, model_parameters=self.model.parameters(), learning_rate=self.config.learning_rate)
        self.criterion = self.resolve_loss_function(self.config.loss_function)

    def train(self):
        for epoch in range(self.config.epochs):
            self.train_one_epoch()
            self.evaluate()
            self.log_results()


class SaeTrainer(BaseTrainer):
    def __init__(self, config: SaeTrainConfig, model: torch.nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        
        self.optimizer = self.resolve_optimizer(self.config.optimizer, model_parameters=self.model.parameters(), learning_rate=self.config.learning_rate)
        self.criterion = self.resolve_loss_function(self.config.loss_function)

    def train(self):    
        # Implement the training loop here
        pass