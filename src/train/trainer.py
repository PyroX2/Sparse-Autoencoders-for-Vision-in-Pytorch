import torch
import tqdm
import logging as basic_logging
from src import logging
from src import metrics
from src import configs


class BaseTrainer:
    def __init__(self, n_classes: int, logger: basic_logging.Logger):
        self.model = None
        self.epoch = 0
        self.metrics_logger = logging.MetricsLogger(logger)
        self.train_metrics_calculator = metrics.MetricsCalculator(num_classes=n_classes)
        self.val_metrics_calculator = metrics.MetricsCalculator(num_classes=n_classes)

    def resolve_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def resolve_loss_function(self, loss_function_name):
        if loss_function_name.lower() == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_function_name.lower() == 'mse':
            return torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def train_one_epoch(self):
        """
        Performs the optimization over the whole dataset once.
        Resets the train_metrics_calculator at the beggining and updates metrics inplace but does not return the metrics directly.
        Those can be accesed outside of this function.

        Returns:
            float: Mean training loss for current epoch.
        """
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

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the model on val dataloader

        Returns:
            float: Mean loss.
        """

        self.val_metrics_calculator.reset()
        self.model.eval()
        total_loss = 0
        n_instances = 0
        
        for inputs, targets in tqdm.tqdm(self.val_dataloader, desc="Evaluating"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            self.val_metrics_calculator.update(outputs.detach(), targets.detach())
            n_instances += inputs.shape[0]
        
        mean_loss = total_loss / n_instances
        return mean_loss

    def log_results(self, results: dict, prefix: str = ""):
        self.metrics_logger.show(results, epoch=self.epoch, prefix=prefix)


class ClsTrainer(BaseTrainer):
    def __init__(self,
                 config: configs.ClsTrainConfig,
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 logger,
                 n_classes: int = 10,
                 device: str = "cpu"):
        super().__init__(n_classes=n_classes, logger=logger)
        self.config = config
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = self.resolve_optimizer(self.config.hyperparams.optimizer, learning_rate=self.config.hyperparams.lr)
        self.criterion = self.resolve_loss_function(self.config.hyperparams.loss_function)

    def train(self):
        for epoch in range(self.config.hyperparams.num_epochs):
            self.train_one_epoch()
            self.evaluate()
            train_accuracy, train_precision, train_recall, train_f1_score, train_auprc, train_auroc = self.train_metrics_calculator.compute_all()
            train_results = {
                "accuracy": train_accuracy,
                "precision": train_precision,
                "recall": train_recall,
                "f1_score": train_f1_score,
                "auprc": train_auprc,
                "auroc": train_auroc
            }
            val_accuracy, val_precision, val_recall, val_f1_score, val_auprc, val_auroc = self.val_metrics_calculator.compute_all()
            val_results = {
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1_score": val_f1_score,
                "auprc": val_auprc,
                "auroc": val_auroc
            }
            self.log_results(train_results, prefix="Train")
            self.log_results(val_results, prefix="Val")
            self.epoch += 1


class SaeTrainer(BaseTrainer):
    def __init__(self, config: configs.SaeTrainConfig, model: torch.nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        
        self.optimizer = self.resolve_optimizer(self.config.optimizer, model_parameters=self.model.parameters(), learning_rate=self.config.learning_rate)
        self.criterion = self.resolve_loss_function(self.config.loss_function)

    def train(self):    
        # Implement the training loop here
        pass