"""Metrics calculator for multi-class classification."""

import torch
from torch import Tensor

from torcheval import metrics
from typing import List, Tuple


class MetricsCalculator:
    """Compute and aggregate metrics for multi-class classification."""

    def __init__(self, num_classes):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes to calculate metrics for.
        """
        self.accuracy = metrics.MulticlassAccuracy(num_classes=num_classes)
        self.f1_score = metrics.MulticlassF1Score(num_classes=num_classes)
        self.auprc = metrics.MulticlassAUPRC(num_classes=num_classes)
        self.auroc = metrics.MulticlassAUROC(num_classes=num_classes)
        self.recall = metrics.MulticlassRecall(num_classes=num_classes)
        self.precision = metrics.MulticlassPrecision(num_classes=num_classes)

    def update(self, outputs: List | Tensor, targets: List | Tensor) -> None:
        """
        Update the metrics state with new data.

        Args:
            outputs: List or Tensor of outputs returned by the model.
            targets: List or Tensor of targets for the current batch.
        """
        # Change dtype of variables
        if isinstance(outputs, List):
            outputs = torch.tensor(outputs)
        if isinstance(targets, List):
            targets = torch.tensor(targets)

        outputs = outputs.to(torch.float32)
        targets = targets.to(torch.long)

        # Update all metrics
        self.accuracy.update(outputs, targets)
        self.f1_score.update(outputs, targets)
        self.auprc.update(outputs, targets)
        self.auroc.update(outputs, targets)
        self.recall.update(outputs, targets)
        self.precision.update(outputs, targets)

    def compute_all(self) -> Tuple:
        """
        Compute all metrics.

        Returns:
            Tuple with accuracy, precision, recall, F1, AUPRC, and AUROC.
        """
        # Compute all metrics
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()
        auprc = self.auprc.compute()
        auroc = self.auroc.compute()
        recall = self.recall.compute()
        precision = self.recall.compute()
        return accuracy, precision, recall, f1_score, auprc, auroc

    def reset(self) -> None:
        """Reset all metric state."""
        self.accuracy.reset()
        self.f1_score.reset()
        self.auprc.reset()
        self.auroc.reset()
        self.recall.reset()
        self.precision.reset()
