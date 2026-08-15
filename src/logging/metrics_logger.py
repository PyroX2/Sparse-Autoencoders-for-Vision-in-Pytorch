import logging
from src import metrics


class MetricsLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def show(
        self,
        results: dict,
        epoch: int = 0,
        prefix: str = "",
    ):
        if prefix:
            prefix = f"[{prefix.strip()}] "

        self.logger.info(
            f"{prefix}Results from epoch: {epoch}{self.parse_results(results)}"
        )

    @staticmethod
    def parse_results(results: dict):
        return f"""
                    Accuracy: {results["accuracy"]:.4f},
                    Precision: {results["precision"]:.4f},
                    Recall: {results["recall"]:.4f},
                    F1: {results["f1_score"]:.4f},
                    AUPRC: {results["auprc"]:.4f},
                    AUROC: {results["auroc"]:.4f}
                """
