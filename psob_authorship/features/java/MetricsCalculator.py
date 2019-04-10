import logging
from typing import Set, List

import torch

from psob_authorship.features.java.ast_metrics.AstMetricsCalculator import AstMetricsCalculator
from psob_authorship.features.java.character_metrics.CharacterMetricsCalculator import CharacterMetricsCalculator
from psob_authorship.features.java.line_metrics.LineMetricsCalculator import LineMetricsCalculator
from psob_authorship.features.utils import configure_logger_by_default


class MetricsCalculator:
    LOGGER = logging.getLogger('metrics_calculator')

    @staticmethod
    def get_metric_names() -> List[str]:
        return LineMetricsCalculator.get_metric_names() + CharacterMetricsCalculator.get_metric_names() + AstMetricsCalculator.get_metric_names()

    def __init__(self, dataset_path: str, ast_path: str) -> None:
        configure_logger_by_default(self.LOGGER)
        self.LOGGER.info("Started calculating metrics")
        super().__init__()
        self.line_metrics_calculator = LineMetricsCalculator(dataset_path)
        self.character_metrics_calculator = CharacterMetricsCalculator(dataset_path)
        self.ast_metrics_calculator = \
            AstMetricsCalculator(ast_path, self.character_metrics_calculator.character_number_for_file)
        self.LOGGER.info("End calculating metrics")

    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.cat((
            self.line_metrics_calculator.get_metrics(filepaths),
            self.character_metrics_calculator.get_metrics(filepaths),
            self.ast_metrics_calculator.get_metrics(filepaths)
        ))
