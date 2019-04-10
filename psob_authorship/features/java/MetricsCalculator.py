import logging
from typing import Set, List

import torch

from psob_authorship.features.java.ast_metrics.AstMetricsCalculator import AstMetricsCalculator
from psob_authorship.features.java.character_metrics.CharacterMetricsCalculator import CharacterMetricsCalculator
from psob_authorship.features.java.line_metrics.LineMetricsCalculator import LineMetricsCalculator
from psob_authorship.features.utils import configure_logger_by_default, ZERO_DIVISION_RETURN


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

    @staticmethod
    def set_mean_to_indices(default_value_indices, mean_values: torch.Tensor, features: torch.Tensor):
        for row in range(features.shape[0]):
            for col in range(features.shape[1]):
                if default_value_indices[row][col] == 1:
                    features[row][col] = mean_values[col]

    @staticmethod
    def set_mean_to_zero_division(mean_values: torch.Tensor, features: torch.Tensor):
        default_value_indices = features == ZERO_DIVISION_RETURN
        MetricsCalculator.set_mean_to_indices(default_value_indices, mean_values, features)

    @staticmethod
    def transform_metrics_default_values_to_mean(default_value, features: torch.Tensor) -> torch.Tensor:
        default_value_indices = features == default_value
        features[default_value_indices] = 0
        mean_values = torch.mean(features, dim=0)
        MetricsCalculator.set_mean_to_indices(default_value_indices, mean_values, features)
        return mean_values

    @staticmethod
    def transform_metrics_zero_division_to_mean(features: torch.Tensor) -> torch.Tensor:
        return MetricsCalculator.transform_metrics_default_values_to_mean(ZERO_DIVISION_RETURN, features)

    @staticmethod
    def transform_metrics_zero_division_to_new_value(new_value, features: torch.Tensor):
        MetricsCalculator.transform_metrics_default_values_to_new_value(ZERO_DIVISION_RETURN, new_value, features)

    @staticmethod
    def transform_metrics_default_values_to_new_value(default_value, new_value, features: torch.Tensor):
        default_value_indices = features == default_value
        features[default_value_indices] = new_value

    @staticmethod
    def transform_metrics_zero_division_to_zero(features: torch.Tensor):
        MetricsCalculator.transform_metrics_zero_division_to_new_value(0, features)

    @staticmethod
    def transform_metrics_zero_division_to_one(features: torch.Tensor):
        MetricsCalculator.transform_metrics_zero_division_to_new_value(1, features)
