from typing import Set

import torch

from psob_authorship.features.java.ast_metrics.AstMetricsCalculator import AstMetricsCalculator
from psob_authorship.features.java.character_metrics.CharacterMetricsCalculator import CharacterMetricsCalculator
from psob_authorship.features.java.line_metrics.LineMetricsCalculator import LineMetricsCalculator


class MetricsCalculator:
    def __init__(self, dataset_path: str, ast_path: str) -> None:
        super().__init__()
        self.line_metrics_calculator = LineMetricsCalculator(dataset_path)
        self.braces_metrics_calculator = CharacterMetricsCalculator(dataset_path)
        self.ast_metrics_calculator = \
            AstMetricsCalculator(ast_path, self.braces_metrics_calculator.character_number_for_file)

    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.cat((
            self.line_metrics_calculator.get_metrics(filepaths),
            self.braces_metrics_calculator.get_metrics(filepaths),
            self.ast_metrics_calculator.get_metrics(filepaths)
        ))
