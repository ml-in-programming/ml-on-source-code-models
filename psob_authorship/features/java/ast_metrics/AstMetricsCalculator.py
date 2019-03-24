from typing import Set

import torch

from psob_authorship.features.java.ast.Ast import FileAst
from psob_authorship.features.java.ast_metrics.StatementsMetricsCalculator import StatementsMetricsCalculator
from psob_authorship.features.java.ast_metrics.VariableMetricsCalculator import VariableMetricsCalculator


class AstMetricsCalculator:
    def __init__(self, ast_path, filenames=None) -> None:
        super().__init__()
        asts = FileAst.load_asts_from_files(ast_path, filenames)
        self.variable_metrics_calculator = VariableMetricsCalculator(asts)
        self.loops_metrics_calculator = StatementsMetricsCalculator(asts)

    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.cat((
            self.variable_metrics_calculator.get_metrics(filepaths),
            self.loops_metrics_calculator.get_metrics(filepaths)
        ))
