from typing import Set

import torch

from psob_authorship.features.java.ast.Ast import FileAst
from psob_authorship.features.java.ast_metrics.VariableMetricsCalculator import VariableMetricsCalculator


class AstMetricsCalculator:
    def __init__(self, ast_path, filenames=None) -> None:
        super().__init__()
        self.asts = FileAst.load_asts_from_files(ast_path, filenames)
        self.variable_metrics_calculator = VariableMetricsCalculator(self.asts)

    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return self.variable_metrics_calculator.get_metrics(filepaths)
