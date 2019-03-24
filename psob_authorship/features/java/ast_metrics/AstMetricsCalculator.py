from typing import Set

import torch

from psob_authorship.features.java.ast.Ast import FileAst
from psob_authorship.features.java.ast_metrics.StatementsMetricsCalculator import StatementsMetricsCalculator
from psob_authorship.features.java.ast_metrics.VariableMetricsCalculator import VariableMetricsCalculator


class AstMetricsCalculator:
    def __init__(self, ast_path, filenames=None) -> None:
        super().__init__()
        self.asts = FileAst.load_asts_from_files(ast_path, filenames)
        self.variable_metrics_calculator = VariableMetricsCalculator(self.asts)
        self.loops_metrics_calculator = StatementsMetricsCalculator(self.asts)

    def maximum_depth_of_an_ast(self, filepaths: Set[str]) -> torch.Tensor:
        """
        Max depth of files ast.
        :param filepaths: paths to files for which metric should be calculated
        :return: files ast max depth.
        """
        return torch.tensor([
            float(max([self.asts[filepath].depth for filepath in filepaths]))
        ])

    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.cat((
            self.variable_metrics_calculator.get_metrics(filepaths),
            self.loops_metrics_calculator.get_metrics(filepaths),
            self.maximum_depth_of_an_ast(filepaths)
        ))
