import logging
from typing import Set, Dict

import torch

from psob_authorship.features.java.ast.Ast import FileAst
from psob_authorship.features.java.ast_metrics.StatementsMetricsCalculator import StatementsMetricsCalculator
from psob_authorship.features.java.ast_metrics.VariableMetricsCalculator import VariableMetricsCalculator


class AstMetricsCalculator:
    LOGGER = logging.getLogger('metrics_calculator')

    def __init__(self, ast_path, character_number_for_file: Dict[str, int], filenames=None) -> None:
        self.LOGGER.info("Started calculating ast metrics")
        super().__init__()
        self.LOGGER.info("Started loading ast to memory")
        self.asts = FileAst.load_asts_from_files(ast_path, filenames)
        self.LOGGER.info("End loading ast to memory")
        self.variable_metrics_calculator = VariableMetricsCalculator(self.asts)
        self.loops_metrics_calculator = StatementsMetricsCalculator(self.asts, character_number_for_file)
        self.LOGGER.info("End calculating ast metrics")

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

    @staticmethod
    def get_metric_names():
        return VariableMetricsCalculator.get_metrics_names() + StatementsMetricsCalculator.get_metrics_names() + \
               ["maximum_depth_of_an_ast"]
