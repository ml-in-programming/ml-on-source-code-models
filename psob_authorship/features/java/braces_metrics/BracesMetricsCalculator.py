import torch
from typing import Dict, Set

from psob_authorship.features.utils import get_absfilepaths, divide_with_handling_zero_division


class BracesMetricsCalculator:
    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.tensor([
            self.percentage_of_open_braces_alone_in_a_line(filepaths),
            self.percentage_of_close_braces_alone_in_a_line(filepaths)
        ])

    def percentage_of_open_braces_alone_in_a_line(self, filepaths: Set[str]) -> float:
        """
        Counts the following: (number of lines with alone open braces) / (number of lines that contain open braces) * 100
        If open brace is located in comment then it will be counted too.
        That is not good but this situation occurs rare.

        :param filepaths: paths to files for which metric should be calculated
        :return: open braces alone in a line metric
        """
        return divide_with_handling_zero_division(
            sum([self.alone_open_braces_for_file[filepath] for filepath in filepaths]),
            sum([self.open_braces_for_file[filepath] for filepath in filepaths]),
            "calculating percentage of open braces alone in a line for " + str(filepaths)
        ) * 100

    def percentage_of_close_braces_alone_in_a_line(self, filepaths: Set[str]) -> float:
        """
        Counts the following: (number of lines with alone close braces) / (number of lines that contain close braces) * 100
        If close brace is located in comment then it will be counted too.
        That is not good but this situation occurs rare.

        :param filepaths: paths to files for which metric should be calculated
        :return: close braces alone in a line metric
        """
        return divide_with_handling_zero_division(
            sum([self.alone_close_braces_for_file[filepath] for filepath in filepaths]),
            sum([self.close_braces_for_file[filepath] for filepath in filepaths]),
            "calculating percentage of close braces alone in a line for " + str(filepaths)
        ) * 100

    OPEN_BRACE = '{'
    CLOSE_BRACE = '}'

    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.alone_open_braces_for_file: Dict[str, int] = {}
        self.open_braces_for_file: Dict[str, int] = {}
        self.alone_close_braces_for_file: Dict[str, int] = {}
        self.close_braces_for_file: Dict[str, int] = {}
        self.calculate_metrics_for_file()

    def calculate_metrics_for_file(self) -> None:
        filepaths = get_absfilepaths(self.dataset_path)
        for filepath in filepaths:
            self.calculate_metrics(filepath)

    def calculate_metrics(self, filepath: str) -> None:
        with open(filepath) as file:
            for line in file:
                self.alone_open_braces_for_file[filepath] += line.strip() == BracesMetricsCalculator.OPEN_BRACE
                self.open_braces_for_file += BracesMetricsCalculator.OPEN_BRACE in line
                self.alone_close_braces_for_file += line.strip() == BracesMetricsCalculator.CLOSE_BRACE
                self.close_braces_for_file += BracesMetricsCalculator.CLOSE_BRACE in line
