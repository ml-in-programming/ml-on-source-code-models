import logging
from collections import defaultdict
from typing import Dict, Set

import torch

from psob_authorship.features.utils import get_absfilepaths, divide_nonnegative_with_handling_zero_division, \
    divide_ratio_with_handling_zero_division


class CharacterMetricsCalculator:
    LOGGER = logging.getLogger('metrics_calculator')

    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.tensor([
            self.ratio_of_open_braces_alone_in_a_line(filepaths),
            self.ratio_of_close_braces_alone_in_a_line(filepaths),
            self.average_character_number_per_java_file(filepaths)
            # TODO: this metric decreases significantly accuracy
        ])

    def ratio_of_open_braces_alone_in_a_line(self, filepaths: Set[str]) -> float:
        """
        Counts the following: (number of lines with alone open braces) / (number of lines that contain open braces) * 100
        If open brace is located in comment then it will be counted too.
        That is not good but this situation occurs rare.

        :param filepaths: paths to files for which metric should be calculated
        :return: open braces alone in a line metric
        """
        return divide_ratio_with_handling_zero_division(
            sum([self.alone_open_braces_for_file[filepath] for filepath in filepaths]),
            sum([self.open_braces_for_file[filepath] for filepath in filepaths]),
            self.LOGGER,
            "calculating ratio of open braces alone in a line for " + str(filepaths)
        )

    def ratio_of_close_braces_alone_in_a_line(self, filepaths: Set[str]) -> float:
        """
        Counts the following: (number of lines with alone close braces) / (number of lines that contain close braces) * 100
        If close brace is located in comment then it will be counted too.
        That is not good but this situation occurs rare.

        :param filepaths: paths to files for which metric should be calculated
        :return: close braces alone in a line metric
        """
        return divide_ratio_with_handling_zero_division(
            sum([self.alone_close_braces_for_file[filepath] for filepath in filepaths]),
            sum([self.close_braces_for_file[filepath] for filepath in filepaths]),
            self.LOGGER,
            "calculating ratio of close braces alone in a line for " + str(filepaths)
        )

    def average_character_number_per_java_file(self, filepaths: Set[str]) -> float:
        """
        Character number is including line separators and etc.
        :param filepaths: paths to files for which metric should be calculated
        :return: character number / len(filepaths)
        """
        return divide_nonnegative_with_handling_zero_division(
            sum([self.character_number_for_file[filepath] for filepath in filepaths]),
            len(filepaths),
            self.LOGGER,
            "calculating average character number per java file for " + str(filepaths),
            take_log10=True
        )

    OPEN_BRACE = '{'
    CLOSE_BRACE = '}'

    def __init__(self, dataset_path: str) -> None:
        self.LOGGER.info("Started calculating character metrics")
        super().__init__()
        self.dataset_path = dataset_path
        self.alone_open_braces_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.open_braces_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.alone_close_braces_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.close_braces_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.character_number_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.calculate_metrics_for_file()
        self.LOGGER.info("End calculating character metrics")

    def calculate_metrics_for_file(self) -> None:
        filepaths = get_absfilepaths(self.dataset_path)
        for filepath in filepaths:
            self.calculate_metrics(filepath)

    def calculate_metrics(self, filepath: str) -> None:
        with open(filepath) as file:
            self.character_number_for_file[filepath] = len(file.read())
            file.seek(0)
            for line in file.read().splitlines():
                self.alone_open_braces_for_file[filepath] += line.strip() == CharacterMetricsCalculator.OPEN_BRACE
                self.open_braces_for_file[filepath] += CharacterMetricsCalculator.OPEN_BRACE in line
                self.alone_close_braces_for_file[filepath] += line.strip() == CharacterMetricsCalculator.CLOSE_BRACE
                self.close_braces_for_file[filepath] += CharacterMetricsCalculator.CLOSE_BRACE in line

    @staticmethod
    def get_metric_names():
        return [
            "ratio_of_open_braces_alone_in_a_line",
            "ratio_of_close_braces_alone_in_a_line",
            "average_character_number_per_java_file"
        ]
