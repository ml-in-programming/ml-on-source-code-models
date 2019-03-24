import os
import subprocess
import torch
from typing import Dict, Set

from psob_authorship.features.utils import divide_with_handling_zero_division, get_absfilepaths


class LineMetricsCalculator:
    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.tensor([
            self.ratio_of_blank_lines_to_code_lines(filepaths),
            self.ratio_of_comment_lines_to_code_lines(filepaths),
            self.percentage_of_block_comments_to_all_comment_lines(filepaths)
        ])

    def ratio_of_blank_lines_to_code_lines(self, filepaths: Set[str]) -> float:
        """
        Blank line is a line consisting only of zero or several whitespaces.
        More formally in UML notation: blank line = whitespace[0..*]
        Code line is any line that is not blank line or comment line.

        :param filepaths: paths to files for which metric should be calculated
        :return: blank lines metric
        """
        return divide_with_handling_zero_division(
            sum([self.blank_lines_for_file[filepath] for filepath in filepaths]),
            sum([self.code_lines_for_file[filepath] for filepath in filepaths]),
            "calculating ratio of blank lines to code lines for " + str(filepaths)
        )

    def ratio_of_comment_lines_to_code_lines(self, filepaths: Set[str]) -> float:
        """
        Comment line is a line that starts (without regard to leading whitespaces) with "//" or located in commented block.
        If any symbol (except whitespace) exists before "//" and it is not located in commented block
        then this line is accounted as code line not a comment line.
        Code line is any line that is not blank line or comment line.

        :param filepaths: paths to files for which metric should be calculated
        :return: comment lines metric
        """
        return divide_with_handling_zero_division(
            sum([self.comment_lines_for_file[filepath] for filepath in filepaths]),
            sum([self.code_lines_for_file[filepath] for filepath in filepaths]),
            "calculating ratio of comment lines to code lines for " + str(filepaths)
        )

    def percentage_of_block_comments_to_all_comment_lines(self, filepaths: Set[str]) -> float:
        """
        Block comment size is number of comment lines in block comment.

        :param filepaths: paths to files for which metric should be calculated
        :return: block comment lines metric
        """
        return divide_with_handling_zero_division(
            sum([self.comment_lines_for_file[filepath] - self.single_line_comments_for_file[filepath]
                 for filepath in filepaths]),
            sum([self.comment_lines_for_file[filepath] for filepath in filepaths]),
            "calculating percentage of block comments to all comment lines for " + str(filepaths)
        ) * 100

    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.single_line_comments_for_file: Dict[str, int] = {}
        self.blank_lines_for_file: Dict[str, int] = {}
        self.comment_lines_for_file: Dict[str, int] = {}
        self.code_lines_for_file: Dict[str, int] = {}
        self.calculate_metrics_for_file()

    def calculate_metrics_for_file(self) -> None:
        self.count_single_line_comments_for_file()
        cloc_output = self.get_cloc_output()
        for cloc_file_output in cloc_output:
            if not cloc_file_output.startswith("Java"):
                continue
            splitted = cloc_file_output.split(',')
            filepath = os.path.abspath(splitted[1])
            self.blank_lines_for_file[filepath] = int(splitted[2])
            self.comment_lines_for_file[filepath] = int(splitted[3])
            self.code_lines_for_file[filepath] = int(splitted[4])

    def get_cloc_output(self) -> str:
        """
        Runs cloc external program and calculates number of blank, comment and code lines by file for dataset_path.
        """
        return subprocess.run(["cloc", self.dataset_path, "--by-file", "--quiet", "--csv"], check=True, stdout=subprocess.PIPE) \
            .stdout.decode("utf-8")

    def count_single_line_comments_for_file(self) -> None:
        """
        Counts number of single line comments.
        Line is single line commented if it starts with "//" (without regard to leading whitespaces) and
        is located not inside comment block.
        Now it doesn't works correctly in such case:
        /*
        //this single line comment will be counted, but shouldn't.
        */
        But this case appears too rare so implementation of it can wait.
        """
        filepaths = get_absfilepaths(self.dataset_path)
        for filepath in filepaths:
            self.single_line_comments_for_file[filepath] = self.count_single_line_comments(filepath)

    def count_single_line_comments(self, filepath: str) -> int:
        with open(filepath) as file:
            return sum(line.lstrip().startswith("//") for line in file)
