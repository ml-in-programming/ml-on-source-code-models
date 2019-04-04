from collections import defaultdict
from typing import Dict, Set, List, Tuple

import torch

from psob_authorship.features.java.ast.Ast import Ast
from psob_authorship.features.java.ast.AstSpecificTokensExtractor import AstSpecificTokensExtractor
from psob_authorship.features.utils import divide_with_handling_zero_division


class VariableMetricsCalculator:
    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.tensor([
            self.percentage_of_variable_naming_without_uppercase_letters(filepaths),
            self.percentage_of_variable_naming_starting_with_lowercase_letters(filepaths),
            self.average_variable_name_length(filepaths),
            self.ratio_of_macro_variables(filepaths),
            self.preference_for_cyclic_variables(filepaths)
        ])

    def percentage_of_variable_naming_without_uppercase_letters(self, filepaths: Set[str]) -> float:
        """
        This metric uses ast of files so you will need to generate files stated below with PathMiner.
        :param filepaths: paths to files for which metric should be calculated
        :return: variables without uppercase letters metric
        """
        return divide_with_handling_zero_division(
            sum([self.number_of_variables_in_lowercase_for_file[filepath] for filepath in filepaths]),
            sum([self.number_of_variables_in_lowercase_for_file[filepath] for filepath in filepaths]),
            "calculating metric percentage of variable naming without uppercase letters for " + str(filepaths)
        ) * 100

    def percentage_of_variable_naming_starting_with_lowercase_letters(self, filepaths: Set[str]) -> float:
        """
        This metric uses ast of files so you will need to generate files stated below with PathMiner.
        :param filepaths: paths to files for which metric should be calculated
        :return: variables starting with lowercase letters metric
        """
        return divide_with_handling_zero_division(
            sum([self.number_of_variables_starting_with_lowercase_for_file[filepath] for filepath in filepaths]),
            sum([self.number_of_variables_in_lowercase_for_file[filepath] for filepath in filepaths]),
            "calculating metric percentage of variable naming starting with lowercase letters for " + str(filepaths)
        ) * 100

    def average_variable_name_length(self, filepaths: Set[str]) -> float:
        """
        This metric uses ast of files so you will need to generate files stated below with PathMiner.
        :param filepaths: paths to files for which metric should be calculated
        :return: average variable name metric
        """
        return divide_with_handling_zero_division(
            sum([self.length_of_variables_for_file[filepath] for filepath in filepaths]),
            sum([self.number_of_variables_for_file[filepath] for filepath in filepaths]),
            "calculating metric average_variable_name_length for " + str(filepaths)
        )

    def preference_for_cyclic_variables(self, filepaths: Set[str]) -> float:
        """
        Strange metric. It is interpreted as ratio of variables declared in for loops
        to all variables.
        :param filepaths: paths to files for which metric should be calculated
        :return: ratio of loops variables to all variables
        """
        return divide_with_handling_zero_division(
            sum([self.number_of_variables_in_for_control_for_file[filepath] for filepath in filepaths]),
            sum([self.number_of_variables_for_file[filepath] for filepath in filepaths]),
            "calculating metric preference_for_cyclic_variables for " + str(filepaths)
        ) * 100

    def ratio_of_macro_variables(self, filepaths: Set[str]) -> float:
        """
        This metric is strange. There is no macros in Java. That is why it always returns 0.
        :param filepaths: paths to files for which metric should be calculated
        :return: 0
        """
        return 0

    def percentage_of_for_statements_to_all_loop_statements(self, filepaths: Set[str]) -> float:
        pass

    # second list is child numbers
    VARIABLE_NODE_NAMES = [["variableDeclarator", "enhancedForControl", "variableDeclaratorId"], [0, 1, 0]]

    @staticmethod
    def get_variable_names_for_file(asts: Dict[str, Ast]) -> Dict[str, List[Tuple[str, bool]]]:
        return {filepath: VariableMetricsCalculator.get_variable_names(ast) for filepath, ast in asts.items()}

    @staticmethod
    def get_variable_names(ast: Ast) -> List[Tuple[str, bool]]:
        return AstSpecificTokensExtractor(ast).extract_tokens_by_nodes(VariableMetricsCalculator.VARIABLE_NODE_NAMES)

    def __init__(self, asts: Dict[str, Ast]) -> None:
        super().__init__()
        self.variable_names_for_file = self.get_variable_names_for_file(asts)
        self.number_of_variables_in_lowercase_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.number_of_variables_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.number_of_variables_in_for_control_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.number_of_variables_starting_with_lowercase_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.length_of_variables_for_file: Dict[str, int] = defaultdict(lambda: 0)
        self.calculate_metrics_for_file()

    def calculate_metrics_for_file(self):
        for filepath in self.variable_names_for_file.keys():
            self.calculate_metrics(filepath)

    def calculate_metrics(self, filepath: str) -> None:
        for variable_name, in_for_control in self.variable_names_for_file[filepath]:
            self.number_of_variables_for_file[filepath] += 1
            self.number_of_variables_in_for_control_for_file[filepath] += in_for_control
            self.number_of_variables_in_lowercase_for_file[filepath] += variable_name.islower()
            self.number_of_variables_starting_with_lowercase_for_file[filepath] += variable_name[0].islower()
            self.length_of_variables_for_file[filepath] += len(variable_name)
