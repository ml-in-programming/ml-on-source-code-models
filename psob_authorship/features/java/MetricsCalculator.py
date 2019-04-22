import logging
from typing import Set, List, Dict

import torch

from psob_authorship.features.java.ast_metrics.AstMetricsCalculator import AstMetricsCalculator
from psob_authorship.features.java.character_metrics.CharacterMetricsCalculator import CharacterMetricsCalculator
from psob_authorship.features.java.line_metrics.LineMetricsCalculator import LineMetricsCalculator
from psob_authorship.features.utils import configure_logger_by_default


class MetricsCalculator:
    LOGGER = logging.getLogger('metrics_calculator')
    ALL_METRICS = [
        "ratio_of_blank_lines_to_code_lines",
        "ratio_of_comment_lines_to_code_lines",
        "ratio_of_block_comments_to_all_comment_lines",
        "ratio_of_open_braces_alone_in_a_line",
        "ratio_of_close_braces_alone_in_a_line",
        "average_character_number_per_java_file",
        "ratio_of_variable_naming_without_uppercase_letters",
        "ratio_of_variable_naming_starting_with_lowercase_letters",
        "average_variable_name_length",
        "ratio_of_macro_variables",
        "preference_for_cyclic_variables",
        "ratio_of_for_statements_to_all_loop_statements",
        "ratio_of_if_statements_to_all_conditional_statements",
        "average_number_of_methods_per_class",
        "ratio_of_catch_statements_when_dealing_with_exceptions",
        "ratio_of_branch_statements",
        "ratio_of_try_structure",
        "average_number_of_interfaces_per_class",
        "maximum_depth_of_an_ast"
    ]

    @staticmethod
    def get_metric_names() -> List[str]:
        return LineMetricsCalculator.get_metric_names() + CharacterMetricsCalculator.get_metric_names() + AstMetricsCalculator.get_metric_names()

    def __init__(self, language_config: Dict[str, str], dataset_path: str, ast_path: str) -> None:
        configure_logger_by_default(self.LOGGER)
        self.LOGGER.info("Started calculating metrics")
        super().__init__()
        self.language_config = language_config
        self.line_metrics_calculator = LineMetricsCalculator(language_config, dataset_path)
        self.character_metrics_calculator = CharacterMetricsCalculator(dataset_path)
        self.ast_metrics_calculator = \
            AstMetricsCalculator(ast_path, self.character_metrics_calculator.character_number_for_file)
        self.LOGGER.info("End calculating metrics")
        self.metrics_functions = {
            "ratio_of_blank_lines_to_code_lines":
                self.line_metrics_calculator.ratio_of_blank_lines_to_code_lines,
            "ratio_of_comment_lines_to_code_lines":
                self.line_metrics_calculator.ratio_of_comment_lines_to_code_lines,
            "ratio_of_block_comments_to_all_comment_lines":
                self.line_metrics_calculator.ratio_of_block_comments_to_all_comment_lines,
            "ratio_of_open_braces_alone_in_a_line":
                self.character_metrics_calculator.ratio_of_open_braces_alone_in_a_line,
            "ratio_of_close_braces_alone_in_a_line":
                self.character_metrics_calculator.ratio_of_close_braces_alone_in_a_line,
            "average_character_number_per_java_file":
                self.character_metrics_calculator.average_character_number_per_java_file,
            "ratio_of_variable_naming_without_uppercase_letters":
                self.ast_metrics_calculator.variable_metrics_calculator.ratio_of_variable_naming_without_uppercase_letters,
            "ratio_of_variable_naming_starting_with_lowercase_letters":
                self.ast_metrics_calculator.variable_metrics_calculator.ratio_of_variable_naming_starting_with_lowercase_letters,
            "average_variable_name_length":
                self.ast_metrics_calculator.variable_metrics_calculator.average_variable_name_length,
            "ratio_of_macro_variables":
                self.ast_metrics_calculator.variable_metrics_calculator.ratio_of_macro_variables,
            "preference_for_cyclic_variables":
                self.ast_metrics_calculator.variable_metrics_calculator.preference_for_cyclic_variables,
            "ratio_of_for_statements_to_all_loop_statements":
                self.ast_metrics_calculator.statements_metrics_calculator.ratio_of_for_statements_to_all_loop_statements,
            "ratio_of_if_statements_to_all_conditional_statements":
                self.ast_metrics_calculator.statements_metrics_calculator.ratio_of_if_statements_to_all_conditional_statements,
            "average_number_of_methods_per_class":
                self.ast_metrics_calculator.statements_metrics_calculator.average_number_of_methods_per_class,
            "ratio_of_catch_statements_when_dealing_with_exceptions":
                self.ast_metrics_calculator.statements_metrics_calculator.ratio_of_catch_statements_when_dealing_with_exceptions,
            "ratio_of_branch_statements":
                self.ast_metrics_calculator.statements_metrics_calculator.ratio_of_branch_statements,
            "ratio_of_try_structure":
                self.ast_metrics_calculator.statements_metrics_calculator.ratio_of_try_structure,
            "average_number_of_interfaces_per_class":
                self.ast_metrics_calculator.statements_metrics_calculator.average_number_of_interfaces_per_class,
            "maximum_depth_of_an_ast":
                self.ast_metrics_calculator.maximum_depth_of_an_ast
        }

    def get_metrics(self, filepaths: Set[str], metrics=None) -> torch.Tensor:
        if metrics is None:
            metrics = self.language_config['metrics']
        return torch.tensor([self.metrics_functions[metric](filepaths) for metric in metrics])
