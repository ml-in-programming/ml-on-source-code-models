import torch
from typing import Dict, Set

from psob_authorship.features.java.ast.Ast import FileAst
from psob_authorship.features.java.ast.AstNode import AstNode
from psob_authorship.features.java.ast.AstVisitor import AstVisitor
from psob_authorship.features.utils import divide_with_handling_zero_division


class StatementsMetricsCalculator:
    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.tensor([
            self.percentage_of_for_statements_to_all_loop_statements(filepaths),
            self.percentage_of_if_statements_to_all_conditional_statements(filepaths),
            self.average_number_of_methods_per_class(filepaths),
            self.percentage_of_catch_statements_when_dealing_with_exceptions(filepaths)
        ])

    def percentage_of_for_statements_to_all_loop_statements(self, filepaths: Set[str]) -> float:
        """
        Loop statements are: while (including do while) and for.
        :param filepaths: paths to files for which metric should be calculated
        :return: (number of for statements) / (number of all loop statements) * 100
        """
        return divide_with_handling_zero_division(
            sum([self.fors_for_file[filepath] for filepath in filepaths]),
            sum([self.fors_for_file[filepath] + self.whiles_for_file[filepath] for filepath in filepaths]),
            "calculating percentage of for statements to all loop statements for " + str(filepaths)
        ) * 100

    def percentage_of_if_statements_to_all_conditional_statements(self, filepaths: Set[str]) -> float:
        """
        Conditional statements are: if (including if else) and switch.
        :param filepaths: paths to files for which metric should be calculated
        :return: (number of if statements) / (number of all conditional statements) * 100
        """
        return divide_with_handling_zero_division(
            sum([self.ifs_for_file[filepath] for filepath in filepaths]),
            sum([self.ifs_for_file[filepath] + self.switches_for_file[filepath] for filepath in filepaths]),
            "calculating percentage of if statements to all conditional statements for " + str(filepaths)
        ) * 100

    def average_number_of_methods_per_class(self, filepaths: Set[str]) -> float:
        """
        Calculates average of methods per class considering all given files
        :param filepaths: paths to files for which metric should be calculated
        :return: (number of methods for all filepaths) / (number of classes for all filepaths)
        """
        return divide_with_handling_zero_division(
            sum([self.methods_for_file[filepath] for filepath in filepaths]),
            sum([self.classes_for_file[filepath] for filepath in filepaths]),
            "calculating average number of methods per class for " + str(filepaths)
        )

    def percentage_of_catch_statements_when_dealing_with_exceptions(self, filepaths: Set[str]) -> float:
        """
        Catches are accounted with tries, that is why we need to subtract catches from tries.
        :param filepaths: paths to files for which metric should be calculated
        :return: (number of catches statements) / (number of tries statements) * 100
        """
        return divide_with_handling_zero_division(
            sum([self.catches_for_file[filepath] for filepath in filepaths]),
            sum([self.tries_for_file[filepath] - self.catches_for_file[filepath] for filepath in filepaths]),
            "calculating percentage of if statements to all conditional statements for " + str(filepaths)
        ) * 100

    def __init__(self, asts: Dict[str, FileAst]) -> None:
        super().__init__()
        self.fors_for_file = {}
        self.whiles_for_file = {}
        self.ifs_for_file = {}
        self.switches_for_file = {}
        self.tries_for_file = {}
        self.catches_for_file = {}
        self.methods_for_file = {}
        self.classes_for_file = {}
        for filepath, ast in asts.items():
            statements_visitor = StatementsVisitor()
            ast.accept(statements_visitor)
            self.fors_for_file[filepath] = statements_visitor.fors
            self.whiles_for_file[filepath] = statements_visitor.whiles
            self.ifs_for_file[filepath] = statements_visitor.ifs
            self.switches_for_file[filepath] = statements_visitor.switches
            self.tries_for_file[filepath] = statements_visitor.tries
            self.catches_for_file[filepath] = statements_visitor.catches
            self.methods_for_file[filepath] = statements_visitor.methods
            self.classes_for_file[filepath] = statements_visitor.classes


class StatementsVisitor(AstVisitor):
    FOR = "for"
    WHILE = "while"  # including do while loops
    IF = "if"
    SWITCH = "switch"
    TRY = "try"
    CATCH = "catch"
    METHOD = "methodDeclaration"
    CLASS = "classDeclaration"

    def __init__(self) -> None:
        super().__init__()
        self.fors = 0
        self.whiles = 0
        self.ifs = 0
        self.switches = 0
        self.tries = 0
        self.catches = 0
        self.methods = 0
        self.classes = 0

    def visit(self, node: AstNode):
        if node.token_name == StatementsVisitor.FOR:
            self.fors += 1
        elif node.token_name == StatementsVisitor.WHILE:
            self.whiles += 1
        elif node.token_name == StatementsVisitor.IF:
            self.ifs += 1
        elif node.token_name == StatementsVisitor.SWITCH:
            self.switches += 1
        elif node.token_name == StatementsVisitor.TRY:
            self.tries += 1
        elif node.token_name == StatementsVisitor.CATCH:
            self.catches += 1
        if node.node_name == StatementsVisitor.METHOD:
            self.methods += 1
        elif node.node_name == StatementsVisitor.CLASS:
            self.classes += 1
        super().visit(node)
