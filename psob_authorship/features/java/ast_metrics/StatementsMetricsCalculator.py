from collections import defaultdict
from typing import Dict, Set

import torch

from psob_authorship.features.java.ast.Ast import Ast
from psob_authorship.features.java.ast.AstNode import AstNode
from psob_authorship.features.java.ast.AstVisitor import AstVisitor
from psob_authorship.features.utils import divide_with_handling_zero_division


class StatementsMetricsCalculator:
    def get_metrics(self, filepaths: Set[str]) -> torch.Tensor:
        return torch.tensor([
            self.percentage_of_for_statements_to_all_loop_statements(filepaths),
            self.percentage_of_if_statements_to_all_conditional_statements(filepaths),
            self.average_number_of_methods_per_class(filepaths),
            self.percentage_of_catch_statements_when_dealing_with_exceptions(filepaths),
            self.ratio_of_branch_statements(filepaths),
            self.ratio_of_try_structure(filepaths),
            self.average_number_of_interfaces_per_class(filepaths)
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

    def ratio_of_branch_statements(self, filepaths: Set[str]) -> float:
        """
        Strange metric. Branch statements are: break, continue, return
        Possible variants of interpretation:
        1. number of branch / all statements
        2. number of break / (number of break + number of continue)
        3. (number of break + number of continue) / (number of break, continue, return)
        4. number of break + number of continue + number of return
        5. number of branch / number of characters
        6. (number of break + number of continue) / number of characters
        6 variant was chosen.
        :param filepaths: paths to files for which metric should be calculated
        :return: (number of break + number of continue) / number of characters
        """
        return divide_with_handling_zero_division(
            sum([self.breaks_for_file[filepath] + self.continues_for_file[filepath] for filepath in filepaths]),
            sum([self.character_number_for_file[filepath] for filepath in filepaths]),
            "calculating ratio of ratio_of_branch_statements for " + str(filepaths)
        )

    def ratio_of_try_structure(self, filepaths: Set[str]) -> float:
        """
        Strange metric. Ratio to what?
        Possible variants of interpretation:
        1. number of tries / number of characters
        2. number of tries
        1 variant was chosen.
        :param filepaths: paths to files for which metric should be calculated
        :return: number of tries / number of characters
        """
        return divide_with_handling_zero_division(
            sum([self.tries_for_file[filepath] for filepath in filepaths]),
            sum([self.character_number_for_file[filepath] for filepath in filepaths]),
            "calculating ratio of ratio_of_try_structure for " + str(filepaths)
        )

    def average_number_of_interfaces_per_class(self, filepaths: Set[str]) -> float:
        """
        TODO: calculate all of interfaces, not only implements words
        Calculates average number of interfaces per class
        :param filepaths: paths to files for which metric should be calculated
        :return: number of implements keywords / number of classes
        """
        return divide_with_handling_zero_division(
            sum([self.implements_for_file[filepath] for filepath in filepaths]),
            sum([self.classes_for_file[filepath] for filepath in filepaths]),
            "calculating average_number_of_interfaces_per_class metric for " + str(filepaths)
        )

    def __init__(self, asts: Dict[str, Ast], character_number_for_file: Dict[str, int]) -> None:
        super().__init__()
        self.character_number_for_file = character_number_for_file
        self.fors_for_file = defaultdict(lambda: 0)
        self.whiles_for_file = defaultdict(lambda: 0)
        self.ifs_for_file = defaultdict(lambda: 0)
        self.switches_for_file = defaultdict(lambda: 0)
        self.tries_for_file = defaultdict(lambda: 0)
        self.catches_for_file = defaultdict(lambda: 0)
        self.breaks_for_file = defaultdict(lambda: 0)
        self.continues_for_file = defaultdict(lambda: 0)
        self.implements_for_file = defaultdict(lambda: 0)
        self.methods_for_file = defaultdict(lambda: 0)
        self.classes_for_file = defaultdict(lambda: 0)
        for filepath, ast in asts.items():
            statements_visitor = StatementsVisitor()
            ast.accept(statements_visitor)
            self.fors_for_file[filepath] = statements_visitor.fors
            self.whiles_for_file[filepath] = statements_visitor.whiles
            self.ifs_for_file[filepath] = statements_visitor.ifs
            self.switches_for_file[filepath] = statements_visitor.switches
            self.tries_for_file[filepath] = statements_visitor.tries
            self.catches_for_file[filepath] = statements_visitor.catches
            self.breaks_for_file[filepath] = statements_visitor.breaks
            self.continues_for_file[filepath] = statements_visitor.continues
            self.implements_for_file[filepath] = statements_visitor.visited_implements
            self.methods_for_file[filepath] = statements_visitor.methods
            self.classes_for_file[filepath] = statements_visitor.classes


class StatementsVisitor(AstVisitor):
    FOR = "for"
    WHILE = "while"  # including do while loops
    IF = "if"
    SWITCH = "switch"
    TRY = "try"
    CATCH = "catch"
    BREAK = "break"
    CONTINUE = "continue"
    IMPLEMENTS = "implements"
    TYPE_LIST = "typeList"
    TERMINAL = "Terminal"
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
        self.breaks = 0
        self.continues = 0
        self.visited_implements = False
        self.implements = 0
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
        elif node.token_name == StatementsVisitor.BREAK:
            self.breaks += 1
        elif node.token_name == StatementsVisitor.CONTINUE:
            self.continues += 1
        elif node.token_name == StatementsVisitor.IMPLEMENTS:
            self.visited_implements = True
        if node.node_name == StatementsVisitor.METHOD:
            self.methods += 1
        elif node.node_name == StatementsVisitor.CLASS:
            self.classes += 1
        if self.visited_implements:
            if node.node_name == self.TYPE_LIST:
                self.implements += len(node.children)
            elif node.node_name == self.TERMINAL:
                self.implements += 1
            else:
                raise ValueError("visitor visited implements statement "
                                 "but next child is not an interface name or type list")
        super().visit(node)
