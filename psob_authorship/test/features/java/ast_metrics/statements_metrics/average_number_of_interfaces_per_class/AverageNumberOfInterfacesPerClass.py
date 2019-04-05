import os
import unittest

from psob_authorship.features.java.ast.Ast import FileAst
from psob_authorship.features.java.ast_metrics.StatementsMetricsCalculator import StatementsMetricsCalculator


class MyTestCase(unittest.TestCase):
    TEST_DATA_ROOT_DIR = \
        os.path.abspath("../../../../../test_data/"
                        "features/java/ast_metrics/statements_metrics/average_number_of_interfaces_per_class/")

    AST_PATH = \
        os.path.join(TEST_DATA_ROOT_DIR, "asts")

    INTERFACE_NUMBER_BY_FILEPATH = {
        os.path.join(TEST_DATA_ROOT_DIR, "1_interface.java"): 1,
        os.path.join(TEST_DATA_ROOT_DIR, "2_interfaces.java"): 2,
        os.path.join(TEST_DATA_ROOT_DIR, "2_classes.java"): 3,
        os.path.join(TEST_DATA_ROOT_DIR, "enum_interfaces.java"): 1,
        os.path.join(TEST_DATA_ROOT_DIR, "generic_interfaces.java"): 7
    }
    CLASSES_NUMBER_BY_FILEPATH = {
        os.path.join(TEST_DATA_ROOT_DIR, "1_interface.java"): 1,
        os.path.join(TEST_DATA_ROOT_DIR, "2_interfaces.java"): 1,
        os.path.join(TEST_DATA_ROOT_DIR, "2_classes.java"): 2,
        os.path.join(TEST_DATA_ROOT_DIR, "enum_interfaces.java"): 1,
        os.path.join(TEST_DATA_ROOT_DIR, "generic_interfaces.java"): 4
    }
    CHARACTER_NUMBER_FOR_FILE = {
        os.path.join(TEST_DATA_ROOT_DIR, "1_interface.java"): 31,
        os.path.join(TEST_DATA_ROOT_DIR, "2_interfaces.java"): 45,
        os.path.join(TEST_DATA_ROOT_DIR, "2_classes.java"): 79,
        os.path.join(TEST_DATA_ROOT_DIR, "enum_interfaces.java"): 65,
        os.path.join(TEST_DATA_ROOT_DIR, "generic_interfaces.java"): 392
    }
    METRICS_CALCULATOR = variable_metrics_calculator = StatementsMetricsCalculator(
        FileAst.load_asts_from_files(AST_PATH), CHARACTER_NUMBER_FOR_FILE
    )

    def test_one_variable(self):
        for filepath in self.INTERFACE_NUMBER_BY_FILEPATH.keys():
            self.assertEqual(
                self.INTERFACE_NUMBER_BY_FILEPATH[filepath] / self.CLASSES_NUMBER_BY_FILEPATH[filepath],
                self.METRICS_CALCULATOR.average_number_of_interfaces_per_class({filepath})
            )

    def test_two_files(self):
        for filepath1 in self.INTERFACE_NUMBER_BY_FILEPATH.keys():
            for filepath2 in self.INTERFACE_NUMBER_BY_FILEPATH.keys():
                self.assertEqual(
                    (self.INTERFACE_NUMBER_BY_FILEPATH[filepath1] + self.INTERFACE_NUMBER_BY_FILEPATH[filepath2]) /
                    (self.CLASSES_NUMBER_BY_FILEPATH[filepath1] + self.CLASSES_NUMBER_BY_FILEPATH[filepath2]),
                    self.METRICS_CALCULATOR.average_number_of_interfaces_per_class({filepath1, filepath2})
                )

    def test_all_files(self):
        self.assertEqual(
            sum(self.INTERFACE_NUMBER_BY_FILEPATH.values()) / sum(self.CLASSES_NUMBER_BY_FILEPATH.values()),
            self.METRICS_CALCULATOR.average_number_of_interfaces_per_class(set(self.INTERFACE_NUMBER_BY_FILEPATH.keys()))
        )


if __name__ == '__main__':
    unittest.main()
