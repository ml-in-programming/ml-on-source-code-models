import os
import unittest

from psob_authorship.features.java.ast.Ast import FileAst
from psob_authorship.features.java.ast_metrics.VariableMetricsCalculator import VariableMetricsCalculator


class AverageVariableNameLength(unittest.TestCase):
    TEST_DATA_ROOT_DIR = \
        os.path.abspath("../../../../../test_data/"
                        "features/java/ast_metrics/variable_metrics/average_variable_name_length/")

    AST_PATH = \
        os.path.join(TEST_DATA_ROOT_DIR, "asts")

    VARIABLE_LENGTH_BY_FILEPATH = {
        os.path.join(TEST_DATA_ROOT_DIR, "1_variable.java"): 1,
        os.path.join(TEST_DATA_ROOT_DIR, "2_variables.java"): 2,
        os.path.join(TEST_DATA_ROOT_DIR, "3_variables.java"): 6
    }
    VARIABLE_NUMBER_BY_FILEPATH = {
        os.path.join(TEST_DATA_ROOT_DIR, "1_variable.java"): 1,
        os.path.join(TEST_DATA_ROOT_DIR, "2_variables.java"): 2,
        os.path.join(TEST_DATA_ROOT_DIR, "3_variables.java"): 3
    }
    METRICS_CALCULATOR = variable_metrics_calculator = VariableMetricsCalculator(FileAst.load_asts_from_files(AST_PATH))

    def test_one_variable(self):
        for filepath in self.VARIABLE_LENGTH_BY_FILEPATH.keys():
            self.assertEqual(
                self.VARIABLE_LENGTH_BY_FILEPATH[filepath] / self.VARIABLE_NUMBER_BY_FILEPATH[filepath],
                self.METRICS_CALCULATOR.average_variable_name_length({filepath})
            )

    def test_two_files(self):
        for filepath1 in self.VARIABLE_LENGTH_BY_FILEPATH.keys():
            for filepath2 in self.VARIABLE_LENGTH_BY_FILEPATH.keys():
                self.assertEqual(
                    (self.VARIABLE_LENGTH_BY_FILEPATH[filepath1] + self.VARIABLE_LENGTH_BY_FILEPATH[filepath2]) /
                    (self.VARIABLE_NUMBER_BY_FILEPATH[filepath1] + self.VARIABLE_NUMBER_BY_FILEPATH[filepath2]),
                    self.METRICS_CALCULATOR.average_variable_name_length({filepath1, filepath2})
                )

    def test_all_files(self):
        self.assertEqual(
            sum(self.VARIABLE_LENGTH_BY_FILEPATH.values()) / sum(self.VARIABLE_NUMBER_BY_FILEPATH.values()),
            self.METRICS_CALCULATOR.average_variable_name_length(set(self.VARIABLE_LENGTH_BY_FILEPATH.keys()))
        )


if __name__ == '__main__':
    unittest.main()
