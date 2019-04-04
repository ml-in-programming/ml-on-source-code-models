import os
import unittest

from psob_authorship.features.java.character_metrics.CharacterMetricsCalculator import CharacterMetricsCalculator


class AverageCharacterNumberPerJavaFileTest(unittest.TestCase):
    TEST_DATA_ROOT_DIR = \
        os.path.abspath("../../../test_data/features/java/character_metrics/average_character_number_per_java_file/")
    CHAR_NUMBER_BY_FILEPATH = {
        os.path.join(TEST_DATA_ROOT_DIR, "empty"): 0,
        os.path.join(TEST_DATA_ROOT_DIR, "one_line"): 15,
        os.path.join(TEST_DATA_ROOT_DIR, "two_lines"): 30
    }

    def test_one_file(self) -> None:
        character_metrics_calculator = CharacterMetricsCalculator(self.TEST_DATA_ROOT_DIR)
        for filepath in self.CHAR_NUMBER_BY_FILEPATH.keys():
            self.assertEqual(
                character_metrics_calculator.average_character_number_per_java_file({filepath}),
                self.CHAR_NUMBER_BY_FILEPATH[filepath]
            )

    def test_two_files(self) -> None:
        character_metrics_calculator = CharacterMetricsCalculator(self.TEST_DATA_ROOT_DIR)
        for filepath1 in self.CHAR_NUMBER_BY_FILEPATH.keys():
            for filepath2 in self.CHAR_NUMBER_BY_FILEPATH.keys():
                self.assertEqual(
                    character_metrics_calculator.average_character_number_per_java_file({filepath1, filepath2}),
                    (self.CHAR_NUMBER_BY_FILEPATH[filepath1] + self.CHAR_NUMBER_BY_FILEPATH[filepath2]) / 2
                )

    def test_all_files(self) -> None:
        character_metrics_calculator = CharacterMetricsCalculator(self.TEST_DATA_ROOT_DIR)
        self.assertEqual(
            character_metrics_calculator.average_character_number_per_java_file(
                set(self.CHAR_NUMBER_BY_FILEPATH.keys())
            ),
            sum(self.CHAR_NUMBER_BY_FILEPATH.values()) / len(self.CHAR_NUMBER_BY_FILEPATH.values())
        )
