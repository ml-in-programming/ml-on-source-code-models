import unittest

from psob_authorship.features.java_language_features import ratio_of_blank_lines_to_code_lines, \
    ratio_of_comment_lines_to_code_lines, percentage_of_block_comments_to_all_comment_lines, \
    percentage_of_open_braces_alone_in_a_line, percentage_of_close_braces_alone_in_a_line, \
    percentage_of_variable_naming_without_uppercase_letters, \
    percentage_of_variable_naming_starting_with_lowercase_letters, average_variable_name_length, \
    ratio_of_macro_variables


class BaseFeaturesTest(unittest.TestCase):
    ast_path = "../test_data/asts"

    def check_blank_lines_ratio(self):
        self.assertEqual(self.blank_lines / self.code_lines, ratio_of_blank_lines_to_code_lines(self.test_file))

    def check_comment_lines_ratio(self):
        self.assertEqual(self.comment_lines / self.code_lines,
                         ratio_of_comment_lines_to_code_lines(self.test_file))

    def check_block_comments_percentage(self):
        self.assertEqual((self.comment_lines - self.single_line_comments_lines) / self.comment_lines * 100,
                         percentage_of_block_comments_to_all_comment_lines(self.test_file))

    def check_open_braces_alone_percentage(self):
        self.assertEqual(self.open_braces_alone / self.open_braces * 100,
                         percentage_of_open_braces_alone_in_a_line(self.test_file))

    def check_close_braces_alone_percentage(self):
        self.assertEqual(self.close_braces_alone / self.close_braces * 100,
                         percentage_of_close_braces_alone_in_a_line(self.test_file))

    def check_variable_naming_without_uppercase_letters(self):
        self.assertEqual(self.lowercase_variables / self.variables * 100,
                         percentage_of_variable_naming_without_uppercase_letters(self.test_file, self.ast_path))

    def check_variable_naming_starting_with_lowercase_letters(self):
        self.assertEqual(self.starting_with_lowercase_variables / self.variables * 100,
                         percentage_of_variable_naming_starting_with_lowercase_letters(self.test_file, self.ast_path))

    def check_average_variable_name(self):
        self.assertEqual(self.all_variables_length / self.variables,
                         average_variable_name_length(self.test_file, self.ast_path))

    def check_macros_variable(self):
        self.assertEqual(0, ratio_of_macro_variables(self.test_file, self.ast_path))

    def run_all_checks(self):
        self.check_blank_lines_ratio()
        self.check_comment_lines_ratio()
        self.check_block_comments_percentage()
        self.check_open_braces_alone_percentage()
        self.check_close_braces_alone_percentage()
        self.check_variable_naming_without_uppercase_letters()
        self.check_variable_naming_starting_with_lowercase_letters()
        self.check_average_variable_name()
        self.check_macros_variable()

    def test_on_all_test_data(self):
        self.blank_lines = 17
        self.single_line_comments_lines = 1
        self.comment_lines = 6
        self.code_lines = 56
        self.open_braces = 14
        self.open_braces_alone = 1
        self.close_braces = 14
        self.close_braces_alone = 13
        self.variables = 2 + 3
        self.all_variables_length = 9 + 23 + 1 + 1 + 1
        self.lowercase_variables = 1 + 2
        self.starting_with_lowercase_variables = 2 + 2
        self.test_file = "../test_data/author1"
        self.run_all_checks()

    def test_on_1_file(self):
        self.blank_lines = 2
        self.single_line_comments_lines = 1
        self.comment_lines = 6
        self.code_lines = 10
        self.open_braces = 2
        self.open_braces_alone = 1
        self.close_braces = 2
        self.all_variables_length = 9 + 23
        self.close_braces_alone = 1
        self.variables = 2
        self.lowercase_variables = 1
        self.starting_with_lowercase_variables = 2
        self.test_file = "../test_data/author1/1.java"
        self.run_all_checks()


if __name__ == '__main__':
    unittest.main()
