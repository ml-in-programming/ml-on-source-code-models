import unittest

from psob_authorship.features.java_language_features import ratio_of_blank_lines_to_code_lines, \
    ratio_of_comment_lines_to_code_lines, percentage_of_block_comments_to_all_comment_lines, \
    percentage_of_open_braces_alone_in_a_line, percentage_of_close_braces_alone_in_a_line


class BaseFeaturesTest(unittest.TestCase):
    single_line_comments_lines = None
    comment_lines = None
    code_lines = None
    open_braces = None
    open_braces_alone = None
    close_braces = None
    close_braces_alone = None
    test_file = None

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

    def run_all_checks(self):
        self.check_blank_lines_ratio()
        self.check_comment_lines_ratio()
        self.check_block_comments_percentage()
        self.check_open_braces_alone_percentage()
        self.check_close_braces_alone_percentage()

    def test_on_all_test_data(self):
        self.blank_lines = 17
        self.single_line_comments_lines = 1
        self.comment_lines = 6
        self.code_lines = 50
        self.open_braces = 14
        self.open_braces_alone = 1
        self.close_braces = 14
        self.close_braces_alone = 13
        self.test_file = "../test_data"
        self.run_all_checks()

    def test_on_1_file(self):
        self.blank_lines = 2
        self.single_line_comments_lines = 1
        self.comment_lines = 6
        self.code_lines = 7
        self.open_braces = 2
        self.open_braces_alone = 1
        self.close_braces = 2
        self.close_braces_alone = 1
        self.test_file = "../test_data/1.java"
        self.run_all_checks()


if __name__ == '__main__':
    unittest.main()
