import os
import unittest

from psob_authorship.features.java.ast.Ast import FileAst


class AstTest(unittest.TestCase):
    ast_path = "../test_data/asts"

    def test_ast_creation_from_one_file(self):
        filenames = ["../test_data/author2/2.java"]
        asts = FileAst.load_asts_from_files(AstTest.ast_path, filenames)
        self.assertEqual(1, len(asts))
        ast_to_check = asts[os.path.abspath(filenames[0])]
        self.assertEqual(os.path.abspath(filenames[0]), ast_to_check.filename)
        self.assertEqual('1', ast_to_check.root.token)
        self.assertEqual("<EOF>", ast_to_check.root.token_name)
        self.assertEqual('1', ast_to_check.root.node)
        self.assertEqual("Terminal", ast_to_check.root.node_name)
        self.assertEqual([], ast_to_check.root.children)

    def test_ast_creation_from_all_files(self):
        # TODO: create complex test checking the stucture of created asts.
        asts = FileAst.load_asts_from_files(AstTest.ast_path)
        self.assertEqual(4, len(asts))
