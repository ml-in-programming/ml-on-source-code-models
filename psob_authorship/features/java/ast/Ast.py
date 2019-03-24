import os
import re

from psob_authorship.features.java.ast.AstNode import AstNode
from psob_authorship.features.java.ast.AstVisitor import AstVisitor


def load_tokens(path_to_ast_data):
    with open(os.path.join(path_to_ast_data, "tokens.csv")) as file:
        return {line.split(',')[0]: line.split(',')[1].rstrip("\n") for line in file}


def load_asts(filenames, path_to_ast_data):
    with open(os.path.join(path_to_ast_data, "asts.csv")) as file:
        if filenames is None:
            return {line.split(',')[0]: line.split(',')[1].rstrip("\n") for line in file if not line.startswith("id")}
        else:
            filenames = [os.path.abspath(filename) for filename in filenames]
            return {line.split(',')[0]: line.split(',')[1].rstrip("\n") for line in file if
                    line.split(',')[0] in filenames}


def load_nodes(path_to_ast_data):
    with open(os.path.join(path_to_ast_data, "node_types.csv")) as file:
        return {line.split(',')[0]: line.split(',')[1].rstrip("\n") for line in file}


class Ast:
    @staticmethod
    def split_on_strings(ast_in_string):
        splitted_on_braces = re.split('([{}])', ast_in_string)
        without_spaces = []
        for splitted in splitted_on_braces:
            without_spaces += splitted.split(' ')
        return [part for part in without_spaces if part != ''][::-1]

    def __init__(self, ast_in_string, tokens, nodes) -> None:
        super().__init__()
        self.root = AstNode(Ast.split_on_strings(ast_in_string), tokens, nodes)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Ast):
            return False
        return self.root == o.root

    def accept(self, ast_visitor: AstVisitor) -> None:
        ast_visitor.visit(self.root)


class FileAst(Ast):
    def __init__(self, filename, ast_in_string, tokens, nodes) -> None:
        super().__init__(ast_in_string, tokens, nodes)
        self.filename = filename

    @staticmethod
    def load_asts_from_files(ast_path, filenames=None) -> dict:
        nodes = load_nodes(ast_path)
        tokens = load_tokens(ast_path)
        return {filename: FileAst(filename, ast, tokens, nodes)
                for filename, ast in load_asts(filenames, ast_path).items()}

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, FileAst):
            return False
        return super().__eq__(o) and self.filename == o.filename
