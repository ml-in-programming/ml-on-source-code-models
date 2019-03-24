from typing import Set

from psob_authorship.features.java.ast.Ast import Node, Ast
from psob_authorship.features.java.ast.AstVisitor import AstVisitor


class AstSpecificTokensExtractor:
    def __init__(self, ast: Ast) -> None:
        super().__init__()
        self.ast = ast

    def extract_tokens_by_nodes(self, node_names: Set[str]) -> Set[str]:
        visitor = AstSpecificNodesVisitor(node_names)
        self.ast.accept(visitor)
        return visitor.token_names


class AstSpecificNodesVisitor(AstVisitor):
    def __init__(self, node_names: Set[str]) -> None:
        self.node_names = node_names
        self.token_names: Set[str] = {}

    def visit(self, node: Node):
        if node.node_name in self.node_names:
            self.token_names.add(node.token_name)
        super().visit(node)
