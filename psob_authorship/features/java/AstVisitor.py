import abc

from psob_authorship.features.Ast import Ast, Node


class AstVisitor:
    @abc.abstractmethod
    def visit(self, node: Node):
        for child in node.children:
            self.visit(child)
        return
