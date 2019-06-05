import abc

from psob_authorship.features.java.ast.AstNode import AstNode


class AstVisitor:
    @abc.abstractmethod
    def visit(self, node: AstNode):
        for child in node.children:
            self.visit(child)
        return
