class AstNode:
    def __init__(self, ast_in_strings, tokens, nodes):
        self.token = ast_in_strings.pop()
        self.token_name = tokens[self.token]
        self.node = ast_in_strings.pop()
        self.node_name = nodes[self.node]
        self.children = []
        ast_in_strings.pop()
        while ast_in_strings[-1] != '}':
            self.children.append(AstNode(ast_in_strings, tokens, nodes))
        ast_in_strings.pop()
        self.depth = 1 + max([child.depth for child in self.children], default=0)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, AstNode):
            return False
        return self.token == o.token and self.node == self.node and \
            self.token_name == self.token_name and self.node_name == o.node_name and \
            self.children == o.children
