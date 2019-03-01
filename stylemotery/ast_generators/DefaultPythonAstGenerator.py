import ast


class DefaultPythonAstGenerator:
    NONE = "None"

    def __init__(self) -> None:
        super().__init__()
        self.node_types = []
        for x in dir(ast):
            try:
                if isinstance(ast.__getattribute__(x)(), ast.AST):
                    self.node_types.append(x)
            except TypeError:
                pass
        self.node_types.append(self.NONE)
        self.node_types_indices = {v: i for i, v in enumerate(self.node_types)}

    def get_node_id(self, node):
        return self.node_types_indices[node]

    def __call__(self, filepath):
        try:
            with open(filepath, 'r', encoding="utf-8") as file:
                tree = ast.parse(file.read())
                return tree
        except Exception as e:
            print("ERROR during creating AST: ", e, " filename", filepath)
