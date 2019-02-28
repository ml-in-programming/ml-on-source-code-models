import ast


class DefaultPythonAstGenerator:
    def __call__(self, filepath):
        try:
            with open(filepath, 'r', encoding="utf-8") as file:
                tree = ast.parse(file.read())
                return tree
        except Exception as e:
            print("ERROR during creating AST: ", e, " filename", filepath)
