from stylemotery.ast_generators.DefaultPythonAstGenerator import DefaultPythonAstGenerator


class AstGeneratorsFactory:
    @staticmethod
    def create(language):
        if language == "python":
            return DefaultPythonAstGenerator()
