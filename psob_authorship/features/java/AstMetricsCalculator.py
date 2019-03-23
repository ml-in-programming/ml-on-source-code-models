from psob_authorship.features.Ast import FileAst

class AstMetricsCalculator:
    def __init__(self, ast_path, filenames=None) -> None:
        super().__init__()
        self.asts = FileAst.load_asts_from_files(ast_path, filenames)

