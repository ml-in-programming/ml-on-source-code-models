from psob_authorship.features.java.MetricsCalculator import MetricsCalculator

JAVA_CONFIG = {
    "language_name": "Java",
    "metrics": MetricsCalculator.ALL_METRICS,
    "TYPE_LIST": "typeList",
    "TERMINAL": "Terminal",
    "CLASS_OR_INTERFACE_TYPE": "classOrInterfaceType",
    "METHOD": "methodDeclaration",
    "CLASS": "classDeclaration",
    "ENUM": "enumDeclaration",
    "VARIABLE_NODE_NAMES": [["variableDeclarator", "enhancedForControl", "variableDeclaratorId"], [0, 1, 0]]
}
