from psob_authorship.features.java.MetricsCalculator import MetricsCalculator

CPP_CONFIG = {
    "language_name": "C++",
    "metrics": [metric for metric in MetricsCalculator.ALL_METRICS if metric != "average_number_of_interfaces_per_class"],
    "TYPE_LIST": None,
    "TERMINAL": "Terminal",
    "CLASS_OR_INTERFACE_TYPE": None,
    "METHOD": "function_def",
    "CLASS": "class_def",
    "ENUM": "class_def",
    "VARIABLE_NODE_NAMES": [["var_decl"], [0]]
}
