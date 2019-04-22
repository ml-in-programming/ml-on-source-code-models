from psob_authorship.features.java.MetricsCalculator import MetricsCalculator

CPP_CONFIG = {
    "language_name": "Cpp",
    "metrics": MetricsCalculator.ALL_METRICS.remove("average_number_of_interfaces_per_class")
}
