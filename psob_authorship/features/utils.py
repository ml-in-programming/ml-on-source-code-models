import logging
import math
import os

from typing import Tuple


def get_log_filepath(log_filename: str) -> str:
    log_root = "../logs"
    log_filename = log_filename + ".log"
    return os.path.join(log_root, log_filename)


def configure_logger_by_default(logger: logging.Logger) -> None:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(get_log_filepath(logger.name))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


def get_absfilepaths(file_or_dir):
    if not os.path.isabs(file_or_dir):
        raise ValueError("passed argument to get_absfilepaths(file_or_dir) function must be absolute path")
    if not os.path.exists(file_or_dir):
        raise ValueError("passed argument to get_absfilepaths(file_or_dir) function must be existing path")
    if os.path.isfile(file_or_dir):
        return [os.path.abspath(file_or_dir)]
    filenames = []
    for root, dirs, files in os.walk(file_or_dir):
        for name in files:
            filenames += [os.path.abspath(os.path.join(root, name))]
    return filenames


def divide_percentage_with_handling_zero_division(numerator, denominator, logger: logging.Logger,
                                                  log_information, zero_division_return=-1.0):
    result = divide_with_handling_zero_division(numerator, denominator, logger,
                                                log_information, (0, 1), zero_division_return)
    return zero_division_return if result == zero_division_return else result * 100


def divide_ratio_with_handling_zero_division(numerator, denominator, logger: logging.Logger,
                                             log_information, zero_division_return=-1.0):
    return divide_with_handling_zero_division(numerator, denominator, logger,
                                              log_information, (0, 1), zero_division_return)


def divide_nonnegative_with_handling_zero_division(numerator, denominator, logger: logging.Logger,
                                                   log_information, take_log10=False, log_from_zero=-100.0,
                                                   zero_division_return=-1.0):
    result = divide_with_handling_zero_division(numerator, denominator, logger,
                                                log_information, (0, float("inf")), zero_division_return)
    if result == zero_division_return:
        return zero_division_return
    if result == 0 and take_log10:
        return log_from_zero
    return math.log10(result) if take_log10 else result


def divide_with_handling_zero_division(numerator, denominator, logger: logging.Logger, log_information,
                                       expected_bounds: Tuple[float, float], zero_division_return=-1.0):
    if denominator == 0:
        logger.warning("SUSPICIOUS ZERO DIVISION: " + log_information)
        return zero_division_return
    else:
        result = numerator / denominator
        if result < expected_bounds[0] or result > expected_bounds[1]:
            out_of_bounds_warning = "VALUE " + str(result) + ". SUSPICIOUS METRIC OUT OF BOUNDS [" + \
                                 str(expected_bounds[0]) + "; " + str(expected_bounds[1]) + "]: "
            logger.warning(out_of_bounds_warning + log_information)
        return result


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    n = min(n, len(l))
    step = int(len(l) / n)
    for i in range(0, n):
        if i == n - 1:
            yield l[i * step:len(l)]
        else:
            yield l[i * step:(i + 1) * step]
