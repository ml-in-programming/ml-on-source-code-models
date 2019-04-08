import logging
import os


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


def divide_with_percentage_and_handling_zero_division(numerator, denominator, logger: logging.Logger,
                                                      log_information, zero_division_return=float(-1)):
    result = divide_with_handling_zero_division(numerator, denominator, logger, log_information, zero_division_return)
    return zero_division_return if result == zero_division_return else result * 100


def divide_with_handling_zero_division(numerator, denominator, logger: logging.Logger, log_information,
                                       zero_division_return=float(-1)):
    if denominator == 0:
        logger.warning("SUSPICIOUS ZERO DIVISION: " + log_information)
        return zero_division_return
    else:
        result = numerator / denominator
        if result < 0:
            logger.warning("SUSPICIOUS METRIC BELOW ZERO: " + log_information)
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
