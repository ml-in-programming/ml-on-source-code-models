import os


def get_absfilepaths(file_or_dir):
    if os.path.isfile(file_or_dir):
        return [os.path.abspath(file_or_dir)]
    filenames = []
    for root, dirs, files in os.walk(file_or_dir):
        for name in files:
            filenames += [os.path.abspath(os.path.join(root, name))]
    return filenames


def divide_with_handling_zero_division(numerator, denominator, log_information, zero_division_return=float(0)):
    return zero_division_return if denominator == 0 else numerator / denominator
