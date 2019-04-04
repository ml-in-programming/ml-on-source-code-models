import os


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


def divide_with_handling_zero_division(numerator, denominator, log_information, zero_division_return=float(-1)):
    if denominator == 0:
        print("SUSPICIOUS ZERO DIVISION: " + log_information)
    return zero_division_return if denominator == 0 else numerator / denominator


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    n = min(n, len(l))
    step = int(len(l) / n)
    for i in range(0, n):
        if i == n - 1:
            yield l[i * step:len(l)]
        else:
            yield l[i * step:(i + 1) * step]
