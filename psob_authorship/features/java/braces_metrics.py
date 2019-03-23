from psob_authorship.features.utils import get_filenames, divide_with_handling_zero_division


def percentage_of_open_braces_alone_in_a_line(filenames, file_asts):
    """
    Counts the following: (number of lines with alone open braces) / (number of lines that contain open braces) * 100
    If open brace is located in comment then it will be counted too.
    That is not good but this situation occurs rare.

    :param filenames: list of paths to files to calculate metric
    :param file_asts: list of ast trees for each file
    :return: open braces alone in a line metric
    """
    return percentage_of_symbol_alone_in_a_line(filenames, '{')


def percentage_of_close_braces_alone_in_a_line(filenames, file_asts):
    """
    Counts the following: (number of lines with alone close braces) / (number of lines that contain close braces) * 100
    If close brace is located in comment then it will be counted too.
    That is not good but this situation occurs rare.

    :param filenames: list of paths to files to calculate metric
    :return: close braces alone in a line metric
    """
    return percentage_of_symbol_alone_in_a_line(filenames, '}')


def percentage_of_symbol_alone_in_a_line(filenames, symbol):
    alone_symbols = sum(number_of_symbols_alone_in_a_line(filename, symbol) for filename in filenames)
    all_symbols = sum(number_of_symbols_in_a_line(filename, symbol) for filename in filenames)
    return divide_with_handling_zero_division(alone_symbols, all_symbols) * 100


def number_of_symbols_alone_in_a_line(filename, symbol):
    with open(filename) as file:
        return sum(line.strip() == symbol for line in file)


def number_of_symbols_in_a_line(filename, symbol):
    with open(filename) as file:
        return sum(symbol in line for line in file)
