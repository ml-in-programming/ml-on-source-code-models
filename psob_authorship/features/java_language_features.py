import os
import subprocess


def get_cloc_output(file_or_dir):
    """
    If file_or_dir is dir then sum is calculated (e.g. file1_blank_lines + file2_blank_lines).
    """
    return subprocess.run(["cloc", file_or_dir, "--quiet", "--csv"], check=True, stdout=subprocess.PIPE) \
        .stdout.decode("utf-8")


def get_blank_lines_from_cloc_output(cloc_output):
    return int(cloc_output.splitlines()[2].split(',')[2])


def get_comment_lines_from_cloc_output(cloc_output):
    return int(cloc_output.splitlines()[2].split(',')[3])


def get_code_lines_from_cloc_output(cloc_output):
    return int(cloc_output.splitlines()[2].split(',')[4])


def get_ratio(file_or_dir, numerator, denominator):
    cloc_output = get_cloc_output(file_or_dir)
    return numerator(cloc_output) / denominator(cloc_output)


def ratio_of_blank_lines_to_code_lines(file_or_dir):
    """
    Blank line is a line consisting only of zero or several whitespaces.
    More formally in UML notation: blank line = whitespace[0..*]
    Code line is any line that is not blank line or comment line.

    :param file_or_dir: must be string containing path to file or directory
    :return: blank lines metric
    """
    return get_ratio(file_or_dir, get_blank_lines_from_cloc_output, get_code_lines_from_cloc_output)


def ratio_of_comment_lines_to_code_lines(file_or_dir):
    """
    Comment line is a line that starts (without regard to leading whitespaces) with "//" or located in commented block.
    If any symbol (except whitespace) exists before "//" and it is not located in commented block
    then this line is accounted as code line not a comment line.
    Code line is any line that is not blank line or comment line.

    :param file_or_dir: must be string containing path to file or directory
    :return: comment lines metric
    """
    return get_ratio(file_or_dir, get_comment_lines_from_cloc_output, get_code_lines_from_cloc_output)


def percentage_of_block_comments_to_all_comment_lines(file_or_dir):
    all_comments = get_comment_lines_from_cloc_output(get_cloc_output(file_or_dir))
    block_comments = all_comments - count_single_line_comments(file_or_dir)
    return block_comments / all_comments * 100


def count_single_line_comments(file_or_dir):
    """
    Counts number of single line comments.
    Line is single line commented if it starts with "//" (without regard to leading whitespaces) and
    is located not inside comment block.
    Now it doesn't works correctly in such case:
    /*
    //this single line comment will be counted, but shouldn't.
    */
    But this case appears too rare so implementation of it can wait.

    :param file_or_dir: must be string containing path to file or directory
    :return: single line comment metric
    """
    if os.path.isfile(file_or_dir):
        return count_single_line_comments_for_single_file(file_or_dir)

    result = 0
    for root, dirs, files in os.walk(file_or_dir):
        for name in files:
            filename = os.path.join(root, name)
            result += count_single_line_comments_for_single_file(filename)
    return result


def count_single_line_comments_for_single_file(filename):
    with open(filename) as file:
        return sum(line.lstrip().startswith("//") for line in file)


def percentage_of_open_braces_alone_in_a_line(file_or_dir):
    """
    Counts the following: (number of lines with alone open braces) / (number of lines that contain open braces) * 100
    If open brace is located in comment then it will be counted too.
    That is not good but this situation occurs rare.

    :param file_or_dir: must be string containing path to file or directory
    :return: open braces alone in a line metric
    """
    return percentage_of_symbol_alone_in_a_line(file_or_dir, '{')


def percentage_of_close_braces_alone_in_a_line(file_or_dir):
    """
    Counts the following: (number of lines with alone close braces) / (number of lines that contain close braces) * 100
    If close brace is located in comment then it will be counted too.
    That is not good but this situation occurs rare.

    :param file_or_dir: must be string containing path to file or directory
    :return: close braces alone in a line metric
    """
    return percentage_of_symbol_alone_in_a_line(file_or_dir, '}')


def percentage_of_symbol_alone_in_a_line(file_or_dir, symbol):
    if os.path.isfile(file_or_dir):
        return number_of_symbols_alone_in_a_line(file_or_dir, symbol) / \
               number_of_symbols_in_a_line(file_or_dir, symbol) * 100

    alone_symbols = 0
    all_symbols = 0
    for root, dirs, files in os.walk(file_or_dir):
        for name in files:
            filename = os.path.join(root, name)
            alone_symbols += number_of_symbols_alone_in_a_line(filename, symbol)
            all_symbols += number_of_symbols_in_a_line(filename, symbol)
    return alone_symbols / all_symbols * 100


def number_of_symbols_alone_in_a_line(filename, symbol):
    with open(filename) as file:
        return sum(line.strip() == symbol for line in file)


def number_of_symbols_in_a_line(filename, symbol):
    with open(filename) as file:
        return sum(symbol in line for line in file)
