import subprocess


def ratio_of_blank_lines_to_code_lines(file_or_dir, ast_path):
    """
    Blank line is a line consisting only of zero or several whitespaces.
    More formally in UML notation: blank line = whitespace[0..*]
    Code line is any line that is not blank line or comment line.

    :param file_or_dir: must be string containing path to file or directory
    :return: blank lines metric
    """
    return get_ratio(file_or_dir, get_blank_lines_from_cloc_output, get_code_lines_from_cloc_output)


def ratio_of_comment_lines_to_code_lines(file_or_dir, ast_path):
    """
    Comment line is a line that starts (without regard to leading whitespaces) with "//" or located in commented block.
    If any symbol (except whitespace) exists before "//" and it is not located in commented block
    then this line is accounted as code line not a comment line.
    Code line is any line that is not blank line or comment line.

    :param file_or_dir: must be string containing path to file or directory
    :return: comment lines metric
    """
    return get_ratio(file_or_dir, get_comment_lines_from_cloc_output, get_code_lines_from_cloc_output)


def percentage_of_block_comments_to_all_comment_lines(file_or_dir, ast_path):
    """
    Block comment size is number of comments in block comment.

    :param file_or_dir: must be string containing path to file or directory
    :return: block comment lines metric
    """
    all_comments = get_comment_lines_from_cloc_output(get_cloc_output(file_or_dir))
    block_comments = all_comments - count_single_line_comments(file_or_dir)
    return divide_with_handling_zero_division(block_comments, all_comments) * 100


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
    filenames = get_filenames(file_or_dir)
    return sum(count_single_line_comments_for_single_file(filename) for filename in filenames)


def get_cloc_output(file_or_dir):
    """
    If file_or_dir is dir then sum is calculated (e.g. file1_blank_lines + file2_blank_lines).
    """
    return subprocess.run(["cloc", file_or_dir, "--quiet", "--csv"], check=True, stdout=subprocess.PIPE) \
        .stdout.decode("utf-8")


def get_blank_lines_from_cloc_output(cloc_output):
    return int(cloc_output.splitlines()[-1].split(',')[2])


def get_comment_lines_from_cloc_output(cloc_output):
    return int(cloc_output.splitlines()[-1].split(',')[3])


def get_code_lines_from_cloc_output(cloc_output):
    return int(cloc_output.splitlines()[-1].split(',')[4])


def get_ratio(file_or_dir, numerator, denominator):
    cloc_output = get_cloc_output(file_or_dir)
    return divide_with_handling_zero_division(numerator(cloc_output), denominator(cloc_output))


def count_single_line_comments_for_single_file(filename):
    with open(filename) as file:
        return sum(line.lstrip().startswith("//") for line in file)