import os
import subprocess


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


def percentage_of_variable_naming_without_uppercase_letters(file_or_dir, ast_path):
    """
    This metric uses ast of files so you will need to generate files stated below with PathMiner.
    :param ast_path: path to folder where asts.csv, node_types.csv, tokens.csv are stored
    :param file_or_dir: must be string containing path to file or directory
    :return: variables without uppercase letters metric
    """
    filenames = get_filenames(file_or_dir)
    asts = load_asts(filenames, ast_path)
    nodes = load_nodes(ast_path)
    tokens = load_tokens(ast_path)
    return number_of_variables_in_lowercase(asts, nodes, tokens) / number_of_variables(asts, nodes, tokens) * 100


def percentage_of_variable_naming_starting_with_lowercase_letters(file_or_dir, ast_path):
    """
    This metric uses ast of files so you will need to generate files stated below with PathMiner.
    :param ast_path: path to folder where asts.csv, node_types.csv, tokens.csv are stored
    :param file_or_dir: must be string containing path to file or directory
    :return: variables starting with lowercase letters metric
    """
    filenames = get_filenames(file_or_dir)
    asts = load_asts(filenames, ast_path)
    nodes = load_nodes(ast_path)
    tokens = load_tokens(ast_path)
    return number_of_variables_starting_with_lowercase(asts, nodes, tokens) / \
        number_of_variables(asts, nodes, tokens) * 100


def average_variable_name_length(file_or_dir, ast_path):
    """
    This metric uses ast of files so you will need to generate files stated below with PathMiner.
    :param ast_path: path to folder where asts.csv, node_types.csv, tokens.csv are stored
    :param file_or_dir: must be string containing path to file or directory
    :return: average variable name metric
    """
    filenames = get_filenames(file_or_dir)
    asts = load_asts(filenames, ast_path)
    nodes = load_nodes(ast_path)
    tokens = load_tokens(ast_path)
    variable_names = get_variables_names(asts, nodes, tokens)
    return sum(len(name) for name in variable_names) / len(variable_names)


def ratio_of_macro_variables(file_or_dir, ast_path):
    """
    This metric is strange. There is no macros in Java. That is why it always returns 0.
    :param ast_path: path to folder where asts.csv, node_types.csv, tokens.csv are stored
    :param file_or_dir: must be string containing path to file or directory
    :return: 0
    """
    return 0


def percentage_of_for_statements_to_all_loop_statements(file_or_dir, ast_path):
    pass


def get_variables_names(asts, nodes, tokens):
    # TODO: add support for fields and parameters. Tree reading will be the most convenient way.
    variable_declarator_id = [node for node in nodes if node[1] == "variableDeclarator"]
    variable_declarator_id_for = [node for node in nodes if node[1] == "variableDeclaratorId"]
    field_declarator_id = [node for node in nodes if node[1] == "fieldDeclaration"]
    parameter_declarator_id = [node for node in nodes if node[1] == "formalParameter"]
    variable_local_declarator_id = [node for node in nodes if node[1] == "localVariableDeclaration"]

    if len(variable_declarator_id) == 0:
        return 0
    if len(variable_declarator_id) > 1:
        raise ValueError("Several variable declarators found in node_types.csv")
    variable_declarator_id = variable_declarator_id[0][0]
    variable_token_ids = []
    for ast in asts:
        variable_token_ids += \
            [piece.split('{')[1] for piece in ast[1].split(' ') if piece.startswith(str(variable_declarator_id) + "{")]
    return [token[1] for token in tokens if token[0] in variable_token_ids]


def number_of_variables(asts, nodes, tokens):
    return len(get_variables_names(asts, nodes, tokens))


def number_of_variables_in_lowercase(asts, nodes, tokens):
    return len([name for name in get_variables_names(asts, nodes, tokens) if name.islower()])


def number_of_variables_starting_with_lowercase(asts, nodes, tokens):
    return len([name for name in get_variables_names(asts, nodes, tokens) if name[0].islower()])


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


def percentage_of_block_comments_to_all_comment_lines(file_or_dir):
    all_comments = get_comment_lines_from_cloc_output(get_cloc_output(file_or_dir))
    block_comments = all_comments - count_single_line_comments(file_or_dir)
    return block_comments / all_comments * 100


def count_single_line_comments_for_single_file(filename):
    with open(filename) as file:
        return sum(line.lstrip().startswith("//") for line in file)


def percentage_of_symbol_alone_in_a_line(file_or_dir, symbol):
    filenames = get_filenames(file_or_dir)
    alone_symbols = sum(number_of_symbols_alone_in_a_line(filename, symbol) for filename in filenames)
    all_symbols = sum(number_of_symbols_in_a_line(filename, symbol) for filename in filenames)
    return alone_symbols / all_symbols * 100


def number_of_symbols_alone_in_a_line(filename, symbol):
    with open(filename) as file:
        return sum(line.strip() == symbol for line in file)


def number_of_symbols_in_a_line(filename, symbol):
    with open(filename) as file:
        return sum(symbol in line for line in file)


def load_nodes(path_to_ast_data):
    with open(os.path.join(path_to_ast_data, "node_types.csv")) as file:
        return [(line.split(',')[0], line.split(',')[1].rstrip("\n")) for line in file][1:]


def load_tokens(path_to_ast_data):
    with open(os.path.join(path_to_ast_data, "tokens.csv")) as file:
        return [(line.split(',')[0], line.split(',')[1].rstrip("\n")) for line in file][1:]


def load_asts(filenames, path_to_ast_data):
    filenames = [os.path.abspath(filename) for filename in filenames]
    with open(os.path.join(path_to_ast_data, "asts.csv")) as file:
        return [(line.split(',')[0], line.split(',')[1].rstrip("\n")) for line in file if
                line.split(',')[0] in filenames]


def get_filenames(file_or_dir):
    if os.path.isfile(file_or_dir):
        return [file_or_dir]
    filenames = []
    for root, dirs, files in os.walk(file_or_dir):
        for name in files:
            filenames += [os.path.join(root, name)]
    return filenames
