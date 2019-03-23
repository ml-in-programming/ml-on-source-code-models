def get_all_metrics():
    return [
        ratio_of_blank_lines_to_code_lines, # slow
        ratio_of_comment_lines_to_code_lines,
        percentage_of_block_comments_to_all_comment_lines,
        percentage_of_open_braces_alone_in_a_line, # fast
        percentage_of_close_braces_alone_in_a_line,
        percentage_of_variable_naming_without_uppercase_letters, # slow
        percentage_of_variable_naming_starting_with_lowercase_letters,
        average_variable_name_length,
        ratio_of_macro_variables
    ]