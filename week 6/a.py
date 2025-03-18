import re

def remove_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Remove single-line and inline comments
    cleaned_lines = [re.sub(r'#.*', '', line) for line in lines]

    # Remove multi-line (block) comments
    inside_multiline_comment = False
    result = []
    for line in cleaned_lines:
        if '"""' in line or "'''" in line:
            count = line.count('"""') + line.count("'''")
            if count % 2 == 1:  # Start or end of a multiline comment
                inside_multiline_comment = not inside_multiline_comment
            continue  # Skip the line containing the comment delimiter
        
        if not inside_multiline_comment:
            result.append(line)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(result)

# Example usage
remove_comments('./nnx.py')
