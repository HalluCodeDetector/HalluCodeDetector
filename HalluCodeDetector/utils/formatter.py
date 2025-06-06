import re

def remove_comments(code):
    pattern = r"(//.*?$|/\*.*?\*/|/\*\*.*?\*/)"
    cleaned_code = re.sub(pattern, '', code, flags=re.DOTALL | re.MULTILINE)
    return cleaned_code

def remove_leading_spaces(code_str):
    lines = code_str.splitlines()
    stripped_lines = [line.lstrip() for line in lines]
    return '\n'.join(stripped_lines)

def format_code_to_single_line(code):
    code = remove_comments(code)
    code = re.sub(r'\s+', ' ', code)
    code = re.sub(r'\s*({|}|;|,)\s*', r'\1', code)
    code = re.sub(r'(\)\s+throws)', r')throws', code)
    code = re.sub(r'\s*\[\s*', '[', code)
    code = re.sub(r'\s*]\s*', ']', code)
    code = re.sub(r'\s*for\s+', 'for', code)
    code = re.sub(r'\s*catch\s+', 'catch', code)
    code = re.sub(r'\s*<\s*', '<', code)
    code = re.sub(r'\s*>\s*', '>', code)
    code = re.sub(r'\s*=\s*', '=', code)
    code = re.sub(r'\s*\(\s*', '(', code)
    code = re.sub(r'\s*\)\s*', ')', code)
    code = re.sub(r'\s*\+\s*', '+', code)
    code = re.sub(r'\s*-\s*', '-', code)
    code = re.sub(r'\s*\*\s*', '*', code)
    code = re.sub(r'\s*/\s*', '/', code)
    code = re.sub(r'\s*\.\s*', '.', code)
    code = re.sub(r'\s*!\s*', '.', code)
    code = re.sub(r'\s*\|\s*', '.', code)
    code = re.sub(r'\s*&\s*', '.', code)
    code = re.sub(r'\s*\?\s*', '.', code)
    code = code.strip()
    return code