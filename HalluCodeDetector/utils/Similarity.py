import difflib
import re
import Levenshtein
import tokenize
from io import BytesIO
import keyword
from codebleu import calc_codebleu
from codebleu.bleu import sentence_bleu

def calculate_codebleu(reference_code, generated_code):
    score = calc_codebleu([reference_code], [generated_code], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return score

def calculate_sentencebleu(reference: str, candidate: str):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    score = sentence_bleu(reference_tokens, candidate_tokens)
    return score

def remove_comments(code):
    single_line_comment_pattern = re.compile(r'#.*')
    multi_line_comment_pattern1 = re.compile(r'\'\'\'[\s\S]*?\'\'\'')
    multi_line_comment_pattern2 = re.compile(r'\"\"\"[\s\S]*?\"\"\"')
    code = re.sub(single_line_comment_pattern, '', code)
    code = re.sub(multi_line_comment_pattern1, '', code)
    code = re.sub(multi_line_comment_pattern2, '', code)
    return code

def codebleu_similarity(code1, code2):
    score = calc_codebleu([code1], [code2], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return score['codebleu']

def bleu_similarity(code1, code2):
    score = calc_codebleu([code1], [code2], lang="python", weights=(1, 0, 0, 0), tokenizer=None)
    return score['codebleu']

def line_similarity(text1, text2):
    text1 = remove_comments(text1)
    text2 = remove_comments(text2)
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    diff_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            diff_count += (i2 - i1) + (j2 - j1)
        elif tag == 'delete':
            diff_count += (i2 - i1)
        elif tag == 'insert':
            diff_count += (j2 - j1)
    if diff_count == 0:
        return 1
    return 1 - diff_count/(len(lines1) + len(lines2))

def ast_similarity(code1, code2):
    score = calc_codebleu([code1], [code2], lang="python", weights=(0, 0, 1, 0), tokenizer=None)
    return score['codebleu']

def dataflow_similarity(code1, code2):
    score = calc_codebleu([code1], [code2], lang="python", weights=(0, 0, 0, 1), tokenizer=None)
    return score['codebleu']

def levenshtein_ratio(code1, code2):
    text1 = remove_comments(code1)
    text2 = remove_comments(code2)
    return Levenshtein.ratio(text1, text2)

def tokenize_code(code):
    tokens = []
    try:
        code_bytes = code.encode('utf-8')
        for tok in tokenize.tokenize(BytesIO(code_bytes).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING):
                continue
            if tok.type == tokenize.NAME:
                if keyword.iskeyword(tok.string):
                    tokens.append("KEYWORD_" + tok.string)
                else:
                    tokens.append("IDENTIFIER")
            elif tok.type == tokenize.OP:
                tokens.append(tok.string)
            elif tok.type == tokenize.NUMBER:
                tokens.append("LITERAL_NUMBER")
            elif tok.type == tokenize.STRING:
                tokens.append("LITERAL_STRING")
            else:
                tokens.append(tok.string)
    except tokenize.TokenError:
        pass
    return tokens

def jaccard_similarity(code1, code2):
    try:
        set1 = tokenize_code(code1)
        set2 = tokenize_code(code2)
        intersection = len(set(set1).intersection(set(set2)))
        union = len(set(set1).union(set(set2)))
        return intersection / union if union != 0 else 0
    except IndentationError:
        return -1

def final_similarity(code1, code2, weights, return_components=False):
    ast_sim = 0
    if weights[0] != 0:
        ast_sim = ast_similarity(code1, code2)

    dataflow_sim = 0
    if weights[1] != 0:
        dataflow_sim = dataflow_similarity(code1, code2)

    jaccard_sim = 0
    if weights[2] != 0:
        jaccard_sim = jaccard_similarity(code1, code2)
        if jaccard_sim == -1:
            return 0

    lev_ratio = 0
    if weights[3] != 0:
        lev_ratio = levenshtein_ratio(code1, code2)

    if return_components:
        return [ast_sim, dataflow_sim, jaccard_sim]
    else:
        return (weights[0] * ast_sim +
                weights[1] * dataflow_sim +
                weights[2] * jaccard_sim +
                weights[3] * lev_ratio
                )