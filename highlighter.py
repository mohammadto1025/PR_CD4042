class SyntaxHighlighter:
    def __init__(self, tokens, ast):
        self.tokens = tokens
        self.ast = ast
        self.token_colors = {}
        self._annotate_ast(ast)

    COLORS = {
        'KEYWORD': ('\033[1;34m', 'kw'),          # Bold Blue
        'TYPE': ('\033[36m', 'type'),             # Cyan
        'VARIABLE': ('\033[37m', 'var'),          # White (default)
        'FUNCTION': ('\033[33m', 'func'),         # Yellow
        'TYPE_NAME': ('\033[32m', 'type_name'),   # Bright Green
        'INTEGER': ('\033[33m', 'int'),           # Orange (using yellow)
        'FLOAT': ('\033[33m', 'float'),           # Orange
        'STRING': ('\033[32m', 'string'),         # Warm Green
        'CHAR': ('\033[32m', 'char'),             # Warm Green
        'OPERATOR': ('\033[37m', 'op'),           # Light Gray
        'COMMENT': ('\033[90m', 'comment'),       # Dim Gray
        'PREPROCESSOR': ('\033[35m', 'preproc'),  # Magenta
        'INVALID': ('\033[41m', 'invalid'),       # Red underline (background red)
        'DELIMITER': ('\033[37m', 'delim'),       # Light Gray
    }