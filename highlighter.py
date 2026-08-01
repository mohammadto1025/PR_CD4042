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

    def _annotate_ast(self, node):
        if node is None:
            return
        if hasattr(node, '__class__') and node.__class__.__name__ == 'FuncDecl':
            self._mark_token(node.name, 'FUNCTION')
            for param in node.params:
                self._mark_token(param.name, 'VARIABLE')
            self._mark_type_spec(node.return_type)
        elif hasattr(node, '__class__') and node.__class__.__name__ == 'CallExpr':
            self._mark_token(node.callee, 'FUNCTION')
        elif hasattr(node, '__class__') and node.__class__.__name__ == 'VarDecl':
            self._mark_token(node.name, 'VARIABLE')
            self._mark_type_spec(node.type_spec)
        elif hasattr(node, '__class__') and node.__class__.__name__ == 'Identifier':
            if node.name not in self.token_colors:
                self._mark_token(node.name, 'VARIABLE')
        for attr in dir(node):
            if attr.startswith('_'):
                continue
            child = getattr(node, attr)
            if isinstance(child, list):
                for item in child:
                    self._annotate_ast(item)
            elif hasattr(child, '__class__') and hasattr(child, '__class__.__name__'):
                if child.__class__.__name__ in ['ASTNode', 'Program', 'FuncDecl', 'Param', 'VarDecl',
                                                 'StructDecl', 'Block', 'IfStmt', 'WhileStmt', 'ForStmt',
                                                 'ReturnStmt', 'ExprStmt', 'BinaryExpr', 'UnaryExpr',
                                                 'CallExpr', 'Identifier', 'IntLiteral', 'FloatLiteral',
                                                 'StringLiteral', 'CharLiteral']:
                    self._annotate_ast(child)

    def _mark_token(self, name, color_key):
        for token in self.tokens:
            if token.type == 'IDENTIFIER' and token.lexeme == name:
                key = (token.line, token.column)
                self.token_colors[key] = color_key
                self.token_colors[name] = color_key
                break

    def _mark_type_spec(self, type_name):
        for token in self.tokens:
            if token.type == 'KEYWORD' and token.lexeme == type_name:
                key = (token.line, token.column)
                self.token_colors[key] = 'TYPE'
                self.token_colors[type_name] = 'TYPE'
                break

    def _get_color_for_token(self, token):
        if token.type == 'WHITESPACE':
            return None
        key = (token.line, token.column)
        if key in self.token_colors:
            return self.token_colors[key]
        if token.lexeme in self.token_colors:
            return self.token_colors[token.lexeme]

        if token.type == 'KEYWORD':
            if token.lexeme in ['int', 'float', 'char', 'void', 'double']:
                return 'TYPE'
            return 'KEYWORD'
        elif token.type == 'IDENTIFIER':
            return 'VARIABLE' 
        elif token.type == 'INTEGER' or token.type == 'FLOAT':
            return 'INTEGER' if token.type == 'INTEGER' else 'FLOAT'
        elif token.type == 'STRING':
            return 'STRING'
        elif token.type == 'CHARACTER':
            return 'CHAR'
        elif token.type == 'OPERATOR':
            return 'OPERATOR'
        elif token.type == 'DELIMITER':
            return 'DELIMITER'
        elif token.type == 'COMMENT':
            return 'COMMENT'
        elif token.type == 'PREPROCESSOR':
            return 'PREPROCESSOR'
        elif token.type == 'INVALID':
            return 'INVALID'
        else:
            return None