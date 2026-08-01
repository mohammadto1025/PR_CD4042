class Token:
    def __init__(self, type, lexeme, line, column, file=None, error=None):
        self.type = type
        self.lexeme = lexeme
        self.line = line
        self.column = column
        self.file = file
        self.error = error

    def __repr__(self):
        return f"Token({self.type}, '{self.lexeme}', {self.line}:{self.column})"

class Lexer:
    def __init__(self, code, filename="<input>"):
        self.code = code
        self.filename = filename
        self.index = 0
        self.line = 1
        self.column = 1
        self.length = len(code)

        self.keywords = {
            'if', 'else', 'while', 'for', 'return', 'int', 'float', 'char',
            'void', 'double', 'struct', 'break', 'continue', 'sizeof',
            'typedef', 'enum', 'union', 'switch', 'case', 'default',
            'do', 'goto', 'const', 'static', 'extern', 'register',
            'volatile', 'signed', 'unsigned', 'short', 'long'
        }

        self.operators = {
            '<=', '>=', '==', '!=', '&&', '||', '++', '--',
            '+=', '-=', '*=', '/=', '%=', '->', '::', '...',
            '<<', '>>', '&', '|', '^', '~', '!', '=',
            '+', '-', '*', '/', '%', '<', '>', '.'
        }
        self.delimiters = {'{', '}', '(', ')', '[', ']', ';', ',', ':'}
        self.operator_list = sorted(self.operators, key=len, reverse=True)

    def peek(self, offset=0):
        pos = self.index + offset
        if pos < self.length:
            return self.code[pos]
        return None

    def advance(self):
        ch = self.peek()
        if ch is not None:
            self.index += 1
            if ch == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
        return ch

    def read_identifier_or_keyword(self):
        start = self.index
        start_line, start_col = self.line, self.column
        ch = self.peek()
        if ch is not None and (ch.isalpha() or ch == '_'):
            while True:
                ch = self.peek()
                if ch is not None and (ch.isalnum() or ch == '_'):
                    self.advance()
                else:
                    break
            lexeme = self.code[start:self.index]
            token_type = 'KEYWORD' if lexeme in self.keywords else 'IDENTIFIER'
            return Token(token_type, lexeme, start_line, start_col, self.filename)
        return None

    def read_number(self):
        start = self.index
        start_line, start_col = self.line, self.column
        ch = self.peek()
        if ch == '0' and self.peek(1) in ('x', 'X'):
            self.advance(); self.advance()
            while True:
                ch = self.peek()
                if ch is not None and ch in '0123456789abcdefABCDEF':
                    self.advance()
                else:
                    break
            lexeme = self.code[start:self.index]
            return Token('INTEGER', lexeme, start_line, start_col, self.filename)
        if ch == '0' and self.peek(1) in ('b', 'B'):
            self.advance(); self.advance()
            while True:
                ch = self.peek()
                if ch is not None and ch in '01':
                    self.advance()
                else:
                    break
            lexeme = self.code[start:self.index]
            return Token('INTEGER', lexeme, start_line, start_col, self.filename)
        is_float = False
        while True:
            ch = self.peek()
            if ch is not None and ch.isdigit():
                self.advance()
            else:
                break
        if self.peek() == '.':
            self.advance()
            is_float = True
            while True:
                ch = self.peek()
                if ch is not None and ch.isdigit():
                    self.advance()
                else:
                    break
        if self.peek() in ('e', 'E'):
            self.advance()
            is_float = True
            if self.peek() in ('+', '-'):
                self.advance()
            while True:
                ch = self.peek()
                if ch is not None and ch.isdigit():
                    self.advance()
                else:
                    break
        if self.peek() in ('f', 'F'):
            self.advance()
            is_float = True
        lexeme = self.code[start:self.index]
        if not lexeme:
            return None
        token_type = 'FLOAT' if is_float else 'INTEGER'
        return Token(token_type, lexeme, start_line, start_col, self.filename)

    def read_string(self):
        start = self.index
        start_line, start_col = self.line, self.column
        self.advance()  
        escaped = False
        while True:
            ch = self.peek()
            if ch is None:
                lexeme = self.code[start:self.index]
                return Token('INVALID', lexeme, start_line, start_col,
                             self.filename, error="Unterminated string literal")
            if ch == '\\' and not escaped:
                escaped = True
                self.advance()
                self.advance()
                continue
            if ch == '"' and not escaped:
                self.advance()
                lexeme = self.code[start:self.index]
                return Token('STRING', lexeme, start_line, start_col, self.filename)
            escaped = False
            self.advance()

    def read_character(self):
        start = self.index
        start_line, start_col = self.line, self.column
        self.advance()  
        escaped = False
        while True:
            ch = self.peek()
            if ch is None:
                lexeme = self.code[start:self.index]
                return Token('INVALID', lexeme, start_line, start_col,
                             self.filename, error="Unterminated character literal")
            if ch == '\\' and not escaped:
                escaped = True
                self.advance()
                self.advance()
                continue
            if ch == "'" and not escaped:
                self.advance()
                lexeme = self.code[start:self.index]
                return Token('CHARACTER', lexeme, start_line, start_col, self.filename)
            escaped = False
            self.advance()

    def read_comment(self):
        start = self.index
        start_line, start_col = self.line, self.column
        if self.peek() == '/' and self.peek(1) == '/':
            self.advance(); self.advance()
            while True:
                ch = self.peek()
                if ch is None or ch == '\n':
                    break
                self.advance()
            return None  

        elif self.peek() == '/' and self.peek(1) == '*':
            self.advance(); self.advance()
            nested = 1
            while True:
                ch = self.peek()
                if ch is None:
                    lexeme = self.code[start:self.index]
                    return Token('INVALID', lexeme, start_line, start_col,
                                 self.filename, error="Unterminated block comment")
                if ch == '*' and self.peek(1) == '/':
                    self.advance(); self.advance()
                    nested -= 1
                    if nested == 0:
                        break
                elif ch == '/' and self.peek(1) == '*':
                    self.advance(); self.advance()
                    nested += 1
                else:
                    self.advance()
            return None  
        return None