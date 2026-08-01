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