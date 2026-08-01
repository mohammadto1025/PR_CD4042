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