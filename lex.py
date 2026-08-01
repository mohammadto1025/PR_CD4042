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