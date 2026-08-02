from diagnostic import DiagnosticBag


class Token:
    def __init__(self, type, lexeme, line, column, file=None, error=None):
        self.type = type
        self.lexeme = lexeme
        self.line = line
        self.column = column
        self.file = file
        self.error = error

    @property
    def length(self):
        return max(1, len(self.lexeme))

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
        self.diagnostics = DiagnosticBag()

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

    def _token(self, token_type, lexeme, line, column, error=None):
        token = Token(
            token_type,
            lexeme,
            line,
            column,
            self.filename,
            error=error,
        )
        if token_type == 'INVALID':
            self.diagnostics.error(
                error or f"Invalid token '{lexeme}'",
                file=self.filename,
                line=max(1, line),
                column=max(1, column),
                length=max(1, len(lexeme)),
            )
        return token

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
            return self._token(token_type, lexeme, start_line, start_col)
        return None

    def read_number(self):
        start = self.index
        start_line, start_col = self.line, self.column
        ch = self.peek()

        if ch == '0' and self.peek(1) in ('x', 'X'):
            self.advance()
            self.advance()
            digit_start = self.index
            while self.peek() is not None and self.peek() in '0123456789abcdefABCDEF':
                self.advance()
            lexeme = self.code[start:self.index]
            if self.index == digit_start:
                return self._token(
                    'INVALID', lexeme, start_line, start_col,
                    "Hexadecimal literal requires at least one digit",
                )
            return self._token('INTEGER', lexeme, start_line, start_col)

        if ch == '0' and self.peek(1) in ('b', 'B'):
            self.advance()
            self.advance()
            digit_start = self.index
            while self.peek() is not None and self.peek() in '01':
                self.advance()
            lexeme = self.code[start:self.index]
            if self.index == digit_start:
                return self._token(
                    'INVALID', lexeme, start_line, start_col,
                    "Binary literal requires at least one digit",
                )
            return self._token('INTEGER', lexeme, start_line, start_col)

        is_float = False
        while self.peek() is not None and self.peek().isdigit():
            self.advance()

        if self.peek() == '.':
            self.advance()
            is_float = True
            while self.peek() is not None and self.peek().isdigit():
                self.advance()

        if self.peek() in ('e', 'E'):
            exponent_index = self.index
            self.advance()
            is_float = True
            if self.peek() in ('+', '-'):
                self.advance()
            exponent_digits = self.index
            while self.peek() is not None and self.peek().isdigit():
                self.advance()
            if self.index == exponent_digits:
                lexeme = self.code[start:self.index]
                return self._token(
                    'INVALID', lexeme, start_line, start_col,
                    "Floating-point exponent requires digits",
                )

        if self.peek() in ('f', 'F'):
            self.advance()
            is_float = True

        lexeme = self.code[start:self.index]
        if not lexeme:
            return None
        return self._token(
            'FLOAT' if is_float else 'INTEGER',
            lexeme,
            start_line,
            start_col,
        )

    def read_string(self):
        start = self.index
        start_line, start_col = self.line, self.column
        self.advance()
        escaped = False

        while True:
            ch = self.peek()
            if ch is None:
                return self._token(
                    'INVALID',
                    self.code[start:self.index],
                    start_line,
                    start_col,
                    "Unterminated string literal",
                )
            if ch == '\n' and not escaped:
                return self._token(
                    'INVALID',
                    self.code[start:self.index],
                    start_line,
                    start_col,
                    "Unterminated string literal before end of line",
                )
            self.advance()
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                return self._token(
                    'STRING',
                    self.code[start:self.index],
                    start_line,
                    start_col,
                )

    def read_character(self):
        start = self.index
        start_line, start_col = self.line, self.column
        self.advance()
        escaped = False

        while True:
            ch = self.peek()
            if ch is None:
                return self._token(
                    'INVALID',
                    self.code[start:self.index],
                    start_line,
                    start_col,
                    "Unterminated character literal",
                )
            if ch == '\n' and not escaped:
                return self._token(
                    'INVALID',
                    self.code[start:self.index],
                    start_line,
                    start_col,
                    "Unterminated character literal before end of line",
                )
            self.advance()
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == "'":
                return self._token(
                    'CHARACTER',
                    self.code[start:self.index],
                    start_line,
                    start_col,
                )

    def read_comment(self):
        start = self.index
        start_line, start_col = self.line, self.column

        if self.peek() == '/' and self.peek(1) == '/':
            self.advance()
            self.advance()
            while self.peek() is not None and self.peek() != '\n':
                self.advance()
            return self._token(
                'COMMENT',
                self.code[start:self.index],
                start_line,
                start_col,
            )

        if self.peek() == '/' and self.peek(1) == '*':
            self.advance()
            self.advance()
            nested = 1
            while True:
                ch = self.peek()
                if ch is None:
                    return self._token(
                        'INVALID',
                        self.code[start:self.index],
                        start_line,
                        start_col,
                        "Unterminated block comment",
                    )
                if ch == '*' and self.peek(1) == '/':
                    self.advance()
                    self.advance()
                    nested -= 1
                    if nested == 0:
                        return self._token(
                            'COMMENT',
                            self.code[start:self.index],
                            start_line,
                            start_col,
                        )
                elif ch == '/' and self.peek(1) == '*':
                    # Supporting nested comments is harmless for this teaching subset.
                    self.advance()
                    self.advance()
                    nested += 1
                else:
                    self.advance()

        return None

    def read_operator_or_delimiter(self):
        start_line, start_col = self.line, self.column
        for op in self.operator_list:
            if self.code.startswith(op, self.index):
                for _ in op:
                    self.advance()
                return self._token('OPERATOR', op, start_line, start_col)

        ch = self.peek()
        if ch in self.delimiters:
            self.advance()
            return self._token('DELIMITER', ch, start_line, start_col)
        return None

    def read_preprocessor(self):
        start = self.index
        start_line, start_col = self.line, self.column
        if self.peek() != '#':
            return None

        self.advance()
        while self.peek() is not None and (self.peek().isalnum() or self.peek() == '_'):
            self.advance()
        return self._token(
            'PREPROCESSOR',
            self.code[start:self.index],
            start_line,
            start_col,
        )

    def next_token(self):
        while self.index < self.length:
            ch = self.peek()

            if ch is not None and ch.isspace():
                start_line, start_col = self.line, self.column
                return self._token(
                    'WHITESPACE',
                    self.advance(),
                    start_line,
                    start_col,
                )

            if ch == '/' and self.peek(1) in ('/', '*'):
                return self.read_comment()

            if ch == '#':
                token = self.read_preprocessor()
                if token is not None:
                    return token

            if ch == '"':
                return self.read_string()

            if ch == "'":
                return self.read_character()

            if ch.isdigit() or (
                ch == '.'
                and self.peek(1) is not None
                and self.peek(1).isdigit()
            ):
                return self.read_number()

            if ch.isalpha() or ch == '_':
                return self.read_identifier_or_keyword()

            token = self.read_operator_or_delimiter()
            if token is not None:
                return token

            start_line, start_col = self.line, self.column
            invalid_ch = self.advance()
            return self._token(
                'INVALID',
                invalid_ch,
                start_line,
                start_col,
                f"Unrecognized character '{invalid_ch}'",
            )

        return None

    def tokenize(self):
        tokens = []
        while True:
            token = self.next_token()
            if token is None:
                break
            tokens.append(token)
        return tokens
