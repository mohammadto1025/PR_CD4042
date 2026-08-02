from dataclasses import dataclass, field
from typing import List, Optional, Any

from diagnostic import DiagnosticBag

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


class ASTNode:
    def __init__(self, loc=None):
        self.loc = loc  
        self.type_annotation = None

    def _indent(self, text, level=1):
        return "  " * level + text

    def _format_loc(self):
        if self.loc:
            return f"loc={self.loc[0]}:{self.loc[1]}"
        return "loc=unknown"

    def __str__(self):
        return f"{self.__class__.__name__}(loc={self._format_loc()})"


class Program(ASTNode):
    def __init__(self, declarations, loc=None):
        super().__init__(loc)
        self.declarations = declarations

    def __str__(self):
        result = f"Program({self._format_loc()})"
        for decl in self.declarations:
            result += "\n" + self._indent(str(decl), 1)
        return result


class FuncDecl(ASTNode):
    def __init__(self, return_type, name, params, body, loc=None):
        super().__init__(loc)
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body

    def __str__(self):
        result = f"FuncDecl(name={self.name}, return={self.return_type}, {self._format_loc()})"
        result += "\n" + self._indent("params:", 1)
        for p in self.params:
            result += "\n" + self._indent(str(p), 2)
        result += "\n" + self._indent("body:", 1)
        result += "\n" + self._indent(str(self.body), 2)
        return result


class Param(ASTNode):
    def __init__(self, type_spec, name, loc=None):
        super().__init__(loc)
        self.type_spec = type_spec
        self.name = name

    def __str__(self):
        return f"Param(type={self.type_spec}, name={self.name}, {self._format_loc()})"


class VarDecl(ASTNode):
    def __init__(self, type_spec, name, init_expr=None, loc=None):
        super().__init__(loc)
        self.type_spec = type_spec
        self.name = name
        self.init_expr = init_expr

    def __str__(self):
        result = f"VarDecl(name={self.name}, type={self.type_spec}, {self._format_loc()})"
        if self.init_expr:
            result += "\n" + self._indent("init:", 1)
            result += "\n" + self._indent(str(self.init_expr), 2)
        return result


class StructDecl(ASTNode):
    def __init__(self, name, fields, loc=None):
        super().__init__(loc)
        self.name = name
        self.fields = fields

    def __str__(self):
        result = f"StructDecl(name={self.name}, {self._format_loc()})"
        result += "\n" + self._indent("fields:", 1)
        for f in self.fields:
            result += "\n" + self._indent(str(f), 2)
        return result


class Block(ASTNode):
    def __init__(self, statements, loc=None):
        super().__init__(loc)
        self.statements = statements

    def __str__(self):
        result = f"Block({self._format_loc()})"
        for stmt in self.statements:
            result += "\n" + self._indent(str(stmt), 1)
        return result


class IfStmt(ASTNode):
    def __init__(self, condition, then_stmt, else_stmt=None, loc=None):
        super().__init__(loc)
        self.condition = condition
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    def __str__(self):
        result = f"IfStmt({self._format_loc()})"
        result += "\n" + self._indent("condition:", 1)
        result += "\n" + self._indent(str(self.condition), 2)
        result += "\n" + self._indent("then:", 1)
        result += "\n" + self._indent(str(self.then_stmt), 2)
        if self.else_stmt:
            result += "\n" + self._indent("else:", 1)
            result += "\n" + self._indent(str(self.else_stmt), 2)
        return result


class WhileStmt(ASTNode):
    def __init__(self, condition, body, loc=None):
        super().__init__(loc)
        self.condition = condition
        self.body = body

    def __str__(self):
        result = f"WhileStmt({self._format_loc()})"
        result += "\n" + self._indent("condition:", 1)
        result += "\n" + self._indent(str(self.condition), 2)
        result += "\n" + self._indent("body:", 1)
        result += "\n" + self._indent(str(self.body), 2)
        return result


class ForStmt(ASTNode):
    def __init__(self, init, condition, increment, body, loc=None):
        super().__init__(loc)
        self.init = init
        self.condition = condition
        self.increment = increment
        self.body = body

    def __str__(self):
        result = f"ForStmt({self._format_loc()})"
        result += "\n" + self._indent("init:", 1)
        result += "\n" + self._indent(str(self.init), 2)
        result += "\n" + self._indent("condition:", 1)
        result += "\n" + self._indent(str(self.condition), 2)
        if self.increment:
            result += "\n" + self._indent("increment:", 1)
            result += "\n" + self._indent(str(self.increment), 2)
        result += "\n" + self._indent("body:", 1)
        result += "\n" + self._indent(str(self.body), 2)
        return result


class ReturnStmt(ASTNode):
    def __init__(self, value=None, loc=None):
        super().__init__(loc)
        self.value = value

    def __str__(self):
        result = f"ReturnStmt({self._format_loc()})"
        if self.value:
            result += "\n" + self._indent("value:", 1)
            result += "\n" + self._indent(str(self.value), 2)
        return result


class ExprStmt(ASTNode):
    def __init__(self, expr=None, loc=None):
        super().__init__(loc)
        self.expr = expr

    def __str__(self):
        result = f"ExprStmt({self._format_loc()})"
        if self.expr:
            result += "\n" + self._indent("expr:", 1)
            result += "\n" + self._indent(str(self.expr), 2)
        return result


class BinaryExpr(ASTNode):
    def __init__(self, left, op, right, loc=None):
        super().__init__(loc)
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        result = f"BinaryExpr(op='{self.op}', {self._format_loc()})"
        result += "\n" + self._indent("left:", 1)
        result += "\n" + self._indent(str(self.left), 2)
        result += "\n" + self._indent("right:", 1)
        result += "\n" + self._indent(str(self.right), 2)
        return result


class UnaryExpr(ASTNode):
    def __init__(self, op, operand, loc=None):
        super().__init__(loc)
        self.op = op
        self.operand = operand

    def __str__(self):
        result = f"UnaryExpr(op='{self.op}', {self._format_loc()})"
        result += "\n" + self._indent("operand:", 1)
        result += "\n" + self._indent(str(self.operand), 2)
        return result


class CallExpr(ASTNode):
    def __init__(self, callee, args, loc=None):
        super().__init__(loc)
        self.callee = callee
        self.args = args

    def __str__(self):
        result = f"CallExpr(callee='{self.callee}', {self._format_loc()})"
        for i, arg in enumerate(self.args):
            result += "\n" + self._indent(f"args[{i}]:", 1)
            result += "\n" + self._indent(str(arg), 2)
        return result


class Identifier(ASTNode):
    def __init__(self, name, loc=None):
        super().__init__(loc)
        self.name = name

    def __str__(self):
        return f"Identifier(name='{self.name}', {self._format_loc()})"


class IntLiteral(ASTNode):
    def __init__(self, value, loc=None):
        super().__init__(loc)
        self.value = value

    def __str__(self):
        return f"IntLiteral(value={self.value}, {self._format_loc()})"


class FloatLiteral(ASTNode):
    def __init__(self, value, loc=None):
        super().__init__(loc)
        self.value = value

    def __str__(self):
        return f"FloatLiteral(value={self.value}, {self._format_loc()})"


class StringLiteral(ASTNode):
    def __init__(self, value, loc=None):
        super().__init__(loc)
        self.value = value

    def __str__(self):
        return f"StringLiteral(value={self.value}, {self._format_loc()})"


class CharLiteral(ASTNode):
    def __init__(self, value, loc=None):
        super().__init__(loc)
        self.value = value

    def __str__(self):
        return f"CharLiteral(value={self.value}, {self._format_loc()})"



class BreakStmt(ASTNode):
    def __init__(self, loc=None):
        super().__init__(loc)

    def __str__(self):
        return f"BreakStmt({self._format_loc()})"


class ContinueStmt(ASTNode):
    def __init__(self, loc=None):
        super().__init__(loc)

    def __str__(self):
        return f"ContinueStmt({self._format_loc()})"


class Parser:
    TYPE_KEYWORDS = {'int', 'float', 'char', 'void', 'double'}
    TRIVIA_TYPES = {'WHITESPACE', 'COMMENT'}
    STATEMENT_KEYWORDS = {
        'if', 'while', 'for', 'return', 'break', 'continue',
        'int', 'float', 'char', 'void', 'double', 'struct',
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors = []
        self.diagnostics = DiagnosticBag()
        self._skip_trivia()
        self.current_token = (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else None
        )

    def _skip_trivia(self):
        while (
            self.pos < len(self.tokens)
            and self.tokens[self.pos].type in self.TRIVIA_TYPES
        ):
            self.pos += 1

    def _significant_token(self, offset=0):
        index = self.pos
        seen = 0
        while index < len(self.tokens):
            token = self.tokens[index]
            if token.type not in self.TRIVIA_TYPES:
                if seen == offset:
                    return token
                seen += 1
            index += 1
        return None

    def peek(self, offset=0):
        self._skip_trivia()
        if offset:
            return self._significant_token(offset)
        self.current_token = (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else None
        )
        return self.current_token

    def advance(self):
        if self.current_token is not None:
            self.pos += 1
        self._skip_trivia()
        self.current_token = (
            self.tokens[self.pos]
            if self.pos < len(self.tokens)
            else None
        )
        return self.current_token

    def _matches(self, token, expected_type, expected_lexeme=None):
        return (
            token is not None
            and token.type == expected_type
            and (
                expected_lexeme is None
                or token.lexeme == expected_lexeme
            )
        )

    def _synthetic_token(self, expected_type, expected_lexeme=None, token=None):
        token = token or self.current_token
        line = token.line if token is not None else 1
        column = token.column if token is not None else 1
        file = getattr(token, 'file', None) if token is not None else None
        label = expected_lexeme or expected_type
        return Token(
            expected_type,
            f"<missing_{label}>",
            max(1, line),
            max(1, column),
            file=file,
            error=f"Missing {label}",
        )

    def _report(self, message, token=None, length=None):
        token = token or self.current_token
        line = max(1, getattr(token, 'line', 1))
        column = max(1, getattr(token, 'column', 1))
        file = getattr(token, 'file', None) or '<input>'
        diagnostic = self.diagnostics.error(
            message=message,
            file=file,
            line=line,
            column=column,
            length=max(
                1,
                length
                if length is not None
                else len(getattr(token, 'lexeme', '') or ''),
            ),
        )
        self.errors.append(str(diagnostic))
        return diagnostic

    def match(self, expected_type, error_msg=None):
        token = self.peek()
        if self._matches(token, expected_type):
            self.advance()
            return token
        self._report(
            error_msg
            or (
                f"Expected '{expected_type}', got "
                f"'{token.type if token else 'EOF'}'"
            ),
            token,
        )
        return None

    def expect(
        self,
        expected_type,
        error_msg=None,
        expected_lexeme=None,
    ):
        token = self.peek()
        if self._matches(
            token,
            expected_type,
            expected_lexeme,
        ):
            self.advance()
            return token

        actual = (
            token.lexeme
            if token is not None
            else 'EOF'
        )
        expected = expected_lexeme or expected_type
        self._report(
            error_msg
            or f"Expected '{expected}', got '{actual}'",
            token,
        )
        # Insertion recovery: keep the current token for the caller.
        return self._synthetic_token(
            expected_type,
            expected_lexeme,
            token,
        )

    def _is_type_start(self, token=None):
        token = token or self.current_token
        if token is None or token.type != 'KEYWORD':
            return False
        return (
            token.lexeme in self.TYPE_KEYWORDS
            or token.lexeme == 'struct'
        )

    def _is_declaration_start(self, token=None):
        return self._is_type_start(token)

    def _is_statement_start(self, token=None):
        token = token or self.current_token
        if token is None:
            return False
        if token.type == 'DELIMITER' and token.lexeme in {'{', ';'}:
            return True
        if token.type == 'KEYWORD':
            return token.lexeme in self.STATEMENT_KEYWORDS
        return token.type in {
            'IDENTIFIER', 'INTEGER', 'FLOAT', 'STRING', 'CHARACTER'
        } or (
            token.type == 'DELIMITER'
            and token.lexeme == '('
        ) or (
            token.type == 'OPERATOR'
            and token.lexeme in {'-', '!', '&', '*'}
        )

    def synchronize_declaration(self):
        while self.current_token is not None:
            if self._is_declaration_start():
                return
            self.advance()

    def synchronize_statement(self):
        while self.current_token is not None:
            token = self.current_token
            if token.type == 'DELIMITER':
                if token.lexeme == ';':
                    self.advance()
                    return
                if token.lexeme == '}':
                    return
            if (
                token.type == 'KEYWORD'
                and token.lexeme in self.STATEMENT_KEYWORDS
            ):
                return
            self.advance()

    def is_sync_point(self):
        token = self.current_token
        if token is None:
            return True
        if token.type == 'DELIMITER' and token.lexeme in {';', '}', '{'}:
            return True
        return (
            token.type == 'KEYWORD'
            and token.lexeme in self.STATEMENT_KEYWORDS
        )

    def sync(self):
        self.synchronize_statement()

    def panic(self, message, token=None):
        self._report(message, token)
        self.synchronize_statement()

    def parse_program(self) -> Program:
        loc_token = self.current_token
        declarations = []

        while self.current_token is not None:
            start_pos = self.pos
            declaration = self.parse_declaration()
            if declaration is not None:
                declarations.append(declaration)

            if self.pos == start_pos:
                self.advance()

        return Program(
            declarations,
            loc=(
                max(1, loc_token.line)
                if loc_token is not None
                else 1,
                max(1, loc_token.column)
                if loc_token is not None
                else 1,
            ),
        )

    def parse_declaration(self):
        token = self.current_token
        if token is None:
            return None

        if token.type == 'INVALID':
            self._report(
                token.error
                or f"Invalid token '{token.lexeme}'",
                token,
            )
            self.advance()
            self.synchronize_declaration()
            return None

        if token.type == 'KEYWORD' and token.lexeme in self.TYPE_KEYWORDS:
            return self.parse_function_or_var_decl()

        if token.type == 'KEYWORD' and token.lexeme == 'struct':
            # struct Name { ... }; is a type declaration.
            # struct Name variable; is an ordinary variable declaration.
            after_name = self.peek(2)
            if (
                after_name is not None
                and after_name.type == 'DELIMITER'
                and after_name.lexeme == '{'
            ):
                return self.parse_struct_decl()
            return self.parse_function_or_var_decl()

        self._report(
            f"Unexpected token '{token.lexeme}' in declaration",
            token,
        )
        self.advance()
        self.synchronize_declaration()
        return None

    def parse_function_or_var_decl(self):
        type_spec = self.parse_type_spec()
        if type_spec is None:
            self.synchronize_declaration()
            return None

        ident_token = self.peek()
        if not self._matches(ident_token, 'IDENTIFIER'):
            self._report(
                "Expected identifier after type specifier",
                ident_token,
            )
            self.synchronize_declaration()
            return None

        self.advance()
        name = ident_token.lexeme
        loc = (ident_token.line, ident_token.column)

        if self._matches(self.current_token, 'DELIMITER', '('):
            self.advance()
            params = self.parse_param_list()
            self.expect(
                'DELIMITER',
                "Expected ')' after parameters",
                ')',
            )
            body = self.parse_block()
            return FuncDecl(
                type_spec,
                name,
                params,
                body,
                loc=loc,
            )

        init_expr = None
        if self._matches(self.current_token, 'OPERATOR', '='):
            self.advance()
            init_expr = self.parse_expr()

        self.expect(
            'DELIMITER',
            "Expected ';' after variable declaration",
            ';',
        )
        return VarDecl(
            type_spec,
            name,
            init_expr,
            loc=loc,
        )

    def parse_type_spec(self):
        token = self.current_token
        if token is None:
            self._report("Expected type specifier, got 'EOF'", token)
            return None

        if token.type != 'KEYWORD':
            self._report(
                f"Expected type specifier, got '{token.lexeme}'",
                token,
            )
            return None

        if token.lexeme == 'struct':
            self.advance()
            name_token = self.peek()
            if not self._matches(name_token, 'IDENTIFIER'):
                self._report(
                    "Expected struct name after 'struct'",
                    name_token,
                )
                return None
            self.advance()
            type_name = f"struct {name_token.lexeme}"
        elif token.lexeme in self.TYPE_KEYWORDS:
            type_name = token.lexeme
            self.advance()
        else:
            self._report(
                f"Expected type specifier, got '{token.lexeme}'",
                token,
            )
            return None

        while self._matches(
            self.current_token,
            'OPERATOR',
            '*',
        ):
            type_name += '*'
            self.advance()

        return type_name

    def parse_struct_decl(self):
        struct_token = self.expect(
            'KEYWORD',
            "Expected 'struct'",
            'struct',
        )
        name_token = self.expect(
            'IDENTIFIER',
            "Expected struct name",
        )
        name = name_token.lexeme
        loc = (name_token.line, name_token.column)

        self.expect(
            'DELIMITER',
            "Expected '{' after struct name",
            '{',
        )
        fields = []

        while (
            self.current_token is not None
            and not self._matches(
                self.current_token,
                'DELIMITER',
                '}',
            )
        ):
            start_pos = self.pos
            type_spec = self.parse_type_spec()
            if type_spec is None:
                self.synchronize_statement()
            else:
                ident_token = self.peek()
                if self._matches(ident_token, 'IDENTIFIER'):
                    self.advance()
                    self.expect(
                        'DELIMITER',
                        "Expected ';' after field",
                        ';',
                    )
                    fields.append(
                        VarDecl(
                            type_spec,
                            ident_token.lexeme,
                            None,
                            loc=(
                                ident_token.line,
                                ident_token.column,
                            ),
                        )
                    )
                else:
                    self._report(
                        "Expected field name",
                        ident_token,
                    )
                    self.synchronize_statement()

            if self.pos == start_pos:
                self.advance()

        self.expect('DELIMITER', "Expected '}'", '}')
        if self._matches(
            self.current_token,
            'DELIMITER',
            ';',
        ):
            self.advance()

        return StructDecl(name, fields, loc=loc)

    def parse_param_list(self):
        params = []
        if self._matches(
            self.current_token,
            'DELIMITER',
            ')',
        ):
            return params

        while self.current_token is not None:
            parameter = self.parse_param()
            if parameter is not None:
                params.append(parameter)

            if self._matches(
                self.current_token,
                'DELIMITER',
                ',',
            ):
                self.advance()
                continue
            break

        return params

    def parse_param(self):
        type_spec = self.parse_type_spec()
        if type_spec is None:
            return None

        ident_token = self.peek()
        if not self._matches(ident_token, 'IDENTIFIER'):
            self._report(
                "Expected parameter name",
                ident_token,
            )
            return None

        self.advance()
        return Param(
            type_spec,
            ident_token.lexeme,
            loc=(ident_token.line, ident_token.column),
        )

    def parse_block(self):
        loc_token = self.current_token
        if not self._matches(
            self.current_token,
            'DELIMITER',
            '{',
        ):
            self._report("Expected '{'", self.current_token)
            return Block(
                [],
                loc=(
                    getattr(loc_token, 'line', 1),
                    getattr(loc_token, 'column', 1),
                ),
            )

        self.advance()
        statements = []

        while (
            self.current_token is not None
            and not self._matches(
                self.current_token,
                'DELIMITER',
                '}',
            )
        ):
            start_pos = self.pos
            statement = self.parse_statement()
            if statement is not None:
                statements.append(statement)
            if self.pos == start_pos:
                self.advance()

        self.expect('DELIMITER', "Expected '}'", '}')
        return Block(
            statements,
            loc=(
                loc_token.line if loc_token is not None else 1,
                loc_token.column if loc_token is not None else 1,
            ),
        )

    def parse_statement(self):
        token = self.current_token
        if token is None:
            return None

        if token.type == 'INVALID':
            self._report(
                token.error
                or f"Invalid token '{token.lexeme}'",
                token,
            )
            self.advance()
            self.synchronize_statement()
            return None

        if token.type == 'KEYWORD':
            if token.lexeme == 'if':
                return self.parse_if_stmt()
            if token.lexeme == 'while':
                return self.parse_while_stmt()
            if token.lexeme == 'for':
                return self.parse_for_stmt()
            if token.lexeme == 'return':
                return self.parse_return_stmt()
            if token.lexeme == 'break':
                return self.parse_break_stmt()
            if token.lexeme == 'continue':
                return self.parse_continue_stmt()
            if token.lexeme in self.TYPE_KEYWORDS or token.lexeme == 'struct':
                return self.parse_var_decl_statement()

            self._report(
                f"Unsupported statement keyword '{token.lexeme}'",
                token,
            )
            self.advance()
            self.synchronize_statement()
            return None

        if self._matches(token, 'DELIMITER', '{'):
            return self.parse_block()

        if self._matches(token, 'DELIMITER', ';'):
            self.advance()
            return ExprStmt(None, loc=(token.line, token.column))

        return self.parse_expr_stmt()

    def parse_var_decl_statement(self):
        type_spec = self.parse_type_spec()
        if type_spec is None:
            self.synchronize_statement()
            return None

        ident_token = self.peek()
        if not self._matches(ident_token, 'IDENTIFIER'):
            self._report("Expected identifier", ident_token)
            self.synchronize_statement()
            return None

        self.advance()
        init_expr = None
        if self._matches(
            self.current_token,
            'OPERATOR',
            '=',
        ):
            self.advance()
            init_expr = self.parse_expr()

        self.expect(
            'DELIMITER',
            "Expected ';' after variable declaration",
            ';',
        )
        return VarDecl(
            type_spec,
            ident_token.lexeme,
            init_expr,
            loc=(ident_token.line, ident_token.column),
        )

    def parse_if_stmt(self):
        loc_token = self.current_token
        self.advance()
        self.expect(
            'DELIMITER',
            "Expected '(' after 'if'",
            '(',
        )
        condition = self.parse_expr()
        self.expect(
            'DELIMITER',
            "Expected ')' after condition",
            ')',
        )
        then_stmt = self.parse_statement()
        else_stmt = None
        if (
            self.current_token is not None
            and self.current_token.type == 'KEYWORD'
            and self.current_token.lexeme == 'else'
        ):
            self.advance()
            else_stmt = self.parse_statement()
        return IfStmt(
            condition,
            then_stmt,
            else_stmt,
            loc=(loc_token.line, loc_token.column),
        )

    def parse_while_stmt(self):
        loc_token = self.current_token
        self.advance()
        self.expect(
            'DELIMITER',
            "Expected '(' after 'while'",
            '(',
        )
        condition = self.parse_expr()
        self.expect(
            'DELIMITER',
            "Expected ')' after condition",
            ')',
        )
        body = self.parse_statement()
        return WhileStmt(
            condition,
            body,
            loc=(loc_token.line, loc_token.column),
        )

    def parse_for_stmt(self):
        loc_token = self.current_token
        self.advance()
        self.expect(
            'DELIMITER',
            "Expected '(' after 'for'",
            '(',
        )

        if self._matches(
            self.current_token,
            'DELIMITER',
            ';',
        ):
            init_stmt = ExprStmt(
                None,
                loc=(
                    self.current_token.line,
                    self.current_token.column,
                ),
            )
            self.advance()
        elif self._is_type_start():
            init_stmt = self.parse_var_decl_statement()
        else:
            init_stmt = self.parse_expr_stmt()

        cond_stmt = self.parse_expr_stmt()

        inc_expr = None
        if not self._matches(
            self.current_token,
            'DELIMITER',
            ')',
        ):
            inc_expr = self.parse_expr()

        self.expect(
            'DELIMITER',
            "Expected ')' after for clauses",
            ')',
        )
        body = self.parse_statement()
        return ForStmt(
            init_stmt,
            cond_stmt,
            inc_expr,
            body,
            loc=(loc_token.line, loc_token.column),
        )

    def parse_return_stmt(self):
        loc_token = self.current_token
        self.advance()
        expression = None
        if not self._matches(
            self.current_token,
            'DELIMITER',
            ';',
        ):
            expression = self.parse_expr()
        self.expect(
            'DELIMITER',
            "Expected ';' after return",
            ';',
        )
        return ReturnStmt(
            expression,
            loc=(loc_token.line, loc_token.column),
        )

    def parse_break_stmt(self):
        loc_token = self.current_token
        self.advance()
        self.expect(
            'DELIMITER',
            "Expected ';' after break",
            ';',
        )
        return BreakStmt(
            loc=(loc_token.line, loc_token.column),
        )

    def parse_continue_stmt(self):
        loc_token = self.current_token
        self.advance()
        self.expect(
            'DELIMITER',
            "Expected ';' after continue",
            ';',
        )
        return ContinueStmt(
            loc=(loc_token.line, loc_token.column),
        )

    def parse_expr_stmt(self):
        loc_token = self.current_token
        expression = None
        if not self._matches(
            self.current_token,
            'DELIMITER',
            ';',
        ):
            expression = self.parse_expr()

        self.expect('DELIMITER', "Expected ';'", ';')
        return ExprStmt(
            expression,
            loc=(
                loc_token.line if loc_token is not None else 1,
                loc_token.column if loc_token is not None else 1,
            ),
        )

    def parse_expr(self):
        return self.parse_assignment()

    def parse_assignment(self):
        left = self.parse_logical_or()
        token = self.current_token
        if (
            isinstance(left, Identifier)
            and token is not None
            and token.type == 'OPERATOR'
            and token.lexeme in {
                '=', '+=', '-=', '*=', '/=', '%='
            }
        ):
            self.advance()
            right = self.parse_assignment()
            return BinaryExpr(
                left,
                token.lexeme,
                right,
                loc=(token.line, token.column),
            )
        return left

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while self._matches(
            self.current_token,
            'OPERATOR',
            '||',
        ):
            token = self.current_token
            self.advance()
            right = self.parse_logical_and()
            left = BinaryExpr(
                left,
                token.lexeme,
                right,
                loc=(token.line, token.column),
            )
        return left

    def parse_logical_and(self):
        left = self.parse_equality()
        while self._matches(
            self.current_token,
            'OPERATOR',
            '&&',
        ):
            token = self.current_token
            self.advance()
            right = self.parse_equality()
            left = BinaryExpr(
                left,
                token.lexeme,
                right,
                loc=(token.line, token.column),
            )
        return left

    def parse_equality(self):
        left = self.parse_relational()
        while (
            self.current_token is not None
            and self.current_token.type == 'OPERATOR'
            and self.current_token.lexeme in {'==', '!='}
        ):
            token = self.current_token
            self.advance()
            right = self.parse_relational()
            left = BinaryExpr(
                left,
                token.lexeme,
                right,
                loc=(token.line, token.column),
            )
        return left

    def parse_relational(self):
        left = self.parse_additive()
        while (
            self.current_token is not None
            and self.current_token.type == 'OPERATOR'
            and self.current_token.lexeme in {
                '<', '>', '<=', '>='
            }
        ):
            token = self.current_token
            self.advance()
            right = self.parse_additive()
            left = BinaryExpr(
                left,
                token.lexeme,
                right,
                loc=(token.line, token.column),
            )
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while (
            self.current_token is not None
            and self.current_token.type == 'OPERATOR'
            and self.current_token.lexeme in {'+', '-'}
        ):
            token = self.current_token
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryExpr(
                left,
                token.lexeme,
                right,
                loc=(token.line, token.column),
            )
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while (
            self.current_token is not None
            and self.current_token.type == 'OPERATOR'
            and self.current_token.lexeme in {'*', '/', '%'}
        ):
            token = self.current_token
            self.advance()
            right = self.parse_unary()
            left = BinaryExpr(
                left,
                token.lexeme,
                right,
                loc=(token.line, token.column),
            )
        return left

    def parse_unary(self):
        token = self.current_token
        if (
            token is not None
            and token.type == 'OPERATOR'
            and token.lexeme in {'-', '!', '&', '*'}
        ):
            self.advance()
            operand = self.parse_unary()
            return UnaryExpr(
                token.lexeme,
                operand,
                loc=(token.line, token.column),
            )
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()

        while True:
            token = self.current_token
            if token is None:
                break

            if self._matches(token, 'DELIMITER', '['):
                self.advance()
                index_expr = self.parse_expr()
                self.expect('DELIMITER', "Expected ']'", ']')
                node = BinaryExpr(
                    node,
                    '[]',
                    index_expr,
                    loc=(token.line, token.column),
                )
                continue

            if self._matches(token, 'DELIMITER', '('):
                self.advance()
                args = self.parse_arg_list()
                self.expect('DELIMITER', "Expected ')'", ')')
                if isinstance(node, Identifier):
                    node = CallExpr(
                        node.name,
                        args,
                        loc=(token.line, token.column),
                    )
                else:
                    self._report(
                        "Invalid function call target",
                        token,
                    )
                    node = CallExpr(
                        "",
                        args,
                        loc=(token.line, token.column),
                    )
                continue

            if (
                token.type == 'OPERATOR'
                and token.lexeme in {'.', '->'}
            ):
                operator = token.lexeme
                self.advance()
                ident_token = self.peek()
                if self._matches(ident_token, 'IDENTIFIER'):
                    self.advance()
                else:
                    self._report(
                        f"Expected identifier after '{operator}'",
                        ident_token or token,
                    )
                    ident_token = self._synthetic_token(
                        'IDENTIFIER',
                        None,
                        ident_token or token,
                    )
                node = BinaryExpr(
                    node,
                    operator,
                    Identifier(
                        ident_token.lexeme,
                        loc=(
                            ident_token.line,
                            ident_token.column,
                        ),
                    ),
                    loc=(token.line, token.column),
                )
                continue

            break

        return node

    def parse_primary(self):
        token = self.current_token
        if token is None:
            self._report("Unexpected EOF in primary expression")
            return IntLiteral('0', loc=(1, 1))

        if token.type == 'INTEGER':
            self.advance()
            return IntLiteral(
                token.lexeme,
                loc=(token.line, token.column),
            )
        if token.type == 'FLOAT':
            self.advance()
            return FloatLiteral(
                token.lexeme,
                loc=(token.line, token.column),
            )
        if token.type == 'STRING':
            self.advance()
            return StringLiteral(
                token.lexeme,
                loc=(token.line, token.column),
            )
        if token.type == 'CHARACTER':
            self.advance()
            return CharLiteral(
                token.lexeme,
                loc=(token.line, token.column),
            )
        if token.type == 'IDENTIFIER':
            self.advance()
            return Identifier(
                token.lexeme,
                loc=(token.line, token.column),
            )
        if self._matches(token, 'DELIMITER', '('):
            self.advance()
            expression = self.parse_expr()
            self.expect(
                'DELIMITER',
                "Expected ')' after expression",
                ')',
            )
            return expression

        if token.type == 'INVALID':
            self._report(
                token.error
                or f"Invalid token '{token.lexeme}'",
                token,
            )
            self.advance()
            return IntLiteral(
                '0',
                loc=(token.line, token.column),
            )

        # Missing-expression recovery: do not consume delimiters that the
        # surrounding production still needs to match.
        if (
            token.type == 'DELIMITER'
            and token.lexeme in {';', ')', ']', '}', ','}
        ):
            self._report(
                f"Expected expression before '{token.lexeme}'",
                token,
            )
            return IntLiteral(
                '0',
                loc=(token.line, token.column),
            )

        self._report(
            f"Unexpected token '{token.lexeme}' in primary expression",
            token,
        )
        self.advance()
        return IntLiteral(
            '0',
            loc=(token.line, token.column),
        )

    def parse_arg_list(self):
        args = []
        if self._matches(
            self.current_token,
            'DELIMITER',
            ')',
        ):
            return args

        while self.current_token is not None:
            args.append(self.parse_expr())
            if self._matches(
                self.current_token,
                'DELIMITER',
                ',',
            ):
                self.advance()
                continue
            break

        return args
