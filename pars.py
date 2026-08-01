from dataclasses import dataclass, field
from typing import List, Optional, Any

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

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self._skip_whitespace()
        self.current_token = self.tokens[self.pos] if self.pos < len(self.tokens) else None
        self.errors = []
        
    def _skip_whitespace(self):
        while self.pos < len(self.tokens) and self.tokens[self.pos].type == 'WHITESPACE':
            self.pos += 1

    def peek(self):
        self._skip_whitespace()
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
        return self.current_token

    def advance(self):
        if self.current_token:
            self.pos += 1
        self._skip_whitespace()
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
        return self.current_token

    def match(self, expected_type, error_msg=None):
        token = self.peek()  
        if token and token.type == expected_type:
            self.advance()
            return token
        else:
            if error_msg is None:
                error_msg = f"Expected '{expected_type}', got '{token.type if token else 'EOF'}'"
            self.panic(error_msg, token)
            return None

    def expect(self, expected_type, error_msg=None, expected_lexeme=None):
        token = self.peek()
        if token and token.type == expected_type:
            if expected_lexeme is not None and token.lexeme != expected_lexeme:
                err_msg = f"Expected '{expected_lexeme}', got '{token.lexeme}'"
                self.panic(err_msg, token)
                return Token(expected_type, f"<missing_{expected_lexeme}>",
                           token.line if token else 0,
                           token.column if token else 0)
            self.advance()
            return token
        else:
            if error_msg is None:
                error_msg = f"Expected '{expected_type}', got '{token.type if token else 'EOF'}'"
            self.panic(error_msg, token)
            return Token(expected_type, f"<missing_{expected_type}>",
                       token.line if token else 0,
                       token.column if token else 0)

    def is_sync_point(self):
        if not self.current_token:
            return True
        sync_set = { 'SEMI', 'RBRACE', 'LBRACE', 'KEYWORD' }  
        if self.current_token.type in sync_set:
            return True
        if self.current_token.type == 'KEYWORD' and self.current_token.lexeme in {'if', 'while', 'for', 'return'}:
            return True
        return False

    def sync(self):
        while self.current_token and not self.is_sync_point():
            self.advance()
        if self.current_token:
            self.advance()

    def panic(self, message, token=None):
        if token is None:
            token = self.current_token
        line = token.line if token else 0
        col = token.column if token else 0
        self.errors.append(f"Syntax Error at {line}:{col} - {message}")
        self.sync()

    def parse_program(self) -> Program:
        loc = self.current_token  
        declarations = []
        while self.current_token is not None:
            decl = self.parse_declaration()
            if decl:
                declarations.append(decl)
            else:
                if self.current_token:
                    self.advance()
        return Program(declarations, loc=(loc.line if loc else 0, loc.column if loc else 0))

    def parse_declaration(self):
        token = self.current_token
        if not token:
            return None
        if token.type == 'KEYWORD' and token.lexeme in {'int', 'float', 'char', 'void', 'double'}:
            return self.parse_function_or_var_decl()
        elif token.type == 'KEYWORD' and token.lexeme == 'struct':
            return self.parse_struct_decl()
        self.panic(f"Unexpected token '{token.lexeme}' in declaration", token)
        return None

    def parse_function_or_var_decl(self):
        type_spec = self.parse_type_spec()
        if not type_spec:
            return None
        ident_token = self.expect('IDENTIFIER', "Expected identifier after type specifier")
        if not ident_token:
            return None
        name = ident_token.lexeme
        loc = (ident_token.line, ident_token.column)
        next_tok = self.current_token
        if next_tok and next_tok.type == 'DELIMITER' and next_tok.lexeme == '(':
            self.advance()  
            params = self.parse_param_list()
            self.expect('DELIMITER', "Expected ')' after parameters", ')')  
            body = self.parse_block()
            return FuncDecl(type_spec, name, params, body, loc=loc)
        else:
            init_expr = None
            if self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme == '=':
                self.advance() 
                init_expr = self.parse_expr()
            self.expect('DELIMITER', "Expected ';' after variable declaration", ';')
            return VarDecl(type_spec, name, init_expr, loc=loc)

    def parse_type_spec(self):
        token = self.current_token
        if not token or token.type != 'KEYWORD' or token.lexeme not in {'int', 'float', 'char', 'void', 'double'}:
            self.panic(f"Expected type specifier, got '{token.lexeme if token else 'EOF'}'", token)
            return None
        type_name = token.lexeme
        self.advance()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme == '*':
            type_name += '*'
            self.advance()
        return type_name

    def parse_struct_decl(self):
        self.expect('KEYWORD', "Expected 'struct'", 'struct')
        name_token = self.expect('IDENTIFIER', "Expected struct name")
        name = name_token.lexeme
        loc = (name_token.line, name_token.column)

        self.expect('DELIMITER', "Expected '{' after struct name", '{')
        fields = []
        while self.current_token and not (self.current_token.type == 'DELIMITER' and self.current_token.lexeme == '}'):
            type_spec = self.parse_type_spec()
            if not type_spec:
                break
            ident_token = self.expect('IDENTIFIER', "Expected field name")
            field_name = ident_token.lexeme
            self.expect('DELIMITER', "Expected ';' after field", ';')
            fields.append(VarDecl(type_spec, field_name, None, loc=(ident_token.line, ident_token.column)))
        self.expect('DELIMITER', "Expected '}'", '}')
        if self.current_token and self.current_token.type == 'DELIMITER' and self.current_token.lexeme == ';':
            self.advance()
        return StructDecl(name, fields, loc=loc)

    def parse_param_list(self):
        params = []
        if self.current_token and self.current_token.type == 'DELIMITER' and self.current_token.lexeme == ')':
            return params
        params.append(self.parse_param())
        while self.current_token and self.current_token.type == 'DELIMITER' and self.current_token.lexeme == ',':
            self.advance()  
            params.append(self.parse_param())
        return params

    def parse_param(self):
        type_spec = self.parse_type_spec()
        if not type_spec:
            return None
        ident_token = self.expect('IDENTIFIER', "Expected parameter name")
        return Param(type_spec, ident_token.lexeme, loc=(ident_token.line, ident_token.column))

    def parse_block(self):
        loc_token = self.current_token
        self.expect('DELIMITER', "Expected '{'", '{')
        statements = []
        while self.current_token and not (self.current_token.type == 'DELIMITER' and self.current_token.lexeme == '}'):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            else:
                if self.current_token:
                    self.advance()
        self.expect('DELIMITER', "Expected '}'", '}')
        return Block(statements, loc=(loc_token.line if loc_token else 0, loc_token.column if loc_token else 0))

    def parse_statement(self):
        token = self.current_token
        if not token:
            return None
        if token.type == 'KEYWORD':
            if token.lexeme == 'if':
                return self.parse_if_stmt()
            elif token.lexeme == 'while':
                return self.parse_while_stmt()
            elif token.lexeme == 'for':
                return self.parse_for_stmt()
            elif token.lexeme == 'return':
                return self.parse_return_stmt()
            elif token.lexeme in {'int', 'float', 'char', 'void', 'double'}:
                return self.parse_var_decl_statement()
        elif token.type == 'DELIMITER' and token.lexeme == '{':
            return self.parse_block()
        elif token.type == 'DELIMITER' and token.lexeme == ';':
            self.advance()
            return ExprStmt(None, loc=(token.line, token.column))
        return self.parse_expr_stmt()

    def parse_var_decl_statement(self):
        type_spec = self.parse_type_spec()
        if not type_spec:
            return None
        ident_token = self.expect('IDENTIFIER', "Expected identifier")
        name = ident_token.lexeme
        loc = (ident_token.line, ident_token.column)
        init_expr = None
        if self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme == '=':
            self.advance()
            init_expr = self.parse_expr()
        self.expect('DELIMITER', "Expected ';' after variable declaration", ';')
        return VarDecl(type_spec, name, init_expr, loc=loc)

    def parse_if_stmt(self):
        loc_token = self.current_token
        self.advance()  
        self.expect('DELIMITER', "Expected '(' after 'if'", '(')
        condition = self.parse_expr()
        self.expect('DELIMITER', "Expected ')' after condition", ')')
        then_stmt = self.parse_statement()
        else_stmt = None
        if self.current_token and self.current_token.type == 'KEYWORD' and self.current_token.lexeme == 'else':
            self.advance()
            else_stmt = self.parse_statement()
        return IfStmt(condition, then_stmt, else_stmt, loc=(loc_token.line, loc_token.column))

    def parse_while_stmt(self):
        loc_token = self.current_token
        self.advance() 
        self.expect('DELIMITER', "Expected '(' after 'while'", '(')
        condition = self.parse_expr()
        self.expect('DELIMITER', "Expected ')' after condition", ')')
        body = self.parse_statement()
        return WhileStmt(condition, body, loc=(loc_token.line, loc_token.column))

    def parse_for_stmt(self):
        loc_token = self.current_token
        self.advance()  
        self.expect('DELIMITER', "Expected '(' after 'for'", '(')
        init_stmt = self.parse_expr_stmt()  
        cond_stmt = self.parse_expr_stmt()
        inc_expr = None
        if self.current_token and self.current_token.type != 'DELIMITER' and self.current_token.lexeme != ')':
            inc_expr = self.parse_expr()
        self.expect('DELIMITER', "Expected ')' after for clauses", ')')
        body = self.parse_statement()
        return ForStmt(init_stmt, cond_stmt, inc_expr, body, loc=(loc_token.line, loc_token.column))

    def parse_return_stmt(self):
        loc_token = self.current_token
        self.advance()  
        expr = None
        if self.current_token and not (self.current_token.type == 'DELIMITER' and self.current_token.lexeme == ';'):
            expr = self.parse_expr()
        self.expect('DELIMITER', "Expected ';' after return", ';')
        return ReturnStmt(expr, loc=(loc_token.line, loc_token.column))

    def parse_expr_stmt(self):
        loc = self.current_token
        expr = None
        if self.current_token and not (self.current_token.type == 'DELIMITER' and self.current_token.lexeme == ';'):
            expr = self.parse_expr()
        self.expect('DELIMITER', "Expected ';'", ';')
        return ExprStmt(expr, loc=(loc.line if loc else 0, loc.column if loc else 0))

    def parse_expr(self):
        return self.parse_assignment()

    def parse_assignment(self):
        left = self.parse_logical_or()
        if left and isinstance(left, Identifier):
            token = self.current_token
            if token and token.type == 'OPERATOR' and token.lexeme in {'=', '+=', '-=', '*='}:
                self.advance()
                right = self.parse_assignment()
                return BinaryExpr(left, token.lexeme, right, loc=(token.line, token.column))
        return left

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme == '||':
            op_token = self.current_token
            self.advance()
            right = self.parse_logical_and()
            left = BinaryExpr(left, '||', right, loc=(op_token.line, op_token.column))
        return left

    def parse_logical_and(self):
        left = self.parse_equality()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme == '&&':
            op_token = self.current_token
            self.advance()
            right = self.parse_equality()
            left = BinaryExpr(left, '&&', right, loc=(op_token.line, op_token.column))
        return left

    def parse_equality(self):
        left = self.parse_relational()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme in {'==', '!='}:
            op_token = self.current_token
            self.advance()
            right = self.parse_relational()
            left = BinaryExpr(left, op_token.lexeme, right, loc=(op_token.line, op_token.column))
        return left

    def parse_relational(self):
        left = self.parse_additive()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme in {'<', '>', '<=', '>='}:
            op_token = self.current_token
            self.advance()
            right = self.parse_additive()
            left = BinaryExpr(left, op_token.lexeme, right, loc=(op_token.line, op_token.column))
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme in {'+', '-'}:
            op_token = self.current_token
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryExpr(left, op_token.lexeme, right, loc=(op_token.line, op_token.column))
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token and self.current_token.type == 'OPERATOR' and self.current_token.lexeme in {'*', '/', '%'}:
            op_token = self.current_token
            self.advance()
            right = self.parse_unary()
            left = BinaryExpr(left, op_token.lexeme, right, loc=(op_token.line, op_token.column))
        return left

    def parse_unary(self):
        token = self.current_token
        if token and token.type == 'OPERATOR' and token.lexeme in {'-', '!', '&', '*'}:
            self.advance()
            operand = self.parse_unary()
            return UnaryExpr(token.lexeme, operand, loc=(token.line, token.column))
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            token = self.current_token
            if not token:
                break
            if token.type == 'DELIMITER' and token.lexeme == '[':
                self.advance()
                index_expr = self.parse_expr()
                self.expect('DELIMITER', "Expected ']'", ']')
                node = BinaryExpr(node, '[]', index_expr, loc=(token.line, token.column))
            elif token.type == 'DELIMITER' and token.lexeme == '(':
                self.advance()
                args = self.parse_arg_list()
                self.expect('DELIMITER', "Expected ')'", ')')
                if isinstance(node, Identifier):
                    node = CallExpr(node.name, args, loc=(token.line, token.column))
                else:
                    self.panic("Invalid function call target", token)
                    node = CallExpr("", args, loc=(token.line, token.column))
            elif token.type == 'OPERATOR' and token.lexeme == '.':
                self.advance()
                ident_token = self.expect('IDENTIFIER', "Expected identifier after '.'")
                node = BinaryExpr(node, '.', Identifier(ident_token.lexeme, loc=(ident_token.line, ident_token.column)), loc=(token.line, token.column))
            elif token.type == 'OPERATOR' and token.lexeme == '->':
                self.advance()
                ident_token = self.expect('IDENTIFIER', "Expected identifier after '->'")
                node = BinaryExpr(node, '->', Identifier(ident_token.lexeme, loc=(ident_token.line, ident_token.column)), loc=(token.line, token.column))
            else:
                break
        return node

    def parse_primary(self):
        token = self.current_token
        if not token:
            self.panic("Unexpected EOF in primary expression")
            return None

        if token.type == 'INTEGER':
            self.advance()
            return IntLiteral(token.lexeme, loc=(token.line, token.column))
        elif token.type == 'FLOAT':
            self.advance()
            return FloatLiteral(token.lexeme, loc=(token.line, token.column))
        elif token.type == 'STRING':
            self.advance()
            return StringLiteral(token.lexeme, loc=(token.line, token.column))
        elif token.type == 'CHARACTER':
            self.advance()
            return CharLiteral(token.lexeme, loc=(token.line, token.column))
        elif token.type == 'IDENTIFIER':
            self.advance()
            return Identifier(token.lexeme, loc=(token.line, token.column))
        elif token.type == 'DELIMITER' and token.lexeme == '(':
            self.advance()
            expr = self.parse_expr()
            self.expect('DELIMITER', "Expected ')' after expression", ')')
            return expr
        else:
            self.panic(f"Unexpected token '{token.lexeme}' in primary expression", token)
            return None

    def parse_arg_list(self):
        args = []
        if self.current_token and self.current_token.type == 'DELIMITER' and self.current_token.lexeme == ')':
            return args
        args.append(self.parse_expr())
        while self.current_token and self.current_token.type == 'DELIMITER' and self.current_token.lexeme == ',':
            self.advance()
            args.append(self.parse_expr())
        return args