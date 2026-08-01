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