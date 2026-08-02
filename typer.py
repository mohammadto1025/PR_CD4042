from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from diagnostic import DiagnosticBag
from pars import (
    BinaryExpr,
    BreakStmt,
    Block,
    CallExpr,
    ContinueStmt,
    CharLiteral,
    ExprStmt,
    FloatLiteral,
    ForStmt,
    FuncDecl,
    Identifier,
    IfStmt,
    IntLiteral,
    Param,
    Program,
    ReturnStmt,
    StringLiteral,
    StructDecl,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from scope import Scope, ScopeKind
from semantic import SemanticAnalysisResult
from symbol import Symbol, SymbolKind


ERROR_TYPE = "<error>"
UNKNOWN_TYPE = "<unknown>"
VOID_TYPE = "void"


@dataclass(frozen=True)
class TypeCheckResult:
    diagnostics: DiagnosticBag
    annotations: Dict[int, str]
    semantic_result: SemanticAnalysisResult

    def type_of(self, node: object) -> Optional[str]:
        return self.annotations.get(id(node))


class TypeChecker:
    """Checks C-like types on the AST produced by ``parskon.py``.

    This module does not change the lexer, parser, highlighter, or semantic
    analyzer. It consumes the AST and the result of ``SemanticAnalyzer`` and
    writes computed types only to each node's existing ``type_annotation``
    field.
    """

    NUMERIC_RANK = {
        "char": 0,
        "int": 1,
        "float": 2,
        "double": 3,
    }

    ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%="}
    ARITHMETIC_OPERATORS = {"+", "-", "*", "/", "%"}
    RELATIONAL_OPERATORS = {"<", ">", "<=", ">="}
    EQUALITY_OPERATORS = {"==", "!="}
    LOGICAL_OPERATORS = {"&&", "||"}

    def __init__(self, filename: str = "<input>") -> None:
        self.filename = filename or "<input>"
        self.diagnostics = DiagnosticBag()
        self._annotations: Dict[int, str] = {}
        self._semantic: Optional[SemanticAnalysisResult] = None
        self._current_function: Optional[FuncDecl] = None

    @property
    def annotations(self) -> Dict[int, str]:
        return dict(self._annotations)

    def check(
        self,
        program: Program,
        semantic_result: SemanticAnalysisResult,
    ) -> TypeCheckResult:
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")
        if not isinstance(semantic_result, SemanticAnalysisResult):
            raise TypeError(
                "semantic_result must be an instance of SemanticAnalysisResult"
            )

        self.diagnostics = DiagnosticBag()
        self._annotations.clear()
        self._semantic = semantic_result
        self._current_function = None

        self._annotate(program, VOID_TYPE)
        for declaration in program.declarations:
            self._check_declaration(declaration)

        return TypeCheckResult(
            diagnostics=self.diagnostics,
            annotations=dict(self._annotations),
            semantic_result=semantic_result,
        )

    def _check_declaration(self, node: object) -> None:
        if isinstance(node, FuncDecl):
            self._check_function(node)
            return

        if isinstance(node, VarDecl):
            self._check_variable_declaration(node)
            return

        if isinstance(node, StructDecl):
            self._check_struct_declaration(node)
            return

        self._annotate(node, ERROR_TYPE)

    def _check_function(self, node: FuncDecl) -> None:
        return_type = self._normalize_type(node.return_type)
        self._validate_declared_type(
            return_type,
            node,
            allow_plain_void=True,
            role=f"return type of function '{node.name}'",
        )

        parameter_types = []
        for parameter in node.params:
            if not isinstance(parameter, Param):
                continue
            parameter_type = self._normalize_type(parameter.type_spec)
            parameter_types.append(parameter_type)
            self._validate_declared_type(
                parameter_type,
                parameter,
                allow_plain_void=False,
                role=f"parameter '{parameter.name}'",
            )
            self._annotate(parameter, parameter_type)

        signature = f"({', '.join(parameter_types)}) -> {return_type}"
        self._annotate(node, signature)

        previous_function = self._current_function
        self._current_function = node
        self._check_statement(node.body)
        self._current_function = previous_function

    def _check_struct_declaration(self, node: StructDecl) -> None:
        self._annotate(node, f"struct {node.name}")
        for field in node.fields:
            if not isinstance(field, VarDecl):
                continue
            field_type = self._normalize_type(field.type_spec)
            self._validate_declared_type(
                field_type,
                field,
                allow_plain_void=False,
                role=f"field '{field.name}'",
            )
            self._annotate(field, field_type)
            if field.init_expr is not None:
                actual_type = self._check_expression(field.init_expr)
                self._check_conversion(
                    expected=field_type,
                    actual=actual_type,
                    expression=field.init_expr,
                    node=field,
                    incompatible_message=(
                        f"Cannot initialize field '{field.name}' of type "
                        f"{field_type} with value of type {actual_type}"
                    ),
                )

    def _check_variable_declaration(self, node: VarDecl) -> None:
        declared_type = self._normalize_type(node.type_spec)
        self._validate_declared_type(
            declared_type,
            node,
            allow_plain_void=False,
            role=f"variable '{node.name}'",
        )
        self._annotate(node, declared_type)

        if node.init_expr is None:
            return

        actual_type = self._check_expression(node.init_expr)
        self._check_conversion(
            expected=declared_type,
            actual=actual_type,
            expression=node.init_expr,
            node=node,
            incompatible_message=(
                f"Cannot initialize '{node.name}' of type {declared_type} "
                f"with value of type {actual_type}"
            ),
        )

    def _check_statement(self, node: object) -> None:
        if node is None:
            return

        if isinstance(node, Block):
            for statement in node.statements:
                self._check_statement(statement)
            self._annotate(node, VOID_TYPE)
            return

        if isinstance(node, VarDecl):
            self._check_variable_declaration(node)
            return

        if isinstance(node, IfStmt):
            condition_type = self._check_expression(node.condition)
            self._require_scalar_condition(node.condition, condition_type, "if")
            self._check_statement(node.then_stmt)
            self._check_statement(node.else_stmt)
            self._annotate(node, VOID_TYPE)
            return

        if isinstance(node, WhileStmt):
            condition_type = self._check_expression(node.condition)
            self._require_scalar_condition(node.condition, condition_type, "while")
            self._check_statement(node.body)
            self._annotate(node, VOID_TYPE)
            return

        if isinstance(node, ForStmt):
            self._check_statement(node.init)

            if isinstance(node.condition, ExprStmt):
                if node.condition.expr is not None:
                    condition_type = self._check_expression(node.condition.expr)
                    self._require_scalar_condition(
                        node.condition.expr,
                        condition_type,
                        "for",
                    )
                self._annotate(node.condition, VOID_TYPE)
            else:
                condition_type = self._check_expression(node.condition)
                if node.condition is not None:
                    self._require_scalar_condition(
                        node.condition,
                        condition_type,
                        "for",
                    )

            self._check_expression(node.increment)
            self._check_statement(node.body)
            self._annotate(node, VOID_TYPE)
            return

        if isinstance(node, (BreakStmt, ContinueStmt)):
            self._annotate(node, VOID_TYPE)
            return

        if isinstance(node, ReturnStmt):
            self._check_return_statement(node)
            return

        if isinstance(node, ExprStmt):
            self._check_expression(node.expr)
            self._annotate(node, VOID_TYPE)
            return

        self._check_expression(node)

    def _check_return_statement(self, node: ReturnStmt) -> None:
        if self._current_function is None:
            self._error("Return statement outside a function", node)
            self._check_expression(node.value)
            self._annotate(node, ERROR_TYPE)
            return

        expected = self._normalize_type(self._current_function.return_type)
        function_name = self._current_function.name

        if node.value is None:
            if expected != VOID_TYPE:
                self._error(
                    f"Non-void function '{function_name}' must return a value",
                    node,
                )
            self._annotate(node, VOID_TYPE)
            return

        actual = self._check_expression(node.value)
        if expected == VOID_TYPE:
            self._error(
                f"Void function '{function_name}' cannot return a value",
                node,
            )
            self._annotate(node, ERROR_TYPE)
            return

        self._check_conversion(
            expected=expected,
            actual=actual,
            expression=node.value,
            node=node,
            incompatible_message=(
                f"Return type mismatch in function '{function_name}': "
                f"expected {expected}, got {actual}"
            ),
        )
        self._annotate(node, expected)

    def _check_expression(self, node: object) -> str:
        if node is None:
            return VOID_TYPE

        if isinstance(node, IntLiteral):
            return self._annotate(node, "int")

        if isinstance(node, FloatLiteral):
            value = str(node.value).strip()
            literal_type = "float" if value.lower().endswith("f") else "double"
            return self._annotate(node, literal_type)

        if isinstance(node, StringLiteral):
            return self._annotate(node, "char*")

        if isinstance(node, CharLiteral):
            return self._annotate(node, "char")

        if isinstance(node, Identifier):
            symbol = self._symbol_for(node)
            if symbol is None:
                return self._annotate(node, ERROR_TYPE)
            if symbol.signature is not None:
                return self._annotate(node, str(symbol.signature))
            return self._annotate(node, self._normalize_type(symbol.type))

        if isinstance(node, CallExpr):
            return self._check_call(node)

        if isinstance(node, UnaryExpr):
            return self._check_unary(node)

        if isinstance(node, BinaryExpr):
            return self._check_binary(node)

        return self._annotate(node, ERROR_TYPE)

    def _check_call(self, node: CallExpr) -> str:
        argument_types = [self._check_expression(argument) for argument in node.args]
        symbol = self._symbol_for(node)

        if symbol is None:
            return self._annotate(node, ERROR_TYPE)

        if symbol.kind not in {
            SymbolKind.FUNCTION,
            SymbolKind.METHOD,
            SymbolKind.CONSTRUCTOR,
        }:
            return self._annotate(node, ERROR_TYPE)

        if symbol.signature is None:
            self._error(
                f"Callable symbol '{node.callee}' has no type signature",
                node,
                length=max(1, len(node.callee)),
            )
            return self._annotate(node, ERROR_TYPE)

        expected_types = tuple(
            self._normalize_type(item)
            for item in symbol.signature.parameter_types
        )
        if len(argument_types) != len(expected_types):
            self._error(
                (
                    f"Function '{node.callee}' expects {len(expected_types)} "
                    f"argument(s), got {len(argument_types)}"
                ),
                node,
                length=max(1, len(node.callee)),
            )

        for index, (actual, expected) in enumerate(
            zip(argument_types, expected_types),
            start=1,
        ):
            argument = node.args[index - 1]
            self._check_conversion(
                expected=expected,
                actual=actual,
                expression=argument,
                node=argument,
                incompatible_message=(
                    f"Argument {index} of '{node.callee}' expects "
                    f"{expected}, got {actual}"
                ),
            )

        return_type = self._normalize_type(symbol.signature.return_type)
        return self._annotate(node, return_type)

    def _check_unary(self, node: UnaryExpr) -> str:
        operand_type = self._check_expression(node.operand)
        if self._is_error_type(operand_type):
            return self._annotate(node, ERROR_TYPE)

        if node.op == "-":
            if not self._is_numeric(operand_type):
                self._error(
                    f"Unary '-' requires a numeric operand, got {operand_type}",
                    node,
                )
                return self._annotate(node, ERROR_TYPE)
            result = "int" if operand_type == "char" else operand_type
            return self._annotate(node, result)

        if node.op == "!":
            if not self._is_scalar(operand_type):
                self._error(
                    f"Unary '!' requires a scalar operand, got {operand_type}",
                    node,
                )
                return self._annotate(node, ERROR_TYPE)
            return self._annotate(node, "int")

        if node.op == "&":
            if not self._is_lvalue(node.operand):
                self._error("Address-of operator requires an lvalue", node)
                return self._annotate(node, ERROR_TYPE)
            return self._annotate(node, self._pointer_to(operand_type))

        if node.op == "*":
            if not self._is_pointer(operand_type):
                self._error(
                    f"Dereference operator requires a pointer, got {operand_type}",
                    node,
                )
                return self._annotate(node, ERROR_TYPE)
            return self._annotate(node, self._pointee_type(operand_type))

        self._error(f"Unsupported unary operator '{node.op}'", node)
        return self._annotate(node, ERROR_TYPE)

    def _check_binary(self, node: BinaryExpr) -> str:
        if node.op in self.ASSIGNMENT_OPERATORS:
            return self._check_assignment(node)

        if node.op == "[]":
            return self._check_index(node)

        if node.op in {".", "->"}:
            return self._check_member_access(node)

        left_type = self._check_expression(node.left)
        right_type = self._check_expression(node.right)

        if self._is_error_type(left_type) or self._is_error_type(right_type):
            return self._annotate(node, ERROR_TYPE)

        if node.op in self.ARITHMETIC_OPERATORS:
            return self._check_arithmetic(node, left_type, right_type)

        if node.op in self.RELATIONAL_OPERATORS:
            if self._valid_relational_operands(left_type, right_type):
                return self._annotate(node, "int")
            self._error(
                (
                    f"Operator '{node.op}' cannot compare {left_type} "
                    f"and {right_type}"
                ),
                node,
            )
            return self._annotate(node, ERROR_TYPE)

        if node.op in self.EQUALITY_OPERATORS:
            if self._valid_equality_operands(
                node.left,
                left_type,
                node.right,
                right_type,
            ):
                return self._annotate(node, "int")
            self._error(
                (
                    f"Operator '{node.op}' cannot compare {left_type} "
                    f"and {right_type}"
                ),
                node,
            )
            return self._annotate(node, ERROR_TYPE)

        if node.op in self.LOGICAL_OPERATORS:
            if self._is_scalar(left_type) and self._is_scalar(right_type):
                return self._annotate(node, "int")
            self._error(
                (
                    f"Operator '{node.op}' requires scalar operands, got "
                    f"{left_type} and {right_type}"
                ),
                node,
            )
            return self._annotate(node, ERROR_TYPE)

        self._error(f"Unsupported binary operator '{node.op}'", node)
        return self._annotate(node, ERROR_TYPE)

    def _check_assignment(self, node: BinaryExpr) -> str:
        left_type = self._check_expression(node.left)
        right_type = self._check_expression(node.right)

        if not self._is_lvalue(node.left):
            self._error("Left side of assignment must be an lvalue", node.left)
            return self._annotate(node, ERROR_TYPE)

        if self._is_error_type(left_type) or self._is_error_type(right_type):
            return self._annotate(node, ERROR_TYPE)

        if node.op == "=":
            self._check_conversion(
                expected=left_type,
                actual=right_type,
                expression=node.right,
                node=node,
                incompatible_message=(
                    f"Cannot assign value of type {right_type} to {left_type}"
                ),
            )
            return self._annotate(node, left_type)

        operation = node.op[0]
        result_type = self._compound_operation_type(
            operation,
            left_type,
            right_type,
            node,
        )
        if self._is_error_type(result_type):
            return self._annotate(node, ERROR_TYPE)

        self._check_conversion(
            expected=left_type,
            actual=result_type,
            expression=node.right,
            node=node,
            incompatible_message=(
                f"Result of '{node.op}' has type {result_type}, "
                f"which cannot be assigned to {left_type}"
            ),
        )
        return self._annotate(node, left_type)

    def _compound_operation_type(
        self,
        operation: str,
        left_type: str,
        right_type: str,
        node: BinaryExpr,
    ) -> str:
        if operation in {"+", "-"} and self._is_pointer(left_type):
            if self._is_integer(right_type):
                return left_type
            self._error(
                (
                    f"Operator '{node.op}' requires an integer right operand "
                    f"for pointer type {left_type}, got {right_type}"
                ),
                node,
            )
            return ERROR_TYPE

        if self._is_numeric(left_type) and self._is_numeric(right_type):
            return self._common_numeric_type(left_type, right_type)

        self._error(
            (
                f"Operator '{node.op}' requires compatible numeric operands, "
                f"got {left_type} and {right_type}"
            ),
            node,
        )
        return ERROR_TYPE

    def _check_arithmetic(
        self,
        node: BinaryExpr,
        left_type: str,
        right_type: str,
    ) -> str:
        if node.op == "%":
            if self._is_integer(left_type) and self._is_integer(right_type):
                return self._annotate(node, "int")
            self._error(
                (
                    "Operator '%' requires integer operands, got "
                    f"{left_type} and {right_type}"
                ),
                node,
            )
            return self._annotate(node, ERROR_TYPE)

        if self._is_numeric(left_type) and self._is_numeric(right_type):
            return self._annotate(
                node,
                self._common_numeric_type(left_type, right_type),
            )

        if node.op == "+":
            if self._is_pointer(left_type) and self._is_integer(right_type):
                return self._annotate(node, left_type)
            if self._is_integer(left_type) and self._is_pointer(right_type):
                return self._annotate(node, right_type)

        if node.op == "-":
            if self._is_pointer(left_type) and self._is_integer(right_type):
                return self._annotate(node, left_type)
            if (
                self._is_pointer(left_type)
                and self._is_pointer(right_type)
                and self._compatible_pointer_types(left_type, right_type)
            ):
                return self._annotate(node, "int")

        self._error(
            (
                f"Operator '{node.op}' cannot be applied to "
                f"{left_type} and {right_type}"
            ),
            node,
        )
        return self._annotate(node, ERROR_TYPE)

    def _check_index(self, node: BinaryExpr) -> str:
        base_type = self._check_expression(node.left)
        index_type = self._check_expression(node.right)

        if self._is_error_type(base_type) or self._is_error_type(index_type):
            return self._annotate(node, ERROR_TYPE)

        if not self._is_pointer(base_type):
            self._error(
                f"Indexing requires a pointer operand, got {base_type}",
                node.left,
            )
            return self._annotate(node, ERROR_TYPE)

        if not self._is_integer(index_type):
            self._error(
                f"Array index must be an integer, got {index_type}",
                node.right,
            )
            return self._annotate(node, ERROR_TYPE)

        return self._annotate(node, self._pointee_type(base_type))

    def _check_member_access(self, node: BinaryExpr) -> str:
        base_type = self._check_expression(node.left)
        if self._is_error_type(base_type):
            self._annotate(node.right, ERROR_TYPE)
            return self._annotate(node, ERROR_TYPE)

        expected_pointer = node.op == "->"
        if expected_pointer:
            if not self._is_pointer(base_type):
                self._error(
                    f"Operator '->' requires a pointer, got {base_type}",
                    node,
                )
                self._annotate(node.right, ERROR_TYPE)
                return self._annotate(node, ERROR_TYPE)
            struct_type = self._pointee_type(base_type)
        else:
            if self._is_pointer(base_type):
                self._error(
                    f"Operator '.' requires a non-pointer struct, got {base_type}",
                    node,
                )
                self._annotate(node.right, ERROR_TYPE)
                return self._annotate(node, ERROR_TYPE)
            struct_type = base_type

        struct_name = self._struct_name(struct_type)
        if struct_name is None:
            self._error(
                f"Member access requires a struct type, got {base_type}",
                node,
            )
            self._annotate(node.right, ERROR_TYPE)
            return self._annotate(node, ERROR_TYPE)

        if not isinstance(node.right, Identifier):
            self._error("Member name must be an identifier", node)
            return self._annotate(node, ERROR_TYPE)

        field_symbol = self._find_struct_field(struct_name, node.right.name)
        if field_symbol is None:
            self._error(
                f"Struct '{struct_name}' has no field '{node.right.name}'",
                node.right,
                length=max(1, len(node.right.name)),
            )
            self._annotate(node.right, ERROR_TYPE)
            return self._annotate(node, ERROR_TYPE)

        field_type = self._normalize_type(field_symbol.type)
        self._annotate(node.right, field_type)
        try:
            setattr(node.right, "resolved_symbol", field_symbol)
        except (AttributeError, TypeError):
            pass
        return self._annotate(node, field_type)

    def _require_scalar_condition(
        self,
        node: object,
        condition_type: str,
        statement_name: str,
    ) -> None:
        if self._is_error_type(condition_type):
            return
        if not self._is_scalar(condition_type):
            self._error(
                (
                    f"Condition of '{statement_name}' must be scalar, "
                    f"got {condition_type}"
                ),
                node,
            )

    def _check_conversion(
        self,
        expected: str,
        actual: str,
        expression: object,
        node: object,
        incompatible_message: str,
    ) -> None:
        expected = self._normalize_type(expected)
        actual = self._normalize_type(actual)

        if self._is_error_type(expected) or self._is_error_type(actual):
            return

        conversion = self._conversion_kind(expected, actual, expression)
        if conversion == "incompatible":
            self._error(incompatible_message, node)
        elif conversion == "narrowing":
            self._warning(
                (
                    f"Implicit conversion from {actual} to {expected} "
                    "may lose precision"
                ),
                node,
            )

    def _conversion_kind(
        self,
        expected: str,
        actual: str,
        expression: object,
    ) -> str:
        if expected == actual:
            return "exact"

        if self._is_numeric(expected) and self._is_numeric(actual):
            if self.NUMERIC_RANK[actual] <= self.NUMERIC_RANK[expected]:
                return "widening"
            return "narrowing"

        if self._is_pointer(expected):
            if self._is_null_pointer_constant(expression, actual):
                return "widening"
            if self._is_pointer(actual):
                if self._compatible_pointer_types(expected, actual):
                    return "widening"
            return "incompatible"

        return "incompatible"

    def _valid_relational_operands(self, left: str, right: str) -> bool:
        if self._is_numeric(left) and self._is_numeric(right):
            return True
        return (
            self._is_pointer(left)
            and self._is_pointer(right)
            and self._compatible_pointer_types(left, right)
        )

    def _valid_equality_operands(
        self,
        left_node: object,
        left_type: str,
        right_node: object,
        right_type: str,
    ) -> bool:
        if self._is_numeric(left_type) and self._is_numeric(right_type):
            return True
        if self._is_pointer(left_type) and self._is_pointer(right_type):
            return self._compatible_pointer_types(left_type, right_type)
        if self._is_pointer(left_type):
            return self._is_null_pointer_constant(right_node, right_type)
        if self._is_pointer(right_type):
            return self._is_null_pointer_constant(left_node, left_type)
        return False

    def _validate_declared_type(
        self,
        type_name: str,
        node: object,
        allow_plain_void: bool,
        role: str,
    ) -> None:
        if type_name == VOID_TYPE:
            if not allow_plain_void:
                self._error(f"{role.capitalize()} cannot have type void", node)
            return

        base_type = type_name.rstrip("*")
        if base_type in self.NUMERIC_RANK:
            return
        if base_type == VOID_TYPE and self._is_pointer(type_name):
            return
        if self._struct_name(base_type) is not None:
            return

        self._error(f"Unknown type '{type_name}' for {role}", node)

    def _symbol_for(self, node: object) -> Optional[Symbol]:
        if self._semantic is None:
            return None
        symbol = self._semantic.symbol_for(node)
        if symbol is not None:
            return symbol
        candidate = getattr(node, "resolved_symbol", None)
        return candidate if isinstance(candidate, Symbol) else None

    def _find_struct_field(
        self,
        struct_name: str,
        field_name: str,
    ) -> Optional[Symbol]:
        if self._semantic is None:
            return None

        for child in self._semantic.global_scope.children:
            if child.kind == ScopeKind.STRUCT and child.name.split("#", 1)[0] == struct_name:
                return child.resolve_local(field_name)
        return None

    def _annotate(self, node: object, type_name: str) -> str:
        if node is None:
            return type_name
        normalized = self._normalize_annotation(type_name)
        self._annotations[id(node)] = normalized
        try:
            setattr(node, "type_annotation", normalized)
        except (AttributeError, TypeError):
            pass
        return normalized

    def _error(self, message: str, node: object, length: int = 1) -> None:
        line, column = self._node_location(node)
        self.diagnostics.error(
            message,
            file=self.filename,
            line=line,
            column=column,
            length=max(1, length),
        )

    def _warning(self, message: str, node: object, length: int = 1) -> None:
        line, column = self._node_location(node)
        self.diagnostics.warning(
            message,
            file=self.filename,
            line=line,
            column=column,
            length=max(1, length),
        )

    @staticmethod
    def _node_location(node: object) -> Tuple[int, int]:
        loc = getattr(node, "loc", None)
        if not loc:
            return 1, 1
        line, column = loc
        return max(1, line), max(1, column)


    @classmethod
    def _normalize_annotation(cls, type_name: str) -> str:
        value = str(type_name).strip()
        if value.startswith("(") and "->" in value:
            parameter_part, return_part = value.split("->", 1)
            raw_parameters = parameter_part.strip()[1:-1].strip()
            if raw_parameters:
                parameters = [
                    cls._normalize_type(item)
                    for item in raw_parameters.split(",")
                ]
            else:
                parameters = []
            return_type = cls._normalize_type(return_part)
            return f"({', '.join(parameters)}) -> {return_type}"
        return cls._normalize_type(value)

    @staticmethod
    def _normalize_type(type_name: str) -> str:
        if not type_name:
            return UNKNOWN_TYPE
        value = "".join(str(type_name).split())
        if value.startswith("struct") and value != "struct":
            # Keep one space between the keyword and the struct name.
            suffix = value[len("struct"):]
            pointer_count = len(suffix) - len(suffix.rstrip("*"))
            name = suffix.rstrip("*")
            return f"struct {name}{'*' * pointer_count}"
        return value

    @classmethod
    def _is_numeric(cls, type_name: str) -> bool:
        return type_name in cls.NUMERIC_RANK

    @staticmethod
    def _is_pointer(type_name: str) -> bool:
        return type_name.endswith("*") and type_name not in {ERROR_TYPE, UNKNOWN_TYPE}

    @classmethod
    def _is_integer(cls, type_name: str) -> bool:
        return type_name in {"char", "int"}

    @classmethod
    def _is_scalar(cls, type_name: str) -> bool:
        return cls._is_numeric(type_name) or cls._is_pointer(type_name)

    @classmethod
    def _common_numeric_type(cls, left: str, right: str) -> str:
        left_promoted = "int" if left == "char" else left
        right_promoted = "int" if right == "char" else right
        return max(
            (left_promoted, right_promoted),
            key=lambda item: cls.NUMERIC_RANK[item],
        )

    @staticmethod
    def _pointer_to(type_name: str) -> str:
        return f"{type_name}*"

    @staticmethod
    def _pointee_type(type_name: str) -> str:
        return type_name[:-1] if type_name.endswith("*") else ERROR_TYPE

    @classmethod
    def _compatible_pointer_types(cls, left: str, right: str) -> bool:
        if left == right:
            return True
        return cls._pointee_type(left) == "void" or cls._pointee_type(right) == "void"

    @staticmethod
    def _is_null_pointer_constant(node: object, type_name: str) -> bool:
        if type_name != "int" or not isinstance(node, IntLiteral):
            return False
        value = str(node.value).strip().lower()
        try:
            if value.startswith("0x"):
                return int(value, 16) == 0
            if value.startswith("0b"):
                return int(value, 2) == 0
            return int(value, 10) == 0
        except ValueError:
            return False

    @staticmethod
    def _is_lvalue(node: object) -> bool:
        if isinstance(node, Identifier):
            return True
        if isinstance(node, BinaryExpr) and node.op in {"[]", ".", "->"}:
            return True
        return isinstance(node, UnaryExpr) and node.op == "*"

    @staticmethod
    def _struct_name(type_name: str) -> Optional[str]:
        value = type_name.rstrip("*")
        prefix = "struct "
        if not value.startswith(prefix):
            return None
        name = value[len(prefix):]
        return name or None

    @staticmethod
    def _is_error_type(type_name: str) -> bool:
        return type_name in {ERROR_TYPE, UNKNOWN_TYPE}
