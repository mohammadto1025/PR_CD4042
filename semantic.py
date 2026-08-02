from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

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
from symbol import (
    FunctionSignature,
    ReferenceKind,
    SourceLocation,
    Symbol,
    SymbolKind,
)


@dataclass(frozen=True)
class SemanticAnalysisResult:
    global_scope: Scope
    diagnostics: DiagnosticBag
    bindings: Dict[int, Symbol]
    node_scopes: Dict[int, Scope]

    def symbol_for(self, node: object) -> Optional[Symbol]:
        return self.bindings.get(id(node))

    def scope_for(self, node: object) -> Optional[Scope]:
        return self.node_scopes.get(id(node))


class SemanticAnalyzer:
    """Builds scopes and symbols, resolves names, and records references.

    Type compatibility rules are intentionally left for ``typekon.py``.  This
    module only performs the name- and scope-related part of Phase Two.
    """

    ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%="}
    CALLABLE_KINDS = {
        SymbolKind.FUNCTION,
        SymbolKind.METHOD,
        SymbolKind.CONSTRUCTOR,
    }

    def __init__(self, filename: str = "<input>") -> None:
        self.filename = filename or "<input>"
        self.global_scope = Scope("global", ScopeKind.GLOBAL)
        self.diagnostics = DiagnosticBag()
        self._bindings: Dict[int, Symbol] = {}
        self._node_scopes: Dict[int, Scope] = {}
        self._declaration_symbols: Dict[int, Symbol] = {}
        self._struct_scopes: Dict[str, Scope] = {}
        self._uninitialized_reports: Set[Tuple[str, int, int]] = set()
        self._loop_depth = 0

    @property
    def bindings(self) -> Dict[int, Symbol]:
        return dict(self._bindings)

    @property
    def node_scopes(self) -> Dict[int, Scope]:
        return dict(self._node_scopes)

    def analyze(self, program: Program) -> SemanticAnalysisResult:
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")

        self._reset()
        self._record_scope(program, self.global_scope)

        # Pass 1: collect every top-level declaration first.  This permits
        # recursion and forward calls without changing the existing parser.
        for declaration in program.declarations:
            self._collect_global_declaration(declaration)

        # Pass 2: resolve global initializers and function bodies.
        for declaration in program.declarations:
            self._resolve_global_declaration(declaration)

        self._report_unused_symbols(self.global_scope)

        return SemanticAnalysisResult(
            global_scope=self.global_scope,
            diagnostics=self.diagnostics,
            bindings=dict(self._bindings),
            node_scopes=dict(self._node_scopes),
        )

    def _reset(self) -> None:
        self.global_scope = Scope("global", ScopeKind.GLOBAL)
        self.diagnostics = DiagnosticBag()
        self._bindings.clear()
        self._node_scopes.clear()
        self._declaration_symbols.clear()
        self._struct_scopes.clear()
        self._uninitialized_reports.clear()
        self._loop_depth = 0

    def _collect_global_declaration(self, node: object) -> None:
        if isinstance(node, FuncDecl):
            parameter_types = tuple(
                parameter.type_spec
                for parameter in node.params
                if isinstance(parameter, Param)
            )
            signature = FunctionSignature(parameter_types, node.return_type)
            symbol = Symbol(
                name=node.name,
                kind=SymbolKind.FUNCTION,
                type=node.return_type,
                definition_loc=self._location(node, len(node.name)),
                signature=signature,
                is_initialized=True,
            )
            self._declare(node, symbol, self.global_scope)
            return

        if isinstance(node, VarDecl):
            self._declare_variable(node, self.global_scope, is_global=True)
            return

        if isinstance(node, StructDecl):
            struct_symbol = Symbol(
                name=node.name,
                kind=SymbolKind.STRUCT,
                type=f"struct {node.name}",
                definition_loc=self._location(node, len(node.name)),
                is_initialized=True,
            )
            declared = self._declare(node, struct_symbol, self.global_scope)
            if declared is None:
                return

            struct_scope = self.global_scope.create_child(
                self._unique_child_name(self.global_scope, node.name),
                ScopeKind.STRUCT,
            )
            self._struct_scopes[node.name] = struct_scope
            for field_node in node.fields:
                if not isinstance(field_node, VarDecl):
                    continue
                field_symbol = Symbol(
                    name=field_node.name,
                    kind=SymbolKind.FIELD,
                    type=field_node.type_spec,
                    definition_loc=self._location(field_node, len(field_node.name)),
                    is_initialized=False,
                )
                self._declare(field_node, field_symbol, struct_scope)
            return

        self._record_scope(node, self.global_scope)

    def _resolve_global_declaration(self, node: object) -> None:
        if isinstance(node, FuncDecl):
            self._visit_function(node)
        elif isinstance(node, VarDecl):
            self._visit_global_variable(node)
        elif isinstance(node, StructDecl):
            self._record_scope(node, self.global_scope)

    def _visit_function(self, node: FuncDecl) -> None:
        function_scope = self.global_scope.create_child(
            self._unique_child_name(self.global_scope, node.name),
            ScopeKind.FUNCTION,
        )
        self._record_scope(node, self.global_scope)

        for parameter in node.params:
            if not isinstance(parameter, Param):
                continue
            parameter_symbol = Symbol(
                name=parameter.name,
                kind=SymbolKind.PARAMETER,
                type=parameter.type_spec,
                definition_loc=self._location(parameter, len(parameter.name)),
                is_initialized=True,
            )
            self._declare(parameter, parameter_symbol, function_scope)

        if isinstance(node.body, Block):
            self._visit_block(node.body, function_scope, f"{node.name}_body")

    def _visit_global_variable(self, node: VarDecl) -> None:
        self._record_scope(node, self.global_scope)
        symbol = self._declaration_symbols.get(id(node))
        if node.init_expr is not None:
            self._visit_expression(node.init_expr, self.global_scope)
            if symbol is not None:
                symbol.mark_initialized()

    def _visit_block(
        self,
        node: Block,
        parent_scope: Scope,
        preferred_name: Optional[str] = None,
    ) -> None:
        base_name = preferred_name or self._block_name(node)
        block_scope = parent_scope.create_child(
            self._unique_child_name(parent_scope, base_name),
            ScopeKind.BLOCK,
        )
        self._record_scope(node, block_scope)

        for statement in node.statements:
            self._visit_statement(statement, block_scope)

    def _visit_statement(self, node: object, scope: Scope) -> None:
        if node is None:
            return

        self._record_scope(node, scope)

        if isinstance(node, Block):
            self._visit_block(node, scope)
            return

        if isinstance(node, VarDecl):
            symbol = self._declare_variable(node, scope, is_global=False)
            if node.init_expr is not None:
                self._visit_expression(node.init_expr, scope)
                if symbol is not None:
                    symbol.mark_initialized()
            return

        if isinstance(node, IfStmt):
            self._visit_expression(node.condition, scope)
            self._visit_statement(node.then_stmt, scope)
            self._visit_statement(node.else_stmt, scope)
            return

        if isinstance(node, WhileStmt):
            self._visit_expression(node.condition, scope)
            self._loop_depth += 1
            try:
                self._visit_statement(node.body, scope)
            finally:
                self._loop_depth -= 1
            return

        if isinstance(node, ForStmt):
            self._visit_statement(node.init, scope)
            self._visit_statement(node.condition, scope)
            self._visit_expression(node.increment, scope)
            self._loop_depth += 1
            try:
                self._visit_statement(node.body, scope)
            finally:
                self._loop_depth -= 1
            return

        if isinstance(node, (BreakStmt, ContinueStmt)):
            if self._loop_depth <= 0:
                keyword = "break" if isinstance(node, BreakStmt) else "continue"
                self._error(
                    f"{keyword.capitalize()} statement outside a loop",
                    getattr(node, "loc", None),
                    len(keyword),
                )
            return

        if isinstance(node, ReturnStmt):
            self._visit_expression(node.value, scope)
            return

        if isinstance(node, ExprStmt):
            self._visit_expression(node.expr, scope)
            return

        self._visit_expression(node, scope)

    def _visit_expression(self, node: object, scope: Scope) -> None:
        if node is None:
            return

        self._record_scope(node, scope)

        if isinstance(node, Identifier):
            self._resolve_identifier(node, scope, ReferenceKind.READ)
            return

        if isinstance(node, CallExpr):
            symbol = self._resolve_name(
                name=node.callee,
                loc=getattr(node, "loc", None),
                length=max(1, len(node.callee)),
                scope=scope,
                reference_kind=ReferenceKind.CALL,
                binding_node=node,
            )
            if symbol is not None and symbol.kind not in self.CALLABLE_KINDS:
                self._error(
                    f"Symbol '{node.callee}' is not callable",
                    getattr(node, "loc", None),
                    max(1, len(node.callee)),
                )
            for argument in node.args:
                self._visit_expression(argument, scope)
            return

        if isinstance(node, BinaryExpr):
            if node.op in self.ASSIGNMENT_OPERATORS:
                self._visit_assignment(node, scope)
                return

            if node.op in {".", "->"}:
                # The right identifier is a member name, not a lexical-scope
                # name.  It will be resolved from its struct type later.
                self._visit_expression(node.left, scope)
                self._record_scope(node.right, scope)
                return

            self._visit_expression(node.left, scope)
            self._visit_expression(node.right, scope)
            return

        if isinstance(node, UnaryExpr):
            self._visit_expression(node.operand, scope)
            return

        if isinstance(
            node,
            (IntLiteral, FloatLiteral, StringLiteral, CharLiteral),
        ):
            return

    def _visit_assignment(self, node: BinaryExpr, scope: Scope) -> None:
        if not isinstance(node.left, Identifier):
            self._visit_expression(node.left, scope)
            self._visit_expression(node.right, scope)
            return

        if node.op == "=":
            self._visit_expression(node.right, scope)
            symbol = self._resolve_identifier(
                node.left,
                scope,
                ReferenceKind.WRITE,
                check_initialization=False,
            )
            if symbol is not None:
                symbol.mark_initialized()
            return

        symbol = self._resolve_identifier(
            node.left,
            scope,
            ReferenceKind.READ,
            check_initialization=True,
        )
        self._visit_expression(node.right, scope)
        if symbol is not None:
            symbol.add_reference(
                self._location(node.left, len(node.left.name)),
                ReferenceKind.WRITE,
            )
            symbol.mark_initialized()

    def _declare_variable(
        self,
        node: VarDecl,
        scope: Scope,
        is_global: bool,
    ) -> Optional[Symbol]:
        symbol = Symbol(
            name=node.name,
            kind=SymbolKind.VARIABLE,
            type=node.type_spec,
            definition_loc=self._location(node, len(node.name)),
            is_initialized=is_global,
        )
        return self._declare(node, symbol, scope)

    def _declare(
        self,
        node: object,
        symbol: Symbol,
        scope: Scope,
    ) -> Optional[Symbol]:
        self._record_scope(node, scope)
        result = scope.declare(symbol)

        if result.is_duplicate:
            original = result.duplicate
            self._error(
                (
                    f"Duplicate declaration of '{symbol.name}'; first declared "
                    f"at {original.definition_loc}"
                ),
                getattr(node, "loc", None),
                len(symbol.name),
            )
            return None

        self._declaration_symbols[id(node)] = symbol
        self._bind(node, symbol)

        if result.causes_shadowing:
            shadowed = result.shadowed
            self._warning(
                (
                    f"Declaration of '{symbol.name}' shadows outer declaration "
                    f"at {shadowed.definition_loc}"
                ),
                getattr(node, "loc", None),
                len(symbol.name),
            )

        return symbol

    def _resolve_identifier(
        self,
        node: Identifier,
        scope: Scope,
        reference_kind: ReferenceKind,
        check_initialization: bool = True,
    ) -> Optional[Symbol]:
        return self._resolve_name(
            name=node.name,
            loc=node.loc,
            length=max(1, len(node.name)),
            scope=scope,
            reference_kind=reference_kind,
            binding_node=node,
            check_initialization=check_initialization,
        )

    def _resolve_name(
        self,
        name: str,
        loc: Optional[Tuple[int, int]],
        length: int,
        scope: Scope,
        reference_kind: ReferenceKind,
        binding_node: object,
        check_initialization: bool = True,
    ) -> Optional[Symbol]:
        symbol = scope.resolve(name)
        if symbol is None:
            self._error(f"Undefined symbol '{name}'", loc, length)
            return None

        location = self._source_location(loc, length)
        symbol.add_reference(location, reference_kind)
        self._bind(binding_node, symbol)

        if (
            check_initialization
            and reference_kind == ReferenceKind.READ
            and symbol.kind in {SymbolKind.VARIABLE, SymbolKind.PARAMETER}
            and not symbol.is_initialized
        ):
            key = (symbol.name, location.line, location.column)
            if key not in self._uninitialized_reports:
                self._uninitialized_reports.add(key)
                self._warning(
                    f"Variable '{symbol.name}' may be used before initialization",
                    loc,
                    length,
                )

        return symbol

    def _report_unused_symbols(self, scope: Scope) -> None:
        for symbol in scope:
            if (
                symbol.kind in {SymbolKind.VARIABLE, SymbolKind.PARAMETER}
                and not symbol.is_used
            ):
                loc = symbol.definition_loc
                self.diagnostics.info(
                    f"Unused {symbol.kind.value} '{symbol.name}'",
                    file=loc.file,
                    line=loc.line,
                    column=loc.column,
                    length=loc.length,
                )

        for child in scope.children:
            self._report_unused_symbols(child)

    def _record_scope(self, node: object, scope: Scope) -> None:
        if node is None:
            return
        self._node_scopes[id(node)] = scope
        try:
            setattr(node, "semantic_scope", scope)
        except (AttributeError, TypeError):
            pass

    def _bind(self, node: object, symbol: Symbol) -> None:
        self._bindings[id(node)] = symbol
        try:
            setattr(node, "resolved_symbol", symbol)
        except (AttributeError, TypeError):
            pass

    def _location(self, node: object, length: int = 1) -> SourceLocation:
        return self._source_location(getattr(node, "loc", None), length)

    def _source_location(
        self,
        loc: Optional[Tuple[int, int]],
        length: int = 1,
    ) -> SourceLocation:
        line, column = loc if loc is not None else (1, 1)
        return SourceLocation(
            file=self.filename,
            line=max(1, line),
            column=max(1, column),
            length=max(1, length),
        )

    def _error(
        self,
        message: str,
        loc: Optional[Tuple[int, int]],
        length: int = 1,
    ) -> None:
        location = self._source_location(loc, length)
        self.diagnostics.error(
            message,
            file=location.file,
            line=location.line,
            column=location.column,
            length=location.length,
        )

    def _warning(
        self,
        message: str,
        loc: Optional[Tuple[int, int]],
        length: int = 1,
    ) -> None:
        location = self._source_location(loc, length)
        self.diagnostics.warning(
            message,
            file=location.file,
            line=location.line,
            column=location.column,
            length=location.length,
        )

    @staticmethod
    def _block_name(node: Block) -> str:
        if node.loc is None:
            return "block"
        return f"block@{node.loc[0]}:{node.loc[1]}"

    @staticmethod
    def _unique_child_name(parent: Scope, base_name: str) -> str:
        existing_names = {child.name for child in parent.children}
        if base_name not in existing_names:
            return base_name

        suffix = 2
        while f"{base_name}#{suffix}" in existing_names:
            suffix += 1
        return f"{base_name}#{suffix}"
