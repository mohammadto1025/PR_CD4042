from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pars import (
    ASTNode,
    BinaryExpr,
    Block,
    CallExpr,
    ExprStmt,
    ForStmt,
    FuncDecl,
    Identifier,
    IfStmt,
    Param,
    Program,
    ReturnStmt,
    StructDecl,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from semantic import SemanticAnalysisResult
from symbol import (
    ReferenceKind,
    SourceLocation,
    Symbol,
)


DECLARATION_NODES = (
    FuncDecl,
    Param,
    VarDecl,
    StructDecl,
)


@dataclass(frozen=True)
class NavigationTarget:
    symbol: str
    kind: str
    detail: str
    scope: str
    defined_at: SourceLocation
    reference_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "kind": self.kind,
            "type": self.detail,
            "scope": self.scope,
            "defined_at": self.defined_at.to_dict(),
            "reference_count": self.reference_count,
        }

    def __str__(self) -> str:
        return (
            f"{self.symbol}\n"
            f"kind: {self.kind}\n"
            f"detail: {self.detail}\n"
            f"scope: {self.scope}\n"
            f"defined at: {self.defined_at}\n"
            f"references: {self.reference_count}"
        )


@dataclass(frozen=True)
class NavigationOccurrence:
    symbol_name: str
    symbol_kind: str
    location: SourceLocation
    is_definition: bool
    reference_kind: Optional[ReferenceKind]
    _symbol: Symbol

    @property
    def role(self) -> str:
        if self.is_definition:
            return "definition"
        if self.reference_kind is None:
            return "reference"
        return self.reference_kind.value

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol_name,
            "kind": self.symbol_kind,
            "role": self.role,
            "location": self.location.to_dict(),
        }

    def __str__(self) -> str:
        return (
            f"{self.role} of '{self.symbol_name}' "
            f"at {self.location}"
        )


class NavigationEngine:
    """Scope-aware Go-to-Definition and Find-All-References index.

    Cursor positions are one-based. The index uses semantic bindings, not
    text matching, so shadowed symbols with the same spelling stay separate.
    """

    ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%="}

    def __init__(
        self,
        source: str,
        program: Program,
        semantic_result: SemanticAnalysisResult,
        filename: str = "<input>",
    ) -> None:
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")
        if not isinstance(
            semantic_result,
            SemanticAnalysisResult,
        ):
            raise TypeError(
                "semantic_result must be a SemanticAnalysisResult"
            )

        self.source = source
        self.program = program
        self.semantic_result = semantic_result
        self.filename = filename or "<input>"
        self._lines = source.splitlines()
        self._occurrences: List[NavigationOccurrence] = []

        self._build_index()

    @property
    def occurrences(self) -> List[NavigationOccurrence]:
        return list(self._occurrences)

    def symbol_at(
        self,
        line: int,
        column: int,
        file: Optional[str] = None,
    ) -> Optional[Symbol]:
        occurrence = self.occurrence_at(line, column, file)
        return occurrence._symbol if occurrence is not None else None

    def occurrence_at(
        self,
        line: int,
        column: int,
        file: Optional[str] = None,
    ) -> Optional[NavigationOccurrence]:
        target_file = file or self.filename
        self._validate_cursor(line, column)

        matches = [
            occurrence
            for occurrence in self._occurrences
            if (
                occurrence.location.file == target_file
                and occurrence.location.line == line
                and self._contains_column(
                    occurrence.location,
                    column,
                )
            )
        ]

        if not matches:
            return None

        matches.sort(
            key=lambda occurrence: (
                occurrence.location.length,
                0 if occurrence.is_definition else 1,
                occurrence.location.column,
            )
        )
        return matches[0]

    def goto_definition(
        self,
        line: int,
        column: int,
        file: Optional[str] = None,
    ) -> Optional[NavigationTarget]:
        symbol = self.symbol_at(line, column, file)
        if symbol is None:
            return None
        return self._target_for_symbol(symbol)

    go_to_definition = goto_definition

    def find_references(
        self,
        line: int,
        column: int,
        include_definition: bool = True,
        file: Optional[str] = None,
    ) -> List[NavigationOccurrence]:
        symbol = self.symbol_at(line, column, file)
        if symbol is None:
            return []
        return self.references_for_symbol(
            symbol,
            include_definition=include_definition,
        )

    find_all_references = find_references

    def references_for_symbol(
        self,
        symbol: Symbol,
        include_definition: bool = True,
    ) -> List[NavigationOccurrence]:
        if not isinstance(symbol, Symbol):
            raise TypeError("symbol must be an instance of Symbol")

        results = [
            occurrence
            for occurrence in self._occurrences
            if (
                occurrence._symbol is symbol
                and (
                    include_definition
                    or not occurrence.is_definition
                )
            )
        ]

        return sorted(
            results,
            key=self._occurrence_sort_key,
        )

    def definition_occurrence(
        self,
        line: int,
        column: int,
        file: Optional[str] = None,
    ) -> Optional[NavigationOccurrence]:
        symbol = self.symbol_at(line, column, file)
        if symbol is None:
            return None

        for occurrence in self.references_for_symbol(
            symbol,
            include_definition=True,
        ):
            if occurrence.is_definition:
                return occurrence
        return None

    def format_references(
        self,
        references: Sequence[NavigationOccurrence],
    ) -> str:
        if not references:
            return "No references found."
        return "\n".join(str(reference) for reference in references)

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.filename,
            "occurrences": [
                occurrence.to_dict()
                for occurrence in sorted(
                    self._occurrences,
                    key=self._occurrence_sort_key,
                )
            ],
        }

    def _build_index(self) -> None:
        self._occurrences.clear()
        self._visit(self.program)

        unique: Dict[
            Tuple[int, str, int, int, int, bool, str],
            NavigationOccurrence,
        ] = {}

        for occurrence in self._occurrences:
            reference_kind = (
                occurrence.reference_kind.value
                if occurrence.reference_kind is not None
                else ""
            )
            key = (
                id(occurrence._symbol),
                occurrence.location.file,
                occurrence.location.line,
                occurrence.location.column,
                occurrence.location.length,
                occurrence.is_definition,
                reference_kind,
            )
            unique[key] = occurrence

        self._occurrences = sorted(
            unique.values(),
            key=self._occurrence_sort_key,
        )

    def _visit(
        self,
        node: Optional[ASTNode],
        reference_kind: ReferenceKind = ReferenceKind.READ,
    ) -> None:
        if node is None:
            return

        if isinstance(node, Program):
            for declaration in node.declarations:
                self._visit(declaration)
            return

        if isinstance(node, FuncDecl):
            self._add_definition(node, node.name)
            for parameter in node.params:
                self._visit(parameter)
            self._visit(node.body)
            return

        if isinstance(node, Param):
            self._add_definition(node, node.name)
            return

        if isinstance(node, VarDecl):
            self._add_definition(node, node.name)
            self._visit(node.init_expr)
            return

        if isinstance(node, StructDecl):
            self._add_definition(node, node.name)
            for field in node.fields:
                self._visit(field)
            return

        if isinstance(node, Block):
            for statement in node.statements:
                self._visit(statement)
            return

        if isinstance(node, IfStmt):
            self._visit(node.condition)
            self._visit(node.then_stmt)
            self._visit(node.else_stmt)
            return

        if isinstance(node, WhileStmt):
            self._visit(node.condition)
            self._visit(node.body)
            return

        if isinstance(node, ForStmt):
            self._visit(node.init)
            self._visit(node.condition)
            self._visit(node.increment)
            self._visit(node.body)
            return

        if isinstance(node, ReturnStmt):
            self._visit(node.value)
            return

        if isinstance(node, ExprStmt):
            self._visit(node.expr)
            return

        if isinstance(node, CallExpr):
            symbol = self._symbol_for(node)
            if symbol is not None:
                self._add_occurrence(
                    symbol=symbol,
                    location=self._call_location(node),
                    is_definition=False,
                    reference_kind=ReferenceKind.CALL,
                )
            for argument in node.args:
                self._visit(argument)
            return

        if isinstance(node, Identifier):
            symbol = self._symbol_for(node)
            if symbol is not None:
                self._add_occurrence(
                    symbol=symbol,
                    location=self._node_location(
                        node,
                        len(node.name),
                    ),
                    is_definition=False,
                    reference_kind=reference_kind,
                )
            return

        if isinstance(node, BinaryExpr):
            if node.op in self.ASSIGNMENT_OPERATORS:
                self._visit(
                    node.left,
                    ReferenceKind.WRITE,
                )
                self._visit(
                    node.right,
                    ReferenceKind.READ,
                )
                return

            self._visit(node.left)
            self._visit(node.right)
            return

        if isinstance(node, UnaryExpr):
            self._visit(node.operand)

    def _add_definition(
        self,
        node: ASTNode,
        name: str,
    ) -> None:
        symbol = self._symbol_for(node)
        if symbol is None:
            return

        self._add_occurrence(
            symbol=symbol,
            location=symbol.definition_loc,
            is_definition=True,
            reference_kind=None,
        )

    def _add_occurrence(
        self,
        symbol: Symbol,
        location: SourceLocation,
        is_definition: bool,
        reference_kind: Optional[ReferenceKind],
    ) -> None:
        self._occurrences.append(
            NavigationOccurrence(
                symbol_name=symbol.name,
                symbol_kind=symbol.kind.value,
                location=location,
                is_definition=is_definition,
                reference_kind=reference_kind,
                _symbol=symbol,
            )
        )

    def _symbol_for(
        self,
        node: object,
    ) -> Optional[Symbol]:
        symbol = self.semantic_result.symbol_for(node)
        if symbol is not None:
            return symbol

        candidate = getattr(
            node,
            "resolved_symbol",
            None,
        )
        return candidate if isinstance(candidate, Symbol) else None

    def _node_location(
        self,
        node: ASTNode,
        length: int,
    ) -> SourceLocation:
        return SourceLocation.from_ast_loc(
            getattr(node, "loc", None),
            file=self.filename,
            length=max(1, length),
        )

    def _call_location(
        self,
        node: CallExpr,
    ) -> SourceLocation:
        line, parser_column = (
            node.loc if node.loc is not None else (1, 1)
        )
        fallback_column = max(
            1,
            parser_column - len(node.callee),
        )

        if 1 <= line <= len(self._lines):
            text = self._lines[line - 1]
            paren_index = max(
                0,
                min(len(text), parser_column - 1),
            )
            prefix = text[:paren_index]
            found = prefix.rfind(node.callee)

            if found >= 0:
                between = prefix[
                    found + len(node.callee):
                ]
                if between.strip() == "":
                    fallback_column = found + 1

        return SourceLocation(
            file=self.filename,
            line=max(1, line),
            column=fallback_column,
            length=max(1, len(node.callee)),
        )

    def _target_for_symbol(
        self,
        symbol: Symbol,
    ) -> NavigationTarget:
        scope_name = "global"
        if symbol.scope is not None:
            scope_name = getattr(
                symbol.scope,
                "full_name",
                getattr(symbol.scope, "name", "global"),
            )

        return NavigationTarget(
            symbol=symbol.name,
            kind=symbol.kind.value,
            detail=symbol.detail,
            scope=scope_name,
            defined_at=symbol.definition_loc,
            reference_count=len(
                self.references_for_symbol(
                    symbol,
                    include_definition=False,
                )
            ),
        )

    def _validate_cursor(
        self,
        line: int,
        column: int,
    ) -> None:
        if not isinstance(line, int) or not isinstance(column, int):
            raise TypeError("line and column must be integers")
        if line < 1 or line > len(self._lines):
            raise ValueError("line is outside the source file")

        max_column = len(self._lines[line - 1]) + 1
        if column < 1 or column > max_column:
            raise ValueError("column is outside the source line")

    @staticmethod
    def _contains_column(
        location: SourceLocation,
        column: int,
    ) -> bool:
        return (
            location.column
            <= column
            < location.column + location.length
        )

    @staticmethod
    def _occurrence_sort_key(
        occurrence: NavigationOccurrence,
    ) -> Tuple[str, int, int, int]:
        return (
            occurrence.location.file,
            occurrence.location.line,
            occurrence.location.column,
            0 if occurrence.is_definition else 1,
        )
