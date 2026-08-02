from __future__ import annotations

from dataclasses import dataclass, field, replace
import difflib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from lex import Lexer
from navigation import NavigationEngine, NavigationOccurrence
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
    Parser,
    Program,
    ReturnStmt,
    StructDecl,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from scope import Scope
from semantic import SemanticAnalysisResult, SemanticAnalyzer
from symbol import SourceLocation, Symbol


C_KEYWORDS = {
    "if", "else", "while", "for", "return",
    "int", "float", "char", "void", "double",
    "struct", "break", "continue", "sizeof",
    "typedef", "enum", "union", "switch", "case",
    "default", "do", "goto", "const", "static",
    "extern", "register", "volatile", "signed",
    "unsigned", "short", "long",
}

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class TextEdit:
    location: SourceLocation
    old_text: str
    new_text: str
    role: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "location": self.location.to_dict(),
            "old_text": self.old_text,
            "new_text": self.new_text,
            "role": self.role,
        }

    def __str__(self) -> str:
        return (
            f"{self.location}: {self.role} "
            f"'{self.old_text}' -> '{self.new_text}'"
        )


@dataclass(frozen=True)
class RenameConflict:
    code: str
    message: str
    location: Optional[SourceLocation] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "location": (
                self.location.to_dict()
                if self.location is not None
                else None
            ),
        }

    def __str__(self) -> str:
        if self.location is None:
            return f"[{self.code}] {self.message}"
        return f"[{self.code}] {self.location} {self.message}"


@dataclass(frozen=True)
class RenamePlan:
    old_name: Optional[str]
    new_name: str
    symbol_kind: Optional[str]
    definition: Optional[SourceLocation]
    edits: Tuple[TextEdit, ...] = ()
    conflicts: Tuple[RenameConflict, ...] = ()
    updated_source: Optional[str] = None
    unified_diff: Optional[str] = None

    @property
    def can_apply(self) -> bool:
        return (
            self.old_name is not None
            and not self.conflicts
        )

    @property
    def applied(self) -> bool:
        return self.updated_source is not None

    def to_dict(self) -> Dict[str, object]:
        return {
            "old_name": self.old_name,
            "new_name": self.new_name,
            "symbol_kind": self.symbol_kind,
            "definition": (
                self.definition.to_dict()
                if self.definition is not None
                else None
            ),
            "can_apply": self.can_apply,
            "applied": self.applied,
            "edits": [edit.to_dict() for edit in self.edits],
            "conflicts": [
                conflict.to_dict()
                for conflict in self.conflicts
            ],
            "updated_source": self.updated_source,
            "unified_diff": self.unified_diff,
        }

    def format(self) -> str:
        lines = [
            "Safe Rename",
            f"Old name: {self.old_name or '(none)'}",
            f"New name: {self.new_name}",
            f"Kind: {self.symbol_kind or '(none)'}",
            "Definition: "
            + (
                str(self.definition)
                if self.definition is not None
                else "(none)"
            ),
        ]

        if self.conflicts:
            lines.append("\nConflicts:")
            for conflict in self.conflicts:
                lines.append(f"  {conflict}")
        else:
            lines.append(
                f"\nEdits: {len(self.edits)}"
            )
            for edit in self.edits:
                lines.append(f"  {edit}")

        if self.unified_diff:
            lines.append("\nUnified Diff:")
            lines.append(self.unified_diff)

        lines.append(
            "\nStatus: "
            + (
                "applied"
                if self.applied
                else (
                    "ready"
                    if self.can_apply
                    else "rejected"
                )
            )
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


class RenameEngine:
    """Scope-aware, preview-first safe rename refactoring."""

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
        self.navigation = NavigationEngine(
            source=source,
            program=program,
            semantic_result=semantic_result,
            filename=self.filename,
        )
        self._line_starts = self._calculate_line_starts(source)

    def preview(
        self,
        line: int,
        column: int,
        new_name: str,
    ) -> RenamePlan:
        symbol = self.navigation.symbol_at(line, column)

        if symbol is None:
            return RenamePlan(
                old_name=None,
                new_name=new_name,
                symbol_kind=None,
                definition=None,
                conflicts=(
                    RenameConflict(
                        code="NO_SYMBOL",
                        message="No renameable symbol exists at the cursor.",
                    ),
                ),
            )

        conflicts: List[RenameConflict] = []
        conflicts.extend(
            self._validate_new_name(symbol, new_name)
        )

        if not conflicts and new_name != symbol.name:
            conflicts.extend(
                self._scope_conflicts(symbol, new_name)
            )
            conflicts.extend(
                self._reference_capture_conflicts(
                    symbol,
                    new_name,
                )
            )

        edits: List[TextEdit] = []
        if not conflicts and new_name != symbol.name:
            edits, edit_conflicts = self._build_edits(
                symbol,
                new_name,
            )
            conflicts.extend(edit_conflicts)

        conflicts = self._deduplicate_conflicts(conflicts)

        return RenamePlan(
            old_name=symbol.name,
            new_name=new_name,
            symbol_kind=symbol.kind.value,
            definition=symbol.definition_loc,
            edits=tuple(edits) if not conflicts else (),
            conflicts=tuple(conflicts),
        )

    preview_rename = preview

    def rename(
        self,
        line: int,
        column: int,
        new_name: str,
        verify: bool = True,
    ) -> RenamePlan:
        plan = self.preview(line, column, new_name)

        if not plan.can_apply:
            return plan

        if plan.old_name == new_name:
            return replace(
                plan,
                updated_source=self.source,
                unified_diff="",
            )

        updated_source = self.apply_edits(plan.edits)

        if verify:
            verification_conflicts = self._verify_source(
                updated_source
            )
            if verification_conflicts:
                return replace(
                    plan,
                    edits=(),
                    conflicts=tuple(
                        verification_conflicts
                    ),
                )

        return replace(
            plan,
            updated_source=updated_source,
            unified_diff=self._make_unified_diff(
                updated_source
            ),
        )

    apply = rename


    def _make_unified_diff(
        self,
        updated_source: str,
    ) -> str:
        original_lines = self.source.splitlines(
            keepends=True
        )
        updated_lines = updated_source.splitlines(
            keepends=True
        )
        diff = difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=self.filename,
            tofile=f"{self.filename}.renamed",
            lineterm="",
        )
        return "\n".join(
            line.rstrip("\n")
            for line in diff
        )

    def apply_edits(
        self,
        edits: Sequence[TextEdit],
    ) -> str:
        indexed: List[Tuple[int, int, TextEdit]] = []

        for edit in edits:
            start = self._offset_for(edit.location)
            end = start + len(edit.old_text)

            if self.source[start:end] != edit.old_text:
                raise ValueError(
                    "Text edit is stale at "
                    f"{edit.location}: expected "
                    f"'{edit.old_text}'"
                )

            indexed.append((start, end, edit))

        indexed.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        updated = self.source
        previous_start = len(self.source) + 1

        for start, end, edit in indexed:
            if end > previous_start:
                raise ValueError("Rename edits overlap")

            updated = (
                updated[:start]
                + edit.new_text
                + updated[end:]
            )
            previous_start = start

        return updated

    def _validate_new_name(
        self,
        symbol: Symbol,
        new_name: str,
    ) -> List[RenameConflict]:
        conflicts: List[RenameConflict] = []

        if not isinstance(new_name, str):
            return [
                RenameConflict(
                    code="INVALID_NAME",
                    message="The new name must be a string.",
                    location=symbol.definition_loc,
                )
            ]

        if not IDENTIFIER_PATTERN.fullmatch(new_name):
            conflicts.append(
                RenameConflict(
                    code="INVALID_IDENTIFIER",
                    message=(
                        f"'{new_name}' is not a valid C identifier."
                    ),
                    location=symbol.definition_loc,
                )
            )

        if new_name in C_KEYWORDS:
            conflicts.append(
                RenameConflict(
                    code="KEYWORD",
                    message=(
                        f"'{new_name}' is a reserved C keyword."
                    ),
                    location=symbol.definition_loc,
                )
            )

        return conflicts

    def _scope_conflicts(
        self,
        symbol: Symbol,
        new_name: str,
    ) -> List[RenameConflict]:
        conflicts: List[RenameConflict] = []
        scope = symbol.scope

        if not isinstance(scope, Scope):
            return conflicts

        local = scope.resolve_local(new_name)
        if local is not None and local is not symbol:
            conflicts.append(
                RenameConflict(
                    code="DUPLICATE_IN_SCOPE",
                    message=(
                        f"Renaming '{symbol.name}' to '{new_name}' "
                        f"would collide with {local.kind.value} "
                        f"'{local.name}' in scope '{scope.full_name}'."
                    ),
                    location=local.definition_loc,
                )
            )

        if scope.parent is not None:
            outer = scope.parent.resolve(new_name)
            if outer is not None and outer is not symbol:
                conflicts.append(
                    RenameConflict(
                        code="SHADOWS_OUTER_SYMBOL",
                        message=(
                            f"Renaming '{symbol.name}' to '{new_name}' "
                            f"would shadow {outer.kind.value} "
                            f"'{outer.name}'."
                        ),
                        location=outer.definition_loc,
                    )
                )

        for descendant in self._descendant_scopes(scope):
            inner = descendant.resolve_local(new_name)
            if inner is None or inner is symbol:
                continue

            conflicts.append(
                RenameConflict(
                    code="SHADOWED_BY_INNER_SYMBOL",
                    message=(
                        f"Renaming '{symbol.name}' to '{new_name}' "
                        f"would create shadowing with "
                        f"{inner.kind.value} '{inner.name}' in "
                        f"scope '{descendant.full_name}'."
                    ),
                    location=inner.definition_loc,
                )
            )

        return conflicts

    def _reference_capture_conflicts(
        self,
        symbol: Symbol,
        new_name: str,
    ) -> List[RenameConflict]:
        conflicts: List[RenameConflict] = []

        for node in self._bound_reference_nodes(symbol):
            scope = self.semantic_result.scope_for(node)
            if not isinstance(scope, Scope):
                continue

            visible = scope.resolve(new_name)
            if visible is None or visible is symbol:
                continue

            location = self._node_reference_location(
                node,
                symbol.name,
            )
            conflicts.append(
                RenameConflict(
                    code="REFERENCE_CAPTURE",
                    message=(
                        f"The reference to '{symbol.name}' would "
                        f"resolve to {visible.kind.value} "
                        f"'{visible.name}' after the rename."
                    ),
                    location=location,
                )
            )

        return conflicts

    def _build_edits(
        self,
        symbol: Symbol,
        new_name: str,
    ) -> Tuple[List[TextEdit], List[RenameConflict]]:
        occurrences = self.navigation.references_for_symbol(
            symbol,
            include_definition=True,
        )

        edits: List[TextEdit] = []
        conflicts: List[RenameConflict] = []
        seen_locations: Set[Tuple[str, int, int, int]] = set()

        for occurrence in occurrences:
            location = occurrence.location

            if location.file != self.filename:
                conflicts.append(
                    RenameConflict(
                        code="MULTI_FILE_UNSUPPORTED",
                        message=(
                            "This rename engine currently applies "
                            "edits to one source file at a time."
                        ),
                        location=location,
                    )
                )
                continue

            key = (
                location.file,
                location.line,
                location.column,
                location.length,
            )
            if key in seen_locations:
                continue
            seen_locations.add(key)

            actual_text = self._text_at(location)
            if actual_text != symbol.name:
                conflicts.append(
                    RenameConflict(
                        code="STALE_LOCATION",
                        message=(
                            f"Expected '{symbol.name}' but found "
                            f"'{actual_text}'."
                        ),
                        location=location,
                    )
                )
                continue

            edits.append(
                TextEdit(
                    location=location,
                    old_text=symbol.name,
                    new_text=new_name,
                    role=occurrence.role,
                )
            )

        edits.sort(
            key=lambda edit: (
                edit.location.file,
                edit.location.line,
                edit.location.column,
            )
        )
        return edits, conflicts

    def _verify_source(
        self,
        updated_source: str,
    ) -> List[RenameConflict]:
        tokens = Lexer(
            updated_source,
            filename=self.filename,
        ).tokenize()
        parser = Parser(tokens)
        updated_program = parser.parse_program()

        if parser.errors:
            return [
                RenameConflict(
                    code="PARSER_VALIDATION_FAILED",
                    message=str(error),
                )
                for error in parser.errors
            ]

        updated_semantic = SemanticAnalyzer(
            self.filename
        ).analyze(updated_program)

        old_error_count = len(
            self.semantic_result.diagnostics.errors
        )
        new_errors = updated_semantic.diagnostics.errors

        if len(new_errors) <= old_error_count:
            return []

        return [
            RenameConflict(
                code="SEMANTIC_VALIDATION_FAILED",
                message=str(diagnostic),
            )
            for diagnostic in new_errors
        ]

    def _bound_reference_nodes(
        self,
        target: Symbol,
    ) -> List[ASTNode]:
        results: List[ASTNode] = []

        def visit(node: Optional[ASTNode]) -> None:
            if node is None:
                return

            if isinstance(node, Program):
                for declaration in node.declarations:
                    visit(declaration)
                return

            if isinstance(node, FuncDecl):
                for parameter in node.params:
                    visit(parameter)
                visit(node.body)
                return

            if isinstance(node, (Param, StructDecl)):
                if isinstance(node, StructDecl):
                    for field in node.fields:
                        visit(field)
                return

            if isinstance(node, VarDecl):
                visit(node.init_expr)
                return

            if isinstance(node, Block):
                for statement in node.statements:
                    visit(statement)
                return

            if isinstance(node, IfStmt):
                visit(node.condition)
                visit(node.then_stmt)
                visit(node.else_stmt)
                return

            if isinstance(node, WhileStmt):
                visit(node.condition)
                visit(node.body)
                return

            if isinstance(node, ForStmt):
                visit(node.init)
                visit(node.condition)
                visit(node.increment)
                visit(node.body)
                return

            if isinstance(node, ReturnStmt):
                visit(node.value)
                return

            if isinstance(node, ExprStmt):
                visit(node.expr)
                return

            if isinstance(node, CallExpr):
                if self.semantic_result.symbol_for(node) is target:
                    results.append(node)
                for argument in node.args:
                    visit(argument)
                return

            if isinstance(node, Identifier):
                if self.semantic_result.symbol_for(node) is target:
                    results.append(node)
                return

            if isinstance(node, BinaryExpr):
                visit(node.left)
                visit(node.right)
                return

            if isinstance(node, UnaryExpr):
                visit(node.operand)

        visit(self.program)
        return results

    def _node_reference_location(
        self,
        node: ASTNode,
        old_name: str,
    ) -> SourceLocation:
        if isinstance(node, CallExpr):
            occurrence = next(
                (
                    item
                    for item in self.navigation.occurrences
                    if (
                        not item.is_definition
                        and item._symbol
                        is self.semantic_result.symbol_for(node)
                        and item.location.line
                        == node.loc[0]
                        and item.symbol_name == old_name
                    )
                ),
                None,
            )
            if occurrence is not None:
                return occurrence.location

        line, column = (
            node.loc if node.loc is not None else (1, 1)
        )
        return SourceLocation(
            file=self.filename,
            line=line,
            column=column,
            length=max(1, len(old_name)),
        )

    def _descendant_scopes(
        self,
        scope: Scope,
    ) -> Iterable[Scope]:
        stack = list(reversed(scope.children))

        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    def _text_at(
        self,
        location: SourceLocation,
    ) -> str:
        start = self._offset_for(location)
        end = start + location.length
        return self.source[start:end]

    def _offset_for(
        self,
        location: SourceLocation,
    ) -> int:
        if location.line < 1 or location.line > len(self._line_starts):
            raise ValueError(
                f"Location line is outside source: {location}"
            )

        line_start = self._line_starts[location.line - 1]
        offset = line_start + location.column - 1

        if offset < 0 or offset > len(self.source):
            raise ValueError(
                f"Location column is outside source: {location}"
            )
        return offset

    @staticmethod
    def _calculate_line_starts(
        source: str,
    ) -> List[int]:
        starts = [0]
        for index, character in enumerate(source):
            if character == "\n":
                starts.append(index + 1)
        return starts

    @staticmethod
    def _deduplicate_conflicts(
        conflicts: Sequence[RenameConflict],
    ) -> List[RenameConflict]:
        unique: Dict[
            Tuple[str, str, Optional[str], Optional[int], Optional[int]],
            RenameConflict,
        ] = {}

        for conflict in conflicts:
            location = conflict.location
            key = (
                conflict.code,
                conflict.message,
                location.file if location else None,
                location.line if location else None,
                location.column if location else None,
            )
            unique[key] = conflict

        return list(unique.values())
