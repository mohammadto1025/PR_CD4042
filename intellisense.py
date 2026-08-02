from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lex import Lexer
from pars import (
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
from scope import Scope, ScopeKind
from semantic import SemanticAnalysisResult
from symbol import SourceLocation, Symbol, SymbolKind
from typer import TypeCheckResult


CursorPosition = Tuple[int, int]


class CompletionContextKind(str, Enum):
    GENERAL = "general"
    MEMBER_DOT = "member_dot"
    MEMBER_ARROW = "member_arrow"
    SCOPE_RESOLUTION = "scope_resolution"
    ARGUMENT = "argument"


@dataclass(frozen=True)
class CompletionContext:
    kind: CompletionContextKind
    prefix: str = ""
    receiver: Optional[str] = None
    callee: Optional[str] = None
    argument_index: Optional[int] = None
    expected_type: Optional[str] = None


@dataclass(frozen=True)
class CompletionItem:
    label: str
    kind: str
    detail: str
    sort_order: int
    definition_loc: SourceLocation
    insert_text: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "detail": self.detail,
            "sortOrder": self.sort_order,
            "insertText": self.insert_text or self.label,
            "definition": self.definition_loc.to_dict(),
        }

    def __str__(self) -> str:
        return (
            f"{self.label} | {self.kind} | {self.detail} "
            f"| sortOrder={self.sort_order}"
        )


@dataclass(frozen=True)
class HoverInfo:
    label: str
    kind: str
    detail: str
    scope: str
    definition_loc: SourceLocation
    reference_count: int
    documentation: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "detail": self.detail,
            "scope": self.scope,
            "definition": self.definition_loc.to_dict(),
            "referenceCount": self.reference_count,
            "documentation": self.documentation,
        }

    def __str__(self) -> str:
        return (
            f"{self.label}\n"
            f"kind: {self.kind}\n"
            f"detail: {self.detail}\n"
            f"scope: {self.scope}\n"
            f"defined at: {self.definition_loc}\n"
            f"references: {self.reference_count}"
            + (
                f"\ndocumentation: {self.documentation}"
                if self.documentation
                else ""
            )
        )


class IntelliSenseEngine:
    """Provides scope-aware completion and hover information.

    Cursor positions are 1-based. ``column`` represents the insertion point:
    column 1 is before the first character, and column N is before character N.
    The engine consumes the AST and semantic/type results without modifying any
    Phase One file.
    """

    IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    PREFIX_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)$")

    def __init__(
        self,
        source: str,
        program: Program,
        semantic_result: SemanticAnalysisResult,
        type_result: Optional[TypeCheckResult] = None,
        filename: str = "<input>",
    ) -> None:
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")
        if not isinstance(semantic_result, SemanticAnalysisResult):
            raise TypeError(
                "semantic_result must be an instance of SemanticAnalysisResult"
            )
        if type_result is not None and not isinstance(type_result, TypeCheckResult):
            raise TypeError("type_result must be an instance of TypeCheckResult")

        self.source = source
        self.program = program
        self.semantic_result = semantic_result
        self.type_result = type_result
        self.filename = filename or "<input>"
        self._lines = source.splitlines()
        self._tokens = Lexer(
            source,
            filename=self.filename,
        ).tokenize()
        self._block_ranges = self._build_block_ranges()

    def completion_context(self, line: int, column: int) -> CompletionContext:
        self._validate_cursor(line, column)
        before_cursor = self._source_before_cursor(line, column)

        prefix_match = self.PREFIX_RE.search(before_cursor)
        prefix = prefix_match.group(1) if prefix_match else ""
        context_text = (
            before_cursor[: prefix_match.start()]
            if prefix_match is not None
            else before_cursor
        )
        context_text = context_text.rstrip()

        for operator, kind in (
            ("->", CompletionContextKind.MEMBER_ARROW),
            (".", CompletionContextKind.MEMBER_DOT),
            ("::", CompletionContextKind.SCOPE_RESOLUTION),
        ):
            if context_text.endswith(operator):
                receiver_text = context_text[: -len(operator)].rstrip()
                receiver_match = self.PREFIX_RE.search(receiver_text)
                return CompletionContext(
                    kind=kind,
                    prefix=prefix,
                    receiver=(
                        receiver_match.group(1)
                        if receiver_match is not None
                        else None
                    ),
                )

        scope = self.scope_at(line, column)
        call_context = self._active_call_context(before_cursor, scope)
        if call_context is not None:
            callee, argument_index, expected_type = call_context
            return CompletionContext(
                kind=CompletionContextKind.ARGUMENT,
                prefix=prefix,
                callee=callee,
                argument_index=argument_index,
                expected_type=expected_type,
            )

        return CompletionContext(
            kind=CompletionContextKind.GENERAL,
            prefix=prefix,
        )

    def complete(
        self,
        line: int,
        column: int,
        limit: int = 50,
    ) -> List[CompletionItem]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        context = self.completion_context(line, column)
        scope = self.scope_at(line, column)

        if context.kind in {
            CompletionContextKind.MEMBER_DOT,
            CompletionContextKind.MEMBER_ARROW,
            CompletionContextKind.SCOPE_RESOLUTION,
        }:
            candidates = self._member_candidates(context, scope)
        else:
            candidates = scope.visible_symbols()

        items: List[CompletionItem] = []
        for symbol in candidates:
            prefix_score = self._prefix_score(symbol.name, context.prefix)
            if prefix_score is None:
                continue

            score = prefix_score
            score += self._scope_distance(scope, symbol) * 3
            score += self._kind_priority(symbol.kind)

            if (
                context.expected_type is not None
                and self._same_type(symbol.type, context.expected_type)
            ):
                score = max(0, score - 15)

            items.append(
                CompletionItem(
                    label=symbol.name,
                    kind=symbol.kind.value,
                    detail=symbol.detail,
                    sort_order=score,
                    definition_loc=symbol.definition_loc,
                    insert_text=symbol.name,
                )
            )

        items.sort(
            key=lambda item: (
                item.sort_order,
                item.label.lower(),
                item.kind,
            )
        )
        return items[:limit]

    def hover(self, line: int, column: int) -> Optional[HoverInfo]:
        self._validate_cursor(line, column)
        word = self._word_at(line, column)
        if word is None:
            return None

        name, _, _ = word
        scope = self.scope_at(line, column)
        symbol = scope.resolve(name)
        if symbol is None:
            return None

        symbol_scope = getattr(symbol, "scope", None)
        scope_name = (
            symbol_scope.full_name
            if isinstance(symbol_scope, Scope)
            else "unknown"
        )

        return HoverInfo(
            label=symbol.name,
            kind=symbol.kind.value,
            detail=symbol.detail,
            scope=scope_name,
            definition_loc=symbol.definition_loc,
            reference_count=len(symbol.references),
            documentation=self._documentation_for_symbol(symbol),
        )


    def _documentation_for_symbol(
        self,
        symbol: Symbol,
    ) -> Optional[str]:
        definition = symbol.definition_loc
        preceding = [
            token
            for token in self._tokens
            if (token.line, token.column)
            < (definition.line, definition.column)
        ]
        if not preceding:
            return None

        index = len(preceding) - 1
        while index >= 0:
            token = preceding[index]
            if token.type == "WHITESPACE":
                index -= 1
                continue
            # Definition locations point at the symbol name, so declaration
            # keywords and pointer stars on the same line may appear between
            # the documentation comment and the symbol.
            if token.line == definition.line and token.type != "COMMENT":
                index -= 1
                continue
            break
        if index < 0 or preceding[index].type != "COMMENT":
            return None

        nearest = preceding[index]
        nearest_end_line = (
            nearest.line
            + nearest.lexeme.count("\n")
        )
        if definition.line - nearest_end_line > 2:
            return None

        comments = []
        while index >= 0:
            token = preceding[index]
            if token.type == "WHITESPACE":
                index -= 1
                continue
            if token.type != "COMMENT":
                break
            if not token.lexeme.lstrip().startswith(
                ("/**", "/*!", "///", "//!")
            ):
                break
            comments.append(token.lexeme)
            index -= 1

        if not comments:
            return None

        comments.reverse()
        cleaned = [
            self._clean_documentation_comment(comment)
            for comment in comments
        ]
        text = " ".join(
            part for part in cleaned if part
        ).strip()
        return text or None

    @staticmethod
    def _clean_documentation_comment(
        comment: str,
    ) -> str:
        text = comment.strip()
        if text.startswith(("///", "//!")):
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line.startswith(("///", "//!")):
                    line = line[3:]
                lines.append(line.strip())
            return " ".join(lines).strip()

        if text.startswith(("/**", "/*!")):
            text = text[3:]
            if text.endswith("*/"):
                text = text[:-2]
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("*"):
                    line = line[1:]
                lines.append(line.strip())
            return " ".join(lines).strip()

        return ""

    def scope_at(self, line: int, column: int) -> Scope:
        self._validate_cursor(line, column)
        cursor = (line, column)
        global_scope = self.semantic_result.global_scope
        declaration = self._declaration_at(cursor)

        if declaration is None:
            return global_scope

        if isinstance(declaration, FuncDecl) and isinstance(declaration.body, Block):
            body_range = self._range_for_block(declaration.body)
            if body_range is not None:
                body_start, body_end = body_range
                if cursor > body_end:
                    return global_scope
                if cursor < body_start:
                    return self._best_scope_before_cursor(
                        [declaration, *declaration.params],
                        cursor,
                        global_scope,
                    )

        if isinstance(declaration, StructDecl):
            struct_range = self._range_for_struct(declaration)
            if struct_range is not None and cursor > struct_range[1]:
                return global_scope

        nodes = list(self._walk_for_cursor(declaration, cursor))
        return self._best_scope_before_cursor(nodes, cursor, global_scope)

    def _best_scope_before_cursor(
        self,
        nodes: Iterable[object],
        cursor: CursorPosition,
        fallback: Scope,
    ) -> Scope:
        best_scope = fallback
        best_score = (-1, -1, -1)

        for node in nodes:
            location = getattr(node, "loc", None)
            if not self._valid_loc(location) or location > cursor:
                continue

            if isinstance(node, Block):
                block_range = self._range_for_block(node)
                if block_range is not None and not self._inside(cursor, block_range):
                    continue

            scope = self.semantic_result.scope_for(node)
            if scope is None:
                continue

            score = (scope.depth, location[0], location[1])
            if score > best_score:
                best_score = score
                best_scope = scope

        return best_scope

    def _declaration_at(self, cursor: CursorPosition) -> Optional[object]:
        declarations = [
            declaration
            for declaration in self.program.declarations
            if self._valid_loc(getattr(declaration, "loc", None))
        ]
        declarations.sort(key=lambda declaration: declaration.loc)

        selected = None
        for declaration in declarations:
            if declaration.loc <= cursor:
                selected = declaration
            else:
                break
        return selected

    def _member_candidates(
        self,
        context: CompletionContext,
        scope: Scope,
    ) -> Sequence[Symbol]:
        if not context.receiver:
            return ()

        struct_name: Optional[str] = None
        receiver_symbol = scope.resolve(context.receiver)

        if context.kind == CompletionContextKind.SCOPE_RESOLUTION:
            if receiver_symbol is not None and receiver_symbol.kind == SymbolKind.STRUCT:
                struct_name = self._struct_name(receiver_symbol.type)
            else:
                struct_name = context.receiver
        else:
            if receiver_symbol is None:
                return ()

            receiver_type = receiver_symbol.type
            is_pointer = self._is_pointer(receiver_type)

            if (
                context.kind == CompletionContextKind.MEMBER_ARROW
                and not is_pointer
            ):
                return ()
            if (
                context.kind == CompletionContextKind.MEMBER_DOT
                and is_pointer
            ):
                return ()

            struct_name = self._struct_name(receiver_type)

        if not struct_name:
            return ()

        struct_scope = self._find_struct_scope(struct_name)
        if struct_scope is None:
            return ()
        return list(struct_scope.symbols.values())

    def _find_struct_scope(self, struct_name: str) -> Optional[Scope]:
        for child in self.semantic_result.global_scope.children:
            if child.kind != ScopeKind.STRUCT:
                continue
            base_name = child.name.split("#", 1)[0]
            if base_name == struct_name:
                return child
        return None

    def _active_call_context(
        self,
        before_cursor: str,
        scope: Scope,
    ) -> Optional[Tuple[str, int, Optional[str]]]:
        open_index = self._nearest_unclosed_parenthesis(before_cursor)
        if open_index is None:
            return None

        callee_text = before_cursor[:open_index].rstrip()
        callee_match = self.PREFIX_RE.search(callee_text)
        if callee_match is None:
            return None

        callee = callee_match.group(1)
        argument_text = before_cursor[open_index + 1 :]
        argument_index = self._top_level_comma_count(argument_text)

        symbol = scope.resolve(callee)
        expected_type = None
        if symbol is not None and symbol.signature is not None:
            parameter_types = symbol.signature.parameter_types
            if argument_index < len(parameter_types):
                expected_type = parameter_types[argument_index]

        return callee, argument_index, expected_type

    def _nearest_unclosed_parenthesis(self, text: str) -> Optional[int]:
        stack: List[int] = []
        quote: Optional[str] = None
        escaped = False
        in_line_comment = False
        in_block_comment = False
        index = 0

        while index < len(text):
            char = text[index]
            next_char = text[index + 1] if index + 1 < len(text) else ""

            if in_line_comment:
                if char == "\n":
                    in_line_comment = False
                index += 1
                continue

            if in_block_comment:
                if char == "*" and next_char == "/":
                    in_block_comment = False
                    index += 2
                else:
                    index += 1
                continue

            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                index += 1
                continue

            if char in {'"', "'"}:
                quote = char
                index += 1
                continue
            if char == "/" and next_char == "/":
                in_line_comment = True
                index += 2
                continue
            if char == "/" and next_char == "*":
                in_block_comment = True
                index += 2
                continue
            if char == "(":
                stack.append(index)
            elif char == ")" and stack:
                stack.pop()

            index += 1

        return stack[-1] if stack else None

    def _top_level_comma_count(self, text: str) -> int:
        parentheses = 0
        brackets = 0
        braces = 0
        count = 0
        quote: Optional[str] = None
        escaped = False

        for char in text:
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue

            if char in {'"', "'"}:
                quote = char
            elif char == "(":
                parentheses += 1
            elif char == ")":
                parentheses = max(0, parentheses - 1)
            elif char == "[":
                brackets += 1
            elif char == "]":
                brackets = max(0, brackets - 1)
            elif char == "{":
                braces += 1
            elif char == "}":
                braces = max(0, braces - 1)
            elif (
                char == ","
                and parentheses == 0
                and brackets == 0
                and braces == 0
            ):
                count += 1

        return count

    def _prefix_score(self, label: str, prefix: str) -> Optional[int]:
        if not prefix:
            return 20

        label_lower = label.lower()
        prefix_lower = prefix.lower()

        if label_lower == prefix_lower:
            return 0
        if label_lower.startswith(prefix_lower):
            return 5 + max(0, len(label) - len(prefix))

        subsequence_score = self._subsequence_score(label_lower, prefix_lower)
        if subsequence_score is not None:
            return 35 + subsequence_score

        ratio = SequenceMatcher(None, prefix_lower, label_lower).ratio()
        if ratio >= 0.55:
            return 80 + int((1.0 - ratio) * 20)

        return None

    def _subsequence_score(
        self,
        label: str,
        prefix: str,
    ) -> Optional[int]:
        position = -1
        gaps = 0
        for character in prefix:
            next_position = label.find(character, position + 1)
            if next_position < 0:
                return None
            if position >= 0:
                gaps += next_position - position - 1
            position = next_position
        return gaps

    def _scope_distance(self, current_scope: Scope, symbol: Symbol) -> int:
        symbol_scope = getattr(symbol, "scope", None)
        if not isinstance(symbol_scope, Scope):
            return 0
        return max(0, current_scope.depth - symbol_scope.depth)

    def _kind_priority(self, kind: SymbolKind) -> int:
        priorities = {
            SymbolKind.PARAMETER: 0,
            SymbolKind.VARIABLE: 1,
            SymbolKind.FIELD: 1,
            SymbolKind.FUNCTION: 4,
            SymbolKind.METHOD: 4,
            SymbolKind.CONSTRUCTOR: 4,
            SymbolKind.STRUCT: 8,
            SymbolKind.TYPE: 8,
            SymbolKind.CLASS: 8,
        }
        return priorities.get(kind, 10)

    def _same_type(self, first: str, second: str) -> bool:
        return self._normalize_type(first) == self._normalize_type(second)

    def _normalize_type(self, type_name: str) -> str:
        return "".join(str(type_name).split())

    def _is_pointer(self, type_name: str) -> bool:
        return self._normalize_type(type_name).endswith("*")

    def _struct_name(self, type_name: str) -> Optional[str]:
        normalized = " ".join(str(type_name).strip().split())
        normalized = normalized.rstrip("*").strip()
        if normalized.startswith("struct "):
            name = normalized[len("struct ") :].strip()
            return name or None
        return None

    def _word_at(
        self,
        line: int,
        column: int,
    ) -> Optional[Tuple[str, int, int]]:
        text = self._line_text(line)
        if not text:
            return None

        index = min(max(column - 1, 0), len(text))
        if index == len(text) or not self._is_identifier_char(text[index]):
            index -= 1
        if index < 0 or not self._is_identifier_char(text[index]):
            return None

        start = index
        while start > 0 and self._is_identifier_char(text[start - 1]):
            start -= 1

        end = index + 1
        while end < len(text) and self._is_identifier_char(text[end]):
            end += 1

        word = text[start:end]
        if not self.IDENTIFIER_RE.fullmatch(word):
            return None
        return word, start + 1, end + 1

    def _is_identifier_char(self, character: str) -> bool:
        return character == "_" or character.isalnum()

    def _source_before_cursor(self, line: int, column: int) -> str:
        current_line = self._line_text(line)
        current_prefix = current_line[: max(0, column - 1)]
        previous_lines = self._lines[: line - 1]
        if previous_lines:
            return "\n".join([*previous_lines, current_prefix])
        return current_prefix

    def _line_text(self, line: int) -> str:
        if 1 <= line <= len(self._lines):
            return self._lines[line - 1]
        return ""

    def _validate_cursor(self, line: int, column: int) -> None:
        if line < 1:
            raise ValueError("line must be at least 1")
        if column < 1:
            raise ValueError("column must be at least 1")
        if line > max(1, len(self._lines)):
            raise ValueError("line is outside the source file")

        text = self._line_text(line)
        if column > len(text) + 1:
            raise ValueError("column is outside the source line")

    def _build_block_ranges(
        self,
    ) -> Dict[CursorPosition, Tuple[CursorPosition, CursorPosition]]:
        stack: List[CursorPosition] = []
        ranges: Dict[
            CursorPosition,
            Tuple[CursorPosition, CursorPosition],
        ] = {}

        for token in Lexer(self.source).tokenize():
            if token.type != "DELIMITER":
                continue
            location = (token.line, token.column)
            if token.lexeme == "{":
                stack.append(location)
            elif token.lexeme == "}" and stack:
                opening = stack.pop()
                ranges[opening] = (opening, location)

        return ranges

    def _range_for_block(
        self,
        block: Block,
    ) -> Optional[Tuple[CursorPosition, CursorPosition]]:
        location = getattr(block, "loc", None)
        if not self._valid_loc(location):
            return None
        return self._block_ranges.get(location)

    def _range_for_struct(
        self,
        node: StructDecl,
    ) -> Optional[Tuple[CursorPosition, CursorPosition]]:
        node_loc = getattr(node, "loc", None)
        if not self._valid_loc(node_loc):
            return None

        possible_ranges = [
            block_range
            for opening, block_range in self._block_ranges.items()
            if opening >= node_loc
        ]
        if not possible_ranges:
            return None
        return min(possible_ranges, key=lambda item: item[0])

    def _inside(
        self,
        cursor: CursorPosition,
        block_range: Tuple[CursorPosition, CursorPosition],
    ) -> bool:
        start, end = block_range
        return start <= cursor <= end

    def _valid_loc(self, location: object) -> bool:
        return (
            isinstance(location, tuple)
            and len(location) == 2
            and isinstance(location[0], int)
            and isinstance(location[1], int)
            and location[0] >= 1
            and location[1] >= 1
        )

    def _walk_for_cursor(
        self,
        node: object,
        cursor: CursorPosition,
    ) -> Iterable[object]:
        if node is None:
            return

        if isinstance(node, Block):
            block_range = self._range_for_block(node)
            if block_range is not None and not self._inside(cursor, block_range):
                return

        yield node

        if isinstance(node, Program):
            children = node.declarations
        elif isinstance(node, FuncDecl):
            children = [*node.params, node.body]
        elif isinstance(node, StructDecl):
            children = node.fields
        elif isinstance(node, Block):
            children = node.statements
        elif isinstance(node, IfStmt):
            children = [node.condition, node.then_stmt, node.else_stmt]
        elif isinstance(node, WhileStmt):
            children = [node.condition, node.body]
        elif isinstance(node, ForStmt):
            children = [
                node.init,
                node.condition,
                node.increment,
                node.body,
            ]
        elif isinstance(node, ReturnStmt):
            children = [node.value]
        elif isinstance(node, ExprStmt):
            children = [node.expr]
        elif isinstance(node, VarDecl):
            children = [node.init_expr]
        elif isinstance(node, BinaryExpr):
            children = [node.left, node.right]
        elif isinstance(node, UnaryExpr):
            children = [node.operand]
        elif isinstance(node, CallExpr):
            children = node.args
        else:
            children = []

        for child in children:
            if child is not None:
                yield from self._walk_for_cursor(child, cursor)

    def _walk(self, node: object) -> Iterable[object]:
        if node is None:
            return

        yield node

        if isinstance(node, Program):
            children = node.declarations
        elif isinstance(node, FuncDecl):
            children = [*node.params, node.body]
        elif isinstance(node, StructDecl):
            children = node.fields
        elif isinstance(node, Block):
            children = node.statements
        elif isinstance(node, IfStmt):
            children = [node.condition, node.then_stmt, node.else_stmt]
        elif isinstance(node, WhileStmt):
            children = [node.condition, node.body]
        elif isinstance(node, ForStmt):
            children = [
                node.init,
                node.condition,
                node.increment,
                node.body,
            ]
        elif isinstance(node, ReturnStmt):
            children = [node.value]
        elif isinstance(node, ExprStmt):
            children = [node.expr]
        elif isinstance(node, VarDecl):
            children = [node.init_expr]
        elif isinstance(node, BinaryExpr):
            children = [node.left, node.right]
        elif isinstance(node, UnaryExpr):
            children = [node.operand]
        elif isinstance(node, CallExpr):
            children = node.args
        elif isinstance(node, Param):
            children = []
        elif isinstance(node, Identifier):
            children = []
        else:
            children = []

        for child in children:
            if child is not None:
                yield from self._walk(child)
