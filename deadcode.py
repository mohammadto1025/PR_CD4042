from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from callgraph import CallGraph, CallGraphBuilder
from cfg import ControlFlowGraph, CFGBuilder
from dataflow import (
    DataFlowAnalyzer,
    FunctionDataFlowResult,
)
from pars import ASTNode, FuncDecl, Program
from scope import Scope
from semantic import SemanticAnalysisResult
from symbol import (
    ReferenceKind,
    SourceLocation,
    Symbol,
    SymbolKind,
)


class DeadCodeKind(str, Enum):
    DEAD_FUNCTION = "dead-function"
    UNREACHABLE_CODE = "unreachable-code"
    UNUSED_VARIABLE = "unused-variable"
    UNUSED_PARAMETER = "unused-parameter"
    DEAD_STORE = "dead-store"


@dataclass(frozen=True)
class DeadCodeIssue:
    kind: DeadCodeKind
    message: str
    location: SourceLocation
    function_name: Optional[str] = None
    block_id: Optional[str] = None
    symbol_name: Optional[str] = None
    statement: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "location": self.location.to_dict(),
            "function": self.function_name,
            "block": self.block_id,
            "symbol": self.symbol_name,
            "statement": self.statement,
        }

    def __str__(self) -> str:
        context: List[str] = []

        if self.function_name is not None:
            context.append(f"function={self.function_name}")
        if self.block_id is not None:
            context.append(f"block={self.block_id}")

        suffix = (
            " [" + ", ".join(context) + "]"
            if context
            else ""
        )

        return (
            f"[{self.kind.value}] {self.location} "
            f"{self.message}{suffix}"
        )


@dataclass
class DeadCodeResult:
    issues: List[DeadCodeIssue]
    entry_function: str = "main"

    def by_kind(
        self,
        kind: DeadCodeKind,
    ) -> List[DeadCodeIssue]:
        return [
            issue
            for issue in self.issues
            if issue.kind == kind
        ]

    @property
    def dead_functions(self) -> List[DeadCodeIssue]:
        return self.by_kind(DeadCodeKind.DEAD_FUNCTION)

    @property
    def unreachable_code(self) -> List[DeadCodeIssue]:
        return self.by_kind(DeadCodeKind.UNREACHABLE_CODE)

    @property
    def unused_variables(self) -> List[DeadCodeIssue]:
        return self.by_kind(DeadCodeKind.UNUSED_VARIABLE)

    @property
    def unused_parameters(self) -> List[DeadCodeIssue]:
        return self.by_kind(DeadCodeKind.UNUSED_PARAMETER)

    @property
    def dead_stores(self) -> List[DeadCodeIssue]:
        return self.by_kind(DeadCodeKind.DEAD_STORE)

    def counts(self) -> Dict[str, int]:
        return {
            kind.value: len(self.by_kind(kind))
            for kind in DeadCodeKind
        }

    def summary(self) -> str:
        counts = self.counts()
        return "\n".join(
            [
                "Dead-Code Analysis Summary",
                f"Dead functions: "
                f"{counts[DeadCodeKind.DEAD_FUNCTION.value]}",
                f"Unreachable code: "
                f"{counts[DeadCodeKind.UNREACHABLE_CODE.value]}",
                f"Unused variables: "
                f"{counts[DeadCodeKind.UNUSED_VARIABLE.value]}",
                f"Unused parameters: "
                f"{counts[DeadCodeKind.UNUSED_PARAMETER.value]}",
                f"Dead stores: "
                f"{counts[DeadCodeKind.DEAD_STORE.value]}",
            ]
        )

    def format(self) -> str:
        lines = [self.summary()]

        for kind in DeadCodeKind:
            category = self.by_kind(kind)
            title = kind.value.replace("-", " ").title()
            lines.append(f"\n{title}:")

            if not category:
                lines.append("  None")
                continue

            for issue in category:
                lines.append(f"  {issue}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_function": self.entry_function,
            "counts": self.counts(),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }

    def __str__(self) -> str:
        return self.format()


class DeadCodeAnalyzer:
    """Combine CFG, data-flow, call-graph and symbols for dead-code checks."""

    def __init__(
        self,
        filename: str = "<input>",
        entry_function: str = "main",
    ) -> None:
        self.filename = filename or "<input>"
        self.entry_function = entry_function or "main"

    def analyze(
        self,
        program: Program,
        semantic_result: SemanticAnalysisResult,
        graphs: Optional[
            Mapping[str, ControlFlowGraph]
        ] = None,
        dataflow_results: Optional[
            Mapping[str, FunctionDataFlowResult]
        ] = None,
        call_graph: Optional[CallGraph] = None,
    ) -> DeadCodeResult:
        if not isinstance(program, Program):
            raise TypeError(
                "program must be an instance of Program"
            )
        if not isinstance(
            semantic_result,
            SemanticAnalysisResult,
        ):
            raise TypeError(
                "semantic_result must be a SemanticAnalysisResult"
            )

        graph_map = (
            dict(graphs)
            if graphs is not None
            else CFGBuilder().build(program)
        )

        flow_map = (
            dict(dataflow_results)
            if dataflow_results is not None
            else DataFlowAnalyzer(
                self.filename
            ).analyze(
                program,
                graph_map,
            )
        )

        resolved_call_graph = (
            call_graph
            if call_graph is not None
            else CallGraphBuilder(
                filename=self.filename,
                entry_function=self.entry_function,
            ).build(
                program,
                semantic_result,
            )
        )

        issues: List[DeadCodeIssue] = []
        issues.extend(
            self._find_dead_functions(
                resolved_call_graph
            )
        )
        issues.extend(
            self._find_unreachable_code(
                graph_map
            )
        )
        issues.extend(
            self._find_unused_symbols(
                semantic_result.global_scope
            )
        )
        issues.extend(
            self._find_dead_stores(
                graph_map,
                flow_map,
            )
        )

        issues = self._deduplicate_and_sort(issues)

        return DeadCodeResult(
            issues=issues,
            entry_function=self.entry_function,
        )

    run = analyze

    def _find_dead_functions(
        self,
        call_graph: CallGraph,
    ) -> List[DeadCodeIssue]:
        issues: List[DeadCodeIssue] = []

        for function_name in call_graph.dead_functions(
            self.entry_function
        ):
            node = call_graph.nodes[function_name]
            issues.append(
                DeadCodeIssue(
                    kind=DeadCodeKind.DEAD_FUNCTION,
                    message=(
                        f"Function '{function_name}' is not "
                        f"reachable from entry function "
                        f"'{self.entry_function}'."
                    ),
                    location=node.definition_loc,
                    function_name=function_name,
                    symbol_name=function_name,
                )
            )

        return issues

    def _find_unreachable_code(
        self,
        graphs: Mapping[str, ControlFlowGraph],
    ) -> List[DeadCodeIssue]:
        issues: List[DeadCodeIssue] = []

        for function_name, graph in graphs.items():
            for block_id in graph.unreachable_block_ids():
                block = graph.blocks[block_id]

                if block.nodes:
                    for index, node in enumerate(block.nodes):
                        statement = (
                            block.statements[index]
                            if index < len(block.statements)
                            else node.__class__.__name__
                        )
                        issues.append(
                            DeadCodeIssue(
                                kind=(
                                    DeadCodeKind.UNREACHABLE_CODE
                                ),
                                message=(
                                    "Statement is unreachable and "
                                    "can never execute."
                                ),
                                location=self._node_location(node),
                                function_name=function_name,
                                block_id=block_id,
                                statement=statement,
                            )
                        )
                    continue

                location = SourceLocation.from_ast_loc(
                    block.loc,
                    file=self.filename,
                )
                issues.append(
                    DeadCodeIssue(
                        kind=DeadCodeKind.UNREACHABLE_CODE,
                        message=(
                            "Basic block is unreachable and "
                            "can never execute."
                        ),
                        location=location,
                        function_name=function_name,
                        block_id=block_id,
                    )
                )

        return issues

    def _find_unused_symbols(
        self,
        global_scope: Scope,
    ) -> List[DeadCodeIssue]:
        issues: List[DeadCodeIssue] = []

        for symbol in self._walk_symbols(global_scope):
            if symbol.kind not in {
                SymbolKind.VARIABLE,
                SymbolKind.PARAMETER,
            }:
                continue

            if self._has_value_use(symbol):
                continue

            if symbol.kind == SymbolKind.PARAMETER:
                kind = DeadCodeKind.UNUSED_PARAMETER
                label = "Parameter"
            else:
                kind = DeadCodeKind.UNUSED_VARIABLE
                label = "Variable"

            issues.append(
                DeadCodeIssue(
                    kind=kind,
                    message=(
                        f"{label} '{symbol.name}' is never read."
                    ),
                    location=symbol.definition_loc,
                    function_name=self._owning_function(symbol),
                    symbol_name=symbol.name,
                )
            )

        return issues

    def _find_dead_stores(
        self,
        graphs: Mapping[str, ControlFlowGraph],
        flow_results: Mapping[
            str,
            FunctionDataFlowResult,
        ],
    ) -> List[DeadCodeIssue]:
        issues: List[DeadCodeIssue] = []

        for function_name, graph in graphs.items():
            flow = flow_results.get(function_name)
            if flow is None:
                continue

            reachable = set(graph.reachable_block_ids())

            for block_id in graph._ordered_block_ids():
                if block_id not in reachable:
                    continue
                if block_id in {
                    graph.entry_id,
                    graph.exit_id,
                }:
                    continue

                facts = flow.blocks[block_id]
                live = set(facts.live_out)

                for event in reversed(facts.events):
                    if event.kind == "use":
                        live.add(event.name)
                        continue

                    if event.kind != "def":
                        continue

                    if event.name not in live:
                        location = SourceLocation.from_ast_loc(
                            event.loc,
                            file=self.filename,
                            length=max(
                                1,
                                len(event.name),
                            ),
                        )
                        issues.append(
                            DeadCodeIssue(
                                kind=DeadCodeKind.DEAD_STORE,
                                message=(
                                    f"The value assigned to "
                                    f"'{event.name}' is never read "
                                    "before being overwritten or "
                                    "leaving the function."
                                ),
                                location=location,
                                function_name=function_name,
                                block_id=block_id,
                                symbol_name=event.name,
                            )
                        )

                    live.discard(event.name)

        return issues

    def _walk_symbols(
        self,
        scope: Scope,
    ) -> Iterable[Symbol]:
        for symbol in scope.symbols.values():
            yield symbol

        for child in scope.children:
            yield from self._walk_symbols(child)

    @staticmethod
    def _has_value_use(
        symbol: Symbol,
    ) -> bool:
        return any(
            reference.kind
            in {
                ReferenceKind.READ,
                ReferenceKind.CALL,
                ReferenceKind.TYPE_USE,
            }
            for reference in symbol.references
        )

    @staticmethod
    def _owning_function(
        symbol: Symbol,
    ) -> Optional[str]:
        scope = symbol.scope

        while isinstance(scope, Scope):
            if scope.kind.value == "function":
                return scope.name
            scope = scope.parent

        return None

    def _node_location(
        self,
        node: ASTNode,
    ) -> SourceLocation:
        return SourceLocation.from_ast_loc(
            getattr(node, "loc", None),
            file=self.filename,
        )

    @staticmethod
    def _deduplicate_and_sort(
        issues: Sequence[DeadCodeIssue],
    ) -> List[DeadCodeIssue]:
        unique: Dict[
            Tuple[
                str,
                str,
                int,
                int,
                Optional[str],
                Optional[str],
                Optional[str],
            ],
            DeadCodeIssue,
        ] = {}

        for issue in issues:
            key = (
                issue.kind.value,
                issue.location.file,
                issue.location.line,
                issue.location.column,
                issue.function_name,
                issue.block_id,
                issue.symbol_name,
            )
            unique[key] = issue

        return sorted(
            unique.values(),
            key=lambda issue: (
                issue.location.file,
                issue.location.line,
                issue.location.column,
                issue.kind.value,
            ),
        )
