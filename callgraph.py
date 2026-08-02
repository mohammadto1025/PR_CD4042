from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from pars import (
    ASTNode,
    BinaryExpr,
    Block,
    CallExpr,
    ExprStmt,
    ForStmt,
    FuncDecl,
    IfStmt,
    Program,
    ReturnStmt,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from semantic import SemanticAnalysisResult
from symbol import SourceLocation, SymbolKind


CALLABLE_KINDS = {
    SymbolKind.FUNCTION,
    SymbolKind.METHOD,
    SymbolKind.CONSTRUCTOR,
}


@dataclass(frozen=True)
class CallGraphNode:
    name: str
    definition_loc: SourceLocation

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "definition": self.definition_loc.to_dict(),
        }


@dataclass(frozen=True)
class CallSite:
    caller: str
    callee: str
    location: SourceLocation
    resolved: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "location": self.location.to_dict(),
            "resolved": self.resolved,
        }

    def __str__(self) -> str:
        status = "resolved" if self.resolved else "unresolved"
        return (
            f"{self.caller} -> {self.callee} "
            f"at {self.location} [{status}]"
        )


class CallGraph:
    """Program-wide directed graph of resolved direct function calls."""

    def __init__(
        self,
        nodes: Optional[Iterable[CallGraphNode]] = None,
        entry_function: str = "main",
    ) -> None:
        self.entry_function = entry_function
        self.nodes: Dict[str, CallGraphNode] = {}
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse: Dict[str, Set[str]] = {}
        self.call_sites: List[CallSite] = []

        for node in nodes or ():
            self.add_node(node)

    def add_node(self, node: CallGraphNode) -> None:
        if not isinstance(node, CallGraphNode):
            raise TypeError("node must be a CallGraphNode")
        if node.name in self.nodes:
            raise ValueError(
                f"Duplicate call-graph function: {node.name}"
            )

        self.nodes[node.name] = node
        self._adjacency[node.name] = set()
        self._reverse[node.name] = set()

    def add_call_site(self, site: CallSite) -> None:
        if not isinstance(site, CallSite):
            raise TypeError("site must be a CallSite")
        if site.caller not in self.nodes:
            raise KeyError(f"Unknown caller: {site.caller}")

        if site not in self.call_sites:
            self.call_sites.append(site)

        if not site.resolved:
            return
        if site.callee not in self.nodes:
            raise KeyError(f"Unknown resolved callee: {site.callee}")

        self._adjacency[site.caller].add(site.callee)
        self._reverse[site.callee].add(site.caller)

    @property
    def unresolved_call_sites(self) -> List[CallSite]:
        return [
            site for site in self.call_sites
            if not site.resolved
        ]

    def has_function(self, name: str) -> bool:
        return name in self.nodes

    def direct_callees(self, function_name: str) -> List[str]:
        self._require_function(function_name)
        return sorted(self._adjacency[function_name])

    def direct_callers(self, function_name: str) -> List[str]:
        self._require_function(function_name)
        return sorted(self._reverse[function_name])

    def call_sites_from(self, function_name: str) -> List[CallSite]:
        self._require_function(function_name)
        return sorted(
            [
                site for site in self.call_sites
                if site.caller == function_name
            ],
            key=self._site_sort_key,
        )

    def call_sites_to(self, function_name: str) -> List[CallSite]:
        self._require_function(function_name)
        return sorted(
            [
                site for site in self.call_sites
                if site.resolved and site.callee == function_name
            ],
            key=self._site_sort_key,
        )

    def reachable_callees(self, function_name: str) -> List[str]:
        """Return all transitively reachable callees, excluding the start."""
        self._require_function(function_name)
        visited = self._walk(function_name, self._adjacency)
        visited.discard(function_name)
        return sorted(visited)

    def reaching_callers(self, function_name: str) -> List[str]:
        """Return all functions that can transitively reach the target."""
        self._require_function(function_name)
        visited = self._walk(function_name, self._reverse)
        visited.discard(function_name)
        return sorted(visited)

    def strongly_connected_components(self) -> List[List[str]]:
        """Tarjan SCC algorithm with deterministic output."""
        index = 0
        indices: Dict[str, int] = {}
        lowlinks: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        components: List[List[str]] = []

        def strong_connect(node: str) -> None:
            nonlocal index

            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for successor in sorted(self._adjacency[node]):
                if successor not in indices:
                    strong_connect(successor)
                    lowlinks[node] = min(
                        lowlinks[node],
                        lowlinks[successor],
                    )
                elif successor in on_stack:
                    lowlinks[node] = min(
                        lowlinks[node],
                        indices[successor],
                    )

            if lowlinks[node] != indices[node]:
                return

            component: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break

            components.append(sorted(component))

        for node in sorted(self.nodes):
            if node not in indices:
                strong_connect(node)

        return sorted(
            components,
            key=lambda component: (
                component[0] if component else "",
                len(component),
            ),
        )

    def recursive_functions(self) -> List[str]:
        recursive: Set[str] = set()

        for component in self.strongly_connected_components():
            if len(component) > 1:
                recursive.update(component)
                continue

            function_name = component[0]
            if function_name in self._adjacency[function_name]:
                recursive.add(function_name)

        return sorted(recursive)

    def dead_functions(
        self,
        entry_function: Optional[str] = None,
    ) -> List[str]:
        """Return functions not reachable from the selected entry.

        If the entry function does not exist, no function is reachable, so
        every defined function is returned.
        """
        entry = entry_function or self.entry_function

        if entry not in self.nodes:
            return sorted(self.nodes)

        reachable = self._walk(entry, self._adjacency)
        return sorted(set(self.nodes) - reachable)

    def edge_count(self) -> int:
        return sum(
            len(callees)
            for callees in self._adjacency.values()
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_function": self.entry_function,
            "nodes": [
                self.nodes[name].to_dict()
                for name in sorted(self.nodes)
            ],
            "edges": [
                {"caller": caller, "callee": callee}
                for caller in sorted(self.nodes)
                for callee in sorted(self._adjacency[caller])
            ],
            "call_sites": [
                site.to_dict()
                for site in sorted(
                    self.call_sites,
                    key=self._site_sort_key,
                )
            ],
            "recursive_functions": self.recursive_functions(),
            "dead_functions": self.dead_functions(),
            "strongly_connected_components": (
                self.strongly_connected_components()
            ),
        }

    def to_dot(self) -> str:
        dead = set(self.dead_functions())
        recursive = set(self.recursive_functions())

        lines = [
            'digraph "callgraph" {',
            "  rankdir=LR;",
            '  node [shape=box, fontname="Courier"];',
        ]

        for name in sorted(self.nodes):
            attributes: List[str] = [f'label="{name}"']

            if name == self.entry_function:
                attributes.append("shape=oval")
            if name in recursive:
                attributes.append('color="purple"')
                attributes.append("penwidth=2")
            if name in dead:
                attributes.append('style="dashed"')
                attributes.append('color="gray"')

            lines.append(
                f'  "{name}" [{", ".join(attributes)}];'
            )

        for caller in sorted(self.nodes):
            for callee in sorted(self._adjacency[caller]):
                lines.append(f'  "{caller}" -> "{callee}";')

        lines.append("}")
        return "\n".join(lines)

    def format(self) -> str:
        lines = [
            "Program Call Graph",
            f"Functions: {len(self.nodes)}",
            f"Edges: {self.edge_count()}",
            f"Entry: {self.entry_function}",
        ]

        for function_name in sorted(self.nodes):
            callees = self.direct_callees(function_name)
            target_text = ", ".join(callees) if callees else "(none)"
            lines.append(f"\n{function_name} -> {target_text}")

        recursive = self.recursive_functions()
        lines.append(
            "\nRecursive functions: "
            + (", ".join(recursive) if recursive else "None")
        )

        dead = self.dead_functions()
        lines.append(
            "Dead functions: "
            + (", ".join(dead) if dead else "None")
        )

        components = [
            component
            for component in self.strongly_connected_components()
            if (
                len(component) > 1
                or component[0] in self._adjacency[component[0]]
            )
        ]
        lines.append("Recursive SCCs:")
        if not components:
            lines.append("  None")
        else:
            for component in components:
                lines.append("  {" + ", ".join(component) + "}")

        unresolved = self.unresolved_call_sites
        lines.append("Unresolved calls:")
        if not unresolved:
            lines.append("  None")
        else:
            for site in sorted(unresolved, key=self._site_sort_key):
                lines.append(f"  {site}")

        return "\n".join(lines)

    def _walk(
        self,
        start: str,
        adjacency: Mapping[str, Set[str]],
    ) -> Set[str]:
        visited: Set[str] = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            stack.extend(
                reversed(sorted(adjacency.get(node, set())))
            )

        return visited

    def _require_function(self, function_name: str) -> None:
        if function_name not in self.nodes:
            raise KeyError(
                f"Unknown call-graph function: {function_name}"
            )

    @staticmethod
    def _site_sort_key(
        site: CallSite,
    ) -> Tuple[str, int, int, str]:
        return (
            site.caller,
            site.location.line,
            site.location.column,
            site.callee,
        )

    def __str__(self) -> str:
        return self.format()


class CallGraphBuilder:
    """Construct a direct static call graph from the AST and Symbol Table."""

    def __init__(
        self,
        filename: str = "<input>",
        entry_function: str = "main",
    ) -> None:
        self.filename = filename or "<input>"
        self.entry_function = entry_function or "main"

    def build(
        self,
        program: Program,
        semantic_result: Optional[
            SemanticAnalysisResult
        ] = None,
    ) -> CallGraph:
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")
        if (
            semantic_result is not None
            and not isinstance(
                semantic_result,
                SemanticAnalysisResult,
            )
        ):
            raise TypeError(
                "semantic_result must be a SemanticAnalysisResult"
            )

        graph = CallGraph(
            entry_function=self.entry_function,
        )

        functions: Dict[str, FuncDecl] = {}

        for declaration in program.declarations:
            if not isinstance(declaration, FuncDecl):
                continue

            if declaration.name in functions:
                # Semantic analysis reports duplicate definitions. The call
                # graph keeps the first valid node rather than crashing.
                continue

            functions[declaration.name] = declaration
            graph.add_node(
                CallGraphNode(
                    name=declaration.name,
                    definition_loc=SourceLocation.from_ast_loc(
                        declaration.loc,
                        file=self.filename,
                        length=max(1, len(declaration.name)),
                    ),
                )
            )

        for caller_name, function in functions.items():
            self._visit(
                function.body,
                caller_name,
                graph,
                semantic_result,
            )

        return graph

    analyze = build

    def _visit(
        self,
        node: Optional[ASTNode],
        caller_name: str,
        graph: CallGraph,
        semantic_result: Optional[SemanticAnalysisResult],
    ) -> None:
        if node is None:
            return

        if isinstance(node, CallExpr):
            resolved_name = self._resolved_callee_name(
                node,
                graph,
                semantic_result,
            )
            graph.add_call_site(
                CallSite(
                    caller=caller_name,
                    callee=resolved_name or node.callee,
                    location=SourceLocation.from_ast_loc(
                        node.loc,
                        file=self.filename,
                        length=max(1, len(node.callee)),
                    ),
                    resolved=resolved_name is not None,
                )
            )

            for argument in node.args:
                self._visit(
                    argument,
                    caller_name,
                    graph,
                    semantic_result,
                )
            return

        if isinstance(node, Block):
            for statement in node.statements:
                self._visit(
                    statement,
                    caller_name,
                    graph,
                    semantic_result,
                )
            return

        if isinstance(node, VarDecl):
            self._visit(
                node.init_expr,
                caller_name,
                graph,
                semantic_result,
            )
            return

        if isinstance(node, IfStmt):
            self._visit(
                node.condition,
                caller_name,
                graph,
                semantic_result,
            )
            self._visit(
                node.then_stmt,
                caller_name,
                graph,
                semantic_result,
            )
            self._visit(
                node.else_stmt,
                caller_name,
                graph,
                semantic_result,
            )
            return

        if isinstance(node, WhileStmt):
            self._visit(
                node.condition,
                caller_name,
                graph,
                semantic_result,
            )
            self._visit(
                node.body,
                caller_name,
                graph,
                semantic_result,
            )
            return

        if isinstance(node, ForStmt):
            self._visit(
                node.init,
                caller_name,
                graph,
                semantic_result,
            )
            self._visit(
                node.condition,
                caller_name,
                graph,
                semantic_result,
            )
            self._visit(
                node.increment,
                caller_name,
                graph,
                semantic_result,
            )
            self._visit(
                node.body,
                caller_name,
                graph,
                semantic_result,
            )
            return

        if isinstance(node, ReturnStmt):
            self._visit(
                node.value,
                caller_name,
                graph,
                semantic_result,
            )
            return

        if isinstance(node, ExprStmt):
            self._visit(
                node.expr,
                caller_name,
                graph,
                semantic_result,
            )
            return

        if isinstance(node, BinaryExpr):
            self._visit(
                node.left,
                caller_name,
                graph,
                semantic_result,
            )
            self._visit(
                node.right,
                caller_name,
                graph,
                semantic_result,
            )
            return

        if isinstance(node, UnaryExpr):
            self._visit(
                node.operand,
                caller_name,
                graph,
                semantic_result,
            )

    def _resolved_callee_name(
        self,
        call: CallExpr,
        graph: CallGraph,
        semantic_result: Optional[SemanticAnalysisResult],
    ) -> Optional[str]:
        if semantic_result is not None:
            symbol = semantic_result.symbol_for(call)
            if (
                symbol is None
                or symbol.kind not in CALLABLE_KINDS
                or symbol.name not in graph.nodes
            ):
                return None
            return symbol.name

        # Fallback for clients that have not run Phase Two. This resolves only
        # direct calls whose names exactly match program function definitions.
        if call.callee in graph.nodes:
            return call.callee
        return None
