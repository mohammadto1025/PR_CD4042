from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from callgraph import CallGraph
from cfg import CFGBuilder, ControlFlowGraph
from lex import Lexer
from pars import (
    ASTNode,
    BinaryExpr,
    Block,
    CallExpr,
    ExprStmt,
    ForStmt,
    FuncDecl,
    IfStmt,
    Param,
    Program,
    ReturnStmt,
    StructDecl,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)


SourceLocation = Tuple[int, int]


@dataclass(frozen=True)
class FunctionMetrics:
    function_name: str
    cyclomatic_complexity: int
    cfg_nodes: int
    cfg_edges: int
    connected_components: int
    nesting_depth: int
    parameter_count: int
    lines_of_code: int
    statement_count: int
    local_variable_count: int
    call_site_count: int
    caller_count: int
    callee_count: int
    high_complexity: bool

    @property
    def formula(self) -> str:
        return (
            f"M = E - N + 2P = "
            f"{self.cfg_edges} - {self.cfg_nodes} "
            f"+ 2({self.connected_components}) "
            f"= {self.cyclomatic_complexity}"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "function": self.function_name,
            "cyclomatic_complexity": (
                self.cyclomatic_complexity
            ),
            "cfg_nodes": self.cfg_nodes,
            "cfg_edges": self.cfg_edges,
            "connected_components": (
                self.connected_components
            ),
            "nesting_depth": self.nesting_depth,
            "parameter_count": self.parameter_count,
            "lines_of_code": self.lines_of_code,
            "statement_count": self.statement_count,
            "local_variable_count": (
                self.local_variable_count
            ),
            "call_site_count": self.call_site_count,
            "caller_count": self.caller_count,
            "callee_count": self.callee_count,
            "high_complexity": self.high_complexity,
        }


@dataclass
class CodeMetricsResult:
    functions: Dict[str, FunctionMetrics]
    complexity_threshold: int = 10

    def for_function(
        self,
        function_name: str,
    ) -> FunctionMetrics:
        try:
            return self.functions[function_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown function metrics: {function_name}"
            ) from error

    @property
    def total_functions(self) -> int:
        return len(self.functions)

    @property
    def total_lines_of_code(self) -> int:
        return sum(
            item.lines_of_code
            for item in self.functions.values()
        )

    @property
    def total_statements(self) -> int:
        return sum(
            item.statement_count
            for item in self.functions.values()
        )

    @property
    def average_complexity(self) -> float:
        if not self.functions:
            return 0.0
        return mean(
            item.cyclomatic_complexity
            for item in self.functions.values()
        )

    @property
    def maximum_complexity(self) -> int:
        if not self.functions:
            return 0
        return max(
            item.cyclomatic_complexity
            for item in self.functions.values()
        )

    @property
    def high_complexity_functions(
        self,
    ) -> List[FunctionMetrics]:
        return sorted(
            [
                item
                for item in self.functions.values()
                if item.high_complexity
            ],
            key=lambda item: (
                -item.cyclomatic_complexity,
                item.function_name,
            ),
        )

    def summary(self) -> str:
        return "\n".join(
            [
                "Code Metrics Summary",
                f"Functions: {self.total_functions}",
                f"Total function LOC: "
                f"{self.total_lines_of_code}",
                f"Total statements: "
                f"{self.total_statements}",
                f"Average cyclomatic complexity: "
                f"{self.average_complexity:.2f}",
                f"Maximum cyclomatic complexity: "
                f"{self.maximum_complexity}",
                f"High-complexity functions "
                f"(M > {self.complexity_threshold}): "
                f"{len(self.high_complexity_functions)}",
            ]
        )

    def format(self) -> str:
        lines = [self.summary()]

        if not self.functions:
            lines.append("\nNo functions found.")
            return "\n".join(lines)

        headers = [
            "Function",
            "M",
            "LOC",
            "Nest",
            "Params",
            "Stmt",
            "Locals",
            "Calls",
            "Fan-in",
            "Fan-out",
            "Status",
        ]

        rows: List[List[str]] = []
        for name in sorted(self.functions):
            item = self.functions[name]
            rows.append(
                [
                    item.function_name,
                    str(item.cyclomatic_complexity),
                    str(item.lines_of_code),
                    str(item.nesting_depth),
                    str(item.parameter_count),
                    str(item.statement_count),
                    str(item.local_variable_count),
                    str(item.call_site_count),
                    str(item.caller_count),
                    str(item.callee_count),
                    (
                        "HIGH"
                        if item.high_complexity
                        else "OK"
                    ),
                ]
            )

        widths = [
            max(
                len(headers[index]),
                max(len(row[index]) for row in rows),
            )
            for index in range(len(headers))
        ]

        def format_row(values: Sequence[str]) -> str:
            return " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(values)
            )

        lines.append("")
        lines.append(format_row(headers))
        lines.append(
            "-+-".join("-" * width for width in widths)
        )
        for row in rows:
            lines.append(format_row(row))

        lines.append("\nMcCabe Formula Details:")
        for name in sorted(self.functions):
            item = self.functions[name]
            lines.append(
                f"  {name}: {item.formula}"
            )

        lines.append("\nHigh-Complexity Warnings:")
        warnings = self.high_complexity_functions
        if not warnings:
            lines.append("  None")
        else:
            for item in warnings:
                lines.append(
                    f"  Function '{item.function_name}' has "
                    f"M={item.cyclomatic_complexity}, which is "
                    f"greater than {self.complexity_threshold}."
                )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "complexity_threshold": (
                self.complexity_threshold
            ),
            "summary": {
                "functions": self.total_functions,
                "total_function_loc": (
                    self.total_lines_of_code
                ),
                "total_statements": self.total_statements,
                "average_complexity": (
                    self.average_complexity
                ),
                "maximum_complexity": (
                    self.maximum_complexity
                ),
                "high_complexity_functions": [
                    item.function_name
                    for item in (
                        self.high_complexity_functions
                    )
                ],
            },
            "functions": {
                name: item.to_dict()
                for name, item in self.functions.items()
            },
        }

    def __str__(self) -> str:
        return self.format()


class CodeMetricsAnalyzer:
    """Compute McCabe complexity and per-function code metrics."""

    CONTROL_NODES = (IfStmt, WhileStmt, ForStmt)
    STATEMENT_NODES = (
        VarDecl,
        IfStmt,
        WhileStmt,
        ForStmt,
        ReturnStmt,
        ExprStmt,
    )

    def __init__(
        self,
        source: str,
        filename: str = "<input>",
        complexity_threshold: int = 10,
    ) -> None:
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if complexity_threshold < 1:
            raise ValueError(
                "complexity_threshold must be positive"
            )

        self.source = source
        self.filename = filename or "<input>"
        self.complexity_threshold = (
            complexity_threshold
        )
        self._tokens = Lexer(
            source,
            filename=self.filename,
        ).tokenize()
        self._code_lines = {
            token.line
            for token in self._tokens
            if token.type != "WHITESPACE"
        }
        self._brace_pairs = self._build_brace_pairs()

    def analyze(
        self,
        program: Program,
        graphs: Optional[
            Mapping[str, ControlFlowGraph]
        ] = None,
        call_graph: Optional[CallGraph] = None,
    ) -> CodeMetricsResult:
        if not isinstance(program, Program):
            raise TypeError(
                "program must be an instance of Program"
            )

        graph_map = (
            dict(graphs)
            if graphs is not None
            else CFGBuilder().build(program)
        )

        functions = {
            declaration.name: declaration
            for declaration in program.declarations
            if isinstance(declaration, FuncDecl)
        }

        results: Dict[str, FunctionMetrics] = {}

        for function_name in sorted(functions):
            function = functions[function_name]
            graph = graph_map.get(function_name)
            if graph is None:
                raise KeyError(
                    f"Missing CFG for function: "
                    f"{function_name}"
                )

            (
                complexity,
                node_count,
                edge_count,
                components,
            ) = self._cyclomatic_complexity(graph)

            call_site_count = self._count_calls(
                function.body
            )
            caller_count = 0
            callee_count = 0

            if call_graph is not None:
                caller_count = len(
                    call_graph.direct_callers(
                        function_name
                    )
                )
                callee_count = len(
                    call_graph.direct_callees(
                        function_name
                    )
                )
                call_site_count = len(
                    call_graph.call_sites_from(
                        function_name
                    )
                )

            results[function_name] = FunctionMetrics(
                function_name=function_name,
                cyclomatic_complexity=complexity,
                cfg_nodes=node_count,
                cfg_edges=edge_count,
                connected_components=components,
                nesting_depth=self._max_nesting_depth(
                    function.body
                ),
                parameter_count=len(function.params),
                lines_of_code=self._function_loc(
                    function
                ),
                statement_count=self._count_statements(
                    function.body
                ),
                local_variable_count=(
                    self._count_local_variables(
                        function.body
                    )
                ),
                call_site_count=call_site_count,
                caller_count=caller_count,
                callee_count=callee_count,
                high_complexity=(
                    complexity
                    > self.complexity_threshold
                ),
            )

        return CodeMetricsResult(
            functions=results,
            complexity_threshold=(
                self.complexity_threshold
            ),
        )

    run = analyze

    def _cyclomatic_complexity(
        self,
        graph: ControlFlowGraph,
    ) -> Tuple[int, int, int, int]:
        reachable = set(graph.reachable_block_ids())

        if not reachable:
            return 1, 0, 0, 1

        edges = [
            edge
            for edge in graph.edges
            if (
                edge.source in reachable
                and edge.target in reachable
            )
        ]

        node_count = len(reachable)
        edge_count = len(edges)
        components = self._weak_components(
            reachable,
            edges,
        )

        complexity = (
            edge_count
            - node_count
            + 2 * components
        )

        return (
            max(1, complexity),
            node_count,
            edge_count,
            components,
        )

    @staticmethod
    def _weak_components(
        nodes: Set[str],
        edges: Sequence[object],
    ) -> int:
        adjacency: Dict[str, Set[str]] = {
            node: set()
            for node in nodes
        }

        for edge in edges:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)

        visited: Set[str] = set()
        components = 0

        for start in sorted(nodes):
            if start in visited:
                continue

            components += 1
            stack = [start]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)
                stack.extend(
                    adjacency[current] - visited
                )

        return max(1, components)

    def _function_loc(
        self,
        function: FuncDecl,
    ) -> int:
        start_line = (
            function.loc[0]
            if function.loc is not None
            else 1
        )

        body_loc = getattr(function.body, "loc", None)
        end_line = self._maximum_line(function)

        if body_loc in self._brace_pairs:
            end_line = self._brace_pairs[body_loc][0]

        return sum(
            1
            for line in self._code_lines
            if start_line <= line <= end_line
        )

    def _build_brace_pairs(
        self,
    ) -> Dict[SourceLocation, SourceLocation]:
        stack: List[SourceLocation] = []
        pairs: Dict[
            SourceLocation,
            SourceLocation,
        ] = {}

        for token in self._tokens:
            if token.type != "DELIMITER":
                continue

            location = (token.line, token.column)

            if token.lexeme == "{":
                stack.append(location)
            elif token.lexeme == "}" and stack:
                opening = stack.pop()
                pairs[opening] = location

        return pairs

    def _maximum_line(
        self,
        node: Optional[ASTNode],
    ) -> int:
        maximum = 1

        def visit(current: Optional[ASTNode]) -> None:
            nonlocal maximum

            if current is None:
                return

            location = getattr(current, "loc", None)
            if (
                isinstance(location, tuple)
                and len(location) == 2
            ):
                maximum = max(
                    maximum,
                    int(location[0]),
                )

            for child in self._children(current):
                visit(child)

        visit(node)
        return maximum

    def _count_statements(
        self,
        node: Optional[ASTNode],
    ) -> int:
        if node is None:
            return 0

        count = (
            1
            if isinstance(node, self.STATEMENT_NODES)
            else 0
        )

        return count + sum(
            self._count_statements(child)
            for child in self._children(node)
        )

    def _count_local_variables(
        self,
        node: Optional[ASTNode],
    ) -> int:
        if node is None:
            return 0

        count = 1 if isinstance(node, VarDecl) else 0
        return count + sum(
            self._count_local_variables(child)
            for child in self._children(node)
        )

    def _count_calls(
        self,
        node: Optional[ASTNode],
    ) -> int:
        if node is None:
            return 0

        count = 1 if isinstance(node, CallExpr) else 0
        return count + sum(
            self._count_calls(child)
            for child in self._children(node)
        )

    def _max_nesting_depth(
        self,
        node: Optional[ASTNode],
        current_depth: int = 0,
    ) -> int:
        if node is None:
            return current_depth

        next_depth = current_depth
        maximum = current_depth

        if isinstance(node, self.CONTROL_NODES):
            next_depth = current_depth + 1
            maximum = next_depth

        for child in self._children(node):
            maximum = max(
                maximum,
                self._max_nesting_depth(
                    child,
                    next_depth,
                ),
            )

        return maximum

    @staticmethod
    def _children(
        node: ASTNode,
    ) -> Iterable[ASTNode]:
        if isinstance(node, Program):
            return tuple(node.declarations)

        if isinstance(node, FuncDecl):
            return (
                *node.params,
                node.body,
            )

        if isinstance(node, Param):
            return ()

        if isinstance(node, VarDecl):
            return (
                (node.init_expr,)
                if node.init_expr is not None
                else ()
            )

        if isinstance(node, StructDecl):
            return tuple(node.fields)

        if isinstance(node, Block):
            return tuple(node.statements)

        if isinstance(node, IfStmt):
            children: List[ASTNode] = [
                node.condition,
                node.then_stmt,
            ]
            if node.else_stmt is not None:
                children.append(node.else_stmt)
            return tuple(children)

        if isinstance(node, WhileStmt):
            return (
                node.condition,
                node.body,
            )

        if isinstance(node, ForStmt):
            children = []
            if node.init is not None:
                children.append(node.init)
            if node.condition is not None:
                children.append(node.condition)
            if node.increment is not None:
                children.append(node.increment)
            children.append(node.body)
            return tuple(children)

        if isinstance(node, ReturnStmt):
            return (
                (node.value,)
                if node.value is not None
                else ()
            )

        if isinstance(node, ExprStmt):
            return (
                (node.expr,)
                if node.expr is not None
                else ()
            )

        if isinstance(node, BinaryExpr):
            return (
                node.left,
                node.right,
            )

        if isinstance(node, UnaryExpr):
            return (node.operand,)

        if isinstance(node, CallExpr):
            return tuple(node.args)

        return ()
