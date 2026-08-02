from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from cfg import BasicBlock, CFGBuilder, ControlFlowGraph
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
    UnaryExpr,
    VarDecl,
    WhileStmt,
)


SourceLocation = Tuple[int, int]
ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%="}
COMPOUND_ASSIGNMENTS = {"+=", "-=", "*="}
MEMBER_OPERATORS = {".", "->"}


@dataclass(frozen=True)
class FlowEvent:
    kind: str
    name: str
    loc: Optional[SourceLocation] = None


@dataclass
class BlockFlowFacts:
    block_id: str
    definitions: Set[str] = field(default_factory=set)
    uses: Set[str] = field(default_factory=set)
    events: List[FlowEvent] = field(default_factory=list)
    definite_in: Set[str] = field(default_factory=set)
    definite_out: Set[str] = field(default_factory=set)
    live_in: Set[str] = field(default_factory=set)
    live_out: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class UninitializedUse:
    variable: str
    block_id: str
    loc: Optional[SourceLocation] = None

    def format(self, filename: str = "<input>") -> str:
        if self.loc is None:
            return (
                f"{filename}:?:? Variable '{self.variable}' may be "
                "used before definite assignment"
            )
        return (
            f"{filename}:{self.loc[0]}:{self.loc[1]} "
            f"Variable '{self.variable}' may be used before "
            "definite assignment"
        )


@dataclass
class FunctionDataFlowResult:
    function_name: str
    cfg: ControlFlowGraph
    initial_assigned: Set[str]
    tracked_variables: Set[str]
    blocks: Dict[str, BlockFlowFacts]
    uninitialized_uses: List[UninitializedUse]
    definite_iterations: int
    liveness_iterations: int
    filename: str = "<input>"

    def facts_for(self, block_id: str) -> BlockFlowFacts:
        return self.blocks[block_id]

    @staticmethod
    def _format_set(values: Iterable[str]) -> str:
        values = sorted(set(values))
        return "{}" if not values else "{" + ", ".join(values) + "}"

    def format(self) -> str:
        lines = [
            f"Data-Flow for function: {self.function_name}",
            "Initial assigned: "
            + self._format_set(self.initial_assigned),
        ]

        unreachable = set(self.cfg.unreachable_block_ids())

        for block_id in self.cfg._ordered_block_ids():
            facts = self.blocks[block_id]
            suffix = " [unreachable]" if block_id in unreachable else ""
            lines.append(f"\n{block_id}:{suffix}")
            lines.append(
                "  DEF      = "
                + self._format_set(facts.definitions)
            )
            lines.append(
                "  USE      = "
                + self._format_set(facts.uses)
            )
            lines.append(
                "  DA-IN    = "
                + self._format_set(facts.definite_in)
            )
            lines.append(
                "  DA-OUT   = "
                + self._format_set(facts.definite_out)
            )
            lines.append(
                "  LIVE-IN  = "
                + self._format_set(facts.live_in)
            )
            lines.append(
                "  LIVE-OUT = "
                + self._format_set(facts.live_out)
            )

        lines.append("\nPotential uninitialized uses:")
        if not self.uninitialized_uses:
            lines.append("  None")
        else:
            for issue in self.uninitialized_uses:
                lines.append("  " + issue.format(self.filename))

        lines.append(
            "\nIterations: "
            f"definite={self.definite_iterations}, "
            f"liveness={self.liveness_iterations}"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


class DataFlowAnalyzer:
    """Definite-assignment and live-variable analysis over CFGs."""

    def __init__(self, filename: str = "<input>") -> None:
        self.filename = filename or "<input>"

    def analyze(
        self,
        program: Program,
        graphs: Optional[Mapping[str, ControlFlowGraph]] = None,
    ) -> Dict[str, FunctionDataFlowResult]:
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")

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
        global_variables = {
            declaration.name
            for declaration in program.declarations
            if isinstance(declaration, VarDecl)
        }

        results: Dict[str, FunctionDataFlowResult] = {}

        for function_name, graph in graph_map.items():
            function = functions.get(function_name)
            if function is None:
                raise KeyError(
                    f"CFG has no matching function AST: {function_name}"
                )

            results[function_name] = self.analyze_function(
                function,
                graph,
                global_variables,
            )

        return results

    run = analyze

    def analyze_function(
        self,
        function: FuncDecl,
        graph: ControlFlowGraph,
        global_variables: Optional[Iterable[str]] = None,
    ) -> FunctionDataFlowResult:
        parameters = {
            parameter.name
            for parameter in function.params
            if isinstance(parameter, Param)
        }
        globals_set = set(global_variables or ())
        local_variables = self._collect_local_variables(function.body)

        tracked_variables = parameters | globals_set | local_variables
        initial_assigned = parameters | globals_set

        block_facts = {
            block_id: self._build_block_facts(block)
            for block_id, block in graph.blocks.items()
        }

        definite_iterations = self._run_definite_assignment(
            graph,
            block_facts,
            tracked_variables,
            initial_assigned,
        )
        uninitialized = self._find_uninitialized_uses(
            graph,
            block_facts,
            tracked_variables,
        )
        liveness_iterations = self._run_liveness(
            graph,
            block_facts,
        )

        return FunctionDataFlowResult(
            function_name=function.name,
            cfg=graph,
            initial_assigned=initial_assigned,
            tracked_variables=tracked_variables,
            blocks=block_facts,
            uninitialized_uses=uninitialized,
            definite_iterations=definite_iterations,
            liveness_iterations=liveness_iterations,
            filename=self.filename,
        )

    def _build_block_facts(
        self,
        block: BasicBlock,
    ) -> BlockFlowFacts:
        events: List[FlowEvent] = []
        for node in block.nodes:
            events.extend(self._events_for_node(node))

        definitions: Set[str] = set()
        uses_before_definition: Set[str] = set()

        for event in events:
            if event.kind == "use":
                if event.name not in definitions:
                    uses_before_definition.add(event.name)
            elif event.kind == "def":
                definitions.add(event.name)

        return BlockFlowFacts(
            block_id=block.block_id,
            definitions=definitions,
            uses=uses_before_definition,
            events=events,
        )

    def _events_for_node(
        self,
        node: Optional[ASTNode],
    ) -> List[FlowEvent]:
        if node is None:
            return []

        if isinstance(node, VarDecl):
            events = self._events_for_expression(node.init_expr)
            if node.init_expr is not None:
                events.append(FlowEvent("def", node.name, node.loc))
            return events

        if isinstance(node, ExprStmt):
            return self._events_for_expression(node.expr)

        if isinstance(node, ReturnStmt):
            return self._events_for_expression(node.value)

        if isinstance(node, IfStmt):
            return self._events_for_expression(node.condition)

        if isinstance(node, WhileStmt):
            return self._events_for_expression(node.condition)

        if isinstance(node, ForStmt):
            events: List[FlowEvent] = []
            events.extend(self._events_for_node(node.init))
            events.extend(self._events_for_node(node.condition))
            events.extend(self._events_for_expression(node.increment))
            return events

        return self._events_for_expression(node)

    def _events_for_expression(
        self,
        node: Optional[ASTNode],
    ) -> List[FlowEvent]:
        if node is None:
            return []

        if isinstance(node, Identifier):
            return [FlowEvent("use", node.name, node.loc)]

        if isinstance(node, CallExpr):
            events: List[FlowEvent] = []
            for argument in node.args:
                events.extend(self._events_for_expression(argument))
            return events

        if isinstance(node, UnaryExpr):
            return self._events_for_expression(node.operand)

        if isinstance(node, BinaryExpr):
            if node.op in ASSIGNMENT_OPERATORS:
                events: List[FlowEvent] = []

                if node.op in COMPOUND_ASSIGNMENTS:
                    events.extend(
                        self._events_for_expression(node.left)
                    )
                elif not isinstance(node.left, Identifier):
                    events.extend(
                        self._events_for_expression(node.left)
                    )

                events.extend(
                    self._events_for_expression(node.right)
                )

                if isinstance(node.left, Identifier):
                    events.append(
                        FlowEvent(
                            "def",
                            node.left.name,
                            node.left.loc,
                        )
                    )
                return events

            if node.op in MEMBER_OPERATORS:
                return self._events_for_expression(node.left)

            events = self._events_for_expression(node.left)
            events.extend(self._events_for_expression(node.right))
            return events

        return []

    def _run_definite_assignment(
        self,
        graph: ControlFlowGraph,
        facts: Dict[str, BlockFlowFacts],
        universe: Set[str],
        initial_assigned: Set[str],
    ) -> int:
        reachable = set(graph.reachable_block_ids())

        for block_id, block_facts in facts.items():
            if block_id == graph.entry_id:
                block_facts.definite_in = set(initial_assigned)
                block_facts.definite_out = set(initial_assigned)
            elif block_id in reachable:
                block_facts.definite_in = set(universe)
                block_facts.definite_out = set(universe)
            else:
                block_facts.definite_in = set()
                block_facts.definite_out = set()

        iterations = 0
        changed = True

        while changed:
            iterations += 1
            changed = False

            for block_id in graph._ordered_block_ids():
                if block_id == graph.entry_id:
                    continue
                if block_id not in reachable:
                    continue

                predecessors = [
                    edge.source
                    for edge in graph.predecessors(block_id)
                    if edge.source in reachable
                ]

                if not predecessors:
                    new_in: Set[str] = set()
                else:
                    predecessor_sets = [
                        facts[pred].definite_out
                        for pred in predecessors
                    ]
                    new_in = set(predecessor_sets[0])
                    for values in predecessor_sets[1:]:
                        new_in.intersection_update(values)

                new_out = new_in | facts[block_id].definitions

                if (
                    new_in != facts[block_id].definite_in
                    or new_out != facts[block_id].definite_out
                ):
                    facts[block_id].definite_in = new_in
                    facts[block_id].definite_out = new_out
                    changed = True

        return iterations

    def _find_uninitialized_uses(
        self,
        graph: ControlFlowGraph,
        facts: Dict[str, BlockFlowFacts],
        tracked_variables: Set[str],
    ) -> List[UninitializedUse]:
        reachable = set(graph.reachable_block_ids())
        issues: List[UninitializedUse] = []
        seen = set()

        for block_id in graph._ordered_block_ids():
            if block_id not in reachable:
                continue

            assigned = set(facts[block_id].definite_in)

            for event in facts[block_id].events:
                if event.name not in tracked_variables:
                    continue

                if event.kind == "use":
                    key = (event.name, block_id, event.loc)
                    if event.name not in assigned and key not in seen:
                        seen.add(key)
                        issues.append(
                            UninitializedUse(
                                variable=event.name,
                                block_id=block_id,
                                loc=event.loc,
                            )
                        )
                elif event.kind == "def":
                    assigned.add(event.name)

        return issues

    def _run_liveness(
        self,
        graph: ControlFlowGraph,
        facts: Dict[str, BlockFlowFacts],
    ) -> int:
        reachable = set(graph.reachable_block_ids())
        ordered = list(reversed(graph._ordered_block_ids()))
        iterations = 0
        changed = True

        while changed:
            iterations += 1
            changed = False

            for block_id in ordered:
                if block_id not in reachable:
                    continue

                successors = [
                    edge.target
                    for edge in graph.successors(block_id)
                    if edge.target in reachable
                ]

                new_out: Set[str] = set()
                for successor in successors:
                    new_out.update(facts[successor].live_in)

                new_in = (
                    facts[block_id].uses
                    | (
                        new_out
                        - facts[block_id].definitions
                    )
                )

                if (
                    new_in != facts[block_id].live_in
                    or new_out != facts[block_id].live_out
                ):
                    facts[block_id].live_in = new_in
                    facts[block_id].live_out = new_out
                    changed = True

        return iterations

    def _collect_local_variables(
        self,
        node: Optional[ASTNode],
    ) -> Set[str]:
        result: Set[str] = set()

        def visit(current: Optional[ASTNode]) -> None:
            if current is None:
                return

            if isinstance(current, VarDecl):
                result.add(current.name)
                visit(current.init_expr)
                return

            if isinstance(current, Block):
                for statement in current.statements:
                    visit(statement)
                return

            if isinstance(current, IfStmt):
                visit(current.condition)
                visit(current.then_stmt)
                visit(current.else_stmt)
                return

            if isinstance(current, WhileStmt):
                visit(current.condition)
                visit(current.body)
                return

            if isinstance(current, ForStmt):
                visit(current.init)
                visit(current.condition)
                visit(current.increment)
                visit(current.body)
                return

            if isinstance(current, ReturnStmt):
                visit(current.value)
                return

            if isinstance(current, ExprStmt):
                visit(current.expr)
                return

            if isinstance(current, BinaryExpr):
                visit(current.left)
                visit(current.right)
                return

            if isinstance(current, UnaryExpr):
                visit(current.operand)
                return

            if isinstance(current, CallExpr):
                for argument in current.args:
                    visit(argument)

        visit(node)
        return result
