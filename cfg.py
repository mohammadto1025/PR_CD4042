from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pars import (
    ASTNode,
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
    Program,
    ReturnStmt,
    StringLiteral,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)


SourceLocation = Tuple[int, int]


class BlockKind(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    BASIC = "basic"
    CONDITION = "condition"
    LOOP_CONDITION = "loop-condition"
    FOR_INCREMENT = "for-increment"


@dataclass
class BasicBlock:
    block_id: str
    kind: BlockKind
    statements: List[str] = field(default_factory=list)
    loc: Optional[SourceLocation] = None
    nodes: List[ASTNode] = field(default_factory=list, repr=False)

    def add_statement(
        self,
        text: str,
        node: Optional[ASTNode] = None,
    ) -> None:
        text = text.strip()
        if text:
            self.statements.append(text)
        if node is not None:
            self.nodes.append(node)


@dataclass(frozen=True)
class CFGEdge:
    source: str
    target: str
    label: str = "next"


@dataclass(frozen=True)
class PendingEdge:
    source: str
    label: str = "next"
    control: Optional[str] = None


@dataclass
class ControlFlowGraph:
    function_name: str
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    edges: List[CFGEdge] = field(default_factory=list)
    entry_id: str = "ENTRY"
    exit_id: str = "EXIT"

    def add_block(self, block: BasicBlock) -> None:
        if block.block_id in self.blocks:
            raise ValueError(
                f"Duplicate basic-block id: {block.block_id}"
            )
        self.blocks[block.block_id] = block

    def add_edge(
        self,
        source: str,
        target: str,
        label: str = "next",
    ) -> None:
        if source not in self.blocks:
            raise KeyError(f"Unknown source block: {source}")
        if target not in self.blocks:
            raise KeyError(f"Unknown target block: {target}")

        edge = CFGEdge(source, target, label or "next")
        if edge not in self.edges:
            self.edges.append(edge)

    def successors(self, block_id: str) -> List[CFGEdge]:
        return [
            edge for edge in self.edges
            if edge.source == block_id
        ]

    def predecessors(self, block_id: str) -> List[CFGEdge]:
        return [
            edge for edge in self.edges
            if edge.target == block_id
        ]

    def reachable_block_ids(self) -> List[str]:
        if self.entry_id not in self.blocks:
            return []

        visited = set()
        order: List[str] = []
        queue = deque([self.entry_id])

        while queue:
            block_id = queue.popleft()
            if block_id in visited:
                continue

            visited.add(block_id)
            order.append(block_id)

            for edge in self.successors(block_id):
                if edge.target not in visited:
                    queue.append(edge.target)

        return order

    def unreachable_block_ids(self) -> List[str]:
        reachable = set(self.reachable_block_ids())
        return [
            block_id
            for block_id in self._ordered_block_ids()
            if block_id not in reachable
        ]

    def _ordered_block_ids(self) -> List[str]:
        numbered = [
            block_id
            for block_id in self.blocks
            if block_id not in {self.entry_id, self.exit_id}
        ]

        def sort_key(block_id: str) -> Tuple[int, str]:
            if block_id.startswith("B"):
                suffix = block_id[1:]
                if suffix.isdigit():
                    return int(suffix), block_id
            return 10**9, block_id

        numbered.sort(key=sort_key)

        ordered: List[str] = []
        if self.entry_id in self.blocks:
            ordered.append(self.entry_id)
        ordered.extend(numbered)
        if self.exit_id in self.blocks:
            ordered.append(self.exit_id)
        return ordered

    def format(self) -> str:
        lines = [f"CFG for function: {self.function_name}"]

        unreachable = set(self.unreachable_block_ids())

        for block_id in self._ordered_block_ids():
            block = self.blocks[block_id]

            suffix = ""
            if block_id in unreachable:
                suffix = " [unreachable]"

            if block.kind in {BlockKind.ENTRY, BlockKind.EXIT}:
                lines.append(f"\n{block.block_id}:{suffix}")
            else:
                lines.append(
                    f"\n{block.block_id} [{block.kind.value}]:{suffix}"
                )

            if block.statements:
                for statement in block.statements:
                    lines.append(f"  {statement}")
            elif block.kind not in {
                BlockKind.ENTRY,
                BlockKind.EXIT,
            }:
                lines.append("  (empty)")

            for edge in self.successors(block_id):
                lines.append(
                    f"  {edge.label} -> {edge.target}"
                )

        return "\n".join(lines)

    def to_dot(self) -> str:
        lines = [
            f'digraph "{self.function_name}" {{',
            "  rankdir=TB;",
            '  node [shape=box, fontname="Courier"];',
        ]

        unreachable = set(self.unreachable_block_ids())

        for block_id in self._ordered_block_ids():
            block = self.blocks[block_id]
            label_parts = [block.block_id]
            label_parts.extend(block.statements)

            if block_id in unreachable:
                label_parts.append("[unreachable]")

            label = "\\n".join(label_parts)
            label = label.replace('"', '\\"')

            shape = "oval" if block.kind in {
                BlockKind.ENTRY,
                BlockKind.EXIT,
            } else "box"

            lines.append(
                f'  "{block_id}" '
                f'[label="{label}", shape={shape}];'
            )

        for edge in self.edges:
            label = edge.label.replace('"', '\\"')
            lines.append(
                f'  "{edge.source}" -> "{edge.target}" '
                f'[label="{label}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


class CFGBuilder:
    """Build one control-flow graph for every function in a Program AST."""

    def __init__(self) -> None:
        self._cfg: Optional[ControlFlowGraph] = None
        self._next_block_number = 1

    def build(
        self,
        program: Program,
    ) -> Dict[str, ControlFlowGraph]:
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")

        graphs: Dict[str, ControlFlowGraph] = {}

        for declaration in program.declarations:
            if not isinstance(declaration, FuncDecl):
                continue

            if declaration.name in graphs:
                raise ValueError(
                    f"Duplicate function CFG: {declaration.name}"
                )

            graphs[declaration.name] = self.build_function(
                declaration
            )

        return graphs

    build_program = build

    def build_function(
        self,
        function: FuncDecl,
    ) -> ControlFlowGraph:
        if not isinstance(function, FuncDecl):
            raise TypeError(
                "function must be an instance of FuncDecl"
            )

        self._next_block_number = 1
        self._cfg = ControlFlowGraph(function.name)

        self._cfg.add_block(
            BasicBlock("ENTRY", BlockKind.ENTRY, loc=function.loc)
        )
        self._cfg.add_block(
            BasicBlock("EXIT", BlockKind.EXIT, loc=function.loc)
        )

        pending = [PendingEdge("ENTRY")]
        pending = self._build_statement(
            function.body,
            pending,
        )

        for edge in pending:
            self._cfg.add_edge(
                edge.source,
                "EXIT",
                edge.label,
            )

        result = self._cfg
        self._cfg = None
        return result

    def _build_statement(
        self,
        statement: Optional[ASTNode],
        incoming: Sequence[PendingEdge],
    ) -> List[PendingEdge]:
        if statement is None:
            return list(incoming)

        if isinstance(statement, Block):
            pending = list(incoming)
            for child in statement.statements:
                control_pending = [
                    edge
                    for edge in pending
                    if edge.control is not None
                ]
                normal_pending = [
                    edge
                    for edge in pending
                    if edge.control is None
                ]
                built = self._build_statement(
                    child,
                    normal_pending,
                )
                pending = control_pending + built
            return pending

        if isinstance(statement, IfStmt):
            return self._build_if(statement, incoming)

        if isinstance(statement, WhileStmt):
            return self._build_while(statement, incoming)

        if isinstance(statement, ForStmt):
            return self._build_for(statement, incoming)

        if isinstance(statement, ReturnStmt):
            text = self._statement_text(statement)
            tails = self._append_linear_statement(
                text,
                incoming,
                statement.loc,
                statement,
            )

            for tail in tails:
                self._require_cfg().add_edge(
                    tail.source,
                    "EXIT",
                    "return",
                )

            return []

        if isinstance(statement, BreakStmt):
            tails = self._append_linear_statement(
                "break;",
                incoming,
                statement.loc,
                statement,
            )
            return [
                PendingEdge(tail.source, "break", "break")
                for tail in tails
            ]

        if isinstance(statement, ContinueStmt):
            tails = self._append_linear_statement(
                "continue;",
                incoming,
                statement.loc,
                statement,
            )
            return [
                PendingEdge(
                    tail.source,
                    "continue",
                    "continue",
                )
                for tail in tails
            ]

        text = self._statement_text(statement)
        if not text:
            return list(incoming)

        return self._append_linear_statement(
            text,
            incoming,
            getattr(statement, "loc", None),
            statement,
        )

    def _build_if(
        self,
        statement: IfStmt,
        incoming: Sequence[PendingEdge],
    ) -> List[PendingEdge]:
        condition_block = self._new_block(
            BlockKind.CONDITION,
            [f"if {self._expression_text(statement.condition)}"],
            statement.loc,
            [statement.condition],
        )
        self._connect_pending(incoming, condition_block.block_id)

        then_pending = self._build_statement(
            statement.then_stmt,
            [PendingEdge(condition_block.block_id, "true")],
        )

        if statement.else_stmt is None:
            else_pending = [
                PendingEdge(condition_block.block_id, "false")
            ]
        else:
            else_pending = self._build_statement(
                statement.else_stmt,
                [PendingEdge(condition_block.block_id, "false")],
            )

        return then_pending + else_pending

    def _build_while(
        self,
        statement: WhileStmt,
        incoming: Sequence[PendingEdge],
    ) -> List[PendingEdge]:
        condition_block = self._new_block(
            BlockKind.LOOP_CONDITION,
            [
                "while "
                + self._expression_text(statement.condition)
            ],
            statement.loc,
            [statement.condition],
        )
        self._connect_pending(incoming, condition_block.block_id)

        body_pending = self._build_statement(
            statement.body,
            [PendingEdge(condition_block.block_id, "true")],
        )

        exits = [
            PendingEdge(condition_block.block_id, "false")
        ]
        for tail in body_pending:
            if tail.control == "continue":
                self._require_cfg().add_edge(
                    tail.source,
                    condition_block.block_id,
                    "continue",
                )
            elif tail.control == "break":
                exits.append(
                    PendingEdge(tail.source, "break")
                )
            else:
                self._require_cfg().add_edge(
                    tail.source,
                    condition_block.block_id,
                    "back",
                )

        return exits

    def _build_for(
        self,
        statement: ForStmt,
        incoming: Sequence[PendingEdge],
    ) -> List[PendingEdge]:
        pending = list(incoming)

        init_text = self._statement_text(statement.init)
        if init_text:
            pending = self._append_linear_statement(
                f"for-init: {init_text}",
                pending,
                getattr(statement.init, "loc", statement.loc),
                statement.init,
            )

        condition_expr = statement.condition
        if isinstance(condition_expr, ExprStmt):
            condition_expr = condition_expr.expr

        condition_text = (
            self._expression_text(condition_expr)
            if condition_expr is not None
            else "true"
        )

        condition_nodes = (
            [condition_expr]
            if condition_expr is not None
            else []
        )
        condition_block = self._new_block(
            BlockKind.LOOP_CONDITION,
            [f"for-condition: {condition_text}"],
            statement.loc,
            condition_nodes,
        )
        self._connect_pending(pending, condition_block.block_id)

        body_pending = self._build_statement(
            statement.body,
            [PendingEdge(condition_block.block_id, "true")],
        )

        exits = [
            PendingEdge(condition_block.block_id, "false")
        ]

        increment_block = None
        if statement.increment is not None:
            increment_block = self._new_block(
                BlockKind.FOR_INCREMENT,
                [
                    "for-increment: "
                    + self._expression_text(statement.increment)
                ],
                getattr(statement.increment, "loc", statement.loc),
                [statement.increment],
            )

        for tail in body_pending:
            if tail.control == "break":
                exits.append(
                    PendingEdge(tail.source, "break")
                )
                continue

            target = (
                increment_block.block_id
                if increment_block is not None
                else condition_block.block_id
            )
            label = (
                "continue"
                if tail.control == "continue"
                else "next"
                if increment_block is not None
                else "back"
            )
            self._require_cfg().add_edge(
                tail.source,
                target,
                label,
            )

        if increment_block is not None:
            self._require_cfg().add_edge(
                increment_block.block_id,
                condition_block.block_id,
                "back",
            )

        return exits

    def _append_linear_statement(
        self,
        text: str,
        incoming: Sequence[PendingEdge],
        loc: Optional[SourceLocation],
        node: Optional[ASTNode] = None,
    ) -> List[PendingEdge]:
        cfg = self._require_cfg()
        incoming = list(incoming)

        if len(incoming) == 1:
            edge = incoming[0]
            source_block = cfg.blocks[edge.source]

            can_append = (
                edge.label == "next"
                and source_block.kind == BlockKind.BASIC
                and not cfg.successors(source_block.block_id)
            )

            if can_append:
                source_block.add_statement(text, node)
                return [PendingEdge(source_block.block_id)]

        block_nodes = [node] if node is not None else []
        block = self._new_block(
            BlockKind.BASIC,
            [text],
            loc,
            block_nodes,
        )
        self._connect_pending(incoming, block.block_id)
        return [PendingEdge(block.block_id)]

    def _connect_pending(
        self,
        pending: Iterable[PendingEdge],
        target: str,
    ) -> None:
        cfg = self._require_cfg()
        for edge in pending:
            cfg.add_edge(edge.source, target, edge.label)

    def _new_block(
        self,
        kind: BlockKind,
        statements: Optional[Sequence[str]] = None,
        loc: Optional[SourceLocation] = None,
        nodes: Optional[Sequence[ASTNode]] = None,
    ) -> BasicBlock:
        cfg = self._require_cfg()

        block_id = f"B{self._next_block_number}"
        self._next_block_number += 1

        block = BasicBlock(
            block_id=block_id,
            kind=kind,
            statements=list(statements or []),
            loc=loc,
            nodes=list(nodes or []),
        )
        cfg.add_block(block)
        return block

    def _require_cfg(self) -> ControlFlowGraph:
        if self._cfg is None:
            raise RuntimeError("No CFG is currently being built")
        return self._cfg

    def _statement_text(
        self,
        statement: Optional[ASTNode],
    ) -> str:
        if statement is None:
            return ""

        if isinstance(statement, VarDecl):
            text = f"{statement.type_spec} {statement.name}"
            if statement.init_expr is not None:
                text += (
                    " = "
                    + self._expression_text(statement.init_expr)
                )
            return text + ";"

        if isinstance(statement, ExprStmt):
            if statement.expr is None:
                return ""
            return self._expression_text(statement.expr) + ";"

        if isinstance(statement, ReturnStmt):
            if statement.value is None:
                return "return;"
            return (
                "return "
                + self._expression_text(statement.value)
                + ";"
            )

        return statement.__class__.__name__

    def _expression_text(
        self,
        expression: Optional[ASTNode],
    ) -> str:
        if expression is None:
            return ""

        if isinstance(expression, Identifier):
            return expression.name

        if isinstance(expression, IntLiteral):
            return str(expression.value)

        if isinstance(expression, FloatLiteral):
            return str(expression.value)

        if isinstance(expression, StringLiteral):
            return str(expression.value)

        if isinstance(expression, CharLiteral):
            return str(expression.value)

        if isinstance(expression, CallExpr):
            arguments = ", ".join(
                self._expression_text(argument)
                for argument in expression.args
            )
            return f"{expression.callee}({arguments})"

        if isinstance(expression, UnaryExpr):
            return (
                expression.op
                + self._expression_text(expression.operand)
            )

        if isinstance(expression, BinaryExpr):
            left = self._expression_text(expression.left)
            right = self._expression_text(expression.right)
            return f"{left} {expression.op} {right}"

        return expression.__class__.__name__
