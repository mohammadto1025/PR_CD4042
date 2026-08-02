from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from callgraph import CallGraph, CallGraphBuilder
from cfg import CFGBuilder, ControlFlowGraph
from dataflow import (
    DataFlowAnalyzer,
    FunctionDataFlowResult,
)
from deadcode import DeadCodeAnalyzer, DeadCodeResult
from navigation import (
    NavigationEngine,
    NavigationOccurrence,
    NavigationTarget,
)
from pars import Program
from phase2 import PhaseTwoPipeline, PhaseTwoResult
from refactor import RenameEngine, RenamePlan


@dataclass(frozen=True)
class PhaseThreeResult:
    source: str
    filename: str
    entry_function: str
    program: Program
    phase2_result: PhaseTwoResult
    cfg_results: Dict[str, ControlFlowGraph]
    dataflow_results: Dict[str, FunctionDataFlowResult]
    call_graph: CallGraph
    navigation: NavigationEngine
    rename_engine: RenameEngine
    dead_code_result: DeadCodeResult

    @property
    def has_phase2_errors(self) -> bool:
        return self.phase2_result.has_errors

    @property
    def function_names(self) -> List[str]:
        return sorted(self.cfg_results)

    @property
    def recursive_functions(self) -> List[str]:
        return self.call_graph.recursive_functions()

    @property
    def dead_function_names(self) -> List[str]:
        return self.call_graph.dead_functions(
            self.entry_function
        )

    @property
    def uninitialized_use_count(self) -> int:
        return sum(
            len(result.uninitialized_uses)
            for result in self.dataflow_results.values()
        )

    def cfg_for(
        self,
        function_name: str,
    ) -> ControlFlowGraph:
        try:
            return self.cfg_results[function_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown function CFG: {function_name}"
            ) from error

    def dataflow_for(
        self,
        function_name: str,
    ) -> FunctionDataFlowResult:
        try:
            return self.dataflow_results[function_name]
        except KeyError as error:
            raise KeyError(
                f"Unknown function data-flow result: "
                f"{function_name}"
            ) from error

    def goto_definition(
        self,
        line: int,
        column: int,
    ) -> Optional[NavigationTarget]:
        return self.navigation.goto_definition(
            line=line,
            column=column,
        )

    go_to_definition = goto_definition

    def find_references(
        self,
        line: int,
        column: int,
        include_definition: bool = True,
    ) -> List[NavigationOccurrence]:
        return self.navigation.find_references(
            line=line,
            column=column,
            include_definition=include_definition,
        )

    find_all_references = find_references

    def format_references(
        self,
        references: List[NavigationOccurrence],
    ) -> str:
        return self.navigation.format_references(references)

    def preview_rename(
        self,
        line: int,
        column: int,
        new_name: str,
    ) -> RenamePlan:
        return self.rename_engine.preview(
            line=line,
            column=column,
            new_name=new_name,
        )

    def rename(
        self,
        line: int,
        column: int,
        new_name: str,
        verify: bool = True,
    ) -> RenamePlan:
        return self.rename_engine.rename(
            line=line,
            column=column,
            new_name=new_name,
            verify=verify,
        )

    def format_cfgs(self) -> str:
        if not self.cfg_results:
            return "No functions found."

        return "\n\n".join(
            self.cfg_results[name].format()
            for name in self.function_names
        )

    def format_dataflow(self) -> str:
        if not self.dataflow_results:
            return "No functions found."

        return "\n\n".join(
            self.dataflow_results[name].format()
            for name in self.function_names
        )

    def format_call_graph(self) -> str:
        return self.call_graph.format()

    def format_dead_code(self) -> str:
        return self.dead_code_result.format()

    def summary(self) -> str:
        return (
            "Phase Three Analysis Summary\n"
            f"Functions: {len(self.function_names)}\n"
            f"CFGs: {len(self.cfg_results)}\n"
            f"Call edges: {self.call_graph.edge_count()}\n"
            f"Recursive functions: "
            f"{len(self.recursive_functions)}\n"
            f"Dead functions: "
            f"{len(self.dead_function_names)}\n"
            f"Potential uninitialized uses: "
            f"{self.uninitialized_use_count}\n"
            f"Dead-code issues: "
            f"{len(self.dead_code_result.issues)}"
        )

    def format_all(self) -> str:
        return "\n".join(
            [
                self.summary(),
                "\n" + "=" * 50,
                "Control Flow Graphs:",
                self.format_cfgs(),
                "\n" + "=" * 50,
                "Data-Flow Analysis:",
                self.format_dataflow(),
                "\n" + "=" * 50,
                "Call Graph:",
                self.format_call_graph(),
                "\n" + "=" * 50,
                "Dead-Code Analysis:",
                self.format_dead_code(),
            ]
        )


class PhaseThreePipeline:
    """Coordinate every Phase Three analysis without changing earlier phases."""

    def __init__(
        self,
        source: str,
        filename: str = "<input>",
        entry_function: str = "main",
    ) -> None:
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")
        if not isinstance(entry_function, str):
            raise TypeError(
                "entry_function must be a string"
            )

        self.source = source
        self.filename = filename or "<input>"
        self.entry_function = entry_function or "main"

    def run(
        self,
        program: Program,
        phase2_result: Optional[PhaseTwoResult] = None,
    ) -> PhaseThreeResult:
        if not isinstance(program, Program):
            raise TypeError(
                "program must be an instance of Program"
            )

        resolved_phase2 = phase2_result

        if resolved_phase2 is None:
            resolved_phase2 = PhaseTwoPipeline(
                source=self.source,
                filename=self.filename,
            ).run(program)
        elif not isinstance(
            resolved_phase2,
            PhaseTwoResult,
        ):
            raise TypeError(
                "phase2_result must be a PhaseTwoResult"
            )
        elif resolved_phase2.program is not program:
            raise ValueError(
                "phase2_result belongs to a different AST"
            )

        cfg_results = CFGBuilder().build(program)

        dataflow_results = DataFlowAnalyzer(
            filename=self.filename,
        ).analyze(
            program=program,
            graphs=cfg_results,
        )

        call_graph = CallGraphBuilder(
            filename=self.filename,
            entry_function=self.entry_function,
        ).build(
            program=program,
            semantic_result=(
                resolved_phase2.semantic_result
            ),
        )

        navigation = NavigationEngine(
            source=self.source,
            program=program,
            semantic_result=(
                resolved_phase2.semantic_result
            ),
            filename=self.filename,
        )

        rename_engine = RenameEngine(
            source=self.source,
            program=program,
            semantic_result=(
                resolved_phase2.semantic_result
            ),
            filename=self.filename,
        )

        dead_code_result = DeadCodeAnalyzer(
            filename=self.filename,
            entry_function=self.entry_function,
        ).analyze(
            program=program,
            semantic_result=(
                resolved_phase2.semantic_result
            ),
            graphs=cfg_results,
            dataflow_results=dataflow_results,
            call_graph=call_graph,
        )

        return PhaseThreeResult(
            source=self.source,
            filename=self.filename,
            entry_function=self.entry_function,
            program=program,
            phase2_result=resolved_phase2,
            cfg_results=cfg_results,
            dataflow_results=dataflow_results,
            call_graph=call_graph,
            navigation=navigation,
            rename_engine=rename_engine,
            dead_code_result=dead_code_result,
        )

    analyze = run
