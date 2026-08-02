from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from diagnostic import Diagnostic, DiagnosticBag
from intellisense import CompletionItem, HoverInfo, IntelliSenseEngine
from pars import Program
from scope import Scope
from semantic import SemanticAnalysisResult, SemanticAnalyzer
from typer import TypeCheckResult, TypeChecker


@dataclass(frozen=True)
class PhaseTwoResult:
    program: Program
    semantic_result: SemanticAnalysisResult
    type_result: TypeCheckResult
    diagnostics: DiagnosticBag
    intellisense: IntelliSenseEngine

    @property
    def global_scope(self) -> Scope:
        return self.semantic_result.global_scope

    @property
    def has_errors(self) -> bool:
        return self.diagnostics.has_errors

    @property
    def errors(self) -> List[Diagnostic]:
        return self.diagnostics.errors

    @property
    def warnings(self) -> List[Diagnostic]:
        return self.diagnostics.warnings

    @property
    def infos(self) -> List[Diagnostic]:
        return self.diagnostics.infos

    def complete(
        self,
        line: int,
        column: int,
        limit: int = 50,
    ) -> List[CompletionItem]:
        return self.intellisense.complete(line, column, limit)

    def hover(self, line: int, column: int) -> Optional[HoverInfo]:
        return self.intellisense.hover(line, column)

    def format_diagnostics(self) -> str:
        return str(self.diagnostics)

    def format_scope_tree(self) -> str:
        return self.global_scope.format_tree()

    def summary(self) -> str:
        return (
            "Phase Two Analysis Summary\n"
            f"Errors: {len(self.errors)}\n"
            f"Warnings: {len(self.warnings)}\n"
            f"Infos: {len(self.infos)}"
        )


class PhaseTwoPipeline:
    """Coordinates all Phase Two modules without changing Phase One.

    The caller passes the original source text and the AST already produced by
    ``parskon.py``. The pipeline then performs semantic analysis, type checking,
    diagnostic aggregation, completion, and hover support.
    """

    def __init__(
        self,
        source: str,
        filename: str = "<input>",
    ) -> None:
        if not isinstance(source, str):
            raise TypeError("source must be a string")

        self.source = source
        self.filename = filename or "<input>"

    def run(self, program: Program) -> PhaseTwoResult:
        if not isinstance(program, Program):
            raise TypeError("program must be an instance of Program")

        semantic_result = SemanticAnalyzer(self.filename).analyze(program)
        type_result = TypeChecker(self.filename).check(
            program,
            semantic_result,
        )

        diagnostics = DiagnosticBag()
        diagnostics.extend(semantic_result.diagnostics)
        diagnostics.extend(type_result.diagnostics)

        intellisense = IntelliSenseEngine(
            source=self.source,
            program=program,
            semantic_result=semantic_result,
            type_result=type_result,
            filename=self.filename,
        )

        return PhaseTwoResult(
            program=program,
            semantic_result=semantic_result,
            type_result=type_result,
            diagnostics=diagnostics,
            intellisense=intellisense,
        )

    analyze = run
