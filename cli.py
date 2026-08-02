from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Iterable, List, Optional, Sequence, TextIO, Tuple

from diagnostic import Diagnostic

from lex import Lexer
from pars import Parser, Program
from phase2 import PhaseTwoPipeline, PhaseTwoResult
from phase3 import PhaseThreePipeline, PhaseThreeResult
from scope import Scope
from symbol import Symbol


@dataclass
class AnalysisSession:
    source: str
    filename: str
    entry_function: str
    tokens: List[object]
    lexer_diagnostics: List[Diagnostic]
    parser_diagnostics: List[Diagnostic]
    parser_errors: List[str]
    program: Program
    phase2_result: PhaseTwoResult
    phase3_result: Optional[PhaseThreeResult]
    phase3_error: Optional[str] = None

    @classmethod
    def from_source(
        cls,
        source: str,
        filename: str = "<input>",
        entry_function: str = "main",
    ) -> "AnalysisSession":
        if not isinstance(source, str):
            raise TypeError("source must be a string")

        filename = filename or "<input>"
        entry_function = entry_function or "main"

        lexer = Lexer(
            source,
            filename=filename,
        )
        tokens = lexer.tokenize()

        parser = Parser(tokens)
        program = parser.parse_program()

        phase2_result = PhaseTwoPipeline(
            source=source,
            filename=filename,
        ).run(program)

        phase3_result: Optional[PhaseThreeResult] = None
        phase3_error: Optional[str] = None

        try:
            phase3_result = PhaseThreePipeline(
                source=source,
                filename=filename,
                entry_function=entry_function,
            ).run(
                program=program,
                phase2_result=phase2_result,
            )
        except Exception as error:
            phase3_error = (
                f"{error.__class__.__name__}: {error}"
            )

        return cls(
            source=source,
            filename=filename,
            entry_function=entry_function,
            tokens=list(tokens),
            lexer_diagnostics=list(lexer.diagnostics),
            parser_diagnostics=list(parser.diagnostics),
            parser_errors=list(parser.errors),
            program=program,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase3_error=phase3_error,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        entry_function: str = "main",
    ) -> "AnalysisSession":
        source_path = Path(path)
        source = source_path.read_text(encoding="utf-8")
        return cls.from_source(
            source=source,
            filename=str(source_path),
            entry_function=entry_function,
        )

    @classmethod
    def from_phase3_result(
        cls,
        result: PhaseThreeResult,
        tokens: Optional[Sequence[object]] = None,
        parser_errors: Optional[Sequence[str]] = None,
        lexer_diagnostics: Optional[Sequence[Diagnostic]] = None,
        parser_diagnostics: Optional[Sequence[Diagnostic]] = None,
    ) -> "AnalysisSession":
        if not isinstance(result, PhaseThreeResult):
            raise TypeError(
                "result must be a PhaseThreeResult"
            )

        return cls(
            source=result.source,
            filename=result.filename,
            entry_function=result.entry_function,
            tokens=list(tokens or ()),
            lexer_diagnostics=list(lexer_diagnostics or ()),
            parser_diagnostics=list(parser_diagnostics or ()),
            parser_errors=list(parser_errors or ()),
            program=result.program,
            phase2_result=result.phase2_result,
            phase3_result=result,
            phase3_error=None,
        )

    @property
    def phase3_ready(self) -> bool:
        return self.phase3_result is not None

    def startup_summary(self) -> str:
        lines = [
            "Compoiler Interactive CLI",
            f"File: {self.filename}",
            f"Entry function: {self.entry_function}",
            f"Lexical errors: {len(self.lexer_diagnostics)}",
            f"Syntax errors: {len(self.parser_diagnostics) or len(self.parser_errors)}",
            f"Semantic/type errors: "
            f"{len(self.phase2_result.errors)}",
            "Phase Three: "
            + (
                "ready"
                if self.phase3_ready
                else "unavailable"
            ),
        ]

        if self.phase3_error is not None:
            lines.append(
                f"Phase Three error: {self.phase3_error}"
            )

        return "\n".join(lines)


class CompilerCLI:
    PROMPT = "compoiler> "

    HELP_TEXT = """\
Available commands:

  help
      Show this command list.

  summary
      Show the Phase 2 and Phase 3 summaries.

  all
      Show CFGs, data-flow, call graph and dead-code results.

  tokens
      Show lexer tokens.

  ast
      Show the parsed AST.

  diagnostics
      Show syntax, semantic and type diagnostics.

  scope
      Show the complete scope tree.

  complete <line> <column>
      Show code-completion candidates.

  hover <line> <column>
      Show hover information for the symbol at a position.

  functions
      List all defined functions.

  show-cfg <function>
      Show the CFG of one function.

  dataflow <function>
      Show definite-assignment and liveness results.

  callgraph
      Show the complete program call graph.

  callees <function>
      Show direct callees.

  callers <function>
      Show direct callers.

  reachable <function>
      Show all transitively reachable callees.

  reaching <function>
      Show all functions that can reach the target.

  recursive
      Show recursive functions.

  scc
      Show strongly connected components.

  dead-functions
      Show functions unreachable from the entry function.

  dead-code
      Show all dead-code categories.

  goto-def [file] <line> <column>
      Go to the exact definition of the symbol at a position.

  find-refs <symbol>
      Find references by symbol name. The name must be unambiguous.

  find-refs-at <line> <column>
      Find references using a source position.

  rename <symbol> <new-name> [--save <path>]
      Safely rename an unambiguous symbol.

  rename-at <line> <column> <new-name> [--save <path>]
      Safely rename the symbol at a source position.

  callgraph-dot [path]
      Print Graphviz DOT or save it to a file.

  source
      Show the current source code with line numbers.

  load <path>
      Load and analyze another source file.

  reload
      Reload the current source file from disk.

  exit
  quit
      Leave the CLI.
"""

    def __init__(
        self,
        session: AnalysisSession,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        if not isinstance(session, AnalysisSession):
            raise TypeError(
                "session must be an AnalysisSession"
            )

        self.session = session
        self.input_stream = input_stream
        self.output_stream = output_stream

    @classmethod
    def from_source(
        cls,
        source: str,
        filename: str = "<input>",
        entry_function: str = "main",
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> "CompilerCLI":
        return cls(
            session=AnalysisSession.from_source(
                source=source,
                filename=filename,
                entry_function=entry_function,
            ),
            input_stream=input_stream,
            output_stream=output_stream,
        )

    @classmethod
    def from_phase3_result(
        cls,
        result: PhaseThreeResult,
        tokens: Optional[Sequence[object]] = None,
        parser_errors: Optional[Sequence[str]] = None,
        lexer_diagnostics: Optional[Sequence[Diagnostic]] = None,
        parser_diagnostics: Optional[Sequence[Diagnostic]] = None,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> "CompilerCLI":
        return cls(
            session=AnalysisSession.from_phase3_result(
                result=result,
                tokens=tokens,
                parser_errors=parser_errors,
                lexer_diagnostics=lexer_diagnostics,
                parser_diagnostics=parser_diagnostics,
            ),
            input_stream=input_stream,
            output_stream=output_stream,
        )

    def run(self, show_banner: bool = True) -> None:
        if show_banner:
            self._write(self.session.startup_summary())
            self._write(
                "Type 'help' to see the available commands."
            )

        while True:
            try:
                self.output_stream.write(self.PROMPT)
                self.output_stream.flush()
                line = self.input_stream.readline()
            except KeyboardInterrupt:
                self._write("\nUse 'exit' to leave the CLI.")
                continue

            if line == "":
                self._write("")
                break

            keep_running = self.execute(line)
            if not keep_running:
                break

    def execute(self, command_line: str) -> bool:
        try:
            parts = shlex.split(command_line)
        except ValueError as error:
            self._write(f"Command parse error: {error}")
            return True

        if not parts:
            return True

        command = parts[0].lower()
        arguments = parts[1:]

        aliases = {
            "?": "help",
            "q": "quit",
            "refs": "find-refs",
            "cfg": "show-cfg",
            "df": "dataflow",
            "goto": "goto-def",
        }
        command = aliases.get(command, command)

        if command in {"exit", "quit"}:
            self._write("Goodbye.")
            return False

        handlers = {
            "help": self._cmd_help,
            "summary": self._cmd_summary,
            "all": self._cmd_all,
            "tokens": self._cmd_tokens,
            "ast": self._cmd_ast,
            "diagnostics": self._cmd_diagnostics,
            "scope": self._cmd_scope,
            "complete": self._cmd_complete,
            "hover": self._cmd_hover,
            "functions": self._cmd_functions,
            "show-cfg": self._cmd_show_cfg,
            "dataflow": self._cmd_dataflow,
            "callgraph": self._cmd_callgraph,
            "callees": self._cmd_callees,
            "callers": self._cmd_callers,
            "reachable": self._cmd_reachable,
            "reaching": self._cmd_reaching,
            "recursive": self._cmd_recursive,
            "scc": self._cmd_scc,
            "dead-functions": self._cmd_dead_functions,
            "dead-code": self._cmd_dead_code,
            "goto-def": self._cmd_goto_definition,
            "find-refs": self._cmd_find_references,
            "find-refs-at": self._cmd_find_references_at,
            "rename": self._cmd_rename,
            "rename-at": self._cmd_rename_at,
            "callgraph-dot": self._cmd_callgraph_dot,
            "source": self._cmd_source,
            "load": self._cmd_load,
            "reload": self._cmd_reload,
        }

        handler = handlers.get(command)
        if handler is None:
            self._write(
                f"Unknown command: {command}. "
                "Type 'help' for the command list."
            )
            return True

        try:
            handler(arguments)
        except (
            ValueError,
            TypeError,
            KeyError,
            OSError,
        ) as error:
            message = (
                error.args[0]
                if isinstance(error, KeyError)
                and error.args
                else str(error)
            )
            self._write(f"Error: {message}")
        except Exception as error:
            self._write(
                "Command failed safely: "
                f"{error.__class__.__name__}: {error}"
            )

        return True

    def _cmd_help(self, arguments: Sequence[str]) -> None:
        self._require_count(arguments, 0, "help")
        self._write(self.HELP_TEXT.rstrip())

    def _cmd_summary(self, arguments: Sequence[str]) -> None:
        self._require_count(arguments, 0, "summary")
        self._write(self.session.phase2_result.summary())

        result = self._phase3()
        if result is not None:
            self._write("")
            self._write(result.summary())

    def _cmd_all(self, arguments: Sequence[str]) -> None:
        self._require_count(arguments, 0, "all")
        result = self._require_phase3()
        self._write(result.format_all())

    def _cmd_tokens(self, arguments: Sequence[str]) -> None:
        self._require_count(arguments, 0, "tokens")

        if not self.session.tokens:
            self._write("Token list is not available.")
            return

        for token in self.session.tokens:
            self._write(str(token))

    def _cmd_ast(self, arguments: Sequence[str]) -> None:
        self._require_count(arguments, 0, "ast")
        self._write(str(self.session.program))

    def _cmd_diagnostics(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(arguments, 0, "diagnostics")

        self._write("Lexical Diagnostics:")
        if self.session.lexer_diagnostics:
            for diagnostic in self.session.lexer_diagnostics:
                self._write(f"  {diagnostic}")
        else:
            self._write("  None")

        self._write("\nSyntax Diagnostics:")
        if self.session.parser_diagnostics:
            for diagnostic in self.session.parser_diagnostics:
                self._write(f"  {diagnostic}")
        elif self.session.parser_errors:
            for error in self.session.parser_errors:
                self._write(f"  {error}")
        else:
            self._write("  None")

        self._write("\nSemantic and Type Diagnostics:")
        if len(self.session.phase2_result.diagnostics) == 0:
            self._write("  None")
        else:
            self._write(
                self.session.phase2_result.format_diagnostics()
            )

        if self.session.phase3_error is not None:
            self._write("\nPhase Three Diagnostics:")
            self._write(
                f"  {self.session.phase3_error}"
            )

    def _cmd_scope(self, arguments: Sequence[str]) -> None:
        self._require_count(arguments, 0, "scope")
        self._write(
            self.session.phase2_result.format_scope_tree()
        )

    def _cmd_complete(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            2,
            "complete <line> <column>",
        )
        line, column = self._parse_position(arguments)

        items = self.session.phase2_result.complete(
            line=line,
            column=column,
        )

        if not items:
            self._write("No completion items.")
            return

        for item in items:
            self._write(str(item))

    def _cmd_hover(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            2,
            "hover <line> <column>",
        )
        line, column = self._parse_position(arguments)

        hover = self.session.phase2_result.hover(
            line=line,
            column=column,
        )
        self._write(
            str(hover)
            if hover is not None
            else "No hover information."
        )

    def _cmd_functions(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(arguments, 0, "functions")
        result = self._require_phase3()

        if not result.function_names:
            self._write("No functions found.")
            return

        for name in result.function_names:
            self._write(name)

    def _cmd_show_cfg(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            1,
            "show-cfg <function>",
        )
        result = self._require_phase3()
        self._write(
            result.cfg_for(arguments[0]).format()
        )

    def _cmd_dataflow(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            1,
            "dataflow <function>",
        )
        result = self._require_phase3()
        self._write(
            result.dataflow_for(arguments[0]).format()
        )

    def _cmd_callgraph(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(arguments, 0, "callgraph")
        self._write(
            self._require_phase3().format_call_graph()
        )

    def _cmd_callees(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._callgraph_query(
            arguments,
            "callees <function>",
            "Direct callees",
            "direct_callees",
        )

    def _cmd_callers(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._callgraph_query(
            arguments,
            "callers <function>",
            "Direct callers",
            "direct_callers",
        )

    def _cmd_reachable(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._callgraph_query(
            arguments,
            "reachable <function>",
            "Reachable callees",
            "reachable_callees",
        )

    def _cmd_reaching(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._callgraph_query(
            arguments,
            "reaching <function>",
            "Reaching callers",
            "reaching_callers",
        )

    def _cmd_recursive(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(arguments, 0, "recursive")
        names = (
            self._require_phase3()
            .call_graph
            .recursive_functions()
        )
        self._write_name_list(
            "Recursive functions",
            names,
        )

    def _cmd_scc(self, arguments: Sequence[str]) -> None:
        self._require_count(arguments, 0, "scc")
        components = (
            self._require_phase3()
            .call_graph
            .strongly_connected_components()
        )

        if not components:
            self._write("No strongly connected components.")
            return

        for component in components:
            self._write(
                "{" + ", ".join(component) + "}"
            )

    def _cmd_dead_functions(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            0,
            "dead-functions",
        )
        result = self._require_phase3()
        names = result.call_graph.dead_functions(
            result.entry_function
        )
        self._write_name_list(
            "Dead functions",
            names,
        )

    def _cmd_dead_code(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(arguments, 0, "dead-code")
        self._write(
            self._require_phase3().format_dead_code()
        )

    def _cmd_goto_definition(
        self,
        arguments: Sequence[str],
    ) -> None:
        if len(arguments) == 2:
            line, column = self._parse_position(arguments)
        elif len(arguments) == 3:
            requested_file = arguments[0]
            self._require_current_file(requested_file)
            line, column = self._parse_position(
                arguments[1:]
            )
        else:
            raise ValueError(
                "Usage: goto-def [file] <line> <column>"
            )

        target = self._require_phase3().goto_definition(
            line=line,
            column=column,
        )
        self._write(
            str(target)
            if target is not None
            else "No symbol found at that position."
        )

    def _cmd_find_references(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            1,
            "find-refs <symbol>",
        )
        result = self._require_phase3()
        symbol = self._resolve_unique_symbol(
            arguments[0]
        )
        if symbol is None:
            return

        references = (
            result.navigation.references_for_symbol(
                symbol,
                include_definition=True,
            )
        )
        self._write(
            result.format_references(references)
        )

    def _cmd_find_references_at(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            2,
            "find-refs-at <line> <column>",
        )
        line, column = self._parse_position(arguments)
        result = self._require_phase3()
        references = result.find_references(
            line=line,
            column=column,
            include_definition=True,
        )
        self._write(
            result.format_references(references)
        )

    def _cmd_rename(
        self,
        arguments: Sequence[str],
    ) -> None:
        positional, save_path = self._parse_save_option(
            arguments
        )
        self._require_count(
            positional,
            2,
            "rename <symbol> <new-name> "
            "[--save <path>]",
        )

        symbol = self._resolve_unique_symbol(
            positional[0]
        )
        if symbol is None:
            return

        self._perform_rename(
            line=symbol.definition_loc.line,
            column=symbol.definition_loc.column,
            new_name=positional[1],
            save_path=save_path,
        )

    def _cmd_rename_at(
        self,
        arguments: Sequence[str],
    ) -> None:
        positional, save_path = self._parse_save_option(
            arguments
        )
        self._require_count(
            positional,
            3,
            "rename-at <line> <column> <new-name> "
            "[--save <path>]",
        )

        line, column = self._parse_position(
            positional[:2]
        )
        self._perform_rename(
            line=line,
            column=column,
            new_name=positional[2],
            save_path=save_path,
        )

    def _cmd_callgraph_dot(
        self,
        arguments: Sequence[str],
    ) -> None:
        if len(arguments) > 1:
            raise ValueError(
                "Usage: callgraph-dot [path]"
            )

        dot = self._require_phase3().call_graph.to_dot()

        if not arguments:
            self._write(dot)
            return

        path = Path(arguments[0])
        path.write_text(dot, encoding="utf-8")
        self._write(
            f"Call graph DOT saved to: {path}"
        )

    def _cmd_source(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(arguments, 0, "source")

        lines = self.session.source.splitlines()
        width = max(1, len(str(len(lines))))

        for number, line in enumerate(lines, start=1):
            self._write(
                f"{number:>{width}} | {line}"
            )

    def _cmd_load(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(
            arguments,
            1,
            "load <path>",
        )
        self.session = AnalysisSession.from_file(
            arguments[0],
            entry_function=self.session.entry_function,
        )
        self._write(self.session.startup_summary())

    def _cmd_reload(
        self,
        arguments: Sequence[str],
    ) -> None:
        self._require_count(arguments, 0, "reload")
        path = Path(self.session.filename)

        if not path.is_file():
            raise ValueError(
                "The current session was created from "
                "in-memory source and cannot be reloaded."
            )

        self.session = AnalysisSession.from_file(
            path,
            entry_function=self.session.entry_function,
        )
        self._write(self.session.startup_summary())

    def _callgraph_query(
        self,
        arguments: Sequence[str],
        usage: str,
        label: str,
        method_name: str,
    ) -> None:
        self._require_count(arguments, 1, usage)
        graph = self._require_phase3().call_graph
        method = getattr(graph, method_name)
        names = method(arguments[0])
        self._write_name_list(label, names)

    def _perform_rename(
        self,
        line: int,
        column: int,
        new_name: str,
        save_path: Optional[str],
    ) -> None:
        plan = self._require_phase3().rename(
            line=line,
            column=column,
            new_name=new_name,
        )
        self._write(plan.format())

        if not plan.applied:
            return

        self._write("\nUpdated Source:")
        self._write(plan.updated_source or "")

        if save_path is not None:
            path = Path(save_path)
            path.write_text(
                plan.updated_source or "",
                encoding="utf-8",
            )
            self._write(
                f"\nRenamed source saved to: {path}"
            )

    def _resolve_unique_symbol(
        self,
        name: str,
    ) -> Optional[Symbol]:
        matches = sorted(
            self._symbols_named(name),
            key=lambda symbol: (
                symbol.definition_loc.file,
                symbol.definition_loc.line,
                symbol.definition_loc.column,
                symbol.kind.value,
            ),
        )

        if not matches:
            self._write(
                f"No symbol named '{name}' was found."
            )
            return None

        if len(matches) > 1:
            self._write(
                f"Symbol name '{name}' is ambiguous:"
            )
            for symbol in matches:
                scope_name = (
                    symbol.scope.full_name
                    if isinstance(symbol.scope, Scope)
                    else "unknown"
                )
                self._write(
                    f"  {symbol.kind.value} at "
                    f"{symbol.definition_loc} "
                    f"[scope={scope_name}]"
                )
            self._write(
                "Use a position-based command such as "
                "'find-refs-at' or 'rename-at'."
            )
            return None

        return matches[0]

    def _symbols_named(
        self,
        name: str,
    ) -> Iterable[Symbol]:
        stack = [
            self.session.phase2_result.global_scope
        ]

        while stack:
            scope = stack.pop()
            symbol = scope.resolve_local(name)
            if symbol is not None:
                yield symbol
            stack.extend(reversed(scope.children))

    def _require_current_file(
        self,
        requested: str,
    ) -> None:
        actual = Path(self.session.filename)

        if (
            requested != self.session.filename
            and Path(requested).name != actual.name
        ):
            raise ValueError(
                f"The current file is '{self.session.filename}', "
                f"not '{requested}'."
            )

    def _phase3(self) -> Optional[PhaseThreeResult]:
        if self.session.phase3_result is None:
            self._write(
                "Phase Three results are unavailable."
            )
            if self.session.phase3_error is not None:
                self._write(
                    self.session.phase3_error
                )
        return self.session.phase3_result

    def _require_phase3(self) -> PhaseThreeResult:
        result = self._phase3()
        if result is None:
            raise ValueError(
                "Phase Three analysis is unavailable."
            )
        return result

    @staticmethod
    def _parse_position(
        arguments: Sequence[str],
    ) -> Tuple[int, int]:
        if len(arguments) != 2:
            raise ValueError(
                "A position requires <line> <column>."
            )

        try:
            line = int(arguments[0])
            column = int(arguments[1])
        except ValueError as error:
            raise ValueError(
                "Line and column must be integers."
            ) from error

        if line < 1 or column < 1:
            raise ValueError(
                "Line and column must be positive."
            )

        return line, column

    @staticmethod
    def _parse_save_option(
        arguments: Sequence[str],
    ) -> Tuple[List[str], Optional[str]]:
        values = list(arguments)

        if "--save" not in values:
            return values, None

        index = values.index("--save")
        if index + 1 >= len(values):
            raise ValueError(
                "--save requires an output path."
            )
        if index + 2 != len(values):
            raise ValueError(
                "--save must be the final option."
            )

        save_path = values[index + 1]
        positional = values[:index]
        return positional, save_path

    @staticmethod
    def _require_count(
        arguments: Sequence[str],
        expected: int,
        usage: str,
    ) -> None:
        if len(arguments) != expected:
            raise ValueError(f"Usage: {usage}")

    def _write_name_list(
        self,
        label: str,
        names: Sequence[str],
    ) -> None:
        self._write(
            f"{label}: "
            + (
                ", ".join(names)
                if names
                else "None"
            )
        )

    def _write(self, text: str) -> None:
        self.output_stream.write(str(text) + "\n")
        self.output_stream.flush()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive compiler and IDE-analysis CLI."
        )
    )
    parser.add_argument(
        "source_file",
        help="Path to the source file to analyze.",
    )
    parser.add_argument(
        "--entry",
        default="main",
        help="Program entry function (default: main).",
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        help=(
            "Execute a command and exit. "
            "May be supplied more than once."
        ),
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        session = AnalysisSession.from_file(
            arguments.source_file,
            entry_function=arguments.entry,
        )
    except OSError as error:
        parser.error(str(error))

    cli = CompilerCLI(session)
    cli._write(session.startup_summary())

    if arguments.command:
        for command in arguments.command:
            if not cli.execute(command):
                break
        return 0

    cli._write(
        "Type 'help' to see the available commands."
    )
    cli.run(show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
