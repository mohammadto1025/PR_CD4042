from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


class DiagnosticSeverity(str, Enum):
    ERROR = "Error"
    WARNING = "Warning"
    INFO = "Info"


@dataclass(frozen=True)
class Diagnostic:
    severity: DiagnosticSeverity
    message: str
    file: str
    line: int
    column: int
    length: int = 1

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("Diagnostic message cannot be empty")
        if self.line < 1:
            raise ValueError("Diagnostic line must be at least 1")
        if self.column < 1:
            raise ValueError("Diagnostic column must be at least 1")
        if self.length < 1:
            raise ValueError("Diagnostic length must be at least 1")

    @classmethod
    def from_location(
        cls,
        severity: DiagnosticSeverity,
        message: str,
        loc: Optional[Tuple[int, int]],
        file: str = "<input>",
        length: int = 1,
    ) -> "Diagnostic":
        line, column = loc if loc is not None else (1, 1)
        return cls(severity, message, file, line, column, length)

    @classmethod
    def error(
        cls,
        message: str,
        file: str,
        line: int,
        column: int,
        length: int = 1,
    ) -> "Diagnostic":
        return cls(DiagnosticSeverity.ERROR, message, file, line, column, length)

    @classmethod
    def warning(
        cls,
        message: str,
        file: str,
        line: int,
        column: int,
        length: int = 1,
    ) -> "Diagnostic":
        return cls(DiagnosticSeverity.WARNING, message, file, line, column, length)

    @classmethod
    def info(
        cls,
        message: str,
        file: str,
        line: int,
        column: int,
        length: int = 1,
    ) -> "Diagnostic":
        return cls(DiagnosticSeverity.INFO, message, file, line, column, length)

    def to_dict(self) -> Dict[str, object]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "length": self.length,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity.value}] {self.file}:{self.line}:{self.column} "
            f"(length={self.length}) {self.message}"
        )


class DiagnosticBag:
    def __init__(self) -> None:
        self._items: List[Diagnostic] = []

    def add(self, diagnostic: Diagnostic) -> Diagnostic:
        if not isinstance(diagnostic, Diagnostic):
            raise TypeError("diagnostic must be an instance of Diagnostic")
        self._items.append(diagnostic)
        return diagnostic

    def error(
        self,
        message: str,
        file: str = "<input>",
        line: int = 1,
        column: int = 1,
        length: int = 1,
    ) -> Diagnostic:
        return self.add(Diagnostic.error(message, file, line, column, length))

    def warning(
        self,
        message: str,
        file: str = "<input>",
        line: int = 1,
        column: int = 1,
        length: int = 1,
    ) -> Diagnostic:
        return self.add(Diagnostic.warning(message, file, line, column, length))

    def info(
        self,
        message: str,
        file: str = "<input>",
        line: int = 1,
        column: int = 1,
        length: int = 1,
    ) -> Diagnostic:
        return self.add(Diagnostic.info(message, file, line, column, length))

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        for diagnostic in diagnostics:
            self.add(diagnostic)

    @property
    def errors(self) -> List[Diagnostic]:
        return [item for item in self._items if item.severity == DiagnosticSeverity.ERROR]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [item for item in self._items if item.severity == DiagnosticSeverity.WARNING]

    @property
    def infos(self) -> List[Diagnostic]:
        return [item for item in self._items if item.severity == DiagnosticSeverity.INFO]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def to_list(self) -> List[Dict[str, object]]:
        return [item.to_dict() for item in self._items]

    def clear(self) -> None:
        self._items.clear()

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __str__(self) -> str:
        if not self._items:
            return "No diagnostics."
        return "\n".join(str(item) for item in self._items)
