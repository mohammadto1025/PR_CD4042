from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SymbolKind(str, Enum):
    VARIABLE = "variable"
    FUNCTION = "function"
    TYPE = "type"
    PARAMETER = "parameter"
    STRUCT = "struct"
    FIELD = "field"
    CLASS = "class"
    METHOD = "method"
    CONSTRUCTOR = "constructor"


class ReferenceKind(str, Enum):
    READ = "read"
    WRITE = "write"
    CALL = "call"
    TYPE_USE = "type_use"


@dataclass(frozen=True)
class SourceLocation:
    file: str = "<input>"
    line: int = 1
    column: int = 1
    length: int = 1

    def __post_init__(self) -> None:
        if not self.file:
            raise ValueError("Source file cannot be empty")
        if self.line < 1:
            raise ValueError("Source line must be at least 1")
        if self.column < 1:
            raise ValueError("Source column must be at least 1")
        if self.length < 1:
            raise ValueError("Source length must be at least 1")

    @classmethod
    def from_ast_loc(
        cls,
        loc: Optional[Tuple[int, int]],
        file: str = "<input>",
        length: int = 1,
    ) -> "SourceLocation":
        line, column = loc if loc is not None else (1, 1)
        return cls(file=file, line=line, column=column, length=length)

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "length": self.length,
        }

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass(frozen=True)
class SymbolReference:
    location: SourceLocation
    kind: ReferenceKind = ReferenceKind.READ

    def to_dict(self) -> Dict[str, object]:
        result = self.location.to_dict()
        result["kind"] = self.kind.value
        return result

    def __str__(self) -> str:
        return f"{self.kind.value} at {self.location}"


@dataclass(frozen=True)
class FunctionSignature:
    parameter_types: Tuple[str, ...] = ()
    return_type: str = "void"

    def __post_init__(self) -> None:
        if any(not parameter_type for parameter_type in self.parameter_types):
            raise ValueError("Function parameter types cannot be empty")
        if not self.return_type:
            raise ValueError("Function return type cannot be empty")

    def to_dict(self) -> Dict[str, object]:
        return {
            "parameter_types": list(self.parameter_types),
            "return_type": self.return_type,
        }

    def __str__(self) -> str:
        parameters = ", ".join(self.parameter_types)
        return f"({parameters}) -> {self.return_type}"


@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    type: str
    definition_loc: SourceLocation
    scope: Any = None
    references: List[SymbolReference] = field(default_factory=list)
    signature: Optional[FunctionSignature] = None
    is_initialized: bool = False
    is_used: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Symbol name cannot be empty")
        if not isinstance(self.kind, SymbolKind):
            raise TypeError("kind must be an instance of SymbolKind")
        if not self.type:
            raise ValueError("Symbol type cannot be empty")
        if not isinstance(self.definition_loc, SourceLocation):
            raise TypeError("definition_loc must be an instance of SourceLocation")
        if self.signature is not None and self.kind not in {
            SymbolKind.FUNCTION,
            SymbolKind.METHOD,
            SymbolKind.CONSTRUCTOR,
        }:
            raise ValueError("Only callable symbols can have a signature")

    def add_reference(
        self,
        location: SourceLocation,
        kind: ReferenceKind = ReferenceKind.READ,
    ) -> SymbolReference:
        if not isinstance(location, SourceLocation):
            raise TypeError("location must be an instance of SourceLocation")
        if not isinstance(kind, ReferenceKind):
            raise TypeError("kind must be an instance of ReferenceKind")

        reference = SymbolReference(location=location, kind=kind)
        if reference not in self.references:
            self.references.append(reference)

        if kind in {
            ReferenceKind.READ,
            ReferenceKind.CALL,
            ReferenceKind.TYPE_USE,
        }:
            self.is_used = True

        return reference

    def mark_initialized(self) -> None:
        self.is_initialized = True

    def mark_used(self) -> None:
        self.is_used = True

    @property
    def detail(self) -> str:
        if self.signature is not None:
            return str(self.signature)
        return self.type

    def to_dict(self) -> Dict[str, object]:
        scope_name = None
        if self.scope is not None:
            scope_name = getattr(self.scope, "name", str(self.scope))

        return {
            "name": self.name,
            "kind": self.kind.value,
            "type": self.type,
            "scope": scope_name,
            "definition_loc": self.definition_loc.to_dict(),
            "references": [reference.to_dict() for reference in self.references],
            "signature": self.signature.to_dict() if self.signature else None,
            "is_initialized": self.is_initialized,
            "is_used": self.is_used,
        }

    def __str__(self) -> str:
        return (
            f"[{self.kind.value}] '{self.name}' : {self.detail} "
            f"defined at {self.definition_loc}"
        )
