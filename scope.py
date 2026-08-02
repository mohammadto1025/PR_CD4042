from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional

from symbol import Symbol


class ScopeKind(str, Enum):
    GLOBAL = "global"
    FUNCTION = "function"
    BLOCK = "block"
    STRUCT = "struct"
    CLASS = "class"


@dataclass(frozen=True)
class DeclarationResult:
    success: bool
    symbol: Symbol
    duplicate: Optional[Symbol] = None
    shadowed: Optional[Symbol] = None

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate is not None

    @property
    def causes_shadowing(self) -> bool:
        return self.shadowed is not None


@dataclass
class Scope:
    name: str
    kind: ScopeKind
    parent: Optional["Scope"] = None
    symbols: Dict[str, Symbol] = field(default_factory=dict, init=False)
    children: List["Scope"] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Scope name cannot be empty")
        if not isinstance(self.kind, ScopeKind):
            raise TypeError("kind must be an instance of ScopeKind")

        if self.parent is None and self.kind != ScopeKind.GLOBAL:
            raise ValueError("Only a global scope may have no parent")

        if self.parent is not None:
            self.parent.children.append(self)

    def create_child(self, name: str, kind: ScopeKind) -> "Scope":
        return Scope(name=name, kind=kind, parent=self)

    def declare(self, symbol: Symbol) -> DeclarationResult:
        if not isinstance(symbol, Symbol):
            raise TypeError("symbol must be an instance of Symbol")

        duplicate = self.resolve_local(symbol.name)
        if duplicate is not None:
            return DeclarationResult(
                success=False,
                symbol=symbol,
                duplicate=duplicate,
            )

        shadowed = self.parent.resolve(symbol.name) if self.parent is not None else None
        self.symbols[symbol.name] = symbol
        symbol.scope = self

        return DeclarationResult(
            success=True,
            symbol=symbol,
            shadowed=shadowed,
        )

    def resolve_local(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name)

    def resolve(self, name: str) -> Optional[Symbol]:
        scope: Optional[Scope] = self
        while scope is not None:
            symbol = scope.resolve_local(name)
            if symbol is not None:
                return symbol
            scope = scope.parent
        return None

    def visible_symbols(self) -> List[Symbol]:
        visible: List[Symbol] = []
        seen_names = set()

        scope: Optional[Scope] = self
        while scope is not None:
            for symbol in scope.symbols.values():
                if symbol.name not in seen_names:
                    visible.append(symbol)
                    seen_names.add(symbol.name)
            scope = scope.parent

        return visible

    def ancestors(self, include_self: bool = True) -> Iterator["Scope"]:
        scope: Optional[Scope] = self if include_self else self.parent
        while scope is not None:
            yield scope
            scope = scope.parent

    @property
    def depth(self) -> int:
        return sum(1 for _ in self.ancestors()) - 1

    @property
    def full_name(self) -> str:
        names = [scope.name for scope in self.ancestors()]
        return "::".join(reversed(names))

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "full_name": self.full_name,
            "symbols": [symbol.to_dict() for symbol in self.symbols.values()],
            "children": [child.to_dict() for child in self.children],
        }

    def format_tree(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}{self.kind.value.title()} Scope [{self.name}]"]

        if self.symbols:
            for symbol in self.symbols.values():
                lines.append(f"{prefix}  {symbol}")
        else:
            lines.append(f"{prefix}  (no declarations)")

        for child in self.children:
            lines.append(child.format_tree(indent + 1))

        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return self.resolve_local(name) is not None

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self.symbols.values())

    def __str__(self) -> str:
        return self.format_tree()
