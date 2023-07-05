from __future__ import annotations

from typing import Any, Dict

_class_type_cache: Dict[type, str] = {}
_fully_qualified_identifier_cache: Dict[str, Any] = {}


def resolve_fully_qualified_identifier(fully_qualified_identifier: str) -> Any:
    return _fully_qualified_identifier_cache[fully_qualified_identifier]


def get_fully_qualified_identifier(cls: type) -> str:
    try:
        return _class_type_cache[cls]
    except KeyError:
        fully_qualified_identifier = f"{cls.__module__}:{cls.__qualname__}"

        _fully_qualified_identifier_cache[fully_qualified_identifier] = cls
        _class_type_cache[cls] = fully_qualified_identifier
        return fully_qualified_identifier
