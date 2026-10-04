"""ASCII FBX 7.4 serialisation primitives.

FBX's ASCII form is a nested ``Name: value { ... }`` document with three node
shapes:

* ``Key: value``                     -- scalar property
* ``Key::id, "name", "type" { ... }`` -- an object in the ``Objects`` section
* ``Key: *N { a: v,v,v }``           -- a packed numeric array

This module builds that tree; :mod:`dazconv.fbx.exporter` fills it in.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

Scalar = Union[str, int, float, bool]


def escape_string(value: str) -> str:
    """Escape a string for the ASCII FBX dialect."""
    text = str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ")


def format_scalar(value: Scalar) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float(value)
    return f'"{escape_string(value)}"'


def format_float(value: float) -> str:
    """Format a float without scientific notation (strict FBX parsers)."""
    if value != value or value in (math.inf, -math.inf):
        return "0"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e-3:
        text = f"{value:.7f}"
    elif magnitude >= 1e-9:
        text = f"{value:.12f}"
    else:
        return "0"
    text = text.rstrip("0").rstrip(".")
    return text if text not in ("", "-", "-0") else "0"


def format_number_list(values: Iterable[float]) -> str:
    return ",".join(format_float(float(v)) for v in values)


class FbxNode:
    """One node of the ASCII FBX tree."""

    __slots__ = ("name", "value", "compound", "array", "children")

    def __init__(
        self,
        name: str,
        value: Optional[Scalar] = None,
        compound: Optional[Tuple] = None,
        array: Optional[Tuple[str, Sequence]] = None,
    ) -> None:
        self.name = name
        self.value = value
        self.compound = compound
        self.array = array
        self.children: List["FbxNode"] = []

    # ------------------------------------------------------------ building --
    def child(
        self,
        name: str,
        value: Optional[Scalar] = None,
        compound: Optional[Tuple] = None,
        array: Optional[Tuple[str, Sequence]] = None,
    ) -> "FbxNode":
        node = FbxNode(name, value=value, compound=compound, array=array)
        self.children.append(node)
        return node

    def numbers(self, name: str, values: Sequence[float]) -> "FbxNode":
        """Add a packed numeric array: ``Name: *N { a: ... }``."""
        return self.child(name, array=("d", values))

    def prop(
        self,
        name: str,
        property_type: str,
        subtype: str = "",
        flags: str = "",
        *values,
    ) -> "FbxNode":
        """Add a ``P:`` property line inside a ``Properties70`` block.

        ``P:`` lines are always flat in ASCII FBX -- even ``Compound``
        properties, whose members are written as further ``P:`` lines with a
        ``Parent|Member`` name rather than nested inside braces.
        """
        head = ", ".join(
            [
                f'"{escape_string(name)}"',
                f'"{escape_string(property_type)}"',
                f'"{escape_string(subtype)}"',
                f'"{escape_string(flags)}"',
            ]
        )
        rendered = []
        for value in values:
            if isinstance(value, str):
                rendered.append(f' "{escape_string(value)}"')
            else:
                rendered.append(format_scalar(value))
        return self.child("P", value=head + "," + ",".join(rendered))

    def properties70(self) -> "FbxNode":
        return self.child("Properties70")

    # -------------------------------------------------------- serialising --
    def header(self) -> str:
        if self.compound is not None:
            object_id, object_name, object_type = self.compound
            if object_id is None:
                # Objects such as SceneInfo carry a name/type but no numeric id.
                return (
                    f'{self.name}: "{escape_string(object_name)}", '
                    f'"{escape_string(object_type)}"'
                )
            return (
                f"{self.name}: {object_id}, "
                f'"{escape_string(object_name)}", "{escape_string(object_type)}"'
            )
        if self.array is not None:
            return f"{self.name}: *{len(self.array[1])}"
        if self.value is not None:
            return f"{self.name}: {self.value}"
        return f"{self.name}: "

    def write(self, out: List[str], indent: int = 0) -> None:
        pad = "\t" * indent
        if self.array is not None:
            kind, values = self.array
            out.append(f"{pad}{self.header()} {{")
            if values:
                if kind == "i":
                    body = ",".join(str(int(v)) for v in values)
                else:
                    body = format_number_list(values)
                # Only the first line carries the "a:" prefix; continuation
                # lines are bare -- repeating "a:" makes FBX readers reject the
                # whole array.
                first = True
                for chunk in _chunk(body, 4096):
                    out.append(f"{pad}\ta: {chunk}" if first else f"{pad}\t{chunk}")
                    first = False
            else:
                out.append(f"{pad}\ta: ")
            out.append(f"{pad}}}")
            return
        if not self.children:
            out.append(f"{pad}{self.header()}")
            return
        out.append(f"{pad}{self.header()} {{")
        for child in self.children:
            child.write(out, indent + 1)
        out.append(f"{pad}}}")

    def serialize(self) -> str:
        lines: List[str] = []
        self.write(lines, 0)
        return "\n".join(lines)


def _chunk(text: str, size: int) -> Iterable[str]:
    """Split a packed array payload into lines without breaking a number.

    FBX readers consume ``a:`` payloads line by line, so every break must land
    immediately after a comma -- splitting mid-number silently corrupts the
    data (``16.5252`` -> ``16`` + ``.5252``).
    """
    start = 0
    total = len(text)
    while start < total:
        end = min(start + size, total)
        if end < total:
            cut = text.rfind(",", start, end)
            if cut <= start:
                nxt = text.find(",", end)
                end = total if nxt < 0 else nxt + 1
            else:
                end = cut + 1
        yield text[start:end]
        start = end


def write_document(nodes: Sequence[FbxNode], header_comment: str = "") -> str:
    lines: List[str] = []
    if header_comment:
        lines.extend(f"; {line}" for line in header_comment.splitlines())
    for node in nodes:
        node.write(lines, 0)
    return "\n".join(lines) + "\n"
