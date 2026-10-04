"""Low-level writer for the FBX 7.4 *binary* container format.

Binary FBX (as opposed to ASCII) is what Blender, Unity, Unreal, Maya, Max and
every FBX SDK based importer actually accept, so the converter writes it
natively -- no Autodesk SDK, no Blender, no external binaries.

File layout::

    "Kaydara FBX Binary  \\x00" \\x1a \\x00   (23 byte magic)
    uint32 version                          (7400)
    <node records...>
    13 zero bytes                           (end-of-record sentinel)
    footer

A node record is::

    uint32 end_offset        absolute offset of the byte after this record
    uint32 num_properties
    uint32 property_list_len bytes of the property block
    uint8  name_len
    char   name[name_len]
    <properties>
    <nested records>         (optional, each terminated by a 13-byte sentinel)
"""

from __future__ import annotations

import struct
import zlib
from typing import Any, BinaryIO, Iterable, List, Optional, Sequence, Tuple

HEADER_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
FOOTER_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
FOOTER_MAGIC = (b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e"
                b"\xec\xe9\x0c\xe3\x75\x8f\x29\x0b")
SENTINEL_LEN = 13
VERSION = 7400

# Name/class separator used inside binary FBX object names ("Body\x00\x01Model")
NAME_CLASS_SEP = b"\x00\x01"

# Arrays larger than this get zlib-compressed (smaller ones aren't worth it).
COMPRESS_THRESHOLD = 128
COMPRESS_LEVEL = 6


def fbx_name(name: str, cls: str) -> bytes:
    """Encode an object name the way binary FBX expects it."""
    return name.encode("utf-8", "replace") + NAME_CLASS_SEP + cls.encode("ascii")


class Node:
    """One FBX node record: a name, a property list and child records."""

    __slots__ = ("name", "props", "prop_types", "children")

    def __init__(self, name: str):
        self.name: bytes = name.encode("ascii")
        self.props: List[Any] = []
        self.prop_types: List[str] = []
        self.children: List["Node"] = []

    # ------------------------------------------------------- property adds
    def add(self, type_code: str, value: Any) -> "Node":
        self.prop_types.append(type_code)
        self.props.append(value)
        return self

    def add_bool(self, v: bool) -> "Node":
        return self.add("C", bool(v))

    def add_int16(self, v: int) -> "Node":
        return self.add("Y", int(v))

    def add_int32(self, v: int) -> "Node":
        return self.add("I", int(v))

    def add_int64(self, v: int) -> "Node":
        return self.add("L", int(v))

    def add_float32(self, v: float) -> "Node":
        return self.add("F", float(v))

    def add_float64(self, v: float) -> "Node":
        return self.add("D", float(v))

    def add_string(self, v: str) -> "Node":
        return self.add("S", v)

    def add_bytes(self, v: bytes) -> "Node":
        return self.add("R", v)

    def add_raw_name(self, v: bytes) -> "Node":
        """Add an already-encoded Name\\x00\\x01Class string."""
        return self.add("S", v)

    def add_int32_array(self, v: Sequence[int]) -> "Node":
        return self.add("i", v)

    def add_int64_array(self, v: Sequence[int]) -> "Node":
        return self.add("l", v)

    def add_float32_array(self, v: Sequence[float]) -> "Node":
        return self.add("f", v)

    def add_float64_array(self, v: Sequence[float]) -> "Node":
        return self.add("d", v)

    def add_bool_array(self, v: Sequence[bool]) -> "Node":
        return self.add("b", v)

    # ------------------------------------------------------------ children
    def child(self, name: str) -> "Node":
        node = Node(name)
        self.children.append(node)
        return node

    def append(self, node: "Node") -> "Node":
        self.children.append(node)
        return node

    def has_child(self, name: str) -> bool:
        bname = name.encode("ascii")
        return any(c.name == bname for c in self.children)

    # Convenience: a child holding a single typed property.
    def single(self, name: str, type_code: str, value: Any) -> "Node":
        return self.child(name).add(type_code, value)

    def prop70(self, name: str, ptype: str, subtype: str, flags: str,
               *values: Any) -> "Node":
        """Emit a ``P:`` entry inside a Properties70 block."""
        p = self.child("P")
        p.add_string(name)
        p.add_string(ptype)
        p.add_string(subtype)
        p.add_string(flags)
        for v in values:
            if isinstance(v, bool):
                p.add_int32(1 if v else 0)
            elif isinstance(v, int):
                # KTime and other large values must go out as int64.
                if -2147483648 <= v <= 2147483647 and ptype != "KTime":
                    p.add_int32(v)
                else:
                    p.add_int64(v)
            elif isinstance(v, float):
                p.add_float64(v)
            elif isinstance(v, str):
                p.add_string(v)
            elif isinstance(v, bytes):
                p.add_bytes(v)
        return p


# ------------------------------------------------------------- serialisation
_ARRAY_FORMATS = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}
_ARRAY_ITEM_SIZE = {"f": 4, "d": 8, "l": 8, "i": 4, "b": 1}


def _encode_array(type_code: str, values: Sequence[Any]) -> bytes:
    fmt = _ARRAY_FORMATS[type_code]
    count = len(values)
    if type_code == "b":
        raw = bytes(1 if v else 0 for v in values)
    else:
        raw = struct.pack("<%d%s" % (count, fmt), *values)
    if count > COMPRESS_THRESHOLD:
        comp = zlib.compress(raw, COMPRESS_LEVEL)
        if len(comp) < len(raw):
            return struct.pack("<III", count, 1, len(comp)) + comp
    return struct.pack("<III", count, 0, len(raw)) + raw


def _encode_property(type_code: str, value: Any) -> bytes:
    tc = type_code.encode("ascii")
    if type_code == "C":
        return tc + struct.pack("<?", bool(value))
    if type_code == "Y":
        return tc + struct.pack("<h", int(value))
    if type_code == "I":
        return tc + struct.pack("<i", int(value))
    if type_code == "L":
        return tc + struct.pack("<q", int(value))
    if type_code == "F":
        return tc + struct.pack("<f", float(value))
    if type_code == "D":
        return tc + struct.pack("<d", float(value))
    if type_code == "S":
        data = value if isinstance(value, bytes) else str(value).encode("utf-8", "replace")
        return tc + struct.pack("<I", len(data)) + data
    if type_code == "R":
        data = value if isinstance(value, bytes) else bytes(value)
        return tc + struct.pack("<I", len(data)) + data
    if type_code in _ARRAY_FORMATS:
        return tc + _encode_array(type_code, value)
    raise ValueError("unknown FBX property type %r" % type_code)


def _write_node(out: BinaryIO, node: Node) -> None:
    start = out.tell()
    out.write(b"\x00" * 12)             # placeholder for the 3 offsets
    out.write(struct.pack("<B", len(node.name)))
    out.write(node.name)

    prop_start = out.tell()
    for tc, value in zip(node.prop_types, node.props):
        out.write(_encode_property(tc, value))
    prop_len = out.tell() - prop_start

    for child in node.children:
        _write_node(out, child)
    if node.children:
        out.write(b"\x00" * SENTINEL_LEN)

    end = out.tell()
    out.seek(start)
    out.write(struct.pack("<III", end, len(node.props), prop_len))
    out.seek(end)


def write_document(out: BinaryIO, roots: Iterable[Node],
                   version: int = VERSION) -> None:
    """Serialise a complete FBX document to a binary stream."""
    out.write(HEADER_MAGIC)
    out.write(struct.pack("<I", version))
    for node in roots:
        _write_node(out, node)
    out.write(b"\x00" * SENTINEL_LEN)

    # Footer: id, 16-byte alignment padding, version, 120 zeros, magic.
    out.write(FOOTER_ID)
    pos = out.tell()
    pad = (16 - (pos % 16)) % 16
    out.write(b"\x00" * pad)
    out.write(b"\x00" * 4)
    out.write(struct.pack("<I", version))
    out.write(b"\x00" * 120)
    out.write(FOOTER_MAGIC)


# ------------------------------------------------------------------ reading
class ParsedNode:
    """Minimal parsed representation, used by the test-suite / validators."""

    __slots__ = ("name", "props", "children")

    def __init__(self, name: bytes, props: List[Any], children: List["ParsedNode"]):
        self.name = name
        self.props = props
        self.children = children

    def find(self, name: str) -> Optional["ParsedNode"]:
        bname = name.encode("ascii")
        for c in self.children:
            if c.name == bname:
                return c
        return None

    def find_all(self, name: str) -> List["ParsedNode"]:
        bname = name.encode("ascii")
        return [c for c in self.children if c.name == bname]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<%s props=%d children=%d>" % (self.name.decode(), len(self.props),
                                              len(self.children))


def _read_array(data: bytes, pos: int, type_code: str) -> Tuple[Any, int]:
    count, encoding, comp_len = struct.unpack_from("<III", data, pos)
    pos += 12
    payload = data[pos:pos + comp_len]
    pos += comp_len
    if encoding == 1:
        payload = zlib.decompress(payload)
    fmt = _ARRAY_FORMATS[type_code]
    values = list(struct.unpack("<%d%s" % (count, fmt), payload[:count * _ARRAY_ITEM_SIZE[type_code]]))
    return values, pos


def _read_property(data: bytes, pos: int) -> Tuple[Any, int]:
    tc = chr(data[pos])
    pos += 1
    if tc == "C":
        return bool(data[pos]), pos + 1
    if tc == "Y":
        return struct.unpack_from("<h", data, pos)[0], pos + 2
    if tc == "I":
        return struct.unpack_from("<i", data, pos)[0], pos + 4
    if tc == "L":
        return struct.unpack_from("<q", data, pos)[0], pos + 8
    if tc == "F":
        return struct.unpack_from("<f", data, pos)[0], pos + 4
    if tc == "D":
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if tc in ("S", "R"):
        n = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return data[pos:pos + n], pos + n
    if tc in _ARRAY_FORMATS:
        return _read_array(data, pos, tc)
    raise ValueError("unknown property type %r at %d" % (tc, pos))


def _read_node(data: bytes, pos: int) -> Tuple[Optional[ParsedNode], int]:
    end_offset, num_props, prop_len = struct.unpack_from("<III", data, pos)
    pos += 12
    if end_offset == 0:
        return None, pos + 1  # null record (sentinel)
    name_len = data[pos]
    pos += 1
    name = data[pos:pos + name_len]
    pos += name_len
    props = []
    for _ in range(num_props):
        val, pos = _read_property(data, pos)
        props.append(val)
    children: List[ParsedNode] = []
    while pos < end_offset - SENTINEL_LEN:
        child, pos = _read_node(data, pos)
        if child is None:
            break
        children.append(child)
    return ParsedNode(name, props, children), end_offset


def parse(path_or_bytes) -> ParsedNode:
    """Parse a binary FBX file into a tree (root node name is b'').

    Only used for validation/tests, but it keeps the writer honest.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as fh:
            data = fh.read()
    if not data.startswith(HEADER_MAGIC[:21]):
        raise ValueError("not a binary FBX file")
    version = struct.unpack_from("<I", data, 23)[0]
    pos = 27
    roots: List[ParsedNode] = []
    while pos < len(data) - SENTINEL_LEN:
        node, pos = _read_node(data, pos)
        if node is None:
            break
        roots.append(node)
    root = ParsedNode(b"", [version], roots)
    return root
