"""dazconv -- convert Daz Studio assets (DSON) to FBX, glTF/GLB and OBJ.

A standalone, dependency-free converter: it reads Daz ``.duf``/``.dsf`` files
straight off disk, so Daz Studio does not need to be installed or running.

Typical use::

    from dazconv import convert

    result = convert("~/My DAZ 3D Library/People/Genesis 8/Char.duf",
                     "out/char.fbx")
    print(result.summary)
"""

from .version import __version__
from .api import ConversionResult, convert, convert_scene, inspect_file
from .convert import ConvertOptions

__all__ = [
    "ConversionResult",
    "ConvertOptions",
    "__version__",
    "convert",
    "convert_scene",
    "inspect_file",
]
