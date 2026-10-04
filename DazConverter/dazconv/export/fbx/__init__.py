"""Binary FBX 7.4 support: a self-contained writer and a reader.

``binary`` implements the container format (node records, typed properties,
zlib-compressed arrays) in both directions, so everything written here can be
read back and verified without the FBX SDK.
"""

from . import binary
from .writer import FbxOptions, FbxWriter, write_fbx

__all__ = ["binary", "FbxOptions", "FbxWriter", "write_fbx"]
