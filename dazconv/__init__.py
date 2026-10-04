"""dazconv -- a dependency-free Daz Studio asset to FBX converter.

Public entry points::

    from dazconv import convert_file, SceneLoader, ContentLibrary

Everything here is pure standard-library Python; no Daz Studio, Blender or
third-party packages are required at runtime.
"""

from .convert import ConversionReport, ExportOptions, convert_files, convert_file
from .dson import DsonError, load_dson, load_dson_file
from .fbx.exporter import FbxExportError, export_fbx
from .geometry import Geometry, Polygon
from .library import ContentLibrary
from .materials import Material
from .morphs import Morph, discover_morphs
from .rig import Bone, Skeleton
from .scene import MeshAsset, SceneLoader, SceneModel
from .skinning import SkinBinding, build_influences
from .uvset import UVSet

__version__ = "1.0.0"

__all__ = [
    "Bone",
    "ContentLibrary",
    "ConversionReport",
    "DsonError",
    "ExportOptions",
    "FbxExportError",
    "Geometry",
    "Material",
    "MeshAsset",
    "Morph",
    "Polygon",
    "SceneLoader",
    "SceneModel",
    "Skeleton",
    "SkinBinding",
    "UVSet",
    "__version__",
    "build_influences",
    "convert_file",
    "convert_files",
    "discover_morphs",
    "export_fbx",
    "load_dson",
    "load_dson_file",
]
