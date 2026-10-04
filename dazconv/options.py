"""Export/conversion options shared by the CLI, the web app and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

FBX_ASCII_VERSION = 7400


@dataclass
class ExportOptions:
    """Everything that changes how a scene is written to FBX."""

    # geometry
    triangulate: bool = False
    smooth_normals: bool = True
    scale: float = 1.0
    unit_scale: float = 1.0          # FBX unit in centimetres; Daz is already cm
    up_axis: str = "y"               # 'y' (Daz/FBX default) or 'z'

    # uv / materials
    flip_uv: bool = True
    export_materials: bool = True
    export_textures: bool = True
    copy_textures: bool = False
    embed_textures: bool = False

    # rig / skin
    export_skeleton: bool = True
    export_weights: bool = True
    max_influences: int = 4
    #: Influences lighter than this are dropped before renormalisation.
    min_weight: float = 1e-5
    split_by_material: bool = False

    # morphs
    export_blendshapes: bool = False
    morph_selections: List[Tuple[str, float]] = field(default_factory=list)
    #: Folders scanned for extra morph assets (Daz keeps them under ``Morphs/``).
    morph_search_dirs: List[str] = field(default_factory=list)
    #: Individual morph files to load, for morphs that live outside those folders.
    morph_files: List[str] = field(default_factory=list)

    # behaviour
    apply_pose: bool = False         # use posed channel values instead of bind pose
    fail_on_warning: bool = False

    def validate(self) -> List[str]:
        problems = []
        if self.max_influences < 1:
            problems.append("max_influences must be at least 1")
        if self.scale <= 0:
            problems.append("scale must be greater than zero")
        if self.unit_scale <= 0:
            problems.append("unit_scale must be greater than zero")
        if self.up_axis not in ("y", "z"):
            problems.append("up_axis must be 'y' or 'z'")
        for morph_id, weight in self.morph_selections:
            if not isinstance(weight, (int, float)):
                problems.append(f"morph '{morph_id}' has a non-numeric weight")
        return problems

    def as_dict(self) -> dict:
        return {
            "triangulate": self.triangulate,
            "smooth_normals": self.smooth_normals,
            "scale": self.scale,
            "unit_scale": self.unit_scale,
            "up_axis": self.up_axis,
            "flip_uv": self.flip_uv,
            "export_materials": self.export_materials,
            "export_textures": self.export_textures,
            "copy_textures": self.copy_textures,
            "embed_textures": self.embed_textures,
            "export_skeleton": self.export_skeleton,
            "export_weights": self.export_weights,
            "max_influences": self.max_influences,
            "min_weight": self.min_weight,
            "split_by_material": self.split_by_material,
            "export_blendshapes": self.export_blendshapes,
            "morphs": [{"id": m, "weight": w} for m, w in self.morph_selections],
            "morph_search_dirs": list(self.morph_search_dirs),
            "morph_files": list(self.morph_files),
            "apply_pose": self.apply_pose,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ExportOptions":
        """Build options from a JSON-ish dict, ignoring unknown keys."""
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        # "morphs" is the JSON alias for the morph_selections field.
        allowed.add("morphs")
        kwargs = {}
        for key, value in data.items():
            if key not in allowed:
                continue
            if key == "morphs" and isinstance(value, list):
                kwargs["morph_selections"] = [
                    (str(item.get("id", "")), float(item.get("weight", 1.0)))
                    for item in value
                    if isinstance(item, dict) and item.get("id")
                ]
            elif key in ("morph_selections",) and isinstance(value, list):
                kwargs["morph_selections"] = [
                    (str(entry[0]), float(entry[1])) for entry in value if len(entry) >= 2
                ]
            elif isinstance(value, bool) or value is None:
                kwargs[key] = bool(value)
            elif key in ("max_influences",):
                kwargs[key] = int(value)
            elif key in ("scale", "unit_scale", "min_weight"):
                kwargs[key] = float(value)
            elif isinstance(value, list):
                kwargs[key] = [str(v) for v in value]
            else:
                kwargs[key] = value
        return cls(**kwargs)
