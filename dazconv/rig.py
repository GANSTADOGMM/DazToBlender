"""Skeleton reconstruction from a DSON ``node_library``.

Daz node transforms compose as (Daz SDK ``DzNode::getLocalMatrix``)::

    local = T(center_point) * R(orientation) * R(rotation, rotation_order)
            * T(translation) * S(scale * general_scale)

``end_point`` only drives how a bone is drawn; it is not part of the transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .mathutil import (
    IDENTITY,
    Mat4,
    Vec3,
    as_vec3,
    euler_matrix,
    mat4_from_scale,
    mat4_from_translation,
    mat4_mul,
)

CHANNEL_KEYS = ("x", "y", "z")


def read_channel_vector(channels, prefer_current: bool = False) -> Vec3:
    """Read an x/y/z channel triplet such as ``center_point`` or ``rotation``.

    ``prefer_current`` selects ``current_value`` (the posed value) over the
    authored default ``value`` -- used when exporting a posed scene.
    """
    if not channels:
        return (0.0, 0.0, 0.0)
    if isinstance(channels, dict):  # single channel with id/name/value
        channels = [channels]
    values = {"x": 0.0, "y": 0.0, "z": 0.0}
    for channel in channels:
        if not isinstance(channel, dict):
            continue
        key = str(channel.get("id") or "").lower()
        if key not in CHANNEL_KEYS:
            name = str(channel.get("name") or "").lower()
            for prefix in CHANNEL_KEYS:
                if name.startswith(prefix):
                    key = prefix
                    break
        if key not in CHANNEL_KEYS:
            continue
        if prefer_current and "current_value" in channel and channel["current_value"] is not None:
            values[key] = float(channel["current_value"])
        elif "value" in channel and channel["value"] is not None:
            values[key] = float(channel["value"])
        elif "current_value" in channel and channel["current_value"] is not None:
            values[key] = float(channel["current_value"])
    return (values["x"], values["y"], values["z"])


@dataclass
class Bone:
    id: str
    name: str = ""
    label: str = ""
    parent_ref: Optional[str] = None
    node_type: str = "bone"
    rotation_order: str = "XYZ"
    inherits_scale: bool = True

    origin: Vec3 = (0.0, 0.0, 0.0)
    end_point: Vec3 = (0.0, 0.0, 0.0)
    orientation: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    translation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    general_scale: float = 1.0

    local_matrix: Mat4 = IDENTITY
    world_matrix: Mat4 = IDENTITY
    index: int = -1

    parent: Optional["Bone"] = field(default=None, repr=False, compare=False)
    children: List["Bone"] = field(default_factory=list, repr=False, compare=False)

    # ------------------------------------------------------------ loading --
    @classmethod
    def from_dson(cls, data: dict, prefer_current: bool = False) -> "Bone":
        return cls(
            id=data.get("id") or data.get("name") or "bone",
            name=data.get("name") or "",
            label=data.get("label") or "",
            parent_ref=data.get("parent"),
            node_type=data.get("type") or "bone",
            rotation_order=data.get("rotation_order") or "XYZ",
            inherits_scale=bool(data.get("inherits_scale", True)),
            origin=read_channel_vector(data.get("center_point"), prefer_current),
            end_point=read_channel_vector(data.get("end_point"), prefer_current),
            orientation=read_channel_vector(data.get("orientation"), prefer_current),
            rotation=read_channel_vector(data.get("rotation"), prefer_current),
            translation=read_channel_vector(data.get("translation"), prefer_current),
            scale=as_vec3(
                [
                    _channel_value(data.get("scale"), "x", 1.0),
                    _channel_value(data.get("scale"), "y", 1.0),
                    _channel_value(data.get("scale"), "z", 1.0),
                ],
                1.0,
            ),
            general_scale=float(_scalar_channel(data.get("general_scale"), 1.0)),
        )

    # ---------------------------------------------------------- transforms --
    def compute_local_matrix(self) -> Mat4:
        scale = (
            self.scale[0] * self.general_scale,
            self.scale[1] * self.general_scale,
            self.scale[2] * self.general_scale,
        )
        matrix = mat4_from_translation(self.origin)
        matrix = mat4_mul(matrix, euler_matrix(*self.orientation, order=self.rotation_order))
        matrix = mat4_mul(matrix, euler_matrix(*self.rotation, order=self.rotation_order))
        matrix = mat4_mul(matrix, mat4_from_translation(self.translation))
        if scale != (1.0, 1.0, 1.0):
            matrix = mat4_mul(matrix, mat4_from_scale(scale))
        self.local_matrix = matrix
        return matrix

    @property
    def length(self) -> float:
        dx = self.end_point[0] - self.origin[0]
        dy = self.end_point[1] - self.origin[1]
        dz = self.end_point[2] - self.origin[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    @property
    def world_position(self) -> Vec3:
        return (self.world_matrix[3], self.world_matrix[7], self.world_matrix[11])


def _channel_value(channels, axis: str, default: float = 0.0) -> float:
    if not channels:
        return default
    if isinstance(channels, (int, float)):
        return float(channels)
    if isinstance(channels, dict):
        channels = [channels]
    for channel in channels:
        if isinstance(channel, dict) and str(channel.get("id") or "").lower() == axis:
            for key in ("value", "current_value"):
                if channel.get(key) is not None:
                    return float(channel[key])
    return default


def _scalar_channel(channel, default: float = 1.0) -> float:
    if channel is None:
        return default
    if isinstance(channel, (int, float)):
        return float(channel)
    if isinstance(channel, dict):
        for key in ("current_value", "value"):
            if channel.get(key) is not None:
                return float(channel[key])
    return default


@dataclass
class Skeleton:
    bones: List[Bone] = field(default_factory=list)
    roots: List[Bone] = field(default_factory=list)
    by_id: Dict[str, Bone] = field(default_factory=dict)

    @classmethod
    def from_node_library(
        cls, nodes: Sequence[dict], prefer_current: bool = False
    ) -> "Skeleton":
        skeleton = cls()
        for data in nodes:
            bone = Bone.from_dson(data, prefer_current)
            skeleton.bones.append(bone)
            skeleton.by_id[bone.id] = bone
        skeleton.link()
        return skeleton

    def link(self) -> None:
        """Resolve ``parent: "#id"`` references and compute world matrices."""
        self.roots = []
        for index, bone in enumerate(self.bones):
            bone.index = index
            bone.children = []
            parent = self._resolve(bone.parent_ref)
            bone.parent = parent
            if parent is None:
                self.roots.append(bone)
            else:
                parent.children.append(bone)

        for bone in self.bones:
            bone.compute_local_matrix()

        def walk(bone: Bone, parent_world: Mat4) -> None:
            bone.world_matrix = mat4_mul(parent_world, bone.local_matrix)
            for child in bone.children:
                walk(child, bone.world_matrix)

        for root in self.roots:
            walk(root, IDENTITY)

    def _resolve(self, ref: Optional[str]) -> Optional[Bone]:
        if not ref:
            return None
        key = ref.split("#")[-1]
        return self.by_id.get(key)

    # ----------------------------------------------------------- queries --
    def find(self, bone_id: str) -> Optional[Bone]:
        return self.by_id.get(bone_id.split("#")[-1])

    def depth_ordered(self) -> List[Bone]:
        """Bones in a parent-before-child order (required by FBX connections)."""
        ordered: List[Bone] = []
        seen = set()

        def visit(bone: Bone) -> None:
            if bone.id in seen:
                return
            seen.add(bone.id)
            ordered.append(bone)
            for child in bone.children:
                visit(child)

        for root in self.roots:
            visit(root)
        # Orphans (parent ref pointed nowhere) still need to be exported.
        for bone in self.bones:
            if bone.id not in seen:
                seen.add(bone.id)
                ordered.append(bone)
        return ordered

    def stats(self) -> dict:
        return {
            "bone_count": len(self.bones),
            "root_count": len(self.roots),
            "max_depth": max((self._depth(b) for b in self.roots), default=0),
        }

    def _depth(self, bone: Bone) -> int:
        if not bone.children:
            return 1
        return 1 + max(self._depth(c) for c in bone.children)
