"""Skin bindings: per-vertex joint weights from a DSON ``skin`` modifier.

The binding lives in ``modifier_library`` as an object holding a ``skin``
member with ``node``, ``geometry``, ``vertex_count`` and ``joints``.  Each joint
carries ``node_weights`` -- a sparse ``float_indexed_array`` of
``[vertex_index, weight]`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .rig import Skeleton


@dataclass
class WeightedJoint:
    id: str
    node_ref: str = ""
    #: vertex_index -> weight
    weights: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_dson(cls, data: dict) -> "WeightedJoint":
        weights: Dict[int, float] = {}
        for key in ("node_weights",):
            block = data.get(key) or {}
            for entry in block.get("values") or []:
                if len(entry) < 2:
                    continue
                weights[int(entry[0])] = float(entry[1])
        return cls(
            id=data.get("id") or "",
            node_ref=data.get("node") or "",
            weights=weights,
        )

    @property
    def bone_id(self) -> str:
        return self.node_ref.split("#")[-1]


@dataclass
class SkinBinding:
    geometry_ref: str = ""
    node_ref: str = ""
    vertex_count: int = 0
    joints: List[WeightedJoint] = field(default_factory=list)
    source_id: str = ""

    @classmethod
    def from_dson(cls, modifier: dict, source_id: str = "") -> Optional["SkinBinding"]:
        skin = modifier.get("skin")
        if not isinstance(skin, dict):
            return None
        joints = [
            WeightedJoint.from_dson(entry)
            for entry in (skin.get("joints") or [])
            if isinstance(entry, dict)
        ]
        return cls(
            geometry_ref=skin.get("geometry") or "",
            node_ref=skin.get("node") or "",
            vertex_count=int(skin.get("vertex_count") or 0),
            joints=joints,
            source_id=source_id or modifier.get("id") or "",
        )

    @property
    def geometry_id(self) -> str:
        return self.geometry_ref.split("#")[-1]

    @property
    def root_node_id(self) -> str:
        return self.node_ref.split("#")[-1]

    def joint_for_bone(self, bone_id: str) -> Optional[WeightedJoint]:
        for joint in self.joints:
            if joint.bone_id == bone_id:
                return joint
        return None

    # ----------------------------------------------------------- analysis --
    def weight_sums(self) -> Dict[int, float]:
        totals: Dict[int, float] = {}
        for joint in self.joints:
            for vertex, weight in joint.weights.items():
                totals[vertex] = totals.get(vertex, 0.0) + weight
        return totals

    def validate(self, expected_vertices: int) -> List[str]:
        problems: List[str] = []
        if self.vertex_count and self.vertex_count != expected_vertices:
            problems.append(
                f"skin binding '{self.source_id}' declares {self.vertex_count} vertices "
                f"but the geometry has {expected_vertices}"
            )
        totals = self.weight_sums()
        unweighted = expected_vertices - len(totals)
        if unweighted > 0:
            problems.append(
                f"skin binding '{self.source_id}': {unweighted} of {expected_vertices} "
                "vertices have no joint weights"
            )
        drifting = [
            (vertex, total)
            for vertex, total in totals.items()
            if abs(total - 1.0) > 1e-3
        ]
        if drifting:
            worst = max(drifting, key=lambda item: abs(item[1] - 1.0))
            problems.append(
                f"skin binding '{self.source_id}': {len(drifting)} vertices have weights "
                f"that do not sum to 1 (worst: vertex {worst[0]} = {worst[1]:.4f})"
            )
        return problems


@dataclass
class VertexInfluences:
    """Dense per-vertex influence list, ready for FBX cluster assembly."""

    joint_indices: List[List[int]] = field(default_factory=list)
    joint_weights: List[List[float]] = field(default_factory=list)
    dropped_influences: int = 0

    @property
    def max_influences(self) -> int:
        return max((len(w) for w in self.joint_weights), default=0)

    def influence_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for weights in self.joint_weights:
            counts[len(weights)] = counts.get(len(weights), 0) + 1
        return counts


def build_influences(
    binding: SkinBinding,
    skeleton: Skeleton,
    vertex_count: int,
    max_influences: int = 4,
    min_weight: float = 1e-5,
    renormalize: bool = True,
) -> VertexInfluences:
    """Convert a skin binding into per-vertex joint indices + weights.

    ``max_influences`` caps how many joints drive a vertex (FBX/Unity are
    happiest at 4).  Kept weights are renormalised so they still sum to 1.
    """
    # Map joint -> skeleton bone index; joints pointing at unknown nodes are skipped.
    joint_to_bone: List[int] = []
    for joint in binding.joints:
        bone = skeleton.find(joint.bone_id)
        joint_to_bone.append(bone.index if bone is not None else -1)

    per_vertex: List[List[Tuple[int, float]]] = [[] for _ in range(vertex_count)]
    for joint_index, joint in enumerate(binding.joints):
        bone_index = joint_to_bone[joint_index]
        if bone_index < 0:
            continue
        for vertex, weight in joint.weights.items():
            if weight <= min_weight:
                continue
            if 0 <= vertex < vertex_count:
                per_vertex[vertex].append((bone_index, weight))

    influences = VertexInfluences()
    for entries in per_vertex:
        if not entries:
            influences.joint_indices.append([])
            influences.joint_weights.append([])
            continue
        entries.sort(key=lambda item: item[1], reverse=True)
        if max_influences and len(entries) > max_influences:
            influences.dropped_influences += len(entries) - max_influences
            entries = entries[:max_influences]
        total = sum(weight for _, weight in entries)
        if renormalize and total > 0:
            entries = [(bone, weight / total) for bone, weight in entries]
        influences.joint_indices.append([bone for bone, _ in entries])
        influences.joint_weights.append([weight for _, weight in entries])
    return influences
