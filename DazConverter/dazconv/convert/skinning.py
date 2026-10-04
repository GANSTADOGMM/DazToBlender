"""DSON skin_binding modifiers -> IR skin weights.

Daz stores weights in one of two ways:

* ``node_weights`` -- a "general" weight map, one weight per vertex per bone.
* ``local_weights`` -- TriAx weights, separate maps for rotation around each
  of x/y/z.  Game engines have no equivalent, so we collapse them by taking
  the maximum per vertex (the convention used by every Daz importer), which
  preserves the influence region.

Weights are then pruned to a maximum influence count and renormalised.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ir import SkinWeights

MAX_INFLUENCES_DEFAULT = 8


def _weight_pairs(block: Any) -> Iterable[Tuple[int, float]]:
    if not isinstance(block, dict):
        return []
    values = block.get("values")
    if not isinstance(values, list):
        return []
    out = []
    for pair in values:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            try:
                out.append((int(pair[0]), float(pair[1])))
            except (TypeError, ValueError):
                continue
    return out


def joint_weight_map(joint: Dict[str, Any]) -> Dict[int, float]:
    """Return {vertex_index: weight} for a single skin joint entry."""
    weights: Dict[int, float] = {}
    node_weights = joint.get("node_weights")
    if node_weights:
        for vi, w in _weight_pairs(node_weights):
            if w > 0.0:
                weights[vi] = max(weights.get(vi, 0.0), w)
        if weights:
            return weights

    local = joint.get("local_weights")
    if isinstance(local, dict):
        for axis in ("x", "y", "z"):
            for vi, w in _weight_pairs(local.get(axis)):
                if w > 0.0:
                    weights[vi] = max(weights.get(vi, 0.0), w)
    if weights:
        return weights

    scale = joint.get("scale_weights")
    if scale:
        for vi, w in _weight_pairs(scale):
            if w > 0.0:
                weights[vi] = max(weights.get(vi, 0.0), w)
    return weights


def parse_skin_binding(skin: Dict[str, Any],
                       vertex_count: int,
                       bone_filter: Optional[Sequence[str]] = None,
                       max_influences: int = MAX_INFLUENCES_DEFAULT,
                       ) -> Tuple[SkinWeights, List[str]]:
    """Convert a DSON ``skin`` block into normalised per-vertex influences.

    Returns (SkinWeights, warnings).
    """
    warnings: List[str] = []
    joints = skin.get("joints")
    if not isinstance(joints, list):
        return SkinWeights(), ["skin binding has no joints"]

    allowed = set(bone_filter) if bone_filter is not None else None
    bone_names: List[str] = []
    raw: List[Dict[int, float]] = []

    for joint in joints:
        if not isinstance(joint, dict):
            continue
        name = _joint_name(joint)
        if not name:
            continue
        if allowed is not None and name not in allowed:
            warnings.append("weights reference unknown bone '%s'" % name)
            continue
        wmap = joint_weight_map(joint)
        if not wmap:
            continue
        bone_names.append(name)
        raw.append(wmap)

    influences: List[List[Tuple[int, float]]] = [[] for _ in range(vertex_count)]
    for bi, wmap in enumerate(raw):
        for vi, w in wmap.items():
            if 0 <= vi < vertex_count:
                influences[vi].append((bi, w))

    skin_weights = SkinWeights(bone_names=bone_names, influences=influences)
    prune_and_normalize(skin_weights, max_influences)
    unweighted = sum(1 for inf in skin_weights.influences if not inf)
    if unweighted:
        warnings.append("%d vertices have no skin weights" % unweighted)
    return skin_weights, warnings


def _joint_name(joint: Dict[str, Any]) -> str:
    node = joint.get("node")
    if isinstance(node, str) and node:
        from ..dson import uri as uri_mod
        u = uri_mod.parse(node)
        if u.asset_id:
            return u.asset_id
        return node.lstrip("#")
    jid = joint.get("id")
    return str(jid) if jid else ""


def prune_and_normalize(skin: SkinWeights, max_influences: int = MAX_INFLUENCES_DEFAULT) -> None:
    """Keep the strongest N influences per vertex and make them sum to 1."""
    for vi, inf in enumerate(skin.influences):
        if not inf:
            continue
        if len(inf) > max_influences:
            inf = sorted(inf, key=lambda p: -p[1])[:max_influences]
        total = sum(w for _, w in inf)
        if total <= 0.0:
            skin.influences[vi] = []
            continue
        skin.influences[vi] = [(b, w / total) for b, w in inf]


def fallback_rigid_bind(skin: SkinWeights, bone_index: int) -> None:
    """Assign every unweighted vertex fully to one bone (keeps meshes attached)."""
    for vi, inf in enumerate(skin.influences):
        if not inf:
            skin.influences[vi] = [(bone_index, 1.0)]


def remap_bones(skin: SkinWeights, mapping: Dict[int, int], new_names: List[str]) -> None:
    """Rewrite bone indices (used when merging meshes onto one skeleton)."""
    out: List[List[Tuple[int, float]]] = []
    for inf in skin.influences:
        row = []
        for bi, w in inf:
            nb = mapping.get(bi)
            if nb is not None:
                row.append((nb, w))
        out.append(row)
    skin.influences = out
    skin.bone_names = new_names


def used_bone_names(skin: SkinWeights) -> List[str]:
    used = {bi for inf in skin.influences for bi, _ in inf}
    return [n for i, n in enumerate(skin.bone_names) if i in used]
