"""Minimal pure-python 3D math (no numpy dependency).

Matrices are row-major 4x4 stored as a flat tuple/list of 16 floats, and are
applied to row vectors:  v' = v * M.  This matches the convention used by
Daz Studio / FBX, which keeps the export code free of transposes.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat4 = List[float]

IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def identity() -> Mat4:
    return list(IDENTITY)


def mat_mul(a: Sequence[float], b: Sequence[float]) -> Mat4:
    """Return a * b (row-vector convention: apply `a` first, then `b`)."""
    out = [0.0] * 16
    for i in range(4):
        ai = i * 4
        a0, a1, a2, a3 = a[ai], a[ai + 1], a[ai + 2], a[ai + 3]
        out[ai + 0] = a0 * b[0] + a1 * b[4] + a2 * b[8] + a3 * b[12]
        out[ai + 1] = a0 * b[1] + a1 * b[5] + a2 * b[9] + a3 * b[13]
        out[ai + 2] = a0 * b[2] + a1 * b[6] + a2 * b[10] + a3 * b[14]
        out[ai + 3] = a0 * b[3] + a1 * b[7] + a2 * b[11] + a3 * b[15]
    return out


def mat_mul_many(*mats: Sequence[float]) -> Mat4:
    result = list(mats[0])
    for m in mats[1:]:
        result = mat_mul(result, m)
    return result


def translation(t: Sequence[float]) -> Mat4:
    m = identity()
    m[12], m[13], m[14] = float(t[0]), float(t[1]), float(t[2])
    return m


def scaling(s: Sequence[float]) -> Mat4:
    m = identity()
    m[0], m[5], m[10] = float(s[0]), float(s[1]), float(s[2])
    return m


def rot_x(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [1, 0, 0, 0,
            0, c, s, 0,
            0, -s, c, 0,
            0, 0, 0, 1]


def rot_y(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [c, 0, -s, 0,
            0, 1, 0, 0,
            s, 0, c, 0,
            0, 0, 0, 1]


def rot_z(a: float) -> Mat4:
    c, s = math.cos(a), math.sin(a)
    return [c, s, 0, 0,
            -s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1]


_AXIS_FN = {"X": rot_x, "Y": rot_y, "Z": rot_z}


def euler_to_mat(angles_deg: Sequence[float], order: str = "XYZ") -> Mat4:
    """Build a rotation matrix from per-axis degrees.

    `order` is the Daz/FBX rotation order string (e.g. "XYZ", "YZX").  The
    first letter is the axis applied first to the vector.
    """
    order = (order or "XYZ").upper()
    rx = _AXIS_FN["X"](angles_deg[0] * DEG2RAD)
    ry = _AXIS_FN["Y"](angles_deg[1] * DEG2RAD)
    rz = _AXIS_FN["Z"](angles_deg[2] * DEG2RAD)
    by_axis = {"X": rx, "Y": ry, "Z": rz}
    m = identity()
    for axis in order:
        m = mat_mul(m, by_axis[axis])
    return m


def mat_to_euler_xyz(m: Sequence[float]) -> Vec3:
    """Inverse of ``euler_to_mat(angles, "XYZ")`` -- returns degrees.

    With the row-vector convention used here, M = Rx * Ry * Rz expands to::

        m[0] = cy*cz   m[1] = cy*sz   m[2]  = -sy
        m[6] = sx*cy   m[10] = cx*cy  m[9]  = -sx*cz + cx*sy*sz
    """
    sy = max(-1.0, min(1.0, -m[2]))
    # Only fall back to the gimbal-lock branch at (near) exact singularity:
    # atan2 stays accurate for very small cos(y), and bones pointing straight
    # down the axis are extremely common in Daz rigs.
    if abs(sy) < 1.0 - 1e-13:
        y = math.asin(sy)
        x = math.atan2(m[6], m[10])
        z = math.atan2(m[1], m[0])
    else:  # gimbal lock: fold x and z together
        y = math.asin(sy)
        x = math.atan2(m[4], m[5]) if sy > 0 else math.atan2(-m[4], m[5])
        z = 0.0
    return (x * RAD2DEG, y * RAD2DEG, z * RAD2DEG)


def transform_point(m: Sequence[float], p: Sequence[float]) -> Vec3:
    x, y, z = p[0], p[1], p[2]
    return (
        x * m[0] + y * m[4] + z * m[8] + m[12],
        x * m[1] + y * m[5] + z * m[9] + m[13],
        x * m[2] + y * m[6] + z * m[10] + m[14],
    )


def transform_vector(m: Sequence[float], p: Sequence[float]) -> Vec3:
    x, y, z = p[0], p[1], p[2]
    return (
        x * m[0] + y * m[4] + z * m[8],
        x * m[1] + y * m[5] + z * m[9],
        x * m[2] + y * m[6] + z * m[10],
    )


def mat_inverse(m: Sequence[float]) -> Mat4:
    """General 4x4 inverse (Cramer).  Falls back to identity if singular."""
    a = list(m)
    inv = [0.0] * 16
    inv[0] = (a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] +
              a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10])
    inv[4] = (-a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] -
              a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10])
    inv[8] = (a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] +
              a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9])
    inv[12] = (-a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] -
               a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9])
    inv[1] = (-a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] -
              a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10])
    inv[5] = (a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] +
              a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10])
    inv[9] = (-a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] -
              a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9])
    inv[13] = (a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] +
               a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9])
    inv[2] = (a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] +
              a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6])
    inv[6] = (-a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] -
              a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6])
    inv[10] = (a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] +
               a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5])
    inv[14] = (-a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] -
               a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5])
    inv[3] = (-a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] -
              a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6])
    inv[7] = (a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] +
              a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6])
    inv[11] = (-a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] -
               a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5])
    inv[15] = (a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] +
               a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5])
    det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12]
    if abs(det) < 1e-16:
        return identity()
    det = 1.0 / det
    return [v * det for v in inv]


def compose(translate: Sequence[float], rot_mat: Sequence[float],
            scale: Sequence[float] = (1.0, 1.0, 1.0)) -> Mat4:
    m = mat_mul(scaling(scale), rot_mat)
    m[12], m[13], m[14] = float(translate[0]), float(translate[1]), float(translate[2])
    return m


def decompose(m: Sequence[float]) -> Tuple[Vec3, Vec3, Vec3]:
    """Return (translation, euler_xyz_degrees, scale)."""
    t = (m[12], m[13], m[14])
    sx = math.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2)
    sy = math.sqrt(m[4] ** 2 + m[5] ** 2 + m[6] ** 2)
    sz = math.sqrt(m[8] ** 2 + m[9] ** 2 + m[10] ** 2)
    sx = sx or 1.0
    sy = sy or 1.0
    sz = sz or 1.0
    r = [m[0] / sx, m[1] / sx, m[2] / sx, 0,
         m[4] / sy, m[5] / sy, m[6] / sy, 0,
         m[8] / sz, m[9] / sz, m[10] / sz, 0,
         0, 0, 0, 1]
    return t, mat_to_euler_xyz(r), (sx, sy, sz)


def normalize(v: Sequence[float]) -> Vec3:
    ln = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if ln < 1e-20:
        return (0.0, 0.0, 0.0)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mat_to_rows(m: Sequence[float]) -> List[List[float]]:
    return [list(m[0:4]), list(m[4:8]), list(m[8:12]), list(m[12:16])]


def column_major(m: Sequence[float]) -> List[float]:
    """Convert our row-major layout to column-major (glTF ordering).

    For row-vector matrices, the glTF/OpenGL column-major array of the
    equivalent column-vector matrix is literally the same 16 numbers.
    """
    return list(m)


def flatten(points: Iterable[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for p in points:
        out.append(float(p[0]))
        out.append(float(p[1]))
        out.append(float(p[2]))
    return out
