"""Minimal dependency-free 3D math for the Daz -> FBX converter.

Conventions
-----------
* Right-handed, Y-up (same as Daz Studio and FBX), angles in degrees.
* A matrix is a 16-tuple of floats in **row-major** order, index ``row*4 + col``.
* Matrices transform *column* vectors: ``p' = M * p``.

FBX's own matrix storage is row-vector based (``p' = p * M``), so anything
handed to an FBX writer must be transposed -- see :func:`transpose`.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec3 = tuple
Mat4 = tuple

IDENTITY: Mat4 = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)

_EPS = 1e-12


# ---------------------------------------------------------------- vectors --
def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return (float(x), float(y), float(z))


def as_vec3(value, default: float = 0.0) -> Vec3:
    """Coerce ``None``/short sequences into a 3-tuple of floats."""
    if value is None:
        return (default, default, default)
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    seq = tuple(value)
    if len(seq) < 3:
        seq = seq + (default,) * (3 - len(seq))
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3:
    n = length(a)
    if n < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def distance(a: Vec3, b: Vec3) -> float:
    return length(sub(a, b))


# --------------------------------------------------------------- matrices --
def mat4_identity() -> Mat4:
    return IDENTITY


def mat4_from_translation(t: Vec3) -> Mat4:
    return (
        1.0, 0.0, 0.0, float(t[0]),
        0.0, 1.0, 0.0, float(t[1]),
        0.0, 0.0, 1.0, float(t[2]),
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_from_scale(s: Vec3) -> Mat4:
    return (
        float(s[0]), 0.0, 0.0, 0.0,
        0.0, float(s[1]), 0.0, 0.0,
        0.0, 0.0, float(s[2]), 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    """Return ``a * b`` (apply ``b`` first, then ``a``)."""
    out = [0.0] * 16
    for r in range(4):
        ar0, ar1, ar2, ar3 = a[r * 4:r * 4 + 4]
        for c in range(4):
            out[r * 4 + c] = (
                ar0 * b[c] + ar1 * b[4 + c] + ar2 * b[8 + c] + ar3 * b[12 + c]
            )
    return tuple(out)


def mat4_mul_many(*mats: Mat4) -> Mat4:
    result = IDENTITY
    for m in mats:
        result = mat4_mul(result, m)
    return result


def rot_x(deg: float) -> Mat4:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def rot_y(deg: float) -> Mat4:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def rot_z(deg: float) -> Mat4:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


_AXIS_FUNCS = {"X": rot_x, "Y": rot_y, "Z": rot_z}


def euler_matrix(rx: float, ry: float, rz: float, order: str = "XYZ") -> Mat4:
    """Build a rotation from Euler angles applied in ``order``.

    Daz stores a ``rotation_order`` such as ``"YZX"``.  The axes are applied in
    the order listed, which in matrix-product terms is the same left-to-right
    order (``R = R_first * R_second * R_third``).
    """
    order = (order or "XYZ").upper().replace("-", "")
    angles = {"X": rx, "Y": ry, "Z": rz}
    mats = [_AXIS_FUNCS[axis](angles.get(axis, 0.0)) for axis in order if axis in _AXIS_FUNCS]
    if not mats:
        mats = [rot_x(rx), rot_y(ry), rot_z(rz)]
    result = IDENTITY
    for m in mats:
        result = mat4_mul(result, m)
    return result


def transform_point(m: Mat4, p: Vec3) -> Vec3:
    x = m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3]
    y = m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7]
    z = m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]
    return (x, y, z)


def transform_vector(m: Mat4, v: Vec3) -> Vec3:
    """Transform a direction (ignores translation)."""
    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
        m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
    )


def mat4_translation(m: Mat4) -> Vec3:
    return (m[3], m[7], m[11])


def mat4_scale(m: Mat4) -> Vec3:
    """Column magnitudes -- the scale part of a TRS matrix."""
    return (
        length((m[0], m[4], m[8])),
        length((m[1], m[5], m[9])),
        length((m[2], m[6], m[10])),
    )


def mat4_inverse(m: Mat4) -> Mat4:
    """Full 4x4 inverse by cofactors (returns identity when singular)."""
    a = list(m)
    inv = [0.0] * 16

    inv[0] = (a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15]
              + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10])
    inv[4] = (-a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15]
              - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10])
    inv[8] = (a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15]
              + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9])
    inv[12] = (-a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14]
               - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9])
    inv[1] = (-a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15]
              - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10])
    inv[5] = (a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15]
              + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10])
    inv[9] = (-a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15]
              - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9])
    inv[13] = (a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14]
               + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9])
    inv[2] = (a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15]
              + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6])
    inv[6] = (-a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15]
              - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6])
    inv[10] = (a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15]
               + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5])
    inv[14] = (-a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14]
               - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5])
    inv[3] = (-a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11]
              - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6])
    inv[7] = (a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11]
              + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6])
    inv[11] = (-a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11]
               - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5])
    inv[15] = (a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10]
               + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5])

    det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12]
    if abs(det) < _EPS:
        return IDENTITY
    inv_det = 1.0 / det
    return tuple(v * inv_det for v in inv)


def decompose_trs(m: Mat4):
    """Split a matrix into ``(translation, euler_xyz_degrees, scale)``.

    Good enough for Daz node transforms, which never contain shear.
    """
    t = mat4_translation(m)
    sx, sy, sz = mat4_scale(m)
    sx = sx or 1.0
    sy = sy or 1.0
    sz = sz or 1.0
    # Normalised rotation basis (row-major storage -> rows are basis vectors).
    r0 = (m[0] / sx, m[1] / sy, m[2] / sz)
    r1 = (m[4] / sx, m[5] / sy, m[6] / sz)
    r2 = (m[8] / sx, m[9] / sy, m[10] / sz)

    sy_e = math.sqrt(max(r0[0] * r0[0] + r1[0] * r1[0], 0.0))
    if sy_e > 1e-6:
        rx = math.degrees(math.atan2(r2[1], r2[2]))
        ry = math.degrees(math.atan2(-r2[0], sy_e))
        rz = math.degrees(math.atan2(r1[0], r0[0]))
    else:  # gimbal lock
        rx = math.degrees(math.atan2(-r1[2], r1[1]))
        ry = math.degrees(math.atan2(-r2[0], sy_e))
        rz = 0.0
    return t, (rx, ry, rz), (sx, sy, sz)


def transpose(m: Mat4) -> Mat4:
    """Transpose -- converts between column-vector and FBX row-vector layout."""
    return (
        m[0], m[4], m[8], m[12],
        m[1], m[5], m[9], m[13],
        m[2], m[6], m[10], m[14],
        m[3], m[7], m[11], m[15],
    )


# ------------------------------------------------------------- utilities --
def bbox(points: Iterable[Vec3]):
    """Return ``(min_corner, max_corner)`` for a point cloud."""
    it = iter(points)
    try:
        first = next(it)
    except StopIteration:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    mn = [first[0], first[1], first[2]]
    mx = [first[0], first[1], first[2]]
    for p in it:
        for i in range(3):
            if p[i] < mn[i]:
                mn[i] = p[i]
            if p[i] > mx[i]:
                mx[i] = p[i]
    return tuple(mn), tuple(mx)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
