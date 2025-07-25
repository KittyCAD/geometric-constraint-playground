from typing import Dict

import jax.numpy as jnp

from newton.constraints import (
    Constraint,
    LineHorizontal,
    LineLineAngle,
    LinesParallel,
    LinesPerpendicular,
    LineVertical,
    PointFixed,
    PointPointDistance,
)


def compute_residual_point_fixed(
    constraint: PointFixed,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    # We can actually return a tuple here and the solver will handle it.
    pos = positions[constraint.point.id]
    pos_init = constraint.fixed_pos
    return pos - pos_init


def compute_residual_point_point_distance(
    constraint: PointPointDistance,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    # Euclidean distance between two points.
    pos1 = positions[constraint.p1.id]
    pos2 = positions[constraint.p2.id]
    current_dist = jnp.sqrt(jnp.sum((pos1 - pos2) ** 2))
    return current_dist - constraint.distance


def compute_residual_lines_parallel(
    constraint: LinesParallel,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    # Get angle from 2D cross product.
    v1 = positions[constraint.line1.p2.id] - positions[constraint.line1.p1.id]
    v2 = positions[constraint.line2.p2.id] - positions[constraint.line2.p1.id]
    return v1[0] * v2[1] - v1[1] * v2[0]


def compute_residual_lines_perpendicular(
    constraint: LinesPerpendicular,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    # Get angle from dot product.
    v1 = positions[constraint.line1.p2.id] - positions[constraint.line1.p1.id]
    v2 = positions[constraint.line2.p2.id] - positions[constraint.line2.p1.id]
    return v1[0] * v2[0] + v1[1] * v2[1]


def compute_residual_line_line_angle(
    constraint: LineLineAngle,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    v1 = positions[constraint.line1.p2.id] - positions[constraint.line1.p1.id]
    v2 = positions[constraint.line2.p2.id] - positions[constraint.line2.p1.id]
    v1_norm = jnp.sqrt(jnp.sum(v1**2))
    v2_norm = jnp.sqrt(jnp.sum(v2**2))
    dot_product = jnp.dot(v1, v2)

    cos_theta = jnp.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0)
    current_angle = jnp.arccos(cos_theta)
    return current_angle - constraint.angle


def compute_residual_line_horizontal(
    constraint: LineHorizontal,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    p1_pos = positions[constraint.line.p1.id]
    p2_pos = positions[constraint.line.p2.id]
    return p1_pos[1] - p2_pos[1]  # Difference in y-coordinates


def compute_residual_line_vertical(
    constraint: LineVertical,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    p1_pos = positions[constraint.line.p1.id]
    p2_pos = positions[constraint.line.p2.id]
    return p1_pos[0] - p2_pos[0]  # Difference in x-coordinates


DISPATCH_TABLE = {
    PointFixed: compute_residual_point_fixed,
    PointPointDistance: compute_residual_point_point_distance,
    LinesParallel: compute_residual_lines_parallel,
    LinesPerpendicular: compute_residual_lines_perpendicular,
    LineLineAngle: compute_residual_line_line_angle,
    LineHorizontal: compute_residual_line_horizontal,
    LineVertical: compute_residual_line_vertical,
}


def compute_residual(
    constraint: Constraint,
    positions: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    constraint_type = type(constraint)
    func = DISPATCH_TABLE.get(constraint_type)

    if func:
        return func(constraint, positions)

    raise TypeError(f"Unknown constraint type: {constraint_type}")
