from dataclasses import dataclass, field
from typing import Union

import jax.numpy as jnp

from newton.primitives import Line, Point


@dataclass
class PointFixed:
    """
    Constraint to fix a point to a specific location.
    """

    point: Point
    fixed_pos: jnp.ndarray = field(init=False)

    def __post_init__(self):
        # Store the initial position as the target fixed position.
        self.fixed_pos = jnp.array([self.point.x, self.point.y])
        self.point.constraints.append(self)

    def residual(self, current_pos: jnp.ndarray) -> jnp.ndarray:
        return current_pos - self.fixed_pos


@dataclass
class PointPointDistance:
    """
    Constraint to maintain a fixed distance between two points.
    If `distance` is not provided, it's calculated from the initial point positions.
    """

    p1: Point
    p2: Point
    distance: Union[float, None] = None

    def __post_init__(self):
        if self.distance is None:
            self.distance = float(
                jnp.sqrt((self.p1.x - self.p2.x) ** 2 + (self.p1.y - self.p2.y) ** 2)
            )
        self.distance = float(self.distance)
        self.p1.constraints.append(self)
        self.p2.constraints.append(self)

    def residual(self, pos1: jnp.ndarray, pos2: jnp.ndarray) -> jnp.ndarray:
        current_dist = jnp.sqrt(jnp.sum((pos1 - pos2) ** 2))
        return current_dist - self.distance


@dataclass
class LinesParallel:
    """
    Constrains two lines to be parallel.
    The residual is the 2D cross-product of the lines' direction vectors.
    """

    line1: Line
    line2: Line

    def residual(self, l1_p1_pos, l1_p2_pos, l2_p1_pos, l2_p2_pos) -> float:
        v1 = l1_p2_pos - l1_p1_pos
        v2 = l2_p2_pos - l2_p1_pos
        return v1[0] * v2[1] - v1[1] * v2[0]  # 2D cross product


@dataclass
class LinesPerpendicular:
    """
    Constrains two lines to be perpendicular.
    The residual is the dot product of the lines' direction vectors.
    """

    line1: Line
    line2: Line

    def residual(self, l1_p1_pos, l1_p2_pos, l2_p1_pos, l2_p2_pos) -> float:
        v1 = l1_p2_pos - l1_p1_pos
        v2 = l2_p2_pos - l2_p1_pos
        return v1[0] * v2[0] + v1[1] * v2[1]  # Dot product.


@dataclass
class LineLineAngle:
    """
    Constrains the angle between two lines to a specific value.
    """

    line1: Line
    line2: Line
    angle: float  # Radians

    def residual(self, l1_p1_pos, l1_p2_pos, l2_p1_pos, l2_p2_pos) -> jnp.ndarray:
        v1 = l1_p2_pos - l1_p1_pos
        v2 = l2_p2_pos - l2_p1_pos
        v1_norm = jnp.sqrt(jnp.sum(v1**2))
        v2_norm = jnp.sqrt(jnp.sum(v2**2))
        dot_product = jnp.dot(v1, v2)

        cos_theta = jnp.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0)
        current_angle = jnp.arccos(cos_theta)
        return current_angle - self.angle


@dataclass
class HorizontalConstraint:
    """Constrains a line to be horizontal."""

    line: Line

    def residual(self, p1_pos: jnp.ndarray, p2_pos: jnp.ndarray) -> jnp.ndarray:
        # The residual is the difference in y-coordinates.
        return p1_pos[1] - p2_pos[1]


@dataclass
class VerticalConstraint:
    """Constrains a line to be vertical."""

    line: Line

    def residual(self, p1_pos: jnp.ndarray, p2_pos: jnp.ndarray) -> jnp.ndarray:
        # The residual is the difference in x-coordinates.
        return p1_pos[0] - p2_pos[0]


Constraint = Union[
    PointFixed,
    PointPointDistance,
    LinesParallel,
    LinesPerpendicular,
    LineLineAngle,
    HorizontalConstraint,
    VerticalConstraint,
]
