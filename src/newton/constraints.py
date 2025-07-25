from dataclasses import dataclass, field
from typing import Union

import jax.numpy as jnp

from newton.primitives import Line, Point


@dataclass
class PointFixed:
    point: Point
    fixed_pos: jnp.ndarray = field(init=False)

    def __post_init__(self):
        self.fixed_pos = jnp.array([self.point.x, self.point.y])


@dataclass
class PointPointDistance:
    p1: Point
    p2: Point
    distance: Union[float, None] = None

    def __post_init__(self):
        if self.distance is None:
            self.distance = float(
                jnp.sqrt((self.p1.x - self.p2.x) ** 2 + (self.p1.y - self.p2.y) ** 2)
            )
        self.distance = float(self.distance)


@dataclass
class LinesParallel:
    line1: Line
    line2: Line


@dataclass
class LinesPerpendicular:
    line1: Line
    line2: Line


@dataclass
class LineLineAngle:
    line1: Line
    line2: Line
    angle: float  # Radians


@dataclass
class LineHorizontal:
    line: Line


@dataclass
class LineVertical:
    line: Line


Constraint = Union[
    PointFixed,
    PointPointDistance,
    LinesParallel,
    LinesPerpendicular,
    LineLineAngle,
    LineHorizontal,
    LineVertical,
]
