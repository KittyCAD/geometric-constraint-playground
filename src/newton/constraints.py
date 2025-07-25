from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Mapping, Tuple, Union

import jax.numpy as jnp
import numpy as np

from newton import backend as nb
from newton.constants import EPS
from newton.primitives import Line, Point

ArrayLike = Union[np.ndarray, jnp.ndarray]


class BaseConstraint(ABC):
    @abstractmethod
    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        pass

    @abstractmethod
    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        pass

    def get_residual_dim(self) -> int:
        return 1

    @abstractmethod
    def get_involved_primitive_ids(self) -> frozenset:
        pass


@dataclass
class PointFixed(BaseConstraint):
    point: Point
    fixed_pos: ArrayLike = field(init=False)

    def __post_init__(self):
        self.fixed_pos = nb.np.array([self.point.x, self.point.y])

    def get_residual_dim(self) -> int:
        return 2

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        return positions[self.point.id] - self.fixed_pos

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        return [
            (self.point.id, "x", 1.0, 0),
            (self.point.id, "y", 1.0, 1),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset([self.point.id])


@dataclass
class PointPointDistance(BaseConstraint):
    p1: Point
    p2: Point
    distance: float

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        pos1 = positions[self.p1.id]
        pos2 = positions[self.p2.id]
        current_dist = nb.np.linalg.norm(pos1 - pos2)
        return nb.np.array([current_dist - self.distance])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        pos1, pos2 = positions[self.p1.id], positions[self.p2.id]
        d_pos = pos1 - pos2
        dist = nb.np.linalg.norm(d_pos)

        if dist < EPS:
            return []

        deriv_x = float(d_pos[0] / dist)
        deriv_y = float(d_pos[1] / dist)

        return [
            (self.p1.id, "x", deriv_x, 0),
            (self.p1.id, "y", deriv_y, 0),
            (self.p2.id, "x", -deriv_x, 0),
            (self.p2.id, "y", -deriv_y, 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset([self.p1.id, self.p2.id])


@dataclass
class LinesParallel(BaseConstraint):
    line1: Line
    line2: Line

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        v1 = positions[self.line1.p2.id] - positions[self.line1.p1.id]
        v2 = positions[self.line2.p2.id] - positions[self.line2.p1.id]
        return nb.np.array([v1[0] * v2[1] - v1[1] * v2[0]])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        v1 = positions[self.line1.p2.id] - positions[self.line1.p1.id]
        v2 = positions[self.line2.p2.id] - positions[self.line2.p1.id]
        return [
            (self.line1.p1.id, "x", float(-v2[1]), 0),
            (self.line1.p1.id, "y", float(v2[0]), 0),
            (self.line1.p2.id, "x", float(v2[1]), 0),
            (self.line1.p2.id, "y", float(-v2[0]), 0),
            (self.line2.p1.id, "x", float(v1[1]), 0),
            (self.line2.p1.id, "y", float(-v1[0]), 0),
            (self.line2.p2.id, "x", float(-v1[1]), 0),
            (self.line2.p2.id, "y", float(v1[0]), 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(
            [
                self.line1.id,
                self.line2.id,
                self.line1.p1.id,
                self.line1.p2.id,
                self.line2.p1.id,
                self.line2.p2.id,
            ]
        )


@dataclass
class LinesPerpendicular(BaseConstraint):
    line1: Line
    line2: Line

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        v1 = positions[self.line1.p2.id] - positions[self.line1.p1.id]
        v2 = positions[self.line2.p2.id] - positions[self.line2.p1.id]
        return nb.np.array([v1[0] * v2[0] + v1[1] * v2[1]])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        v1 = positions[self.line1.p2.id] - positions[self.line1.p1.id]
        v2 = positions[self.line2.p2.id] - positions[self.line2.p1.id]
        return [
            (self.line1.p1.id, "x", float(-v2[0]), 0),
            (self.line1.p1.id, "y", float(-v2[1]), 0),
            (self.line1.p2.id, "x", float(v2[0]), 0),
            (self.line1.p2.id, "y", float(v2[1]), 0),
            (self.line2.p1.id, "x", float(-v1[0]), 0),
            (self.line2.p1.id, "y", float(-v1[1]), 0),
            (self.line2.p2.id, "x", float(v1[0]), 0),
            (self.line2.p2.id, "y", float(v1[1]), 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(
            [
                self.line1.id,
                self.line2.id,
                self.line1.p1.id,
                self.line1.p2.id,
                self.line2.p1.id,
                self.line2.p2.id,
            ]
        )


@dataclass
class LineLineAngle(BaseConstraint):
    line1: Line
    line2: Line
    angle: float = field()

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        # Get direction vectors for both lines.
        v1 = positions[self.line1.p2.id] - positions[self.line1.p1.id]
        v2 = positions[self.line2.p2.id] - positions[self.line2.p1.id]

        # Calculate magnitudes.
        mag1 = nb.np.linalg.norm(v1)
        mag2 = nb.np.linalg.norm(v2)

        # Avoid division by zero.
        if mag1 < EPS or mag2 < EPS:
            return nb.np.array([0.0])

        # Calculate dot product and clip to valid range for arccos.
        dot_product = nb.np.dot(v1, v2)
        cos_angle = nb.np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)

        # Calculate current angle.
        current_angle = nb.np.arccos(cos_angle)

        # Return difference between current and target angle.
        return nb.np.array([current_angle - self.angle])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        # HERE BE DRAGONS... I don't think I trust this.

        # Get direction vectors for both lines
        v1 = positions[self.line1.p2.id] - positions[self.line1.p1.id]
        v2 = positions[self.line2.p2.id] - positions[self.line2.p1.id]

        # Calculate magnitudes.
        mag1 = nb.np.linalg.norm(v1)
        mag2 = nb.np.linalg.norm(v2)

        # Avoid division by zero.
        if mag1 < EPS or mag2 < EPS:
            return []

        # Calculate dot product and normalized dot product.
        dot_product = nb.np.dot(v1, v2)
        cos_angle = nb.np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)

        # Avoid numerical issues at exact -1 or 1.
        if abs(abs(cos_angle) - 1.0) < EPS:
            cos_angle = 0.99 * nb.np.sign(cos_angle)

        # Derivative of arccos: d(arccos(x))/dx = -1/sqrt(1-x^2).
        denom = 1.0 - cos_angle**2
        if denom < EPS:
            denom = EPS
        d_arccos = -1.0 / nb.np.sqrt(denom)

        # For each point, calculate partial derivatives.
        result = []

        # Derivatives with respect to first line's first point.
        dx1 = float(
            d_arccos
            * ((-v2[0] / (mag1 * mag2)) + (dot_product * v1[0]) / (mag1**3 * mag2))
        )
        dy1 = float(
            d_arccos
            * ((-v2[1] / (mag1 * mag2)) + (dot_product * v1[1]) / (mag1**3 * mag2))
        )
        result.append((self.line1.p1.id, "x", dx1, 0))
        result.append((self.line1.p1.id, "y", dy1, 0))

        # Derivatives with respect to first line's second point.
        dx2 = float(
            d_arccos
            * ((v2[0] / (mag1 * mag2)) - (dot_product * v1[0]) / (mag1**3 * mag2))
        )
        dy2 = float(
            d_arccos
            * ((v2[1] / (mag1 * mag2)) - (dot_product * v1[1]) / (mag1**3 * mag2))
        )
        result.append((self.line1.p2.id, "x", dx2, 0))
        result.append((self.line1.p2.id, "y", dy2, 0))

        # Derivatives with respect to second line's first point.
        dx3 = float(
            d_arccos
            * ((-v1[0] / (mag1 * mag2)) + (dot_product * v2[0]) / (mag1 * mag2**3))
        )
        dy3 = float(
            d_arccos
            * ((-v1[1] / (mag1 * mag2)) + (dot_product * v2[1]) / (mag1 * mag2**3))
        )
        result.append((self.line2.p1.id, "x", dx3, 0))
        result.append((self.line2.p1.id, "y", dy3, 0))

        # Derivatives with respect to second line's second point.
        dx4 = float(
            d_arccos
            * ((v1[0] / (mag1 * mag2)) - (dot_product * v2[0]) / (mag1 * mag2**3))
        )
        dy4 = float(
            d_arccos
            * ((v1[1] / (mag1 * mag2)) - (dot_product * v2[1]) / (mag1 * mag2**3))
        )
        result.append((self.line2.p2.id, "x", dx4, 0))
        result.append((self.line2.p2.id, "y", dy4, 0))

        return result

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(
            [
                self.line1.id,
                self.line2.id,
                self.line1.p1.id,
                self.line1.p2.id,
                self.line2.p1.id,
                self.line2.p2.id,
            ]
        )


@dataclass
class LineHorizontal(BaseConstraint):
    line: Line

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        p1_pos = positions[self.line.p1.id]
        p2_pos = positions[self.line.p2.id]
        return nb.np.array([p1_pos[1] - p2_pos[1]])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        return [
            (self.line.p1.id, "y", 1.0, 0),
            (self.line.p2.id, "y", -1.0, 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset([self.line.id, self.line.p1.id, self.line.p2.id])


@dataclass
class LineVertical(BaseConstraint):
    line: Line

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        p1_pos = positions[self.line.p1.id]
        p2_pos = positions[self.line.p2.id]
        return nb.np.array([p1_pos[0] - p2_pos[0]])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        return [
            (self.line.p1.id, "x", 1.0, 0),
            (self.line.p2.id, "x", -1.0, 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset([self.line.id, self.line.p1.id, self.line.p2.id])


Constraint = Union[
    PointFixed,
    PointPointDistance,
    LinesParallel,
    LinesPerpendicular,
    LineLineAngle,
    LineHorizontal,
    LineVertical,
]
