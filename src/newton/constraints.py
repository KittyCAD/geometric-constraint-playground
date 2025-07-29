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
class PointPointEuclideanDistance(BaseConstraint):
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
class PointPointXDistance(BaseConstraint):
    p1: Point
    p2: Point
    distance: float

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        p1_pos = positions[self.p1.id]
        p2_pos = positions[self.p2.id]
        return nb.np.array([abs(p1_pos[0] - p2_pos[0]) - self.distance])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        p1_pos, p2_pos = positions[self.p1.id], positions[self.p2.id]
        sign = 1.0 if (p1_pos[0] - p2_pos[0]) > 0 else -1.0
        return [
            (self.p1.id, "x", sign, 0),
            (self.p2.id, "x", -sign, 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset([self.p1.id, self.p2.id])


@dataclass
class PointPointYDistance(BaseConstraint):
    p1: Point
    p2: Point
    distance: float

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        p1_pos = positions[self.p1.id]
        p2_pos = positions[self.p2.id]
        return nb.np.array([abs(p1_pos[1] - p2_pos[1]) - self.distance])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        p1_pos, p2_pos = positions[self.p1.id], positions[self.p2.id]
        sign = 1.0 if (p1_pos[1] - p2_pos[1]) > 0 else -1.0
        return [
            (self.p1.id, "y", sign, 0),
            (self.p2.id, "y", -sign, 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset([self.p1.id, self.p2.id])


@dataclass
class LineLength(BaseConstraint):
    line: Line
    length: float

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        pos1 = positions[self.line.p1.id]
        pos2 = positions[self.line.p2.id]
        current_dist = nb.np.linalg.norm(pos1 - pos2)
        return nb.np.array([current_dist - self.length])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        # This is identical to PointPointDistance jacobian.
        pos1, pos2 = positions[self.line.p1.id], positions[self.line.p2.id]
        d_pos = pos1 - pos2
        dist = np.linalg.norm(d_pos)
        if dist < EPS:
            return []
        deriv_x = float(d_pos[0] / dist)
        deriv_y = float(d_pos[1] / dist)
        return [
            (self.line.p1.id, "x", deriv_x, 0),
            (self.line.p1.id, "y", deriv_y, 0),
            (self.line.p2.id, "x", -deriv_x, 0),
            (self.line.p2.id, "y", -deriv_y, 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset({self.line.id, self.line.p1.id, self.line.p2.id})


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

        # Create a condition to check for zero-length lines.
        # This must be done with JAX-aware primitives, and we need`|` for element-wise OR.
        is_invalid = (mag1 < EPS) | (mag2 < EPS)

        # To prevent division by zero, create a 'safe' denominator.
        # We replace the product with 1.0 if it's invalid. The result of this
        # branch will be discarded by the final `where` anyway.
        safe_mag_product = nb.np.where(is_invalid, 1.0, mag1 * mag2)

        # Calculate dot product and clip to valid range for arccos.
        dot_product = nb.np.dot(v1, v2)
        cos_angle = nb.np.clip(dot_product / safe_mag_product, -1.0, 1.0)

        # Calculate current angle.
        current_angle = nb.np.arccos(cos_angle)
        angle_residual = nb.np.array([current_angle - self.angle])

        # Return 0.0 if the lines are invalid, otherwise return the calculated residual.
        return nb.np.where(is_invalid, nb.np.array([0.0]), angle_residual)

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


@dataclass
class LinesEqualLength(BaseConstraint):
    line1: Line
    line2: Line

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        p0, p1 = positions[self.line1.p1.id], positions[self.line1.p2.id]
        p2, p3 = positions[self.line2.p1.id], positions[self.line2.p2.id]

        len1 = nb.np.linalg.norm(p1 - p0)
        len2 = nb.np.linalg.norm(p3 - p2)

        return nb.np.array([len1 - len2])

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        p0, p1 = positions[self.line1.p1.id], positions[self.line1.p2.id]
        p2, p3 = positions[self.line2.p1.id], positions[self.line2.p2.id]

        d_pos1 = p1 - p0
        dist1 = np.linalg.norm(d_pos1)

        d_pos2 = p3 - p2
        dist2 = np.linalg.norm(d_pos2)

        entries = []
        # Derivatives for line1 (positive contribution)
        if dist1 > EPS:
            deriv_x1 = float(d_pos1[0] / dist1)
            deriv_y1 = float(d_pos1[1] / dist1)
            entries.extend(
                [
                    (self.line1.p2.id, "x", deriv_x1, 0),
                    (self.line1.p2.id, "y", deriv_y1, 0),
                    (self.line1.p1.id, "x", -deriv_x1, 0),
                    (self.line1.p1.id, "y", -deriv_y1, 0),
                ]
            )

        # Derivatives for line2 (negative contribution)
        if dist2 > EPS:
            deriv_x2 = float(d_pos2[0] / dist2)
            deriv_y2 = float(d_pos2[1] / dist2)
            entries.extend(
                [
                    (self.line2.p2.id, "x", -deriv_x2, 0),
                    (self.line2.p2.id, "y", -deriv_y2, 0),
                    (self.line2.p1.id, "x", deriv_x2, 0),
                    (self.line2.p1.id, "y", deriv_y2, 0),
                ]
            )

        return entries

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(
            {
                self.line1.id,
                self.line1.p1.id,
                self.line1.p2.id,
                self.line2.id,
                self.line2.p1.id,
                self.line2.p2.id,
            }
        )


@dataclass
class LineLineDistance(BaseConstraint):
    line1: Line
    line2: Line
    distance: float

    def get_residual(self, positions: Mapping[str, ArrayLike]) -> ArrayLike:
        # Assumes lines are parallel, enforced by a LinesParallel constraint.
        p0, p1 = positions[self.line1.p1.id], positions[self.line1.p2.id]
        p2 = positions[self.line2.p1.id]

        v = p1 - p0
        w = p2 - p0

        # Distance = |v x w| / |v|
        mag_v = nb.np.linalg.norm(v)
        is_valid = mag_v > EPS
        safe_mag_v = nb.np.where(mag_v < EPS, 1.0, mag_v)

        cross_product_mag = abs(v[0] * w[1] - v[1] * w[0])
        current_dist = cross_product_mag / safe_mag_v

        return nb.np.where(
            is_valid, nb.np.array([current_dist - self.distance]), nb.np.array([0.0])
        )

    def get_jacobian_section(
        self, positions: Mapping[str, ArrayLike]
    ) -> List[Tuple[str, str, float, int]]:
        # TODO...
        raise NotImplementedError(
            "LineLineDistance is only supported by the dense (JAX) solver."
        )

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(
            {
                self.line1.id,
                self.line1.p1.id,
                self.line1.p2.id,
                self.line2.id,
                self.line2.p1.id,
                self.line2.p2.id,
            }
        )


Constraint = Union[
    PointFixed,
    PointPointEuclideanDistance,
    PointPointXDistance,
    PointPointYDistance,
    LineLength,
    LinesParallel,
    LinesPerpendicular,
    LineLineAngle,
    LineHorizontal,
    LineVertical,
    LinesEqualLength,
    LineLineDistance,
]
