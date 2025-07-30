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
        # This method is used to calculate (part of) a row of the Jacobian matrix.
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
        # Residuals: R1 = px - fx, R2 = py - fy
        # ∂R1/∂px = 1
        # ∂R1/∂py = 0
        # ∂R2/∂px = 0
        # ∂R2/∂py = 1

        # Derivatives with respect to the fixed point's coordinates.
        dr1_dx = 1.0
        # dr1_dy = 0.0
        # dr2_dx = 0.0
        dr2_dy = 1.0

        # Indices for the residuals. This is a 2D constraint, so we have two residuals. Basically all other constraints will have 1 residual.
        i_x = 0
        i_y = 1

        return [
            (self.point.id, "x", dr1_dx, i_x),
            (self.point.id, "y", dr2_dy, i_y),
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
        # Residual: R = sqrt((x1-x2)² + (y1-y2)²) - d
        # ∂R/∂x1 = (x1 - x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂y1 = (y1 - y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂x2 = (-x1 + x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂y2 = (-y1 + y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # Derivatives with respect to p1 and p2 and the x/y coordinates thereof.
        p1 = positions[self.p1.id]
        p2 = positions[self.p2.id]

        # Handle zero-length vectors gracefully.
        dist = nb.np.linalg.norm(p1 - p2)  # sqrt((x1 - x2)**2 + (y1 - y2)**2)

        if dist < EPS:
            return []

        # Set out actual derivatives.
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        dr_dx1 = (x1 - x2) / dist
        dr_dy1 = (y1 - y2) / dist
        dr_dx2 = (-x1 + x2) / dist
        dr_dy2 = (-y1 + y2) / dist

        # Get as floats.
        dr_dx1 = float(dr_dx1)
        dr_dy1 = float(dr_dy1)
        dr_dx2 = float(dr_dx2)
        dr_dy2 = float(dr_dy2)

        # This constraint has a scalar residual.
        i_residual = 0

        return [
            (self.p1.id, "x", dr_dx1, i_residual),
            (self.p1.id, "y", dr_dy1, i_residual),
            (self.p2.id, "x", dr_dx2, i_residual),
            (self.p2.id, "y", dr_dy2, i_residual),
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
    ) -> List[Tuple[str, str, float, int]]:  #
        # Residual: R = |x1 - x2| - d
        # When (x1 - x2) >= 0:
        # ∂R/∂x1 = 1
        # ∂R/∂x2 = -1
        # When (x1 - x2) < 0:
        # ∂R/∂x1 = -1
        # ∂R/∂x2 = 1

        # Symbolic derivatives:
        # ∂R/∂x1 = Piecewise((1, x1 - x2 >= 0), (-1, True))
        # ∂R/∂x2 = Piecewise((-1, x1 - x2 >= 0), (1, True))

        # Get our derivatives.
        p1 = positions[self.p1.id]
        p2 = positions[self.p2.id]

        x1 = p1[0]
        x2 = p2[0]

        sign = 1.0 if (x1 - x2) > 0 else -1.0

        dr_dx1 = sign
        dr_dx2 = -sign

        # This constraint has a scalar residual.
        i_residual = 0

        return [
            (self.p1.id, "x", dr_dx1, i_residual),
            (self.p2.id, "x", dr_dx2, i_residual),
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
        # Residual: R = |y1 - y2| - d
        # When (y1 - y2) >= 0:
        # ∂R/∂y1 = 1
        # ∂R/∂y2 = -1
        # When (y1 - y2) < 0:
        # ∂R/∂y1 = -1
        # ∂R/∂y2 = 1

        # Symbolic derivatives:
        # ∂R/∂y1 = Piecewise((1, y1 - y2 >= 0), (-1, True))
        # ∂R/∂y2 = Piecewise((-1, y1 - y2 >= 0), (1, True))

        # Get our derivatives.
        p1 = positions[self.p1.id]
        p2 = positions[self.p2.id]

        y1 = p1[1]
        y2 = p2[1]

        sign = 1.0 if (y1 - y2) > 0 else -1.0

        dr_dy1 = sign
        dr_dy2 = -sign

        # This constraint has a scalar residual.
        i_residual = 0

        return [
            (self.p1.id, "y", dr_dy1, i_residual),
            (self.p2.id, "y", dr_dy2, i_residual),
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
        # Reuse the implementation from PointPointEuclideanDistance.
        temp_constraint = PointPointEuclideanDistance(
            p1=self.line.p1, p2=self.line.p2, distance=self.length
        )
        return temp_constraint.get_jacobian_section(positions)

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset({self.line.id, self.line.p1.id, self.line.p2.id})


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
        # Residual: R = y1 - y2
        # ∂R/∂y1 = 1
        # ∂R/∂y2 = -1

        dr_dy1 = 1.0
        dr_dy2 = -1.0

        # This constraint has a scalar residual.
        i_residual = 0

        return [
            (self.line.p1.id, "y", dr_dy1, i_residual),
            (self.line.p2.id, "y", dr_dy2, i_residual),
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
        # Residual: R = x1 - x2
        # ∂R/∂x1 = 1
        # ∂R/∂x2 = -1

        dr_dx1 = 1.0
        dr_dx2 = -1.0

        # This constraint has a scalar residual.
        i_residual = 0

        return [
            (self.line.p1.id, "x", dr_dx1, i_residual),
            (self.line.p2.id, "x", dr_dx2, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset([self.line.id, self.line.p1.id, self.line.p2.id])


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
        # Residual: R = (x2-x1)*(y4-y3) - (y2-y1)*(x4-x3)
        # ∂R/∂x1 = y3 - y4
        # ∂R/∂y1 = -x3 + x4
        # ∂R/∂x2 = -y3 + y4
        # ∂R/∂y2 = x3 - x4
        # ∂R/∂x3 = -y1 + y2
        # ∂R/∂y3 = x1 - x2
        # ∂R/∂x4 = y1 - y2
        # ∂R/∂y4 = -x1 + x2

        # Get points.
        p1 = positions[self.line1.p1.id]
        p2 = positions[self.line1.p2.id]
        p3 = positions[self.line2.p1.id]
        p4 = positions[self.line2.p2.id]

        # Get their components.
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]

        # Calculate derivatives.
        dr_dx1 = y3 - y4
        dr_dy1 = -x3 + x4
        dr_dx2 = -y3 + y4
        dr_dy2 = x3 - x4
        dr_dx3 = -y1 + y2
        dr_dy3 = x1 - x2
        dr_dx4 = y1 - y2
        dr_dy4 = -x1 + x2

        # Make floats.
        dr_dx1 = float(dr_dx1)
        dr_dy1 = float(dr_dy1)
        dr_dx2 = float(dr_dx2)
        dr_dy2 = float(dr_dy2)
        dr_dx3 = float(dr_dx3)
        dr_dy3 = float(dr_dy3)
        dr_dx4 = float(dr_dx4)
        dr_dy4 = float(dr_dy4)

        # This constraint has a scalar residual.
        i_residual = 0

        return [
            (self.line1.p1.id, "x", dr_dx1, i_residual),
            (self.line1.p1.id, "y", dr_dy1, i_residual),
            (self.line1.p2.id, "x", dr_dx2, i_residual),
            (self.line1.p2.id, "y", dr_dy2, i_residual),
            (self.line2.p1.id, "x", dr_dx3, i_residual),
            (self.line2.p1.id, "y", dr_dy3, i_residual),
            (self.line2.p2.id, "x", dr_dx4, i_residual),
            (self.line2.p2.id, "y", dr_dy4, i_residual),
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
        # Residual: R = (x2-x1)*(x4-x3) + (y2-y1)*(y4-y3)
        # ∂R/∂x1 = x3 - x4
        # ∂R/∂y1 = y3 - y4
        # ∂R/∂x2 = -x3 + x4
        # ∂R/∂y2 = -y3 + y4
        # ∂R/∂x3 = x1 - x2
        # ∂R/∂y3 = y1 - y2
        # ∂R/∂x4 = -x1 + x2
        # ∂R/∂y4 = -y1 + y2

        # Get points.
        p1 = positions[self.line1.p1.id]
        p2 = positions[self.line1.p2.id]
        p3 = positions[self.line2.p1.id]
        p4 = positions[self.line2.p2.id]

        # Get their components.
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]

        # Calculate derivatives.
        dr_dx1 = x3 - x4
        dr_dy1 = y3 - y4
        dr_dx2 = -x3 + x4
        dr_dy2 = -y3 + y4
        dr_dx3 = x1 - x2
        dr_dy3 = y1 - y2
        dr_dx4 = -x1 + x2
        dr_dy4 = -y1 + y2

        # Make floats.
        dr_dx1 = float(dr_dx1)
        dr_dy1 = float(dr_dy1)
        dr_dx2 = float(dr_dx2)
        dr_dy2 = float(dr_dy2)
        dr_dx3 = float(dr_dx3)
        dr_dy3 = float(dr_dy3)
        dr_dx4 = float(dr_dx4)
        dr_dy4 = float(dr_dy4)

        # This constraint has a scalar residual.
        i_residual = 0

        return [
            (self.line1.p1.id, "x", dr_dx1, i_residual),
            (self.line1.p1.id, "y", dr_dy1, i_residual),
            (self.line1.p2.id, "x", dr_dx2, i_residual),
            (self.line1.p2.id, "y", dr_dy2, i_residual),
            (self.line2.p1.id, "x", dr_dx3, i_residual),
            (self.line2.p1.id, "y", dr_dy3, i_residual),
            (self.line2.p2.id, "x", dr_dx4, i_residual),
            (self.line2.p2.id, "y", dr_dy4, i_residual),
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

        # Check for zero-length lines.
        is_invalid = (mag1 < EPS) | (mag2 < EPS)

        # 2D cross product and dot product.
        cross_2d = v1[0] * v2[1] - v1[1] * v2[0]
        dot_product = nb.np.dot(v1, v2)

        # Current angle using atan2.
        current_angle = nb.np.atan2(cross_2d, dot_product)

        # Compute angle difference.
        angle_residual = nb.np.array([current_angle - self.angle])

        # Return 0.0 if invalid, otherwise return residual.
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
