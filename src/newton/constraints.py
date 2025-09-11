from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Mapping, Tuple, Union

import jax.numpy as jnp
import numpy as np

from newton import backend as nb
from newton.constants import EPS
from newton.primitives import Circle, CircularArc, Line, Point

# Note that for get_residual methods, we can't have if statements because JAX
# doesn't support control flow in JIT-compiled functions. Instead, we use
# numpy's where function to handle conditional logic.


class BaseConstraint(ABC):
    @abstractmethod
    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        pass

    @abstractmethod
    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # This method is used to calculate (part of) a row of the Jacobian matrix.
        # Note that variable_values is now a map of specific variable IDs to their scalar values,
        # not a map of IDs to position arrays.
        pass

    @property
    def n_residual_rows(self) -> int:
        return 1

    @abstractmethod
    def get_involved_primitive_ids(self) -> frozenset:
        pass


@dataclass
class PointFixed(BaseConstraint):
    point: Point
    fixed_pos: nb.Vector = field(init=False)

    def __post_init__(self):
        self.fixed_pos = nb.np.array([self.point.x, self.point.y])

    @property
    def n_residual_rows(self) -> int:
        return 2

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p = self.point.get_state(variable_values)
        return p - self.fixed_pos

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
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

        # Row indices for the residuals. This is a 2D constraint, so we have two residuals. Basically all other constraints will have 1 residual.
        i_x = 0
        i_y = 1

        p_vars = self.point.get_variable_ids()

        return [
            (p_vars[0], dr1_dx, i_x),
            (p_vars[1], dr2_dy, i_y),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(self.point.get_involved_primitive_ids())


@dataclass
class PointPointCoincident(BaseConstraint):
    p1: Point
    p2: Point

    @property
    def n_residual_rows(self) -> int:
        # Two residual terms for this: one for delta x, one for delta y.
        return 2

    def get_residual(self, variable_values: Mapping[str, float]):
        p1_pos = self.p1.get_state(variable_values)
        p2_pos = self.p2.get_state(variable_values)

        residual_x = p1_pos[0] - p2_pos[0]  # x1 - x2
        residual_y = p1_pos[1] - p2_pos[1]  # y1 - y2

        return nb.np.array([residual_x, residual_y])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residuals: R1 = x1 - x2, R2 = y1 - y2.
        #
        # For R1 = x1 - x2:
        # ∂R1/∂x1 = 1
        # ∂R1/∂y1 = 0
        # ∂R1/∂x2 = -1
        # ∂R1/∂y2 = 0
        #
        # For R2 = y1 - y2:
        # ∂R2/∂x1 = 0
        # ∂R2/∂y1 = 1
        # ∂R2/∂x2 = 0
        # ∂R2/∂y2 = -1

        # Row indices for the two residuals.
        i_x_residual = 0
        i_y_residual = 1

        # Get variable IDs for both points.
        p1_vars = self.p1.get_variable_ids()
        p2_vars = self.p2.get_variable_ids()

        dr1_dx1 = 1.0
        # dr1_dy1 = 0.0
        dr1_dx2 = -1.0
        # dr1_dy2 = 0.0

        # dr2_dx1 = 0.0
        dr2_dy1 = 1.0
        # dr2_dx2 = 0.0
        dr2_dy2 = -1.0

        # We only care about nonzero derivs here.
        return [
            (p1_vars[0], dr1_dx1, i_x_residual),  # ∂R1/∂x1
            (p2_vars[0], dr1_dx2, i_x_residual),  # ∂R1/∂x2
            (p1_vars[1], dr2_dy1, i_y_residual),  # ∂R2/∂y1
            (p2_vars[1], dr2_dy2, i_y_residual),  # ∂R2/∂y2
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.p1.get_involved_primitive_ids()
        ids_2 = self.p2.get_involved_primitive_ids()
        return frozenset(ids_1.union(ids_2))


@dataclass
class PointPointEuclideanDistance(BaseConstraint):
    p1: Point
    p2: Point
    distance: float

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p1 = self.p1.get_state(variable_values)
        p2 = self.p2.get_state(variable_values)
        current_dist = nb.np.linalg.norm(p1 - p2)
        return nb.np.array([current_dist - self.distance])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = sqrt((x1-x2)**2 + (y1-y2)**2) - d
        # ∂R/∂x1 = (x1 - x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂y1 = (y1 - y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂x2 = (-x1 + x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂y2 = (-y1 + y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # Derivatives with respect to p1 and p2 and the x/y coordinates thereof.
        p1 = self.p1.get_state(variable_values)
        p2 = self.p2.get_state(variable_values)

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

        p1_vars = self.p1.get_variable_ids()
        p2_vars = self.p2.get_variable_ids()

        return [
            (p1_vars[0], dr_dx1, i_residual),
            (p1_vars[1], dr_dy1, i_residual),
            (p2_vars[0], dr_dx2, i_residual),
            (p2_vars[1], dr_dy2, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.p1.get_involved_primitive_ids()
        ids_2 = self.p2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class PointPointXDistance(BaseConstraint):
    p1: Point
    p2: Point
    distance: float

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p1_pos = self.p1.get_state(variable_values)
        p2_pos = self.p2.get_state(variable_values)
        return nb.np.array([abs(p1_pos[0] - p2_pos[0]) - self.distance])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:  #
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
        p1 = self.p1.get_state(variable_values)
        p2 = self.p2.get_state(variable_values)

        x1 = p1[0]
        x2 = p2[0]

        sign = 1.0 if (x1 - x2) > 0 else -1.0

        dr_dx1 = sign
        dr_dx2 = -sign

        # This constraint has a scalar residual.
        i_residual = 0

        p1_vars = self.p1.get_variable_ids()
        p2_vars = self.p2.get_variable_ids()

        return [
            (p1_vars[0], dr_dx1, i_residual),
            (p2_vars[0], dr_dx2, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.p1.get_involved_primitive_ids()
        ids_2 = self.p2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class PointPointYDistance(BaseConstraint):
    p1: Point
    p2: Point
    distance: float

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p1_pos = self.p1.get_state(variable_values)
        p2_pos = self.p2.get_state(variable_values)
        return nb.np.array([abs(p1_pos[1] - p2_pos[1]) - self.distance])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
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
        p1 = self.p1.get_state(variable_values)
        p2 = self.p2.get_state(variable_values)

        y1 = p1[1]
        y2 = p2[1]

        sign = 1.0 if (y1 - y2) > 0 else -1.0

        dr_dy1 = sign
        dr_dy2 = -sign

        # This constraint has a scalar residual.
        i_residual = 0

        p1_vars = self.p1.get_variable_ids()
        p2_vars = self.p2.get_variable_ids()

        return [
            (p1_vars[1], dr_dy1, i_residual),
            (p2_vars[1], dr_dy2, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.p1.get_involved_primitive_ids()
        ids_2 = self.p2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class LineLength(BaseConstraint):
    line: Line
    length: float

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        pos1 = self.line.p1.get_state(variable_values)
        pos2 = self.line.p2.get_state(variable_values)
        current_dist = nb.np.linalg.norm(pos1 - pos2)
        return nb.np.array([current_dist - self.length])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Reuse the implementation from PointPointEuclideanDistance.
        temp_constraint = PointPointEuclideanDistance(
            p1=self.line.p1, p2=self.line.p2, distance=self.length
        )
        return temp_constraint.get_jacobian_row_values(variable_values)

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(self.line.get_involved_primitive_ids())


@dataclass
class LineHorizontal(BaseConstraint):
    line: Line

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p1_pos = self.line.p1.get_state(variable_values)
        p2_pos = self.line.p2.get_state(variable_values)
        return nb.np.array([p1_pos[1] - p2_pos[1]])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = y1 - y2
        # ∂R/∂y1 = 1
        # ∂R/∂y2 = -1

        dr_dy1 = 1.0
        dr_dy2 = -1.0

        # This constraint has a scalar residual.
        i_residual = 0

        # Get the 'y' variable ID for the line's points.
        p1_y_var = self.line.p1.get_variable_ids()[1]
        p2_y_var = self.line.p2.get_variable_ids()[1]

        return [
            (p1_y_var, dr_dy1, i_residual),
            (p2_y_var, dr_dy2, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(self.line.get_involved_primitive_ids())


@dataclass
class LineVertical(BaseConstraint):
    line: Line

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p1_pos = self.line.p1.get_state(variable_values)
        p2_pos = self.line.p2.get_state(variable_values)
        return nb.np.array([p1_pos[0] - p2_pos[0]])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = x1 - x2
        # ∂R/∂x1 = 1
        # ∂R/∂x2 = -1

        dr_dx1 = 1.0
        dr_dx2 = -1.0

        # This constraint has a scalar residual.
        i_residual = 0

        # Get the 'x' variable ID for the line's points.
        p1_x_var = self.line.p1.get_variable_ids()[0]
        p2_x_var = self.line.p2.get_variable_ids()[0]

        return [
            (p1_x_var, dr_dx1, i_residual),
            (p2_x_var, dr_dx2, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(self.line.get_involved_primitive_ids())


@dataclass
class LinesParallel(BaseConstraint):
    line1: Line
    line2: Line

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)

        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

        # Calculate the vectors.
        v1 = p2 - p1
        v2 = p4 - p3

        return nb.np.array([v1[0] * v2[1] - v1[1] * v2[0]])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
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
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

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

        # Get the variable IDs for the points involved.
        p1_x_var = self.line1.p1.get_variable_ids()[0]
        p1_y_var = self.line1.p1.get_variable_ids()[1]
        p2_x_var = self.line1.p2.get_variable_ids()[0]
        p2_y_var = self.line1.p2.get_variable_ids()[1]
        p3_x_var = self.line2.p1.get_variable_ids()[0]
        p3_y_var = self.line2.p1.get_variable_ids()[1]
        p4_x_var = self.line2.p2.get_variable_ids()[0]
        p4_y_var = self.line2.p2.get_variable_ids()[1]

        return [
            (p1_x_var, dr_dx1, i_residual),
            (p1_y_var, dr_dy1, i_residual),
            (p2_x_var, dr_dx2, i_residual),
            (p2_y_var, dr_dy2, i_residual),
            (p3_x_var, dr_dx3, i_residual),
            (p3_y_var, dr_dy3, i_residual),
            (p4_x_var, dr_dx4, i_residual),
            (p4_y_var, dr_dy4, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.line1.get_involved_primitive_ids()
        ids_2 = self.line2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class LinesPerpendicular(BaseConstraint):
    line1: Line
    line2: Line

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)

        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

        # Calculate the vectors.
        v1 = p2 - p1
        v2 = p4 - p3

        # The residual is the dot product of the two vectors.
        # If they are perpendicular, this should be zero.
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        return nb.np.array([dot])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
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
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

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

        # Get the variable IDs for the points involved.
        p1_x_var = self.line1.p1.get_variable_ids()[0]
        p1_y_var = self.line1.p1.get_variable_ids()[1]
        p2_x_var = self.line1.p2.get_variable_ids()[0]
        p2_y_var = self.line1.p2.get_variable_ids()[1]
        p3_x_var = self.line2.p1.get_variable_ids()[0]
        p3_y_var = self.line2.p1.get_variable_ids()[1]
        p4_x_var = self.line2.p2.get_variable_ids()[0]
        p4_y_var = self.line2.p2.get_variable_ids()[1]

        return [
            (p1_x_var, dr_dx1, i_residual),
            (p1_y_var, dr_dy1, i_residual),
            (p2_x_var, dr_dx2, i_residual),
            (p2_y_var, dr_dy2, i_residual),
            (p3_x_var, dr_dx3, i_residual),
            (p3_y_var, dr_dy3, i_residual),
            (p4_x_var, dr_dx4, i_residual),
            (p4_y_var, dr_dy4, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.line1.get_involved_primitive_ids()
        ids_2 = self.line2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class LinesEqualLength(BaseConstraint):
    line1: Line
    line2: Line

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # Get points.
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

        # Calculate lengths.
        len1 = nb.np.linalg.norm(p2 - p1)
        len2 = nb.np.linalg.norm(p4 - p3)

        return nb.np.array([len1 - len2])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = |L1| - |L2|
        # ∂R/∂x1 = (x1 - x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂y1 = (y1 - y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂x2 = (-x1 + x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂y2 = (-y1 + y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂x3 = (-x3 + x4)/sqrt((x3 - x4)**2 + (y3 - y4)**2)
        # ∂R/∂y3 = (-y3 + y4)/sqrt((x3 - x4)**2 + (y3 - y4)**2)
        # ∂R/∂x4 = (x3 - x4)/sqrt((x3 - x4)**2 + (y3 - y4)**2)
        # ∂R/∂y4 = (y3 - y4)/sqrt((x3 - x4)**2 + (y3 - y4)**2)

        # Get points.
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

        # Get their components.
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]

        # Calculate lengths.
        length_l1 = nb.np.linalg.norm(p2 - p1)  # sqrt((x1 - x2)**2 + (y1 - y2)**2)
        length_l2 = nb.np.linalg.norm(p4 - p3)  # sqrt((x3 - x4)**2 + (y3 - y4)**2)

        # Avoid division by zero.
        if length_l1 < EPS or length_l2 < EPS:
            return []

        # Calculate derivatives.
        dr_dx1 = (x1 - x2) / length_l1
        dr_dy1 = (y1 - y2) / length_l1
        dr_dx2 = (-x1 + x2) / length_l1
        dr_dy2 = (-y1 + y2) / length_l1
        dr_dx3 = (-x3 + x4) / length_l2
        dr_dy3 = (-y3 + y4) / length_l2
        dr_dx4 = (x3 - x4) / length_l2
        dr_dy4 = (y3 - y4) / length_l2

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

        # Get the variable IDs for the points involved.
        p1_x_var = self.line1.p1.get_variable_ids()[0]
        p1_y_var = self.line1.p1.get_variable_ids()[1]
        p2_x_var = self.line1.p2.get_variable_ids()[0]
        p2_y_var = self.line1.p2.get_variable_ids()[1]
        p3_x_var = self.line2.p1.get_variable_ids()[0]
        p3_y_var = self.line2.p1.get_variable_ids()[1]
        p4_x_var = self.line2.p2.get_variable_ids()[0]
        p4_y_var = self.line2.p2.get_variable_ids()[1]

        return [
            (p1_x_var, dr_dx1, i_residual),
            (p1_y_var, dr_dy1, i_residual),
            (p2_x_var, dr_dx2, i_residual),
            (p2_y_var, dr_dy2, i_residual),
            (p3_x_var, dr_dx3, i_residual),
            (p3_y_var, dr_dy3, i_residual),
            (p4_x_var, dr_dx4, i_residual),
            (p4_y_var, dr_dy4, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.line1.get_involved_primitive_ids()
        ids_2 = self.line2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class LineLineAngle(BaseConstraint):
    line1: Line
    line2: Line
    angle: float = field()

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # Get direction vectors for both lines.
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

        v1 = p2 - p1
        v2 = p4 - p3

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

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = atan2(v1×v2, v1·v2) - α
        # ∂R/∂x1 = (y1 - y2)/(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)
        # ∂R/∂y1 = (-x1 + x2)/(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)
        # ∂R/∂x2 = (-y1 + y2)/(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)
        # ∂R/∂y2 = (x1 - x2)/(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)
        # ∂R/∂x3 = (-y3 + y4)/(x3**2 - 2*x3*x4 + x4**2 + y3**2 - 2*y3*y4 + y4**2)
        # ∂R/∂y3 = (x3 - x4)/(x3**2 - 2*x3*x4 + x4**2 + y3**2 - 2*y3*y4 + y4**2)
        # ∂R/∂x4 = (y3 - y4)/(x3**2 - 2*x3*x4 + x4**2 + y3**2 - 2*y3*y4 + y4**2)
        # ∂R/∂y4 = (-x3 + x4)/(x3**2 - 2*x3*x4 + x4**2 + y3**2 - 2*y3*y4 + y4**2)

        # Get points.
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        p3 = self.line2.p1.get_state(variable_values)
        p4 = self.line2.p2.get_state(variable_values)

        # Get their components.
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]
        x4, y4 = p4[0], p4[1]

        # Calculate magnitudes.
        mag1 = nb.np.linalg.norm(p2 - p1)  # sqrt((x1 - x2)**2 + (y1 - y2)**2)
        mag2 = nb.np.linalg.norm(p4 - p3)  # sqrt((x3 - x4)**2 + (y3 - y4)**2)

        # Avoid division by zero.
        if mag1 < EPS or mag2 < EPS:
            return []

        # Calculate derivatives.

        # Note that our denominator terms for the partial derivatives above are
        # the squared magnitudes of the vectors, i.e.:
        # x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2 == (x1 - x2)²  + (y1 - y2)²
        # x3**2 - 2*x3*x4 + x4**2 + y3**2 - 2*y3*y4 + y4**2 == (x3 - x4)²  + (y3 - y4)²
        mag1_squared = mag1**2
        mag2_squared = mag2**2

        dr_dx1 = (y1 - y2) / mag1_squared
        dr_dy1 = (-x1 + x2) / mag1_squared
        dr_dx2 = (-y1 + y2) / mag1_squared
        dr_dy2 = (x1 - x2) / mag1_squared
        dr_dx3 = (-y3 + y4) / mag2_squared
        dr_dy3 = (x3 - x4) / mag2_squared
        dr_dx4 = (y3 - y4) / mag2_squared
        dr_dy4 = (-x3 + x4) / mag2_squared

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

        # Get the variable IDs for the points involved.
        p1_x_var = self.line1.p1.get_variable_ids()[0]
        p1_y_var = self.line1.p1.get_variable_ids()[1]
        p2_x_var = self.line1.p2.get_variable_ids()[0]
        p2_y_var = self.line1.p2.get_variable_ids()[1]
        p3_x_var = self.line2.p1.get_variable_ids()[0]
        p3_y_var = self.line2.p1.get_variable_ids()[1]
        p4_x_var = self.line2.p2.get_variable_ids()[0]
        p4_y_var = self.line2.p2.get_variable_ids()[1]

        return [
            (p1_x_var, dr_dx1, i_residual),
            (p1_y_var, dr_dy1, i_residual),
            (p2_x_var, dr_dx2, i_residual),
            (p2_y_var, dr_dy2, i_residual),
            (p3_x_var, dr_dx3, i_residual),
            (p3_y_var, dr_dy3, i_residual),
            (p4_x_var, dr_dx4, i_residual),
            (p4_y_var, dr_dy4, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.line1.get_involved_primitive_ids()
        ids_2 = self.line2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class LineLineDistance(BaseConstraint):
    line1: Line
    line2: Line
    distance: float

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # Assumes lines are parallel, enforced by a LinesParallel constraint.
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        p3 = self.line2.p1.get_state(variable_values)
        # p4 = self.line2.p2.get_state(variable_values)

        v = p2 - p1
        w = p3 - p1

        # Distance = |v x w| / |v|
        mag_v = nb.np.linalg.norm(v)
        is_valid = mag_v > EPS
        safe_mag_v = nb.np.where(mag_v < EPS, 1.0, mag_v)

        cross_product_mag = abs(v[0] * w[1] - v[1] * w[0])
        current_dist = cross_product_mag / safe_mag_v

        return nb.np.where(
            is_valid, nb.np.array([current_dist - self.distance]), nb.np.array([0.0])
        )

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = (|v × w| / |v|) - d
        # ∂R/∂x1 = (-(x1 - x2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)) + (y2 - yp)*((x1 - x2)**2 + (y1 - y2)**2))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂y1 = ((-x2 + xp)*((x1 - x2)**2 + (y1 - y2)**2) - (y1 - y2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂x2 = ((x1 - x2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)) + (-y1 + yp)*((x1 - x2)**2 + (y1 - y2)**2))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂y2 = ((x1 - xp)*((x1 - x2)**2 + (y1 - y2)**2) + (y1 - y2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂xp = (y1 - y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂yp = (-x1 + x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # Get points.
        p1 = self.line1.p1.get_state(variable_values)
        p2 = self.line1.p2.get_state(variable_values)
        pp = self.line2.p1.get_state(variable_values)

        # Get their components.
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        xp, yp = pp[0], pp[1]

        # Calculate magnitudes.
        mag1 = nb.np.linalg.norm(p2 - p1)  # sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # Avoid division by zero.
        if mag1 < EPS:
            return []

        # I think we can basically drop in the formulae above.
        # fmt: off
        # ruff: noqa
        dr_dx1 = (-(x1 - x2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)) + (y2 - yp)*((x1 - x2)**2 + (y1 - y2)**2))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        dr_dy1 = ((-x2 + xp)*((x1 - x2)**2 + (y1 - y2)**2) - (y1 - y2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        dr_dx2 = ((x1 - x2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)) + (-y1 + yp)*((x1 - x2)**2 + (y1 - y2)**2))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        dr_dy2 = ((x1 - xp)*((x1 - x2)**2 + (y1 - y2)**2) + (y1 - y2)*((x1 - x2)*(y1 - yp) - (x1 - xp)*(y1 - y2)))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        dr_dxp = (y1 - y2)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        dr_dyp = (-x1 + x2)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # fmt: on
        # ruff: enable

        # Make floats.
        dr_dx1 = float(dr_dx1)
        dr_dy1 = float(dr_dy1)
        dr_dx2 = float(dr_dx2)
        dr_dy2 = float(dr_dy2)
        dr_dxp = float(dr_dxp)
        dr_dyp = float(dr_dyp)

        # This constraint has a scalar residual.
        i_residual = 0

        # Get the variable IDs for the points involved.
        p1_x_var = self.line1.p1.get_variable_ids()[0]
        p1_y_var = self.line1.p1.get_variable_ids()[1]
        p2_x_var = self.line1.p2.get_variable_ids()[0]
        p2_y_var = self.line1.p2.get_variable_ids()[1]
        pp_x_var = self.line2.p1.get_variable_ids()[0]
        pp_y_var = self.line2.p1.get_variable_ids()[1]

        return [
            (p1_x_var, dr_dx1, i_residual),
            (p1_y_var, dr_dy1, i_residual),
            (p2_x_var, dr_dx2, i_residual),
            (p2_y_var, dr_dy2, i_residual),
            (pp_x_var, dr_dxp, i_residual),
            (pp_y_var, dr_dyp, i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        ids_1 = self.line1.get_involved_primitive_ids()
        ids_2 = self.line2.get_involved_primitive_ids()

        return frozenset(ids_1.union(ids_2))


@dataclass
class CircleRadius(BaseConstraint):
    circle: Circle
    radius: float

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # We don't need get_state, as we only care about the radius variable.

        # Ask the circle for the ID of its radius variable.
        radius_var_id = self.circle.get_variable_ids()[0]

        # Look up the current value of the radius in the map.
        current_radius = variable_values[radius_var_id]

        return nb.np.array([current_radius - self.radius])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # The residual is R = r_current - r_target.
        # The only partial derivative that is non-zero is ∂R/∂r_current, which is 1.
        radius_var_id = self.circle.get_variable_ids()[0]

        # This constraint has a scalar residual.
        i_residual = 0

        return [(radius_var_id, 1.0, i_residual)]

    def get_involved_primitive_ids(self) -> frozenset:
        # This constraint involves the circle itself.
        return frozenset(self.circle.get_involved_primitive_ids())


@dataclass
class LineTangentToCircle(BaseConstraint):
    line: Line
    circle: Circle
    directional: bool = False

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # Get the current state of the primitives.
        p1 = self.line.p1.get_state(variable_values)
        p2 = self.line.p2.get_state(variable_values)

        # Unpack the circle's state: center position and radius.
        circle_state = self.circle.get_state(variable_values)
        center, radius = circle_state[:2], circle_state[2]

        # Calculate the signed distance from the circle's center to the line
        # Formula: distance = (v × w) / |v|
        # where v is the line vector and w is the vector from p1 to the center.
        v = p2 - p1
        w = center - p1

        mag_v = nb.np.linalg.norm(v)

        if mag_v < EPS:
            # TODO: Handle degenerate line case better.
            return nb.np.array([0.0])

        # Avoid division by zero for a zero-length line segment.
        safe_mag_v = nb.np.where(mag_v < EPS, 1.0, mag_v)

        # Signed cross product (no absolute value).
        cross_product = v[0] * w[1] - v[1] * w[0]
        distance_signed = cross_product / safe_mag_v

        # Smooth non-directional version.
        # https://math.stackexchange.com/questions/1284946/soft-absolute-value
        # This delta value means we avoid non-differentiability at zero,
        # but it also means that the constraint is never perfectly satisfied.
        # TODO: We might want to revisit or tune this.
        delta = 1e-8 * (safe_mag_v + 1.0)
        distance_abs = nb.np.sqrt(distance_signed**2 + delta**2)

        distance = nb.np.where(self.directional, distance_signed, distance_abs)

        # The residual is the difference between this distance and the circle's radius.
        residual = distance - radius

        return nb.np.array([residual])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = ((x2-x1)*(yc-y1) - (y2-y1)*(xc-x1)) / sqrt((x2-x1)**2 + (y2-y1)**2) - r
        # ∂R/∂x1 = (-(x1 - x2)*((x1 - x2)*(y1 - yc) - (x1 - xc)*(y1 - y2)) + (y2 - yc)*((x1 - x2)**2 + (y1 - y2)**2))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂y1 = ((-x2 + xc)*((x1 - x2)**2 + (y1 - y2)**2) - (y1 - y2)*((x1 - x2)*(y1 - yc) - (x1 - xc)*(y1 - y2)))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂x2 = ((x1 - x2)*((x1 - x2)*(y1 - yc) - (x1 - xc)*(y1 - y2)) + (-y1 + yc)*((x1 - x2)**2 + (y1 - y2)**2))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂y2 = ((x1 - xc)*((x1 - x2)**2 + (y1 - y2)**2) + (y1 - y2)*((x1 - x2)*(y1 - yc) - (x1 - xc)*(y1 - y2)))/((x1 - x2)**2 + (y1 - y2)**2)**(3/2)
        # ∂R/∂xc = (y1 - y2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂yc = (-x1 + x2)/sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # ∂R/∂r = -1

        p1 = self.line.p1.get_state(variable_values)
        p2 = self.line.p2.get_state(variable_values)
        circle_state = self.circle.get_state(variable_values)
        center, _ = circle_state[:2], circle_state[2]

        x1, y1 = p1
        x2, y2 = p2
        xc, yc = center

        # Calculate common terms.
        dx = x1 - x2
        dy = y1 - y2
        mag_v_sq = dx**2 + dy**2

        if mag_v_sq < EPS:
            # TODO: Handle degenerate line case better.
            return []

        mag_v = np.sqrt(mag_v_sq)
        mag_v_cubed = mag_v_sq * mag_v

        # Use same safe_mag_v and delta pattern as residual.
        safe_mag_v = mag_v if mag_v >= EPS else 1.0
        delta = 1e-8 * (safe_mag_v + 1.0)

        # Cross product term that appears in the derivatives.
        cross_product = dx * (y1 - yc) - (x1 - xc) * dy
        distance_signed = cross_product / safe_mag_v
        distance_abs = np.sqrt(distance_signed**2 + delta**2)  # Smoothed.

        # This scale factor will mop up sign between directional and non-directional
        # cases. If directional is True, scale is 1 (R = d - r),
        # otherwise we use d/(sqrt(d^2+delta^2) (smooths |d| at 0).
        # Note that we don't account for the derivative of the delta bit in our
        # results here, but we should otherwise match the gradient of the residual
        # function above and the delta term is very small.
        # See https://math.stackexchange.com/questions/1284946/soft-absolute-value
        scale = 1.0 if self.directional else (distance_signed / distance_abs)

        # Apply sign multiplier to all derivatives except radius
        # fmt: off
        # ruff: noqa
        dr_dx1 = scale * (-dx * cross_product + (y2 - yc) * mag_v_sq) / mag_v_cubed
        dr_dy1 = scale * ((-x2 + xc) * mag_v_sq - dy * cross_product) / mag_v_cubed
        dr_dx2 = scale * (dx * cross_product + (-y1 + yc) * mag_v_sq) / mag_v_cubed
        dr_dy2 = scale * ((x1 - xc) * mag_v_sq + dy * cross_product) / mag_v_cubed
        dr_dxc = scale * (y1 - y2) / mag_v
        dr_dyc = scale * (-x1 + x2) / mag_v
        dr_dr = -1.0
        # fmt: on
        # ruff: enable

        p1_vars = self.line.p1.get_variable_ids()
        p2_vars = self.line.p2.get_variable_ids()
        center_vars = self.circle.center.get_variable_ids()  # Get from centre point.
        radius_var = self.circle.get_variable_ids()[0]  # Radius is the only variable.

        return [
            (p1_vars[0], float(dr_dx1), 0),
            (p1_vars[1], float(dr_dy1), 0),
            (p2_vars[0], float(dr_dx2), 0),
            (p2_vars[1], float(dr_dy2), 0),
            (center_vars[0], float(dr_dxc), 0),
            (center_vars[1], float(dr_dyc), 0),
            (radius_var, float(dr_dr), 0),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        # This constraint involves the line, its points, the circle, and its center.
        return frozenset(
            self.line.get_involved_primitive_ids().union(
                self.circle.get_involved_primitive_ids()
            )
        )


@dataclass
class PointsEquidistant(BaseConstraint):
    """
    Constrains two points, p1 and p2, to be the same distance from a third
    point, center. This is the key definitional constraint for a circular arc.
    """

    center: Point
    p1: Point
    p2: Point

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # Get the current position of all three points.
        center_pos = self.center.get_state(variable_values)
        p1_pos = self.p1.get_state(variable_values)
        p2_pos = self.p2.get_state(variable_values)

        # For numerical stability and simpler derivatives, we compare the squared
        # distances. The residual is zero if the distances are equal.
        # R = distance(center, p1)² - distance(center, p2)²
        dist1_sq = (p1_pos[0] - center_pos[0]) ** 2 + (p1_pos[1] - center_pos[1]) ** 2
        dist2_sq = (p2_pos[0] - center_pos[0]) ** 2 + (p2_pos[1] - center_pos[1]) ** 2

        return nb.np.array([dist1_sq - dist2_sq])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Residual: R = (x1-xc)²+(y1-yc)² - (x2-xc)²-(y2-yc)²
        # The partial derivatives are:
        # ∂R/∂x1 = 2*(x1-xc)
        # ∂R/∂y1 = 2*(y1-yc)
        # ∂R/∂x2 = -2*(x2-xc)
        # ∂R/∂y2 = -2*(y2-yc)
        # ∂R/∂xc = 2*(x2-x1)
        # ∂R/∂yc = 2*(y2-y1)

        center_pos = self.center.get_state(variable_values)
        p1_pos = self.p1.get_state(variable_values)
        p2_pos = self.p2.get_state(variable_values)

        xc, yc = center_pos
        x1, y1 = p1_pos
        x2, y2 = p2_pos

        # Calculate derivative values.
        dr_dx1 = 2 * (x1 - xc)
        dr_dy1 = 2 * (y1 - yc)
        dr_dx2 = -2 * (x2 - xc)
        dr_dy2 = -2 * (y2 - yc)
        dr_dxc = 2 * (x2 - x1)
        dr_dyc = 2 * (y2 - y1)

        # Get the variable IDs to build the Jacobian row entries.
        center_vars = self.center.get_variable_ids()
        p1_vars = self.p1.get_variable_ids()
        p2_vars = self.p2.get_variable_ids()

        # This constraint has a single residual, so the local row index is always 0.
        i_residual = 0

        return [
            (p1_vars[0], float(dr_dx1), i_residual),
            (p1_vars[1], float(dr_dy1), i_residual),
            (p2_vars[0], float(dr_dx2), i_residual),
            (p2_vars[1], float(dr_dy2), i_residual),
            (center_vars[0], float(dr_dxc), i_residual),
            (center_vars[1], float(dr_dyc), i_residual),
        ]

    def get_involved_primitive_ids(self) -> frozenset:
        # This constraint involves all three points.
        return frozenset(
            self.center.get_involved_primitive_ids().union(
                self.p1.get_involved_primitive_ids(),
                self.p2.get_involved_primitive_ids(),
            )
        )


@dataclass
class ArcRadius(BaseConstraint):
    arc: CircularArc
    radius: float

    @property
    def n_residual_rows(self) -> int:
        # We need two residuals: one for center to start distance, one for center to end distance.
        return 2

    def get_residual(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # Reuse existing implementation from PointPointEuclideanDistance.
        center_to_start = PointPointEuclideanDistance(
            self.arc.center, self.arc.start, self.radius
        )
        center_to_end = PointPointEuclideanDistance(
            self.arc.center, self.arc.end, self.radius
        )

        r1 = center_to_start.get_residual(variable_values)
        r2 = center_to_end.get_residual(variable_values)

        return nb.np.array([r1[0], r2[0]])

    def get_jacobian_row_values(
        self, variable_values: Mapping[str, float]
    ) -> List[Tuple[str, float, int]]:
        # Reuse existing implementation from PointPointEuclideanDistance.
        center_to_start = PointPointEuclideanDistance(
            self.arc.center, self.arc.start, self.radius
        )
        center_to_end = PointPointEuclideanDistance(
            self.arc.center, self.arc.end, self.radius
        )

        # Get Jacobian entries for both constraints.
        start_entries = center_to_start.get_jacobian_row_values(variable_values)
        end_entries = center_to_end.get_jacobian_row_values(variable_values)

        # For start constraint entries, use residual row 0.
        start_entries_with_row = [
            (var_id, value, 0) for var_id, value, _ in start_entries
        ]

        # For end constraint entries, use residual row 1.
        end_entries_with_row = [(var_id, value, 1) for var_id, value, _ in end_entries]

        return start_entries_with_row + end_entries_with_row

    def get_involved_primitive_ids(self) -> frozenset:
        return frozenset(self.arc.get_involved_primitive_ids())


Constraint = Union[
    ArcRadius,
    CircleRadius,
    LineHorizontal,
    LineLength,
    LineLineAngle,
    LineLineDistance,
    LinesEqualLength,
    LinesParallel,
    LinesPerpendicular,
    LineTangentToCircle,
    LineVertical,
    PointFixed,
    PointPointCoincident,
    PointPointEuclideanDistance,
    PointPointXDistance,
    PointPointYDistance,
    PointsEquidistant,
]
