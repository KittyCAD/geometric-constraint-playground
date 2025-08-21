"""
Tests the actual solving of geometric constraint systems using both solvers.
"""

import pytest

from newton.constraints import (
    LineHorizontal,
    PointFixed,
    PointPointEuclideanDistance,
)
from newton.primitives import Line, Point
from newton.solver_dense import Solver2DDense
from newton.solver_sparse import Solver2DSparse

SOLVER_CLASSES = [Solver2DSparse, Solver2DDense]


@pytest.mark.parametrize("Solver", SOLVER_CLASSES)
def test_point_fixed_constraint(Solver):
    # Initial state: a point.
    p1 = Point(x=5.0, y=10.0, id="P1")

    # Fix.
    constraints = [PointFixed(point=p1)]

    solver = Solver(primitives=[p1], constraints=constraints)
    solver.solve()

    # Shouldn't move.
    assert p1.x == pytest.approx(5.0)
    assert p1.y == pytest.approx(10.0)


@pytest.mark.parametrize("Solver", SOLVER_CLASSES)
def test_line_horizontal_constraint(Solver):
    # Initial state: a line that is not horizontal.
    p1 = Point(x=0.0, y=0.0, id="P1")
    p2 = Point(x=5.0, y=10.0, id="P2")
    line = Line(p1, p2, id="L1")

    # Constraints:
    constraints = [
        PointFixed(point=p1),
        LineHorizontal(line=line),
    ]

    solver = Solver(primitives=[p1, p2, line], constraints=constraints)
    solver.solve()

    # Assert that the y-coordinates of both points are now equal.
    # Since p1 was fixed at y=0, p2.y should now be 0.
    assert p1.y == pytest.approx(p2.y)
    assert p2.y == pytest.approx(0.0)

    # The x-coordinate of p2 should not have changed (minimum norm solution).
    assert p2.x == pytest.approx(5.0)


@pytest.mark.parametrize("Solver", SOLVER_CLASSES)
def test_point_point_distance_constraint(Solver):
    # Initial state: p2 is 10 units away from p1.
    p1 = Point(x=0.0, y=0.0, id="P1")
    p2 = Point(x=10.0, y=0.0, id="P2")

    # Constraint: Fix p1 and enforce a distance of 5.0 between p1 and p2.
    target_distance = 5.0
    constraints = [
        PointFixed(point=p1),
        PointPointEuclideanDistance(p1, p2, distance=target_distance),
    ]

    solver = Solver(primitives=[p1, p2], constraints=constraints)
    solver.solve()

    # The solver should find the closest solution to the initial state.
    # Moving p2 from (10,0) to (5,0) is the minimum change required.
    assert p2.x == pytest.approx(5.0)
    assert p2.y == pytest.approx(0.0)

    # Verify the final distance.
    final_dist = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
    assert final_dist == pytest.approx(target_distance)
