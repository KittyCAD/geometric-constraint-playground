import pytest

from newton.constraints import (
    LineHorizontal,
    LinesPerpendicular,
    PointFixed,
    PointPointXDistance,
    PointPointYDistance,
)
from newton.primitives import Line, Point
from newton.solver_sparse import Solver2DSparse
from newton.symbolic_substitution import perform_symbolic_substitution


def test_perpendicular_with_shared_vertex():
    """
    Build a simple polyline p1-p2-p3 with two segments:
    L1 = (p1, p2), L2 = (p2, p3)

    Then constrain L1 perp L2 at the shared vertex p2.
    """

    p1 = Point(x=0.0, y=0.0, id="P1")
    p2 = Point(x=3.2, y=0.7, id="P2")
    p3 = Point(x=4.8, y=2.1, id="P3")

    l1 = Line(p1, p2, id="L1")
    l2 = Line(p2, p3, id="L2")

    assert l1.p2 is l2.p1, "Segments must share the middle vertex"

    primitives = [p1, p2, p3, l1, l2]

    constraints = [
        PointFixed(point=p1),
        LineHorizontal(l1),
        PointPointXDistance(p1, p2, distance=4.0),
        PointPointYDistance(p2, p3, distance=3.0),
        LinesPerpendicular(l1, l2),
    ]

    sub = perform_symbolic_substitution(constraints, primitives)

    # Solve.
    solver = Solver2DSparse(primitives=primitives, constraints=sub.active_constraints)
    solver.solve()

    assert p1.x == pytest.approx(0.0)
    assert p1.y == pytest.approx(0.0)

    assert p2.x == pytest.approx(4.0)
    assert p2.y == pytest.approx(0.0)

    assert p3.x == pytest.approx(4.0)
    assert p3.y == pytest.approx(3.0)

    # Check perpendicularity explicitly via dot product.
    v1 = (p2.x - p1.x, p2.y - p1.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    assert dot == pytest.approx(0.0, abs=1e-9)
