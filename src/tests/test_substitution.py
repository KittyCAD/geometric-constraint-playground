import pytest

from newton.constraints import PointFixed, PointPointCoincident
from newton.primitives import Point
from newton.solver_sparse import Solver2DSparse
from newton.symbolic_substitution import perform_symbolic_substitution


def test_substitution_without_constraint_rewriting():
    # Define three points, two of which are coincident.
    p1 = Point(x=0.0, y=0.0, id="P1")
    p2 = Point(x=0.0, y=0.0, id="P2")  # Coincident with P1.
    p3 = Point(x=5.0, y=0.0, id="P3")  # Unused in constraints.

    # Define constraints.
    constraints = [
        PointPointCoincident(p1=p1, p2=p2),  # Should eliminate P2_x and P2_y.
        PointFixed(point=p1),  # Fix P1 in place.
    ]

    primitives = [p1, p2, p3]

    # Apply symbolic substitution.
    results = perform_symbolic_substitution(constraints, primitives)

    # Substitution map should include variable redirections for P2 -> P1
    assert "P2_x" in results.substitution_map
    assert "P2_y" in results.substitution_map
    assert results.substitution_map["P2_x"].startswith("P1_")
    assert results.substitution_map["P2_y"].startswith("P1_")

    # One constraint (coincidence) should be eliminated, one kept (fixed).
    assert results.constraints_eliminated == 1
    assert results.constraints_unchanged == 1

    # Run solver and verify P2 follows P1
    solver = Solver2DSparse(
        primitives=primitives, constraints=results.active_constraints
    )
    solver.solve()

    assert p2.x == pytest.approx(p1.x)
    assert p2.y == pytest.approx(p1.y)
