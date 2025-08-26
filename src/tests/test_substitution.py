import pytest

from newton.constraints import PointFixed, PointPointCoincident
from newton.primitives import Point
from newton.solver_dense import Solver2DDense
from newton.symbolic_substitution import SymbolicSubstitution


def test_substitution_without_constraint_rewriting():
    # Define three points, two of which are coincident.
    p1 = Point(x=0.0, y=0.0, id="P1")
    p2 = Point(x=0.0, y=0.0, id="P2")  # Coincident with P1.
    p3 = Point(x=5.0, y=0.0, id="P3")

    # Define constraints.
    constraints = [
        PointPointCoincident(p1=p1, p2=p2),  # Should eliminate P2_x and P2_y.
        PointFixed(point=p1),  # Fix P1 in place.
    ]

    primitives = [p1, p2, p3]  # P3 is unused.

    # Apply symbolic substitution (no rewriting).
    substitution = SymbolicSubstitution()
    results = substitution.apply_substitutions(constraints, primitives)

    # Verify that substitution occurred.
    sub_map = results.substitution_map
    assert "P2_x" in sub_map and sub_map["P2_x"].startswith("P1_")
    assert "P2_y" in sub_map and sub_map["P2_y"].startswith("P1_")

    # Constraint elimination should have occurred.
    assert results.constraints_eliminated == 1
    assert results.constraints_unchanged == 1

    # Solve with the substituted constraints.
    solver = Solver2DDense(
        primitives=primitives, constraints=results.active_constraints
    )
    solver.solve()

    # P2 should match P1 exactly due to substitution
    assert p2.x == pytest.approx(p1.x)
    assert p2.y == pytest.approx(p1.y)
