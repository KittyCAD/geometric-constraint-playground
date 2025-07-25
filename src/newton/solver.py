from typing import List

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from scipy.optimize import least_squares

from newton.constraints import (
    Constraint,
    LineHorizontal,
    LineLineAngle,
    LinesParallel,
    LinesPerpendicular,
    LineVertical,
    PointFixed,
    PointPointDistance,
)
from newton.primitives import Point
from newton.residuals import compute_residual

SOLVE_TOLERANCE = 1e-10
DEBUG_LOG = True


class Solver2D:
    def __init__(self, points: List[Point], constraints: List[Constraint]):
        self.points = points
        self.constraints = constraints
        self.free_points = self.identify_free_points()
        self.point_map = {p.id: p for p in self.points}
        self.free_point_indices = {p.id: i for i, p in enumerate(self.free_points)}

    def identify_free_points(self) -> List[Point]:
        # Find the non-fixed points our solver can play tunes with.
        fixed_point_ids = set()
        for c in self.constraints:
            if isinstance(c, PointFixed):
                fixed_point_ids.add(c.point.id)

        return [p for p in self.points if p.id not in fixed_point_ids]

    def build_dependency_graph(self):
        graph = nx.Graph()

        # Get all variables (the coordinates of free points the solver can play with).
        variable_nodes = []
        for p in self.free_points:
            variable_nodes.extend([f"{p.id}_x", f"{p.id}_y"])

        graph.add_nodes_from(variable_nodes, bipartite=0)

        # Add constraint nodes and edges
        for i, c in enumerate(self.constraints):
            constraint_id = f"C_{i}_{type(c).__name__}"
            graph.add_node(constraint_id, bipartite=1)

            # Find which points this constraint depends on.
            points_involved = []

            match c:
                case PointFixed():
                    points_involved.append(c.point)
                case PointPointDistance():
                    points_involved.extend([c.p1, c.p2])
                case LinesParallel() | LinesPerpendicular() | LineLineAngle():
                    points_involved.extend(
                        [c.line1.p1, c.line1.p2, c.line2.p1, c.line2.p2]
                    )
                case LineHorizontal() | LineVertical():
                    points_involved.extend([c.line.p1, c.line.p2])
                case _:
                    raise ValueError(f"Unknown constraint type: {type(c)}")

            # Add edges only for free points
            unique_points_involved = {p.id: p for p in points_involved}.values()

            # Add edges only for free points
            for p in unique_points_involved:
                if p in self.free_points:
                    graph.add_edge(constraint_id, f"{p.id}_x")
                    graph.add_edge(constraint_id, f"{p.id}_y")

        return graph

    def solve(self):
        if not self.free_points:
            print("No free points to solve for. System is fully constrained or empty.")
            return None

        # Build the dependency graph to understand the structure of the problem.
        graph = self.build_dependency_graph()

        # Create the initial guess vector from the current positions of free points.
        initial_guess = np.array([[p.x, p.y] for p in self.free_points]).flatten()

        def get_all_positions(free_vars: jnp.ndarray) -> dict:
            # Array to dict.
            positions = {}
            for p in self.points:
                if p not in self.free_points:
                    positions[p.id] = jnp.array([p.x, p.y])

            for i, p in enumerate(self.free_points):
                positions[p.id] = free_vars[i * 2 : i * 2 + 2]

            return positions

        def residuals_vector(free_vars: jnp.ndarray) -> jnp.ndarray:
            # This function returns a flat vector of all constraint residuals.
            # We don't need to worry about doing residual squaring here.
            positions = get_all_positions(free_vars)

            # Compute all residual parts and concatenate them into a single vector.
            residual_parts = [compute_residual(c, positions) for c in self.constraints]
            return jnp.concatenate([jnp.atleast_1d(res) for res in residual_parts])

        # JIT compile the residuals function and its Jacobian.
        jit_residuals = jax.jit(residuals_vector)
        jit_jacobian = jax.jit(jax.jacfwd(residuals_vector))

        if DEBUG_LOG:
            print(
                f"Solving for {len(self.free_points)} free points: {[p.id for p in self.free_points]}"
            )
            print(f"Initial guess: {initial_guess}")

        # Actual solve magic using the least_squares method.
        # Not sure which is most appropriate here... CC Dave Reeves: current thinking
        # Levenberg-Marquardt (lm) is good because least squares and Newton's method.
        result = least_squares(
            fun=jit_residuals,
            x0=initial_guess,
            jac=jit_jacobian,  # type: ignore
            method="lm",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            verbose=2 if DEBUG_LOG else 0,
        )

        if result.success:
            # Check that all constraints are satisfied within the tolerance.
            final_residuals = result.fun
            max_error = np.max(np.abs(final_residuals))

            if max_error > SOLVE_TOLERANCE:
                raise ValueError(
                    f"Solver failed to meet tolerance. "
                    f"Max error: {max_error} > Tolerance: {SOLVE_TOLERANCE}"
                )

            # If passed, update the points with the new positions.
            final_vars = result.x
            for i, p in enumerate(self.free_points):
                p.x, p.y = final_vars[i * 2], final_vars[i * 2 + 1]
        else:
            raise ValueError(f"Solver failed to find a solution: {result.message}")
        return result
