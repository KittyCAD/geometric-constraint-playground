from typing import Any, Dict, List

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
            graph.add_node(constraint_id, bipartite=1, constraint_index=i)

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

    def analyze_structure(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        # Processes the dependency graph to find and define independent subproblems.
        subproblems = []

        for component in nx.connected_components(graph):
            # Find all constraints that are part of this subproblem.
            sub_constraints = [
                self.constraints[graph.nodes[node]["constraint_index"]]
                for node in component
                if graph.nodes[node].get("bipartite") == 1
            ]

            # Find all free points that are part of this subproblem.
            sub_free_point_ids = {
                node.split("_")[0]
                for node in component
                if graph.nodes[node].get("bipartite") == 0
            }

            # Only create a subproblem if it has variables to solve for.
            if not sub_free_point_ids:
                continue

            sub_free_points = [self.point_map[pid] for pid in sub_free_point_ids]
            subproblems.append(
                {"constraints": sub_constraints, "free_points": sub_free_points}
            )

        return subproblems

    def solve_subproblem(self, subproblem: Dict[str, Any]):
        # Solves a single, independent group of constraints and variables.
        free_points = subproblem["free_points"]
        constraints = subproblem["constraints"]

        if DEBUG_LOG:
            point_ids = [p.id for p in free_points]
            print(f"Solving Subproblem: {point_ids}")

        # Create the initial guess vector from the current positions of relevant free points.
        initial_guess = np.array([[p.x, p.y] for p in free_points]).flatten()

        def get_all_positions(subproblem_free_vars: jnp.ndarray) -> dict:
            # Create a dictionary of all point positions for the residual functions.
            # Start with the current state of all points.
            positions = {p.id: jnp.array([p.x, p.y]) for p in self.points}

            # Overwrite the positions of the points being solved in this subproblem.
            for i, p in enumerate(free_points):
                positions[p.id] = subproblem_free_vars[i * 2 : i * 2 + 2]

            return positions

        def residuals_vector(free_vars: jnp.ndarray) -> jnp.ndarray:
            # This function returns a flat vector of all constraint residuals.
            # We don't need to worry about doing residual squaring here.
            positions = get_all_positions(free_vars)
            residual_parts = [compute_residual(c, positions) for c in constraints]
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
            xtol=SOLVE_TOLERANCE,
            ftol=SOLVE_TOLERANCE,
            gtol=SOLVE_TOLERANCE,
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
            for i, p in enumerate(free_points):
                p.x, p.y = final_vars[i * 2], final_vars[i * 2 + 1]
        else:
            raise ValueError(f"Solver failed to find a solution: {result.message}")

    def solve(self):
        if not self.free_points:
            print("No free points to solve for. System is fully constrained or empty.")
            return

        # Build the dependency graph to understand the structure of the problem.
        graph = self.build_dependency_graph()
        subproblems = self.analyze_structure(graph)

        if DEBUG_LOG:
            print(f"Graph analysis found {len(subproblems)} subproblem(s).")

        # Solve each independent subproblem sequentially.
        for subproblem in subproblems:
            self.solve_subproblem(subproblem)
