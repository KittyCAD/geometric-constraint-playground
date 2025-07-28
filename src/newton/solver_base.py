from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import OptimizeResult

from newton.constraints import BaseConstraint, Constraint, PointFixed
from newton.preprocessor import Preprocessor
from newton.primitives import Point

SOLVE_TOLERANCE = 1e-8
DEBUG_LOG = True

if DEBUG_LOG:
    np.set_printoptions(precision=3, suppress=True, linewidth=120)


class Solver2D(ABC):
    def __init__(self, points: List[Point], constraints: List[Constraint]):
        self.points = points
        self.constraints: Sequence[BaseConstraint] = constraints
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

            from newton.constraints import (
                LineHorizontal,
                LineLineAngle,
                LinesParallel,
                LinesPerpendicular,
                LineVertical,
                PointPointDistance,
            )

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

    def find_separable_systems(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        # This finds wholly disconnected problems in the graph.
        # This is different from independently soluble subproblems, which we need
        # to tackle later.
        constraint_systems = []

        for component in nx.connected_components(graph):
            sub_constraint_indices = {
                graph.nodes[node]["constraint_index"]
                for node in component
                if graph.nodes[node].get("bipartite") == 1
            }
            sub_constraints = [self.constraints[i] for i in sub_constraint_indices]

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
            constraint_systems.append(
                {"constraints": sub_constraints, "free_points": sub_free_points}
            )

        return constraint_systems

    def update_points_from_result(
        self, result: OptimizeResult, free_points: List[Point]
    ):
        if result.success:
            max_error = np.max(np.abs(result.fun))
            if max_error > SOLVE_TOLERANCE:
                raise ValueError(f"Solver failed tolerance. Max error: {max_error}")

            final_vars = result.x
            for i, p in enumerate(free_points):
                p.x, p.y = final_vars[i * 2], final_vars[i * 2 + 1]
        else:
            raise ValueError(f"Solver failed to find a solution: {result.message}")

    def validate_constraint_systems(self, subproblems: List[Dict[str, Any]]) -> None:
        preprocessor = Preprocessor()

        if DEBUG_LOG:
            print("Validating constraints for each subproblem...")

        for i, subproblem in enumerate(subproblems):
            # The preprocessor will raise a ConflictError if any issues are found.
            # If it returns, the subproblem's constraints are considered valid.
            preprocessor.run(subproblem["constraints"])
            if DEBUG_LOG:
                print(f"  - Subproblem {i + 1} is valid.")

    @abstractmethod
    def solve_constraint_system(self, subproblem: Dict[str, Any]):
        # Each concrete solver must implement its own subproblem solving logic.
        pass

    def solve(self):
        if not self.free_points:
            print("No free points to solve for. System is fully constrained or empty.")
            return

        # Split the problem into wholly disconnected problems.
        graph = self.build_dependency_graph()
        constraint_systems = self.find_separable_systems(graph)

        # Validate the constraints in each disconnected system before solving.
        self.validate_constraint_systems(constraint_systems)

        # If validation passes, proceed with the numerical solve.
        if DEBUG_LOG:
            print(
                f"Graph analysis found {len(constraint_systems)} valid disconnected systems(s)."
            )
            print(f"Using {self.__class__.__name__}.")

        for constraint_system in constraint_systems:
            self.solve_constraint_system(constraint_system)
