import logging
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import OptimizeResult

from newton.constants import NONZERO_RANK_TOLERANCE
from newton.constraint_validator import ConstraintValidator
from newton.constraints import (
    BaseConstraint,
    Constraint,
    LineLength,
    LineLineDistance,
    LinesEqualLength,
    PointFixed,
    PointPointXDistance,
    PointPointYDistance,
)
from newton.logging_config import logger
from newton.matrix_utils import compute_rank
from newton.primitives import Point
from newton.structural_analyzer import StructuralAnalyzer

SOLVE_VALIDATION_TOLERANCE = 1e-6  ## Our maximum allowed error on any constraint.
SOLVER_CONVERGENCE_TOLERANCE = 1e-10  ## The tolerance for convergence in the solver.

# Configure numpy print options for debug output
if logger.isEnabledFor(logging.DEBUG):
    np.set_printoptions(precision=3, suppress=True, linewidth=120)


class Solver2D(ABC):
    def __init__(self, points: List[Point], constraints: List[Constraint]):
        self.points = points
        self.constraints: Sequence[BaseConstraint] = constraints
        self.free_points = self.identify_free_points()
        self.point_map = {p.id: p for p in self.points}
        self.module: ModuleType = (
            np  # Default to numpy, can be overridden by subclasses.
        )

    def identify_free_points(self) -> List[Point]:
        # Find the non-fixed points our solver can play tunes with.
        # However... we actually do want to feed fixed points to the structural analyzer,
        # because they may be help in solving other parts of the system.
        fixed_point_ids = set()
        for c in self.constraints:
            if isinstance(c, PointFixed):
                fixed_point_ids.add(c.point.id)

        return [p for p in self.points if p.id not in fixed_point_ids]

    def build_dependency_graph(self) -> nx.Graph:
        # The graph must be built with ALL points to correctly identify components
        # that may be connected by a fixed point.
        graph = nx.Graph()
        variable_nodes = [f"{p.id}_{c}" for p in self.points for c in ("x", "y")]
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
                PointPointEuclideanDistance,
            )

            match c:
                case PointFixed():
                    points_involved.append(c.point)
                case (
                    PointPointEuclideanDistance()
                    | PointPointXDistance()
                    | PointPointYDistance()
                ):
                    points_involved.extend([c.p1, c.p2])
                case LinesParallel() | LinesPerpendicular() | LineLineAngle():
                    points_involved.extend(
                        [c.line1.p1, c.line1.p2, c.line2.p1, c.line2.p2]
                    )
                case LineHorizontal() | LineVertical() | LineLength():
                    points_involved.extend([c.line.p1, c.line.p2])
                case LinesEqualLength() | LineLineDistance():
                    points_involved.extend(
                        [c.line1.p1, c.line1.p2, c.line2.p1, c.line2.p2]
                    )
                case _:
                    raise ValueError(f"Unknown constraint type: {type(c)}")

            # Add edges only for all unique points involved in the constraint.
            unique_points_involved = {p.id: p for p in points_involved}.values()
            for p in unique_points_involved:
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
            if not sub_constraint_indices:
                continue

            sub_constraints = [self.constraints[i] for i in sub_constraint_indices]
            sub_point_ids = {
                node.split("_")[0]
                for node in component
                if graph.nodes[node].get("bipartite") == 0
            }
            sub_points = [self.point_map[pid] for pid in sub_point_ids]

            constraint_systems.append(
                {"constraints": sub_constraints, "points": sub_points}
            )
        return constraint_systems

    def update_points_from_result(
        self, result: OptimizeResult, free_points: List[Point]
    ):
        if result.success:
            max_error = np.max(np.abs(result.fun))
            if max_error > SOLVE_VALIDATION_TOLERANCE:
                raise ValueError(f"Solver failed tolerance. Max error: {max_error}")

            final_vars = result.x
            for i, p in enumerate(free_points):
                p.x, p.y = final_vars[i * 2], final_vars[i * 2 + 1]
        else:
            raise ValueError(f"Solver failed to find a solution: {result.message}")

    def validate_constraint_systems(self, systems: List[Dict[str, Any]]) -> None:
        validator = ConstraintValidator()

        logger.debug("Validating constraints for each disconnected system...")

        for i, system in enumerate(systems):
            # The validator will raise a ConflictError if any issues are found.
            # If it returns, the subproblem's constraints are considered valid.
            validator.run(system["constraints"])
            logger.debug(f"  - Disconnected system {i + 1} is valid.")

    def check_system_state(
        self, jacobian, n_variables, n_equations, tolerance=NONZERO_RANK_TOLERANCE
    ):
        rank = compute_rank(jacobian, tolerance, self.module)

        if rank < n_equations:
            logger.warning(
                f"Initial Jacobian {jacobian.shape} has rank {rank} < {n_equations}. "
                "This is likely due to redundant equations."
            )

        if rank < n_variables:
            logger.warning(
                f"Initial Jacobian {jacobian.shape} has rank {rank} < {n_variables}. "
                f"This may lead to convergence issues."
            )

        logger.debug(
            f"Initial Jacobian is {jacobian.shape} with full rank ({rank}). System is well-posed. Starting solver."
        )

    @abstractmethod
    def solve_constraint_system(self, system: Dict[str, Any]):
        # Each concrete solver must implement its own system solving logic.
        pass

    def solve(self):
        if not self.constraints:
            print("No constraints to solve.")
            return

        # Split the problem into wholly disconnected problems.
        graph = self.build_dependency_graph()
        constraint_systems = self.find_separable_systems(graph)

        # Validate the constraints in each disconnected system before solving.
        self.validate_constraint_systems(constraint_systems)

        # If validation passes, proceed with the numerical solve.
        logger.debug(
            f"Graph analysis found {len(constraint_systems)} valid disconnected system(s)."
        )
        logger.info(f"Using {self.__class__.__name__}.")

        # Then, for each separable system, find the sequential solving order.
        for system in constraint_systems:
            if not system["constraints"]:
                continue

            # Pass the complete list of points for the subsystem to the analyser.
            # ! TODO: This is non-deterministic and can lead to underdetermined systems
            # ! failing to solve.
            analyzer = StructuralAnalyzer(system["constraints"], system["points"])
            sequential_blocks = analyzer.find_solving_sequence_full()

            # Now, for each sequential block, we determine its specific free points
            # before passing that to the numerical solver.
            for block in sequential_blocks:
                # A block contains all points involved in its solution.
                all_points_in_block = block["points"]

                # We filter this list against the globally free points to find
                # which variables can be solved in this block.
                free_points_in_block = [
                    p for p in all_points_in_block if p in self.free_points
                ]

                # Only call the numerical solver if there are actual variables to solve for.
                # A block might only contain a PointFixed constraint, which has no free points.
                if free_points_in_block:
                    solver_block = {
                        "free_points": free_points_in_block,
                        "constraints": block["constraints"],
                    }
                    self.solve_constraint_system(solver_block)
