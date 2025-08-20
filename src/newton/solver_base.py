import logging
from abc import ABC, abstractmethod
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List

import networkx as nx
import numpy as np
from scipy.optimize import OptimizeResult

from newton.constants import CONFIG_USE_SYMB_SUB, NONZERO_RANK_TOLERANCE
from newton.constraint_validator import ConstraintValidator
from newton.constraints import (
    Constraint,
    PointFixed,
)
from newton.logging_config import logger
from newton.matrix_utils import compute_rank
from newton.primitives import Circle, Point, Primitive
from newton.symbolic_substitution import find, perform_symbolic_substitution

SOLVE_VALIDATION_TOLERANCE = 1e-6  ## Our maximum allowed error on any constraint.
SOLVER_CONVERGENCE_TOLERANCE = 1e-10  ## The tolerance for convergence in the solver.


class SystemState(Enum):
    UNDERDETERMINED = "underdetermined"
    OVERDETERMINED = "overdetermined"
    FULLY_DETERMINED = "fully determined"


# Configure numpy print options for debug output
if logger.isEnabledFor(logging.DEBUG):
    np.set_printoptions(precision=3, suppress=True, linewidth=120)


class Solver2D(ABC):
    def __init__(self, primitives: List[Primitive], constraints: List[Constraint]):
        # Build our full set of constraints: this will be the user-defined constraints
        # passed in, plus our definitional constraints from the primitives.
        all_constraints = list(constraints)
        for p in primitives:
            all_constraints.extend(p.build_definitional_constraints())

        # Store the primitives and constraints.
        self.primitives = primitives
        self.primitive_map = {p.id: p for p in primitives}
        self.constraints: List[Constraint] = all_constraints
        self.free_primitives = self.identify_free_primitives()

        self.module: ModuleType = (
            np  # Default to numpy, can be overridden by subclasses.
        )

    def identify_free_primitives(self) -> List[Primitive]:
        # Find the non-fixed elements our solver can play tunes with.
        # However... we actually do want to feed fixed points to the structural analyzer,
        # because they may be help in solving other parts of the system.
        fixed_primitive_ids = set()
        for c in self.constraints:
            if isinstance(c, PointFixed):
                fixed_primitive_ids.add(c.point.id)

        return [p for p in self.primitives if p.id not in fixed_primitive_ids]

    def build_dependency_graph(self) -> nx.Graph:
        graph = nx.Graph()

        # Add all variables from all primitives as nodes.
        for p in self.primitives:
            for var_id in p.get_variable_ids():
                graph.add_node(var_id, bipartite=0)  # Variable nodes

        # Add constraint nodes and connect them to the variables they affect.
        for i, c in enumerate(self.constraints):
            constraint_node_id = f"C_{i}_{type(c).__name__}"
            graph.add_node(constraint_node_id, bipartite=1, constraint_index=i)

            # The constraint itself tells us which primitives are involved.
            involved_primitive_ids = c.get_involved_primitive_ids()

            # For each primitive the constraint touches, connect the constraint to all
            # of that primitive's variables.
            for prim_id in involved_primitive_ids:
                if prim_id in self.primitive_map:
                    primitive = self.primitive_map[prim_id]
                    for var_id in primitive.get_variable_ids():
                        graph.add_edge(constraint_node_id, var_id)

        return graph

    def find_disconnected_systems(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        # This finds wholly disconnected problems in the graph.
        # This is different from independently soluble subproblems, which we need
        # to tackle later.
        constraint_systems = []

        for component in nx.connected_components(graph):
            i_comp_constraints = {
                graph.nodes[node]["constraint_index"]
                for node in component
                if graph.nodes[node].get("bipartite") == 1
            }
            if not i_comp_constraints:
                continue

            comp_constraints = [self.constraints[i] for i in i_comp_constraints]
            comp_primitive_ids = {
                node.split("_")[0]
                for node in component
                if graph.nodes[node].get("bipartite") == 0
            }
            comp_primitives = [self.primitive_map[pid] for pid in comp_primitive_ids]

            constraint_systems.append(
                {"constraints": comp_constraints, "primitives": comp_primitives}
            )
        return constraint_systems

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

        # Determine system state based on rank vs variables
        if rank < n_variables:
            system_state = SystemState.UNDERDETERMINED
        elif rank > n_variables:
            system_state = SystemState.OVERDETERMINED
        else:
            system_state = SystemState.FULLY_DETERMINED

        # Report redundant constraints if present.
        if rank < n_equations:
            logger.info(
                f"System has {rank} linearly independent equations (from {n_equations} total) for {n_variables} variables. "
                f"System contains {n_equations - rank} redundant constraint(s)."
            )
        else:
            # All equations are linearly independent.
            logger.info(
                f"System has {rank} linearly independent equations for {n_variables} variables."
            )

        # Report specific information about the system state.
        if system_state == SystemState.UNDERDETERMINED:
            logger.warning(
                f"System is underdetermined with {n_variables - rank} degree(s) of freedom remaining."
            )
        elif system_state == SystemState.OVERDETERMINED:
            logger.warning(
                f"System is overdetermined with {rank - n_variables} extra constraint(s). This may cause conflicts."
            )
        else:
            logger.info("System is fully determined and well-posed.")

    @abstractmethod
    def solve_constraint_system(self, system: Dict[str, Any]):
        # Each concrete solver must implement its own system solving logic.
        pass

    def solve(self):
        if CONFIG_USE_SYMB_SUB:
            self.solve_with_subtitution()
        else:
            self.solve_without_substitution()

    def solve_without_substitution(self):
        if not self.constraints:
            print("No constraints to solve.")
            return

        # Build a 1:1 point mapping for consistency with the substitution method.
        self.primitive_map = {p.id: p for p in self.primitives}

        # Split the problem into wholly disconnected problems.
        graph = self.build_dependency_graph()
        constraint_systems = self.find_disconnected_systems(graph)

        # Validate the constraints in each disconnected system before solving.
        self.validate_constraint_systems(constraint_systems)

        # If validation passes, proceed with the numerical solve.
        logger.debug(
            f"Graph analysis found {len(constraint_systems)} valid disconnected system(s)."
        )
        logger.info(f"Using {self.__class__.__name__}.")

        # Then, for each separable system, solve.
        for system in constraint_systems:
            if not system["constraints"]:
                continue

            if self.free_primitives:
                solver_block = {
                    "free_primitives": self.free_primitives,
                    "constraints": self.constraints,
                    "substitution_map": {},
                }
                self.solve_constraint_system(solver_block)

    def solve_with_subtitution(self):
        if not self.constraints:
            print("No constraints to solve.")
            return

        # Split the problem into wholly disconnected problems.
        graph = self.build_dependency_graph()
        constraint_systems = self.find_disconnected_systems(graph)

        # Validate the constraints in each disconnected system before solving.
        self.validate_constraint_systems(constraint_systems)

        # If validation passes, proceed with the numerical solve.
        logger.debug(
            f"Graph analysis found {len(constraint_systems)} valid disconnected system(s)."
        )
        logger.info(f"Using {self.__class__.__name__}.")

        # Then, for each separable system, solve.
        for system in constraint_systems:
            if not system["constraints"]:
                continue

            # Get both simplified constraints and the primitive mapping.
            simplified_constraints, substitution_map = perform_symbolic_substitution(
                system["constraints"], system["primitives"]
            )

            # Get the simplified primitive list.
            simplified_primitives = self.get_simplified_primitives(
                simplified_constraints
            )

            # Use simplified primitives for the free primitive calculation
            free_primitives = self.get_free_primitives_from_simplified(
                simplified_primitives, substitution_map
            )

            if free_primitives:
                solver_block = {
                    "free_primitives": free_primitives,
                    "constraints": simplified_constraints,
                    "substitution_map": substitution_map,
                }
                self.solve_constraint_system(solver_block)

    def get_simplified_primitives(
        self, simplified_constraints: List[Constraint]
    ) -> List[Primitive]:
        primitive_ids = set()
        for c in simplified_constraints:
            primitive_ids.update(c.get_involved_primitive_ids())

        return [
            self.primitive_map[pid]
            for pid in primitive_ids
            if pid in self.primitive_map
        ]

    def get_free_primitives_from_simplified(
        self, simplified_primitives: List[Primitive], substitution_map: Dict[str, str]
    ) -> List[Primitive]:
        fixed_primitive_ids = set()
        for c in self.constraints:
            if isinstance(c, PointFixed):
                fixed_primitive_ids.add(c.point.id)

        truly_free_primitives = []
        for p in simplified_primitives:
            if p.id in fixed_primitive_ids:
                continue

            # Check if any of the primitive's variables are "roots"
            # (i.e., they are not a key in the substitution map).
            is_truly_free = False
            for var_id in p.get_variable_ids():
                if var_id not in substitution_map:
                    is_truly_free = True
                    break  # Found a free variable, so the primitive is free.

            if is_truly_free:
                truly_free_primitives.append(p)

        return truly_free_primitives

    def compute_final_variable_values(
        self,
        result: OptimizeResult,
        free_primitives: List[Primitive],
        substitution_map: Dict[str, str],
    ) -> Dict[str, float]:
        # Start with the initial state of all variables in the system.
        final_variable_values: Dict[str, float] = {}
        for p in self.primitives:
            final_variable_values.update(p.get_initial_variable_values())

        # Get the variables that were actually part of the numerical solve.
        solved_var_ids = [
            var_id
            for p in free_primitives
            for var_id in p.get_variable_ids()
            if var_id not in substitution_map
        ]

        # Duplicate check, but we shouldn't need this.
        if len(solved_var_ids) != len(list(dict.fromkeys(solved_var_ids))):
            raise ValueError(
                "Duplicate variable IDs found in the solved variables. "
                "This should not happen."
            )

        # Create a map of the new, solved values.
        solved_values = {var_id: result.x[i] for i, var_id in enumerate(solved_var_ids)}

        # Update the full map with the solved values.
        final_variable_values.update(solved_values)

        # Handle any symbolic substitutions to ensure all variables are consistent.
        # (This part handles cases where, e.g., P2.x was substituted by P1.x)
        all_vars_with_dupes = [v for p in self.primitives for v in p.get_variable_ids()]
        all_vars = list(dict.fromkeys(all_vars_with_dupes))

        for var_id in all_vars:
            root = find(var_id, substitution_map)
            if root in final_variable_values:
                final_variable_values[var_id] = final_variable_values[root]

        return final_variable_values

    def assess_solver_result(
        self,
        final_variable_values: Dict[str, float],
        constraints_solved: List[Constraint],
    ) -> None:
        # Assess the quality of the result using the final variable map.

        # Recalculate residuals using only the geometric constraints that were solved.
        # This ignores the regularization terms included in `result.fun`.
        constraint_residuals = [
            c.get_residual(final_variable_values) for c in constraints_solved
        ]
        constraint_errors = [np.max(np.abs(r)) for r in constraint_residuals]
        geometric_residuals = np.concatenate(constraint_residuals)
        max_error = np.max(np.abs(geometric_residuals))

        if max_error > SOLVE_VALIDATION_TOLERANCE:
            # Find which constraint(s) failed the tolerance check.
            failing_constraints = []
            for i, (c, error) in enumerate(zip(constraints_solved, constraint_errors)):
                if error > SOLVE_VALIDATION_TOLERANCE:
                    points_involved = c.get_involved_primitive_ids()
                    failing_constraints.append(
                        (i, type(c).__name__, error, points_involved)
                    )

            # Log the constraint and points that failed.
            if failing_constraints:
                logger.warning(f"Solver failed tolerance. Max error: {max_error}")
                for i, c_type, error, points in failing_constraints:
                    logger.warning(
                        f"  - Constraint {i} ({c_type}) failed with error {error:.6f}"
                    )
                    logger.warning(f"    Points involved: {points}")

        return

    def update_primitives_from_map(
        self, final_variable_values: Dict[str, float]
    ) -> None:
        for p in self.primitives:
            if isinstance(p, Point):
                p.x = final_variable_values.get(f"{p.id}_x", p.x)
                p.y = final_variable_values.get(f"{p.id}_y", p.y)

            elif isinstance(p, Circle):
                # The circle primitive is responsible for its radius only.
                radius_var_id = p.get_variable_ids()[0]

                # Update the object's radius attribute with the solved value.
                p.radius = final_variable_values.get(radius_var_id, p.radius)

        logger.debug("Updated all primitives from final variable map.")
        return
