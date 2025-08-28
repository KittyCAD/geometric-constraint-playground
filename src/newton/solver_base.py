import logging
from abc import ABC, abstractmethod
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import OptimizeResult

from newton.backend import Vector
from newton.constants import CONFIG_USE_SYMB_SUB, NONZERO_RANK_TOLERANCE
from newton.constraint_validator import ConstraintValidator
from newton.constraints import (
    Constraint,
    PointFixed,
)
from newton.logging_config import logger
from newton.matrix_utils import compute_rank
from newton.primitives import Circle, Point, Primitive
from newton.symbolic_substitution import (
    SubstitutionStats,
    find,
    perform_symbolic_substitution,
)

SOLVE_VALIDATION_TOLERANCE = 1e-6  # Our maximum allowed error on any constraint.
SOLVER_CONVERGENCE_TOLERANCE = 1e-10  # The tolerance for convergence in the solver.


class SystemState(Enum):
    UNDERDETERMINED = "underdetermined"
    OVERDETERMINED = "overdetermined"
    FULLY_DETERMINED = "fully determined"


# Configure numpy print options for debug output
if logger.isEnabledFor(logging.DEBUG):
    np.set_printoptions(precision=3, suppress=True, linewidth=120)


class Solver2D(ABC):
    def __init__(
        self, primitives: Sequence[Primitive], constraints: Sequence[Constraint]
    ):
        # Build our full set of constraints: this will be the user-defined constraints
        # passed in, plus our definitional constraints from the primitives.
        all_constraints = list(constraints)
        for p in primitives:
            all_constraints.extend(p.build_definitional_constraints())

        # Store the primitives and constraints.
        self.primitives = primitives
        self.primitive_map = {p.id: p for p in primitives}
        self.original_constraints: Sequence[Constraint] = all_constraints
        self.active_constraints: Sequence[Constraint] = []
        self.substitution_map: Dict[str, str] = {}
        self.substitution_stats = SubstitutionStats()

        self.module: ModuleType = (
            np  # Default to numpy, can be overridden by subclasses.
        )

    def prepare_constraints(self) -> None:
        # Prepare constraints for solving, applying substitutions if enabled.
        if CONFIG_USE_SYMB_SUB:
            self.apply_symbolic_substitution()
        else:
            self.active_constraints = self.original_constraints
            self.substitution_map = {}
            logger.info("Symbolic substitution disabled.")

    def apply_symbolic_substitution(self) -> None:
        # Apply substitution.
        results = perform_symbolic_substitution(
            self.original_constraints, self.primitives
        )

        # Extract results.
        self.active_constraints = results.active_constraints
        self.substitution_map = results.substitution_map

        # Use the detailed statistics from the substitution process.
        self.substitution_stats.constraints_eliminated = results.constraints_eliminated
        self.substitution_stats.constraints_unchanged = results.constraints_unchanged
        self.substitution_stats.variables_eliminated = len(self.substitution_map)
        self.substitution_stats.substitution_map_size = len(self.substitution_map)

        # Report results.
        self.substitution_stats.report()
        logger.debug(f"Substitution map: {self.substitution_map}")

    def identify_free_primitives(
        self, primitives_to_check: List[Primitive]
    ) -> List[Primitive]:
        # Find the non-fixed elements our solver can play tunes with.
        # However... we actually do want to feed fixed points to the structural analyzer,
        # because they may be help in solving other parts of the system.
        fixed_primitive_ids = set()
        for c in self.active_constraints:
            if isinstance(c, PointFixed):
                fixed_primitive_ids.add(c.point.id)

        return [p for p in primitives_to_check if p.id not in fixed_primitive_ids]

    def build_dependency_graph(self) -> nx.Graph:
        graph = nx.Graph()

        # Add all variables from all primitives as nodes.
        for p in self.primitives:
            for var_id in p.get_variable_ids():
                graph.add_node(var_id, bipartite=0)  # Variable nodes

        # Add constraint nodes and connect them to the variables they affect.
        for i, c in enumerate(self.active_constraints):  # Use active_constraints
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
            # Gather all primitives involved in a component.
            i_comp_constraints = {
                graph.nodes[node]["constraint_index"]
                for node in component
                if graph.nodes[node].get("bipartite") == 1
            }
            if not i_comp_constraints:
                continue

            comp_constraints = [self.active_constraints[i] for i in i_comp_constraints]

            # Then, gather all unique primitives touched by these constraints.
            comp_primitive_ids = set()
            for c in comp_constraints:
                comp_primitive_ids.update(c.get_involved_primitive_ids())

            comp_primitives = [
                self.primitive_map[pid]
                for pid in comp_primitive_ids
                if pid in self.primitive_map
            ]

            constraint_systems.append(
                {
                    "constraints": comp_constraints,
                    "primitives": comp_primitives,
                    "substitution_map": self.substitution_map,  # Include substitution map.
                }
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
        rank = compute_rank(jacobian, tolerance)

        # Determine system state based on rank vs variables.
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

    def get_independent_variables(self, free_primitives: List[Primitive]) -> List[str]:
        """
        Get the list of independent variables for solving.
        These are variables from free primitives that have not been substituted.
        """
        independent_vars = []

        for primitive in free_primitives:
            for var_id in primitive.get_variable_ids():
                # Only include variables that aren't substituted by others.
                if var_id not in self.substitution_map:
                    independent_vars.append(var_id)

        # Ensure deterministic order.
        independent_vars.sort()
        return independent_vars

    def find_root_variable(self, var_id: str) -> str:
        #  Find the root variable in the substitution chain.
        return find(var_id, self.substitution_map)

    def get_all_variable_ids(self) -> List[str]:
        # Get all variable IDs from all primitives.
        all_vars = []
        for primitive in self.primitives:
            all_vars.extend(primitive.get_variable_ids())
        return all_vars

    @abstractmethod
    def solve_constraint_system(self, system: Dict[str, Any]):
        # Each concrete solver must implement its own system solving logic.
        pass

    def solve(self):
        if not self.original_constraints:
            logger.info("No constraints to solve.")
            return

        # Step 1: Prepare constraints (apply substitutions if enabled).
        self.prepare_constraints()

        # Step 2: Split the problem into wholly disconnected problems.
        graph = self.build_dependency_graph()
        constraint_systems = self.find_disconnected_systems(graph)

        # Step 3: Validate the constraints in each disconnected system before solving.
        self.validate_constraint_systems(constraint_systems)

        # Step 4: If validation passes, proceed with the numerical solve.
        logger.debug(
            f"Graph analysis found {len(constraint_systems)} valid disconnected system(s)."
        )
        logger.info(f"Using {self.__class__.__name__}.")

        # Step 5: Then, for each separable system, solve.
        for i, system in enumerate(constraint_systems):
            if not system["constraints"]:
                continue

            free_primitives = self.identify_free_primitives(system["primitives"])
            if free_primitives:
                # Add free_primitives to the system dict for the concrete solver.
                system["free_primitives"] = free_primitives
                logger.debug(f"Solving system {i + 1}/{len(constraint_systems)}")
                self.solve_constraint_system(system)

    def build_variable_values_map(
        self,
        independent_vars_values: Vector,
        independent_vars: List[str],
        initial_values: Dict[str, float],
        substitution_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build a complete mapping of all variable IDs to their current values.

        This is used during solve iterations to provide a complete set of variable
        state vals to constraint evaluation functions.

        Args:
            independent_vars_values: Current values of variables being solved for.
            independent_vars: Ordered list of independent variable IDs.
            initial_values: Initial values for all variables in the system.
            substitution_map: Map from substituted variables to their root variables.

        Returns:
            Complete mapping of all variable IDs to their current values.
        """
        # Start with the initial state of all variables in the system.
        variable_values: Dict[str, Any] = dict(initial_values)

        # Create a dictionary of the variables the solver is currently working on.
        solved_vars = {
            var_id: independent_vars_values[i]
            for i, var_id in enumerate(independent_vars)
        }

        # Update the initial values with the current solved values.
        variable_values.update(solved_vars)

        # Apply symbolic substitutions to ensure consistency.
        for var_id, root_id in substitution_map.items():
            if root_id in variable_values:
                variable_values[var_id] = variable_values[root_id]

        return variable_values

    def fan_out_solved_variable_values(
        self,
        result: OptimizeResult,
        solved_var_ids: List[str],
        substitution_map: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Distribute the solved variable values through the substitution map, ensuring
        that all variables, including those eliminated via substitution, are updated and
        have a type we can work with.
        """
        # Get initial values for all variables.
        initial_values = self.get_initial_values_for_all_variables()

        # Use our consolidated method to build the complete map.
        # Convert result.x to ensure we have the right array type.
        final_values = self.build_variable_values_map(
            result.x, solved_var_ids, initial_values, substitution_map
        )

        # Ensure all values are Python floats (not JAX tracers or numpy scalars).
        fanned_out_values = {
            var_id: float(value) for var_id, value in final_values.items()
        }
        return fanned_out_values

    def get_initial_values_for_all_variables(self) -> Dict[str, float]:
        """
        Get initial values for all variables across all primitives.
        """
        initial_values: Dict[str, float] = {}
        for p in self.primitives:
            initial_values.update(p.get_initial_variable_values())

        return initial_values

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
