from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Sequence, Set

from newton.constants import EPS
from newton.constraints import (
    Constraint,
    LineHorizontal,
    LinesParallel,
    LineVertical,
    PointPointCoincident,
    PointPointEuclideanDistance,
    PointPointXDistance,
    PointPointYDistance,
)
from newton.logging_config import logger
from newton.primitives import Line, Point, Primitive


class SubstitutionAction(Enum):
    ELIMINATE = auto()  # Remove the constraint entirely.
    SUBSTITUTE_AND_KEEP = auto()  # Apply substitutions but keep the constraint.
    KEEP_UNCHANGED = auto()  # No substitutions apply, keep as-is.


@dataclass
class SubstitutionRule:
    # Represents a variable substitution rule.
    target_var: str  # Variable to be substituted.
    replacement_var: str  # Variable to substitute with.
    action: SubstitutionAction  # What to do with the constraint that created this rule.


@dataclass
class SubstitutionStats:
    # Statistics about the substitution process for reporting.
    def __init__(self):
        self.variables_eliminated = 0
        self.constraints_eliminated = 0
        self.constraints_rewritten = 0
        self.constraints_unchanged = 0
        self.substitution_map_size = 0

    def report(self):
        # Log substitution statistics.
        if self.substitution_map_size == 0:
            logger.debug("No symbolic substitutions performed.")
            return

        logger.debug("Symbolic substitution results:")
        logger.debug(f"  • {self.variables_eliminated} variables eliminated")
        logger.debug(f"  • {self.constraints_eliminated} constraints eliminated")
        logger.debug(f"  • {self.constraints_rewritten} constraints rewritten")
        logger.debug(f"  • {self.constraints_unchanged} constraints unchanged")


@dataclass
class SubstitutionResults:
    # Results from the substitution process.
    active_constraints: Sequence[Constraint]
    substitution_map: Dict[str, str]
    constraints_eliminated: int
    constraints_rewritten: int
    constraints_unchanged: int


def find(var_id: str, parent_map: dict[str, str]) -> str:
    # Find the root of the set.
    root = var_id
    while root in parent_map:
        root = parent_map[root]

    # Make every node in the path point directly to the root.
    current = var_id
    while current in parent_map:
        parent = parent_map[current]
        parent_map[current] = root
        current = parent

    return root


def union(var1_id: str, var2_id: str, parent_map: dict[str, str]):
    root1 = find(var1_id, parent_map)
    root2 = find(var2_id, parent_map)

    if root1 != root2:
        # Simple union: make one root the parent of the other. Choose the lexicographically
        # smaller one as the new root for consistency.
        if root1 < root2:
            parent_map[root2] = root1
        else:
            parent_map[root1] = root2


def get_substituted_primitive(
    p: Primitive,
    parent_map: dict[str, str],
    primitive_map: dict[str, Primitive],
) -> Primitive:
    # Finds the single primitive that `p` has been substituted with.
    # It's used to rewrite constraints that require a full primitive object.

    # Find the root primitive ID for the first variable of p. This is our candidate
    # for the new representative primitive.
    variable_ids = p.get_variable_ids()
    if not variable_ids:
        return p

    first_var_root = find(variable_ids[0], parent_map)
    representative_prim_id = first_var_root.split("_")[0]

    # Check if all other variables of p map to this same representative primitive.
    for var_id in variable_ids[1:]:
        root = find(var_id, parent_map)
        prim_id = root.split("_")[0]
        if prim_id != representative_prim_id:
            # This is a partial substitution (e.g., only x-coords are equal).
            # We cannot rewrite the constraint with a single new primitive, so we return
            # the original.
            return p

    return primitive_map[representative_prim_id]


def rewrite_constraint(
    c: Constraint, parent_map: dict[str, str], primitive_map: dict[str, Primitive]
) -> Constraint:
    # Creates a new constraint object with substituted primitives.
    match c:
        case (
            PointPointEuclideanDistance()
            | PointPointXDistance()
            | PointPointYDistance()
        ):
            p1_sub = get_substituted_primitive(c.p1, parent_map, primitive_map)
            p2_sub = get_substituted_primitive(c.p2, parent_map, primitive_map)

            if not isinstance(p1_sub, Point) or not isinstance(p2_sub, Point):
                return c

            return type(c)(p1=p1_sub, p2=p2_sub, distance=c.distance)

        case LineHorizontal() | LineVertical():
            p1_sub = get_substituted_primitive(c.line.p1, parent_map, primitive_map)
            p2_sub = get_substituted_primitive(c.line.p2, parent_map, primitive_map)

            if not isinstance(p1_sub, Point) or not isinstance(p2_sub, Point):
                return c

            new_line = Line(p1=p1_sub, p2=p2_sub, id=c.line.id)
            return type(c)(line=new_line)

        case LinesParallel():
            l1_p1_sub = get_substituted_primitive(c.line1.p1, parent_map, primitive_map)
            l1_p2_sub = get_substituted_primitive(c.line1.p2, parent_map, primitive_map)
            l2_p1_sub = get_substituted_primitive(c.line2.p1, parent_map, primitive_map)
            l2_p2_sub = get_substituted_primitive(c.line2.p2, parent_map, primitive_map)

            if (
                not isinstance(l1_p1_sub, Point)
                or not isinstance(l1_p2_sub, Point)
                or not isinstance(l2_p1_sub, Point)
                or not isinstance(l2_p2_sub, Point)
            ):
                return c

            new_line1 = Line(p1=l1_p1_sub, p2=l1_p2_sub, id=c.line1.id)
            new_line2 = Line(p1=l2_p1_sub, p2=l2_p2_sub, id=c.line2.id)
            return LinesParallel(line1=new_line1, line2=new_line2)

        case _:
            # Default case - return the original constraint
            return c

    # TODO: Add other constraint types.

    return c


class SymbolicSubstitution:
    """
    Handles symbolic substitution with clear separation between:
    1. Rule discovery (which constraints create substitution rules).
    2. Rule application (how to apply rules to other constraints).
    3. Constraint filtering (which constraints to keep/eliminate).
    """

    def __init__(self):
        self.parent_map: dict[str, str] = {}
        self.substitution_rules: List[SubstitutionRule] = []
        self.constraints_to_eliminate: Set[int] = set()
        self.constraints_rewritten_count = 0

    def analyze_constraint(
        self, constraint: Constraint, index: int
    ) -> List[SubstitutionRule]:
        """
        Analyse a single constraint to extract substitution rules.
        Returns list of rules this constraint creates.
        """
        rules = []

        match constraint:
            # Pure equality constraints: can be used for substitution and then skipped.
            # --------------------------------------------------------------------------
            case PointPointCoincident():
                for v1, v2 in zip(
                    constraint.p1.get_variable_ids(), constraint.p2.get_variable_ids()
                ):
                    rules.append(SubstitutionRule(v1, v2, SubstitutionAction.ELIMINATE))

                self.constraints_to_eliminate.add(index)

            case PointPointEuclideanDistance() if constraint.distance < EPS:
                for v1, v2 in zip(
                    constraint.p1.get_variable_ids(), constraint.p2.get_variable_ids()
                ):
                    rules.append(SubstitutionRule(v1, v2, SubstitutionAction.ELIMINATE))

                self.constraints_to_eliminate.add(index)

            # Partial equality constraints: use for substitution, but do not skip, because:
            # - They only establish equality for some coordinates, not all.
            # - The solver still needs the constraint equation to enforce the relationship.
            # - They represent geometric conditions that must be maintained even after substitution.
            # --------------------------------------------------------------------------
            case PointPointXDistance() if constraint.distance < EPS:
                p1_x_var = constraint.p1.get_variable_ids()[0]
                p2_x_var = constraint.p2.get_variable_ids()[0]
                rules.append(
                    SubstitutionRule(
                        p1_x_var, p2_x_var, SubstitutionAction.SUBSTITUTE_AND_KEEP
                    )
                )
                # Don't skip.

            case PointPointYDistance() if constraint.distance < EPS:
                p1_y_var = constraint.p1.get_variable_ids()[1]
                p2_y_var = constraint.p2.get_variable_ids()[1]
                rules.append(
                    SubstitutionRule(
                        p1_y_var, p2_y_var, SubstitutionAction.SUBSTITUTE_AND_KEEP
                    )
                )
                # Don't skip.

            case LineHorizontal():
                p1_y_var = constraint.line.p1.get_variable_ids()[1]
                p2_y_var = constraint.line.p2.get_variable_ids()[1]
                rules.append(
                    SubstitutionRule(
                        p1_y_var, p2_y_var, SubstitutionAction.SUBSTITUTE_AND_KEEP
                    )
                )
                # Don't skip.

            case LineVertical():
                p1_x_var = constraint.line.p1.get_variable_ids()[0]
                p2_x_var = constraint.line.p2.get_variable_ids()[0]
                rules.append(
                    SubstitutionRule(
                        p1_x_var, p2_x_var, SubstitutionAction.SUBSTITUTE_AND_KEEP
                    )
                )
                # Don't skip.

        # TODO: Handle other constraint types.

        return rules

    def build_substitution_map(
        self, constraints: Sequence[Constraint], primitives: Sequence[Primitive]
    ) -> Dict[str, str]:
        """
        Build the final substitution map using union-find.
        Returns mapping from variable_id -> canonical_variable_id.
        """
        # Extract all substitution rules.
        for i, constraint in enumerate(constraints):
            rules = self.analyze_constraint(constraint, i)
            self.substitution_rules.extend(rules)

        # Build union-find structure using the original union/find functions
        # https://www.geeksforgeeks.org/dsa/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/
        for rule in self.substitution_rules:
            union(rule.target_var, rule.replacement_var, self.parent_map)

        # Build final substitution map
        all_variables = self._get_all_variables(primitives)
        substitution_map = {}

        # Now build the substitution map at the variable level.
        # For each variable, find its ultimate root/representative.
        for var_id in all_variables:
            root = find(var_id, self.parent_map)
            if root != var_id:
                substitution_map[var_id] = root

        return substitution_map

    def apply_substitutions(
        self, constraints: Sequence[Constraint], primitives: Sequence[Primitive]
    ) -> SubstitutionResults:
        # Apply symbolic substitution and return nice, structured results.
        original_constraint_count = len(constraints)
        substitution_map = self.build_substitution_map(constraints, primitives)

        if not substitution_map:
            logger.info("No substitutions found. Returning original constraints.")
            return SubstitutionResults(
                active_constraints=constraints,
                substitution_map={},
                constraints_eliminated=0,
                constraints_rewritten=0,
                constraints_unchanged=len(constraints),
            )

        logger.info(f"Found {len(substitution_map)} variable substitutions to perform.")
        logger.debug(f"Substitution map: {substitution_map}")

        # Rewrite the constraint list using the new primitive set.
        simplified_constraints: list[Constraint] = []
        primitive_map = {p.id: p for p in primitives}

        for i, constraint in enumerate(constraints):
            if i in self.constraints_to_eliminate:
                logger.debug(f"Eliminating constraint {i}: {type(constraint).__name__}")
                continue

            # Apply substitutions to this constraint.
            new_constraint = rewrite_constraint(
                constraint, self.parent_map, primitive_map
            )
            simplified_constraints.append(new_constraint)

            # Track if this constraint was actually rewritten.
            if new_constraint is not constraint:
                self.constraints_rewritten_count += 1

        constraints_eliminated = original_constraint_count - len(simplified_constraints)
        constraints_unchanged = (
            len(simplified_constraints) - self.constraints_rewritten_count
        )

        return SubstitutionResults(
            active_constraints=simplified_constraints,
            substitution_map=substitution_map,
            constraints_eliminated=constraints_eliminated,
            constraints_rewritten=self.constraints_rewritten_count,
            constraints_unchanged=constraints_unchanged,
        )

    def _get_all_variables(self, primitives: Sequence[Primitive]) -> Set[str]:
        # Get all variables from all primitives.
        all_var_ids = {var_id for p in primitives for var_id in p.get_variable_ids()}
        return all_var_ids


def perform_symbolic_substitution(
    constraints: Sequence[Constraint], primitives: Sequence[Primitive]
) -> SubstitutionResults:
    substitution = SymbolicSubstitution()
    result = substitution.apply_substitutions(constraints, primitives)
    return result
