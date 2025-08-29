from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Sequence, Set

from newton.constants import EPS
from newton.constraints import (
    Constraint,
    LineHorizontal,
    LineVertical,
    PointPointCoincident,
    PointPointEuclideanDistance,
    PointPointXDistance,
    PointPointYDistance,
)
from newton.logging_config import logger
from newton.primitives import Primitive


class SubstitutionAction(Enum):
    ELIMINATE = auto()  # Remove the constraint entirely.
    SUBSTITUTE_AND_KEEP = auto()  # Apply substitutions but keep the constraint.


@dataclass
class SubstitutionRule:
    # Represents a variable substitution rule.
    target_var: str
    replacement_var: str
    action: SubstitutionAction


@dataclass
class SubstitutionStats:
    # Statistics about the substitution process for reporting.
    def __init__(self):
        self.variables_eliminated = 0
        self.constraints_eliminated = 0
        self.constraints_unchanged = 0
        self.substitution_map_size = 0

    def report(self):
        # Log substitution statistics.
        if self.substitution_map_size == 0:
            logger.debug("No symbolic substitutions performed.")
            return

        logger.debug("Symbolic substitution results:")
        logger.debug(f" * {self.variables_eliminated} variables eliminated")
        logger.debug(f" * {self.constraints_eliminated} constraints eliminated")
        logger.debug(f" * {self.constraints_unchanged} constraints unchanged")


@dataclass
class SubstitutionResults:
    # Results from the substitution process.
    active_constraints: Sequence[Constraint]
    substitution_map: Dict[str, str]
    constraints_eliminated: int
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


def analyze_constraint(
    constraint: Constraint,
    index: int,
):
    """
    Analyse a single constraint to extract substitution rules.
    Returns list of rules this constraint creates.
    """
    rules = []
    constraints_to_eliminate = set()

    match constraint:
        # Pure equality constraints: can be used for substitution and then skipped.
        # --------------------------------------------------------------------------
        case PointPointCoincident():
            for v1, v2 in zip(
                constraint.p1.get_variable_ids(), constraint.p2.get_variable_ids()
            ):
                rules.append(SubstitutionRule(v1, v2, SubstitutionAction.ELIMINATE))

            constraints_to_eliminate.add(index)

        case PointPointEuclideanDistance() if constraint.distance < EPS:
            for v1, v2 in zip(
                constraint.p1.get_variable_ids(), constraint.p2.get_variable_ids()
            ):
                rules.append(SubstitutionRule(v1, v2, SubstitutionAction.ELIMINATE))

            constraints_to_eliminate.add(index)

        # Partial equality constraints, so no elimination, just substitution of variables:
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

        case PointPointYDistance() if constraint.distance < EPS:
            p1_y_var = constraint.p1.get_variable_ids()[1]
            p2_y_var = constraint.p2.get_variable_ids()[1]
            rules.append(
                SubstitutionRule(
                    p1_y_var, p2_y_var, SubstitutionAction.SUBSTITUTE_AND_KEEP
                )
            )

        case LineHorizontal():
            p1_y_var = constraint.line.p1.get_variable_ids()[1]
            p2_y_var = constraint.line.p2.get_variable_ids()[1]
            rules.append(
                SubstitutionRule(
                    p1_y_var, p2_y_var, SubstitutionAction.SUBSTITUTE_AND_KEEP
                )
            )

        case LineVertical():
            p1_x_var = constraint.line.p1.get_variable_ids()[0]
            p2_x_var = constraint.line.p2.get_variable_ids()[0]
            rules.append(
                SubstitutionRule(
                    p1_x_var, p2_x_var, SubstitutionAction.SUBSTITUTE_AND_KEEP
                )
            )

    # TODO: Handle other constraint types.

    return rules, constraints_to_eliminate


def perform_symbolic_substitution(
    constraints: Sequence[Constraint], primitives: Sequence[Primitive]
) -> SubstitutionResults:
    substitution_rules: List[SubstitutionRule] = []
    constraints_to_eliminate: Set[int] = set()
    parent_map: Dict[str, str] = {}
    stats = SubstitutionStats()

    # Analyse constraints for substitution opportunities.
    for i_constraint, constraint in enumerate(constraints):
        _rules, _constraints_to_eliminate = analyze_constraint(constraint, i_constraint)
        substitution_rules.extend(_rules)
        constraints_to_eliminate.update(_constraints_to_eliminate)

    # Build union-find from substitution rules.
    for rule in substitution_rules:
        union(rule.target_var, rule.replacement_var, parent_map)

    # Build final substitution map.
    all_vars = {var_id for p in primitives for var_id in p.get_variable_ids()}
    substitution_map = {}

    for var_id in all_vars:
        root = find(var_id, parent_map)
        if root != var_id:
            substitution_map[var_id] = root

    # Prune constraints.
    simplified_constraints = [
        c for idx, c in enumerate(constraints) if idx not in constraints_to_eliminate
    ]

    # Record stats.
    stats.substitution_map_size = len(substitution_map)
    stats.constraints_eliminated = len(constraints_to_eliminate)
    stats.constraints_unchanged = len(simplified_constraints)
    stats.variables_eliminated = len(substitution_map)

    return SubstitutionResults(
        active_constraints=simplified_constraints,
        substitution_map=substitution_map,
        constraints_eliminated=stats.constraints_eliminated,
        constraints_unchanged=stats.constraints_unchanged,
    )
