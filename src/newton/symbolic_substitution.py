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
        # Simple union: make one root the parent of the other.
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


def perform_symbolic_substitution(
    constraints: list[Constraint], primitives: list[Primitive]
) -> tuple[list[Constraint], dict[str, str]]:
    """
    Performs a symbolic substitution pass on a constraint system.

    This function interrogates the constraints to find equivalent variables.
    It returns a list of constraints that are still active (i.e., not
    fully redundant) and a substitution map that maps each substitutable
    variable to its root representative.

    Returns:
        (active_constraints, substitution_map)
        - active_constraints: A list of constraints for the solver.
        - substitution_map: A dict mapping var_id -> root_var_id.
    """
    primitive_map = {p.id: p for p in primitives}

    # https://www.geeksforgeeks.org/dsa/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/
    parent_map: dict[str, str] = {}
    substitution_map: dict[str, str] = {}
    constraints_to_skip: set[int] = set()

    # Build the equivalence sets using 'Union-Find'.
    for i, c in enumerate(constraints):
        match c:
            # Pure equality constraints: can be used for substitution and then skipped.
            case PointPointCoincident():
                for v1, v2 in zip(c.p1.get_variable_ids(), c.p2.get_variable_ids()):
                    union(v1, v2, parent_map)
                constraints_to_skip.add(i)

            case PointPointEuclideanDistance() if c.distance < EPS:
                for v1, v2 in zip(c.p1.get_variable_ids(), c.p2.get_variable_ids()):
                    union(v1, v2, parent_map)
                constraints_to_skip.add(i)

            # Partial equality constraints: use for substitution, but do not skip, because:
            # - They only establish equality for some coordinates, not all.
            # - The solver still needs the constraint equation to enforce the relationship.
            # - They represent geometric conditions that must be maintained even after substitution.

            case PointPointXDistance() if c.distance < EPS:
                p1_x_var = c.p1.get_variable_ids()[0]
                p2_x_var = c.p2.get_variable_ids()[0]
                union(p1_x_var, p2_x_var, parent_map)
                # Don't skip.

            case PointPointYDistance() if c.distance < EPS:
                p1_y_var = c.p1.get_variable_ids()[1]
                p2_y_var = c.p2.get_variable_ids()[1]
                union(p1_y_var, p2_y_var, parent_map)
                # Don't skip.

            case LineHorizontal():
                p1_y_var = c.line.p1.get_variable_ids()[1]
                p2_y_var = c.line.p2.get_variable_ids()[1]
                union(p1_y_var, p2_y_var, parent_map)
                # Don't skip.

            case LineVertical():
                p1_x_var = c.line.p1.get_variable_ids()[0]
                p2_x_var = c.line.p2.get_variable_ids()[0]
                union(p1_x_var, p2_x_var, parent_map)
                # Don't skip.

        # TODO: Handle other constraint types.

    # Now build the substitution map at the variable level.

    # Get all variables from all primitives.
    all_var_ids = {
        var_id for p in primitive_map.values() for var_id in p.get_variable_ids()
    }

    # For each variable, find its ultimate root/representative.
    for var_id in all_var_ids:
        root = find(var_id, parent_map)
        if root != var_id:
            substitution_map[var_id] = root

    if not substitution_map:
        logger.info("No substitutions found. Returning original constraints.")
        return constraints, {}

    logger.info(f"Found {len(substitution_map)} variable substitutions to perform.")
    logger.debug(f"Substitution map: {substitution_map}")

    # Rewrite the constraint list using the new primitive set.
    simplified_constraints: list[Constraint] = []
    for i, c in enumerate(constraints):
        if i in constraints_to_skip:
            continue
        new_c = rewrite_constraint(c, parent_map, primitive_map)
        simplified_constraints.append(new_c)

    return simplified_constraints, substitution_map
